use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::sync::{Arc, atomic::AtomicI32};

use crate::{
    CommandBuffer, CommandRecorder, Device, Extent2D, Format, GPUError, Image, ImageLayout,
    ImageLayoutTransition, ImageView, PipelineStageFlags, Queue, Semaphore, SubmitInfo,
    SurfaceFormat,
    macros::vk_value_enum,
    raw::{DeviceImpl, ImageImpl, ImageViewImpl, QueueImpl, RawAdapter, RawDevice, SemaphoreImpl},
};

/// How many CPU frame slots to record ahead of the GPU. Two is enough to keep
/// the GPU busy while the next frame is recorded, without adding input latency.
const PREFERRED_FRAMES_IN_FLIGHT: usize = 2;

vk_value_enum! {
    /// How a presented image is scheduled against the display.
    PresentMode: vk::PresentModeKHR,
    default = Fifo => FIFO,
    {
        Immediate => IMMEDIATE,
        Mailbox => MAILBOX,
        FifoRelaxed => FIFO_RELAXED,
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Frame {
    pub index: u32,
    pub flight_index: usize,
    pub suboptimal: bool,
    pub out_of_date: bool,
}

pub struct Swapchain {
    pub inner: SwapchainImpl,
    /// Timeline value the queue reached after the last submission for each
    /// ring slot. `0` means "nothing submitted in this slot yet".
    frame_values: Vec<u64>,
    needs_recreate: bool,
}

/// How to pick a surface format from the ones the surface reports.
pub enum FormatSelector {
    /// Prefer sRGB formats, falling back to linear, then to the first entry.
    SrgbFirst,
    /// Prefer linear (UNORM) formats, falling back to the first entry.
    LinearFirst,
    /// Prefer a specific format, falling back to the first entry.
    Preferred(Format),
    /// Anything else.
    Custom(Box<dyn Fn(&[SurfaceFormat]) -> SurfaceFormat>),
}

impl Default for FormatSelector {
    fn default() -> Self {
        Self::SrgbFirst
    }
}

impl FormatSelector {
    pub fn select(&self, formats: &[SurfaceFormat]) -> SurfaceFormat {
        if formats.is_empty() {
            return SurfaceFormat::default();
        }

        let find = |candidates: &[Format]| {
            candidates.iter().find_map(|wanted| {
                formats
                    .iter()
                    .find(|candidate| {
                        candidate.format == *wanted
                            && candidate.color_space == crate::ColorSpace::SRGB_NONLINEAR
                    })
                    .or_else(|| formats.iter().find(|candidate| candidate.format == *wanted))
                    .copied()
            })
        };

        match self {
            Self::SrgbFirst => find(&[
                Format::Bgra8UnormSrgb,
                Format::Rgba8UnormSrgb,
                Format::Bgra8Unorm,
                Format::Rgba8Unorm,
            ])
            .unwrap_or(formats[0]),
            Self::LinearFirst => find(&[
                Format::Bgra8Unorm,
                Format::Rgba8Unorm,
                Format::Bgra8UnormSrgb,
                Format::Rgba8UnormSrgb,
            ])
            .unwrap_or(formats[0]),
            Self::Preferred(preferred) => *formats
                .iter()
                .find(|candidate| candidate.format == *preferred)
                .unwrap_or(&formats[0]),
            Self::Custom(selector) => selector(formats),
        }
    }
}

pub struct SwapchainCreateInfo {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
    pub preferred_extent: Extent2D,
    pub preferred_image_count: usize,
    pub preferred_present_mode: PresentMode,
    pub format_selector: FormatSelector,
}

impl SwapchainCreateInfo {
    pub fn new(display: RawDisplayHandle, window: RawWindowHandle, extent: Extent2D) -> Self {
        Self {
            display,
            window,
            preferred_extent: extent,
            preferred_image_count: 3,
            // FIFO is the only mode Vulkan guarantees, and the only one that
            // is vsync'd everywhere. Asking for MAILBOX on a driver that lacks
            // it lands in IMMEDIATE, which tears.
            preferred_present_mode: PresentMode::Fifo,
            format_selector: FormatSelector::default(),
        }
    }
}

#[derive(Debug)]
pub struct SwapchainImplResources {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub views: Vec<ImageView>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub extent: Extent2D,
    pub present_mode: PresentMode,
    pub device: RawDevice,
}

pub struct SwapchainImpl {
    pub device: RawDevice,
    pub adapter: RawAdapter,
    pub loader: ash::khr::swapchain::Device,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,

    pub available: Vec<Semaphore>,
    pub finished: Vec<Semaphore>,
    pub frame: usize,

    pub resources: SwapchainImplResources,

    /// Number of CPU frame slots recorded ahead of the GPU. Capped at the
    /// swapchain image count; 2 is enough to keep the GPU busy without adding
    /// input latency.
    pub frames_in_flight: usize,
    pub preferred_extent: Extent2D,
    pub preferred_image_count: u32,
    pub preferred_present_mode: PresentMode,
    pub formats: Arc<[SurfaceFormat]>,
    pub format: SurfaceFormat,
    pub present_modes: Arc<[PresentMode]>,
}

impl SwapchainImpl {
    fn new(device: RawDevice, info: &SwapchainCreateInfo) -> Result<Self, GPUError> {
        let adapter = device.adapter.clone();
        let surface = match Self::create_surface(&device, info.display, info.window) {
            Ok(surface) => surface,
            Err(e) => panic!("{:?}", e),
        };

        let surface_loader =
            ash::khr::surface::Instance::new(&device.instance.entry, &device.instance.handle);
        let loader = ash::khr::swapchain::Device::new(&device.instance.handle, &device.handle);

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(adapter.handle, surface)
                .map_err(GPUError::from)?
        };
        let formats = formats
            .into_iter()
            .map(SurfaceFormat::from)
            .collect::<Vec<_>>();
        let format = info.format_selector.select(&formats);
        log::info!("format: {:?}", format);

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(adapter.handle, surface)
                .map_err(GPUError::from)?
        };
        let present_modes = present_modes
            .into_iter()
            .map(PresentMode::from)
            .collect::<Vec<_>>();

        let resources = Self::create_resources(
            device.clone(),
            &loader,
            surface,
            &surface_loader,
            adapter.handle,
            info.preferred_extent,
            info.preferred_image_count as u32,
            info.preferred_present_mode,
            &present_modes,
            format,
        )?;
        // Presentation parallelism vs CPU parallelism are separate knobs: the
        // swapchain may own 3 images while we only keep 2 frames in flight.
        let frames_in_flight = PREFERRED_FRAMES_IN_FLIGHT.min(resources.images.len());

        let (available, finished) =
            Self::create_syncs(device.clone(), frames_in_flight, resources.images.len());

        let new = Self {
            device,
            adapter,
            loader,
            surface,
            surface_loader,

            available,
            finished,
            frame: 0,

            resources,

            frames_in_flight,
            preferred_extent: info.preferred_extent,
            preferred_image_count: info.preferred_image_count as u32,
            preferred_present_mode: info.preferred_present_mode,
            formats: Arc::from(formats),
            format,
            present_modes: Arc::from(present_modes),
        };

        Ok(new)
    }

    /// `available` (acquire) semaphores are per in-flight slot; `finished`
    /// (present) semaphores are per *image*, because `vkAcquireNextImageKHR`
    /// can hand images back out of order.
    fn create_syncs(
        device: RawDevice,
        frames_in_flight: usize,
        image_count: usize,
    ) -> (Vec<Semaphore>, Vec<Semaphore>) {
        let mut available = Vec::with_capacity(frames_in_flight);
        let mut finished = Vec::with_capacity(image_count);

        for _ in 0..frames_in_flight {
            available.push(Semaphore {
                inner: Arc::new(unsafe { SemaphoreImpl::new_signal(device.clone()) }),
            });
        }

        for _ in 0..image_count {
            finished.push(Semaphore {
                inner: Arc::new(unsafe { SemaphoreImpl::new_signal(device.clone()) }),
            });
        }

        (available, finished)
    }

    fn create_resources(
        device: RawDevice,
        loader: &ash::khr::swapchain::Device,
        surface_handle: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        adapter_handle: vk::PhysicalDevice,
        preferred_extent: Extent2D,
        preferred_image_count: u32,
        preferred_present_mode: PresentMode,
        present_modes: &[PresentMode],
        format: SurfaceFormat,
    ) -> Result<SwapchainImplResources, GPUError> {
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(adapter_handle, surface_handle)
                .map_err(GPUError::from)?
        };

        let min_images = capabilities.min_image_count;
        let image_count = if capabilities.max_image_count == 0 {
            preferred_image_count.max(min_images)
        } else {
            preferred_image_count
                .max(min_images)
                .min(capabilities.max_image_count)
        };

        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: preferred_extent
                    .width
                    .max(capabilities.min_image_extent.width)
                    .min(capabilities.max_image_extent.width),
                height: preferred_extent
                    .height
                    .max(capabilities.min_image_extent.height)
                    .min(capabilities.max_image_extent.height),
            }
        } else {
            capabilities.current_extent
        };

        let present_mode = if present_modes.contains(&preferred_present_mode) {
            preferred_present_mode
        } else {
            // The requested mode carries intent: FIFO/FIFO_RELAXED mean "sync to
            // vblank", MAILBOX/IMMEDIATE mean "present as fast as possible".
            // Fall back within the same intent rather than silently switching
            // vsync on or off.
            let wants_vsync = matches!(
                preferred_present_mode,
                PresentMode::Fifo | PresentMode::FifoRelaxed
            );

            let preferred_fallbacks: &[PresentMode] = if wants_vsync {
                &[PresentMode::Fifo, PresentMode::FifoRelaxed]
            } else {
                &[PresentMode::Immediate, PresentMode::Mailbox]
            };

            let fallback = preferred_fallbacks
                .iter()
                .copied()
                .find(|mode| present_modes.contains(mode))
                .or_else(|| {
                    present_modes
                        .iter()
                        .copied()
                        .find(|&mode| mode == PresentMode::Fifo)
                })
                .or_else(|| present_modes.first().copied())
                .unwrap_or(PresentMode::Fifo);

            log::warn!(
                "Present mode {:?} not supported, falling back to {:?}",
                preferred_present_mode,
                fallback
            );
            fallback
        };

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_handle)
            .min_image_count(image_count)
            .image_format(format.format.into())
            .image_color_space(format.color_space.into())
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode.into())
            .clipped(true);

        let handle = unsafe {
            loader
                .create_swapchain(&info, None)
                .map_err(GPUError::from)?
        };

        let images = unsafe {
            loader
                .get_swapchain_images(handle)
                .map_err(GPUError::from)?
        };

        let images = images
            .iter()
            .copied()
            .map(|handle| Image {
                format: format.format,
                inner: Arc::new(ImageImpl {
                    handle,
                    device: device.clone(),
                    allocation: None,
                    current_layout: AtomicI32::new(vk::ImageLayout::UNDEFINED.as_raw()),
                }),
            })
            .collect::<Vec<_>>();

        let views = Self::create_image_views(device.clone(), &images, format);

        let resources = SwapchainImplResources {
            handle,
            images,
            views,
            capabilities,
            extent: extent.into(),
            present_mode,
            device,
        };

        Ok(resources)
    }

    fn create_image_views(
        device: RawDevice,
        images: &[Image],
        format: SurfaceFormat,
    ) -> Vec<ImageView> {
        images
            .iter()
            .map(|img| unsafe {
                let info = vk::ImageViewCreateInfo::default()
                    .image(img.inner.handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format.into())
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        base_array_layer: 0,
                        base_mip_level: 0,
                    });

                let handle = device
                    .handle
                    .create_image_view(&info, None)
                    .expect("Create Image View");

                ImageView {
                    sampler: None,
                    inner: Arc::new(ImageViewImpl {
                        handle,
                        device: device.clone(),
                        image: img.inner.clone(),
                    }),
                }
            })
            .collect()
    }

    fn create_surface(
        device: &DeviceImpl,
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<vk::SurfaceKHR, GPUError> {
        unsafe {
            ash_window::create_surface(
                &device.instance.entry,
                &device.instance.handle,
                display,
                window,
                None,
            )
            .map_err(GPUError::from)
        }
    }

    pub fn acquire_next(&mut self, timeout: Option<u64>) -> Result<Frame, GPUError> {
        let flight_index = self.frame;
        let available_semaphore = &self.available[flight_index];

        let timeout_ns = timeout.unwrap_or(u64::MAX);
        let (image_index, suboptimal) = unsafe {
            match self.loader.acquire_next_image(
                self.resources.handle,
                timeout_ns,
                available_semaphore.inner.handle,
                vk::Fence::null(),
            ) {
                Ok((idx, suboptimal)) => (idx, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Ok(Frame {
                        index: 0,
                        flight_index,
                        suboptimal: true,
                        out_of_date: true,
                    });
                }
                Err(e) => return Err(GPUError::Vulkan(e)),
            }
        };

        Ok(Frame {
            index: image_index,
            flight_index,
            suboptimal,
            out_of_date: false,
        })
    }

    pub fn image(&self, frame: Frame) -> &Image {
        &self.resources.images[frame.index as usize]
    }

    pub fn view(&self, frame: Frame) -> &ImageView {
        &self.resources.views[frame.index as usize]
    }

    pub fn present(&mut self, queue: &QueueImpl, frame: Frame) -> Result<bool, GPUError> {
        if frame.out_of_date {
            return Ok(true);
        }

        let finished_semaphore = &self.finished[frame.index as usize];

        let swapchains = [self.resources.handle];
        let image_indices = [frame.index];
        let wait_semaphores = [finished_semaphore.inner.handle];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let result = unsafe { self.loader.queue_present(queue.handle, &present_info) };
        // SUBOPTIMAL_KHR means "still presentable, but a new swapchain would be
        // better". Rebuilding on it recreates the swapchain every frame on
        // drivers that report it persistently (which looks like flicker), so
        // only OUT_OF_DATE_KHR triggers a rebuild.
        let needs_recreation = match result {
            Ok(_) => false,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(e) => return Err(GPUError::Vulkan(e)),
        };

        // The image is in PRESENT_SRC_KHR from here on, whether or not the
        // caller recorded an explicit transition to it.
        self.resources.images[frame.index as usize].set_layout(ImageLayout::Present);

        self.frame = (self.frame + 1) % self.frames_in_flight;
        Ok(needs_recreation)
    }

    pub fn available_semaphore(&self, frame: Frame) -> &Semaphore {
        &self.available[frame.flight_index]
    }

    pub fn finished_semaphore(&self, frame: Frame) -> &Semaphore {
        &self.finished[frame.index as usize]
    }

    pub fn recreate(&mut self) -> Result<(), GPUError> {
        log::debug!(
            "recreating swapchain: {}x{}",
            self.preferred_extent.width,
            self.preferred_extent.height
        );
        unsafe { self.device.wait_idle() };

        // Destroy the old swapchain before creating the new one, and never pass
        // it as `oldSwapchain`: the image views must die before the images they
        // point at, and MoltenVK's layer miscarries a swapchain handed back
        // through `oldSwapchain` (it presents undefined images afterwards).
        let old_views = std::mem::take(&mut self.resources.views);
        drop(old_views);
        unsafe {
            self.loader.destroy_swapchain(self.resources.handle, None);
        }

        let new = Self::create_resources(
            self.device.clone(),
            &self.loader,
            self.surface,
            &self.surface_loader,
            self.device.adapter.handle,
            self.preferred_extent,
            self.preferred_image_count,
            self.preferred_present_mode,
            self.present_modes.as_ref(),
            self.format,
        )?;

        let frames_in_flight = PREFERRED_FRAMES_IN_FLIGHT.min(new.images.len());
        let (available, finished) =
            Self::create_syncs(self.device.clone(), frames_in_flight, new.images.len());

        self.resources = new;
        self.available = available;
        self.finished = finished;
        self.frames_in_flight = self.available.len();
        self.frame = 0;
        Ok(())
    }
}

impl Swapchain {
    pub fn set_preferred_extent(&mut self, extent: Extent2D) {
        self.inner.preferred_extent = extent;
    }

    pub fn set_preferred_present_mode(&mut self, present_mode: PresentMode) {
        self.inner.preferred_present_mode = present_mode;
    }

    #[inline]
    pub fn acquire_next(&mut self, timeout: Option<u64>) -> Result<Frame, GPUError> {
        self.inner.acquire_next(timeout)
    }

    #[inline]
    pub fn present(&mut self, queue: &Queue, frame: Frame) -> Result<bool, GPUError> {
        self.inner.present(&queue.inner, frame)
    }

    #[inline]
    pub fn image(&self, frame: Frame) -> &Image {
        self.inner.image(frame)
    }

    /// Transitions the acquired image to `to` on `recorder`.
    ///
    /// This is `recorder.image_transition(swapchain.image(frame), ..)`, which
    /// every frame needs twice: into [`ImageLayoutTransition::COLOR`] before
    /// rendering and into [`ImageLayoutTransition::PRESENT`] before
    /// [`Swapchain::end_frame`].
    pub fn transition_frame(
        &self,
        recorder: &mut CommandRecorder,
        frame: Frame,
        to: ImageLayoutTransition,
    ) {
        recorder.image_transition(self.image(frame), to);
    }

    #[inline]
    pub fn view(&self, frame: Frame) -> &ImageView {
        self.inner.view(frame)
    }

    #[inline]
    pub fn images(&self) -> &[Image] {
        &self.inner.resources.images
    }

    #[inline]
    pub fn available_semaphore(&self, frame: Frame) -> &Semaphore {
        self.inner.available_semaphore(frame)
    }

    #[inline]
    pub fn finished_semaphore(&self, frame: Frame) -> &Semaphore {
        self.inner.finished_semaphore(frame)
    }

    pub fn recreate(&mut self) -> Result<(), GPUError> {
        self.inner.recreate()?;
        self.frame_values = vec![0; self.inner.frames_in_flight];
        self.needs_recreate = false;
        Ok(())
    }

    /// Waits for this frame slot's previous submission, then acquires an image.
    ///
    /// Recreates the swapchain internally when the surface reports out-of-date,
    /// so callers only see the happy path. Update the preferred extent on
    /// window resize and this picks it up.
    pub fn begin_frame(&mut self, queue: &Queue) -> Result<Frame, GPUError> {
        if self.needs_recreate {
            self.recreate()?;
        }

        for _ in 0..3 {
            let flight = self.inner.frame;
            if let Some(&value) = self.frame_values.get(flight)
                && value != 0
            {
                queue.timeline.wait(value, None);
            }

            let frame = self.inner.acquire_next(None)?;
            if !frame.out_of_date {
                return Ok(frame);
            }

            self.recreate()?;
        }

        Err(GPUError::Validation(
            "swapchain could not acquire an image after recreating",
        ))
    }

    /// Submits `records`, presents `frame`, and records the timeline value for
    /// the frame slot.
    pub fn end_frame(
        &mut self,
        queue: &Queue,
        frame: Frame,
        records: &[CommandBuffer],
    ) -> Result<(), GPUError> {
        let available = self.inner.available_semaphore(frame);
        let finished = self.inner.finished_semaphore(frame);

        let value = queue.submit(SubmitInfo {
            records,
            wait_binary: &[(available, PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)],
            signal_binary: &[finished],
            ..Default::default()
        });

        if let Some(slot) = self.frame_values.get_mut(frame.flight_index) {
            *slot = value;
        }

        if self.inner.present(&queue.inner, frame)? {
            self.needs_recreate = true;
        }

        Ok(())
    }

    #[inline]
    pub fn preferred_present_mode(&self) -> PresentMode {
        self.inner.preferred_present_mode
    }

    #[inline]
    pub fn present_mode(&self) -> PresentMode {
        self.inner.resources.present_mode
    }

    #[inline]
    pub fn supported_present_modes(&self) -> &[PresentMode] {
        self.inner.present_modes.as_ref()
    }

    #[inline]
    pub fn format(&self) -> Format {
        self.inner.format.format
    }

    #[inline]
    pub fn extent(&self) -> Extent2D {
        self.inner.resources.extent
    }

    #[inline]
    pub fn flight_index(&self) -> usize {
        self.inner.frame
    }

    #[inline]
    pub fn max_frames_in_flight(&self) -> usize {
        self.inner.frames_in_flight
    }
}

impl Device {
    /// Creates a swapchain for a window, filling in the raw handles for you.
    ///
    /// This is the ergonomic Rust entry point. Use
    /// [`Device::create_swapchain`] when you manage the surface handles
    /// yourself (or from another language).
    pub fn create_swapchain_for_window<W>(
        &self,
        window: &W,
        extent: Extent2D,
    ) -> Result<Swapchain, GPUError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let display = window
            .display_handle()
            .map_err(|_| GPUError::Validation("window has no display handle"))?
            .as_raw();
        let handle = window
            .window_handle()
            .map_err(|_| GPUError::Validation("window has no window handle"))?
            .as_raw();

        self.create_swapchain(&SwapchainCreateInfo::new(display, handle, extent))
    }

    pub fn create_swapchain(&self, info: &SwapchainCreateInfo) -> Result<Swapchain, GPUError> {
        let inner = SwapchainImpl::new(self.inner.clone(), info)?;
        let frame_values = vec![0; inner.frames_in_flight];
        Ok(Swapchain {
            inner,
            frame_values,
            needs_recreate: false,
        })
    }
}

impl Drop for SwapchainImplResources {
    fn drop(&mut self) {
        unsafe {
            self.device.wait_idle();
        }
    }
}

impl Drop for SwapchainImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.wait_idle();
            self.loader.destroy_swapchain(self.resources.handle, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
