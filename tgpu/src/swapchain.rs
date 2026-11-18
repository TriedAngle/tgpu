use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;

use crate::{
    Device, Extent2D, Format, GPUError, Image, ImageView, Queue, Semaphore, SurfaceFormat,
    raw::{DeviceImpl, ImageImpl, ImageViewImpl, QueueImpl, RawAdapter, RawDevice, SemaphoreImpl},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum PresentMode {
    Immediate,
    Mailbox,
    #[default]
    Fifo,
    FifoRelaxed,
    Custom(vk::PresentModeKHR),
}

impl From<PresentMode> for vk::PresentModeKHR {
    fn from(value: PresentMode) -> Self {
        match value {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
            PresentMode::Custom(raw) => raw,
        }
    }
}

impl From<vk::PresentModeKHR> for PresentMode {
    fn from(value: vk::PresentModeKHR) -> Self {
        match value {
            vk::PresentModeKHR::IMMEDIATE => Self::Immediate,
            vk::PresentModeKHR::MAILBOX => Self::Mailbox,
            vk::PresentModeKHR::FIFO => Self::Fifo,
            vk::PresentModeKHR::FIFO_RELAXED => Self::FifoRelaxed,
            raw => Self::Custom(raw),
        }
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
}

pub struct SwapchainCreateInfo {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
    pub preferred_extent: Extent2D,
    pub preferred_image_count: usize,
    pub preferred_present_mode: PresentMode,
    pub format_selector: Box<dyn Fn(&[SurfaceFormat]) -> SurfaceFormat>,
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

    pub max_flight: usize,
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
        let format = (info.format_selector)(&formats);
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
            format,
            None,
        )?;
        let max_flight = resources.images.len();

        let (available, finished) = Self::create_syncs(device.clone(), max_flight);

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

            max_flight,
            preferred_extent: info.preferred_extent,
            preferred_image_count: info.preferred_image_count as u32,
            preferred_present_mode: info.preferred_present_mode,
            formats: Arc::from(formats),
            format,
            present_modes: Arc::from(present_modes),
        };

        Ok(new)
    }

    fn create_syncs(device: RawDevice, max_flight: usize) -> (Vec<Semaphore>, Vec<Semaphore>) {
        let mut available = Vec::with_capacity(max_flight);
        let mut finished = Vec::with_capacity(max_flight);

        for _ in 0..max_flight {
            let inner_available = unsafe { SemaphoreImpl::new_signal(device.clone()) };
            let availabe_semaphore = Semaphore {
                inner: Arc::new(inner_available),
            };

            let inner_finished = unsafe { SemaphoreImpl::new_signal(device.clone()) };
            let finished_semaphore = Semaphore {
                inner: Arc::new(inner_finished),
            };

            available.push(availabe_semaphore);
            finished.push(finished_semaphore);
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
        format: SurfaceFormat,
        old_swapchain: Option<vk::SwapchainKHR>,
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

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(adapter_handle, surface_handle)?
        };
        let present_modes = present_modes
            .into_iter()
            .map(PresentMode::from)
            .collect::<Vec<_>>();

        let present_mode = if present_modes.contains(&preferred_present_mode) {
            preferred_present_mode
        } else {
            present_modes
                .iter()
                .copied()
                .find(|&mode| mode == PresentMode::Mailbox)
                .or_else(|| {
                    log::warn!("Present mode: MAILBOX not found, falling back to Immediate");
                    present_modes
                        .iter()
                        .copied()
                        .find(|&mode| mode == PresentMode::Immediate)
                })
                .unwrap_or_else(|| {
                    log::warn!("Present mode: IMMEDIATE not found, falling back to Fifo");
                    PresentMode::Fifo
                })
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
            .clipped(true)
            .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()));

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
        let needs_recreation = match result {
            Ok(suboptimal) => suboptimal || frame.suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(e) => return Err(GPUError::Vulkan(e)),
        };

        self.frame = (self.frame + 1) % self.max_flight;
        Ok(needs_recreation)
    }

    pub fn available_semaphore(&self, frame: Frame) -> &Semaphore {
        &self.available[frame.flight_index]
    }

    pub fn finished_semaphore(&self, frame: Frame) -> &Semaphore {
        &self.finished[frame.index as usize]
    }

    pub fn recreate(&mut self) -> Result<(), GPUError> {
        unsafe { self.device.wait_idle() };
        let new = Self::create_resources(
            self.device.clone(),
            &self.loader,
            self.surface,
            &self.surface_loader,
            self.device.adapter.handle,
            self.preferred_extent,
            self.preferred_image_count,
            self.preferred_present_mode,
            self.format,
            Some(self.resources.handle),
        )?;

        let (available, finished) = Self::create_syncs(self.device.clone(), new.images.len());

        unsafe {
            self.loader.destroy_swapchain(self.resources.handle, None);
        }

        self.resources = new;
        self.available = available;
        self.finished = finished;
        self.max_flight = self.available.len();
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
        self.inner.max_flight
    }
}

impl Device {
    pub fn create_swapchain(&self, info: &SwapchainCreateInfo) -> Result<Swapchain, GPUError> {
        let inner = SwapchainImpl::new(self.inner.clone(), info)?;
        Ok(Swapchain { inner })
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
