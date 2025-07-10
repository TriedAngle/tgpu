use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;

use crate::{
    Device, GPUError, Image, ImageView, Queue,
    raw::{DeviceImpl, ImageImpl, ImageViewImpl, QueueImpl, RawAdapter, RawDevice},
};

#[derive(Debug, Copy, Clone)]
pub struct Frame {
    pub index: u32,
    pub suboptimal: bool,
}

pub struct Swapchain {
    pub inner: SwapchainImpl,
    pub extent: vk::Extent2D, // TODO: make this own type
}

pub struct SwapchainCreateInfo {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
    pub preferred_image_count: usize,
    pub preferred_present_mode: vk::PresentModeKHR,
    pub format_selector: Box<dyn Fn(&[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR>,
}

#[derive(Debug)]
pub struct SwapchainImplResources {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<Image>,
    pub views: Vec<ImageView>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

pub struct SwapchainImpl {
    pub device: RawDevice,
    pub adapter: RawAdapter,
    pub loader: ash::khr::swapchain::Device,
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,

    pub available: Vec<vk::Semaphore>,
    pub finished: Vec<vk::Semaphore>,
    pub flight: Vec<vk::Fence>,
    pub frame: usize,

    pub resources: SwapchainImplResources,

    pub max_flight: usize,
    pub formats: Arc<[vk::SurfaceFormatKHR]>,
    pub format: vk::SurfaceFormatKHR,
    pub present_modes: Arc<[vk::PresentModeKHR]>,
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
        let format = (info.format_selector)(&formats);
        log::info!("format: {:?}", format);

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(adapter.handle, surface)
                .map_err(GPUError::from)?
        };

        let (available, finished, flight) =
            Self::create_syncs(&device, info.preferred_image_count)?;

        let resources = Self::create_resources(
            device.clone(),
            &loader,
            surface,
            &surface_loader,
            adapter.handle,
            &info,
            format,
            None,
        )?;

        let new = Self {
            device,
            adapter,
            loader,
            surface,
            surface_loader,

            available,
            finished,
            flight,
            frame: 0,

            resources,

            max_flight: info.preferred_image_count,
            formats: Arc::from(formats),
            format,
            present_modes: Arc::from(present_modes),
        };

        Ok(new)
    }

    fn create_syncs(
        device: &DeviceImpl,
        max_flight: usize,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>), GPUError> {
        let mut available = Vec::with_capacity(max_flight);
        let mut finished = Vec::with_capacity(max_flight);
        let mut flight = Vec::with_capacity(max_flight);

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            for _ in 0..max_flight {
                let availabe_semaphore = device
                    .handle
                    .create_semaphore(&semaphore_info, None)
                    .map_err(GPUError::from)?;

                let finished_semaphore = device
                    .handle
                    .create_semaphore(&semaphore_info, None)
                    .map_err(GPUError::from)?;

                let flight_fence = device
                    .handle
                    .create_fence(&fence_info, None)
                    .map_err(GPUError::from)?;

                available.push(availabe_semaphore);
                finished.push(finished_semaphore);
                flight.push(flight_fence);
            }
        }

        Ok((available, finished, flight))
    }

    fn create_resources(
        device: RawDevice,
        loader: &ash::khr::swapchain::Device,
        surface_handle: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        adapter_handle: vk::PhysicalDevice,
        config: &SwapchainCreateInfo,
        format: vk::SurfaceFormatKHR,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<SwapchainImplResources, GPUError> {
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(adapter_handle, surface_handle)
                .map_err(GPUError::from)?
        };

        let min_images = capabilities.min_image_count;
        let max_images = if capabilities.max_image_count == 0 {
            config.preferred_image_count as u32
        } else {
            capabilities.max_image_count
        };

        let image_count = (config.preferred_image_count as u32)
            .max(min_images)
            .min(max_images);

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(adapter_handle, surface_handle)?
        };

        let present_mode = if present_modes.contains(&config.preferred_present_mode) {
            config.preferred_present_mode
        } else {
            present_modes
                .iter()
                .copied()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .or_else(|| {
                    log::warn!("Present mode: MAILBOX not found, falling back to Immediate");
                    present_modes
                        .iter()
                        .copied()
                        .find(|&mode| mode == vk::PresentModeKHR::IMMEDIATE)
                })
                .unwrap_or_else(|| {
                    log::warn!("Present mode: IMMEDIATE not found, falling back to Fifo");
                    vk::PresentModeKHR::FIFO
                })
        };

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_handle)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
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

        let views = Self::create_image_views(device, &images, format);

        let resources = SwapchainImplResources {
            handle,
            images,
            views,
            capabilities,
        };

        Ok(resources)
    }

    fn create_image_views(
        device: RawDevice,
        images: &Vec<Image>,
        format: vk::SurfaceFormatKHR,
    ) -> Vec<ImageView> {
        images
            .iter()
            .map(|img| unsafe {
                let info = vk::ImageViewCreateInfo::default()
                    .image(img.inner.handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
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

                let image_view = ImageView {
                    sampler: None,
                    inner: ImageViewImpl {
                        handle,
                        device: device.clone(),
                        image: img.inner.clone(),
                    },
                };
                image_view
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
        let flight_fence = self.flight[self.frame];
        let available_semaphore = self.available[self.frame];

        unsafe { self.device.wait_fence(flight_fence, timeout) };
        unsafe { self.device.reset_fence(flight_fence) };

        let timeout_ns = timeout.unwrap_or(u64::MAX);
        let (image_index, suboptimal) = unsafe {
            match self.loader.acquire_next_image(
                self.resources.handle,
                timeout_ns,
                available_semaphore,
                vk::Fence::null(),
            ) {
                Ok((idx, suboptimal)) => (idx, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Ok(Frame {
                        index: 0,
                        suboptimal: false,
                    });
                }
                Err(e) => return Err(GPUError::Vulkan(e)),
            }
        };

        Ok(Frame {
            index: image_index,
            suboptimal,
        })
    }

    pub fn image(&self, frame: Frame) -> &Image {
        &self.resources.images[frame.index as usize]
    }

    pub fn view(&self, frame: Frame) -> &ImageView {
        &self.resources.views[frame.index as usize]
    }

    pub fn present(&mut self, queue: &QueueImpl, frame: Frame) -> Result<bool, GPUError> {
        let finished_semaphore = self.finished[self.frame];

        let swapchains = [self.resources.handle];
        let image_indices = [frame.index];
        let wait_semaphores = [finished_semaphore];
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

    pub fn current_available_semaphore(&self) -> vk::Semaphore {
        self.available[self.frame]
    }

    pub fn current_finished_semaphore(&self) -> vk::Semaphore {
        self.finished[self.frame]
    }

    pub fn current_fence(&self) -> vk::Fence {
        self.flight[self.frame]
    }
}

impl Swapchain {
    pub fn acquire_next(&mut self, timeout: Option<u64>) -> Result<Frame, GPUError> {
        self.inner.acquire_next(timeout)
    }

    pub fn present(&mut self, queue: &Queue, frame: Frame) -> Result<bool, GPUError> {
        self.inner.present(&queue.inner, frame)
    }
    pub fn image(&self, frame: Frame) -> &Image {
        self.inner.image(frame)
    }

    pub fn view(&self, frame: Frame) -> &ImageView {
        self.inner.view(frame)
    }
}

impl Device {
    pub fn create_swapchain(&self, info: &SwapchainCreateInfo) -> Result<Swapchain, GPUError> {
        let inner = SwapchainImpl::new(self.inner.clone(), info)?;
        let extent = inner.resources.capabilities.current_extent;
        Ok(Swapchain { inner, extent })
    }
}

impl Drop for SwapchainImpl {
    fn drop(&mut self) {
        unsafe { self.device.wait_idle() };
        unsafe {
            // for &view in &self.resources.views {
            //     self.device.handle.destroy_image_view(view, None);
            // }
            for &semaphore in &self.available {
                self.device.handle.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.finished {
                self.device.handle.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.flight {
                self.device.handle.destroy_fence(fence, None);
            }
            self.loader.destroy_swapchain(self.resources.handle, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
