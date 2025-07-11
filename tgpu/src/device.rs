use std::{
    collections::HashMap, mem::ManuallyDrop, sync::{atomic::AtomicU64, Arc}, time::Duration
};

use ash::vk;
use parking_lot::Mutex;

use crate::{
    Adapter, CommandPools, GPUError, Instance, Queue, QueueFamilyInfo, QueueRequest, Semaphore,
    raw::{QueueImpl, RawAdapter, RawInstance, SemaphoreImpl},
};

#[derive(Debug)]
pub struct Device {
    pub inner: RawDevice,
    pub adapter: Adapter,
}

pub type RawDevice = Arc<DeviceImpl>;

pub struct Extensions {
    pub debug: ash::ext::debug_utils::Device,
    pub sync2: ash::khr::synchronization2::Device,
    pub dynamic: ash::khr::dynamic_rendering::Device,
}

pub struct DeviceImpl {
    pub handle: ash::Device,
    pub instance: RawInstance,
    pub adapter: RawAdapter,
    pub ext: Extensions,
    pub allocator: Arc<ManuallyDrop<vkm::Allocator>>,
}

#[derive(Default)]
pub struct DeviceCreateInfo {}

impl Device {}

impl DeviceImpl {
    pub fn new(
        _info: &DeviceCreateInfo,
        instance: RawInstance,
        adapter: RawAdapter,
        queue_requests: &[QueueRequest],
    ) -> Result<(RawDevice, Vec<QueueImpl>), GPUError> {
        let mut pdev_features2 = vk::PhysicalDeviceFeatures2::default().features(
            vk::PhysicalDeviceFeatures::default()
                .shader_sampled_image_array_dynamic_indexing(true)
                .shader_storage_image_array_dynamic_indexing(true)
                .shader_storage_buffer_array_dynamic_indexing(true)
                .shader_uniform_buffer_array_dynamic_indexing(true),
        );

        // let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
        //     .dynamic_rendering(true)
        //     .synchronization2(true);

        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut timeline_semaphore_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);

        let mut descriptor_indexing_features =
            vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .runtime_descriptor_array(true)
                .descriptor_binding_update_unused_while_pending(true);

        let mut synchronization_two_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        // TODO: once apple engineers actually use their own stuff
        // we can remove all of them except swapchain
        let mut device_extensions = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::timeline_semaphore::NAME.as_ptr(),
            ash::khr::dynamic_rendering::NAME.as_ptr(),
            ash::khr::synchronization2::NAME.as_ptr(),
        ];

        #[cfg(target_os = "macos")]
        {
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        let queue_family_infos =
            QueueImpl::find_queue_families(&instance, &adapter, queue_requests)
                .expect("Find Queues");

        let mut family_queue_counts: HashMap<u32, u32> = HashMap::new();
        for info in &queue_family_infos {
            let count = family_queue_counts.entry(info.family_index).or_default();
            *count = (*count).max(info.queue_index + 1);
        }

        let queue_families: Vec<(u32, Vec<f32>)> = family_queue_counts
            .into_iter()
            .map(|(family_index, count)| {
                let priorities = vec![1.0; count as usize];
                (family_index, priorities)
            })
            .collect();

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_families
            .iter()
            .map(|(family_index, priorities)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*family_index)
                    .queue_priorities(priorities)
            })
            .collect();

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            // enable this and remove all other probably once apple swes stop blueskying
            // .push_next(&mut vulkan_1_3_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut pdev_features2)
            .push_next(&mut timeline_semaphore_features)
            .push_next(&mut synchronization_two_features)
            .push_next(&mut descriptor_indexing_features);

        let handle = unsafe { instance.create_device_handle(&device_info, adapter.handle) };

        let ext = unsafe { Self::new_extensions(&instance.handle, &handle) };

        let allocator = unsafe {
            vkm::Allocator::new(vkm::AllocatorCreateInfo::new(
                &instance.handle,
                &handle,
                adapter.handle(),
            ))
        }?;

        let new = Self {
            handle,
            instance,
            adapter,
            ext,
            allocator: Arc::new(ManuallyDrop::new(allocator)),
        };

        let new = Arc::new(new);

        let queues = queue_family_infos
            .into_iter()
            .map(|info| QueueImpl::new(new.clone(), info))
            .collect::<Vec<_>>();

        Ok((new, queues))
    }

    pub unsafe fn new_extensions(instance: &ash::Instance, device: &ash::Device) -> Extensions {
        let debug = ash::ext::debug_utils::Device::new(instance, device);
        let sync2 = ash::khr::synchronization2::Device::new(instance, device);
        let dynamic = ash::khr::dynamic_rendering::Device::new(instance, device);

        Extensions {
            debug,
            sync2,
            dynamic,
        }
    }

    pub unsafe fn create_queue(&self, info: &QueueFamilyInfo) -> vk::Queue {
        unsafe {
            self.handle
                .get_device_queue(info.family_index, info.queue_index)
        }
    }

    pub unsafe fn wait_idle(&self) {
        let _ = unsafe { self.handle.device_wait_idle() };
    }

    pub unsafe fn wait_fence(&self, fence: vk::Fence, timeout: Option<u64>) {
        let timeout = timeout.unwrap_or(u64::MAX);
        unsafe {
            self.handle
                .wait_for_fences(&[fence], true, timeout)
                .expect("Wait for fence")
        }
    }

    pub unsafe fn reset_fence(&self, fence: vk::Fence) {
        unsafe {
            self.handle.reset_fences(&[fence]).expect("Reset Fence");
        }
    }

    pub unsafe fn get_semaphore_value(&self, handle: vk::Semaphore) -> u64 {
        unsafe { self.handle.get_semaphore_counter_value(handle).unwrap() }
    }

    pub unsafe fn signal_semaphore(&self, handle: vk::Semaphore, value: u64) {
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(handle)
            .value(value);

        unsafe {
            let _ = self.handle.signal_semaphore(&info);
        }
    }

    pub unsafe fn wait_semaphore(
        &self,
        handle: vk::Semaphore,
        value: u64,
        timeout: Option<Duration>,
    ) {
        let handles = [handle];
        let values = [value];
        let timeout_ns = timeout.map_or(u64::MAX, |d| d.as_nanos() as u64);
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&handles)
            .values(&values);

        unsafe {
            let _ = self.handle.wait_semaphores(&info, timeout_ns);
        }
    }
}

impl Instance {
    pub fn request_device(
        &self,
        info: &DeviceCreateInfo,
        adapter: Adapter,
        queue_requests: &[QueueRequest],
    ) -> Result<(Device, impl Iterator<Item = Queue> + use<>), GPUError> {
        let (device, queues) = DeviceImpl::new(
            info,
            self.inner.clone(),
            adapter.inner.clone(),
            queue_requests,
        )?;

        let inner = device.clone();

        let device = Device {
            inner: device,
            adapter,
        };

        let queues = queues.into_iter().map(move |queue| Queue {
            inner: Arc::new(queue),
            pools: CommandPools::new(inner.clone()),
            state: Mutex::new(()),
            submission_counter: AtomicU64::new(1),
            timeline: Semaphore {
                inner: Arc::new(unsafe { SemaphoreImpl::new_timeline(inner.clone(), 0) }),
            },
        });

        Ok((device, queues))
    }
}

impl Drop for DeviceImpl {
    fn drop(&mut self) {
        unsafe {
            let _ = self.handle.device_wait_idle();
            let allocator = Arc::get_mut(&mut self.allocator).expect("Get Allocator");
            ManuallyDrop::drop(allocator);
            self.handle.destroy_device(None);
        }
    }
}

impl std::fmt::Debug for DeviceImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceImpl")
            .field("handle", &self.handle.handle())
            .finish()
    }
}
