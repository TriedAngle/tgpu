use std::sync::Arc;

use crate::{GPUError, raw::InstanceImpl};
use ash::vk;

#[derive(Debug)]
pub struct Adapter {
    pub(crate) inner: RawAdapter,
}

pub type RawAdapter = Arc<AdapterImpl>;

#[derive(Debug)]
pub struct AdapterImpl {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_properties: Arc<[vk::QueueFamilyProperties]>,
    pub features: vk::PhysicalDeviceFeatures,
    pub formats: Arc<[(vk::Format, vk::FormatProperties)]>,
}

impl AdapterImpl {
    pub unsafe fn new(
        instance: &InstanceImpl,
        pdev: vk::PhysicalDevice,
        formats: &[vk::Format],
    ) -> Self {
        let properties = unsafe { instance.properties(pdev) };
        let features = unsafe { instance.features(pdev) };
        let queue_properties = unsafe { instance.queue_properties(pdev) };
        let format_properties = unsafe { instance.format_properties(pdev, formats) };

        Self {
            handle: pdev,
            properties,
            queue_properties: Arc::from(queue_properties),
            features,
            formats: Arc::from(format_properties),
        }
    }

    pub unsafe fn max_buffer_mapping_size(&self) -> u32 {
        self.properties.limits.max_storage_buffer_range
    }

    pub unsafe fn max_descriptor_images(&self) -> u32 {
        self.properties
            .limits
            .max_per_stage_descriptor_sampled_images
    }

    pub unsafe fn max_descriptor_resources(&self) -> u32 {
        self.properties.limits.max_per_stage_resources
    }

    pub unsafe fn handle(&self) -> vk::PhysicalDevice {
        self.handle
    }
}

impl InstanceImpl {
    pub unsafe fn adapters(&self, formats: &[vk::Format]) -> Result<Vec<AdapterImpl>, GPUError> {
        let pdevs = unsafe {
            self.handle
                .enumerate_physical_devices()
                .map_err(GPUError::from)?
        };

        let adapters = pdevs
            .into_iter()
            .map(|physical_device| unsafe { AdapterImpl::new(&self, physical_device, formats) })
            .collect::<Vec<_>>();

        Ok(adapters)
    }
}
