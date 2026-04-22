use std::{ffi::CStr, fmt, sync::Arc};

use ash::vk;

use crate::{GPUError, raw::InstanceImpl};

#[derive(Debug, Clone)]
pub struct Adapter {
    pub(crate) inner: RawAdapter,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AdapterFeatures {
    pub fill_mode_non_solid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdapterDeviceType {
    #[default]
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

impl AdapterDeviceType {
    fn from_vk(value: vk::PhysicalDeviceType) -> Self {
        match value {
            vk::PhysicalDeviceType::INTEGRATED_GPU => Self::IntegratedGpu,
            vk::PhysicalDeviceType::DISCRETE_GPU => Self::DiscreteGpu,
            vk::PhysicalDeviceType::VIRTUAL_GPU => Self::VirtualGpu,
            vk::PhysicalDeviceType::CPU => Self::Cpu,
            _ => Self::Other,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Other => "Other",
            Self::IntegratedGpu => "Integrated GPU",
            Self::DiscreteGpu => "Discrete GPU",
            Self::VirtualGpu => "Virtual GPU",
            Self::Cpu => "CPU",
        }
    }
}

impl fmt::Display for AdapterDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterLimits {
    pub max_push_constants_size: u32,
    pub max_uniform_buffer_range: u32,
    pub min_uniform_buffer_offset_alignment: u64,
    pub max_storage_buffer_range: u32,
    pub max_per_stage_resources: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: AdapterDeviceType,
    pub api_version: u32,
    pub driver_version: u32,
    pub limits: AdapterLimits,
    pub queue_family_count: usize,
}

impl AdapterInfo {
    pub fn default_score(&self) -> u64 {
        let device_type_score = match self.device_type {
            AdapterDeviceType::DiscreteGpu => 5,
            AdapterDeviceType::IntegratedGpu => 4,
            AdapterDeviceType::VirtualGpu => 3,
            AdapterDeviceType::Other => 2,
            AdapterDeviceType::Cpu => 1,
        } as u64;

        let limits = &self.limits;
        (device_type_score << 48)
            | ((limits.max_per_stage_resources as u64) << 20)
            | ((limits.max_uniform_buffer_range as u64) << 4)
            | (limits.max_push_constants_size as u64)
    }
}

pub type RawAdapter = Arc<AdapterImpl>;

#[derive(Debug)]
pub struct AdapterImpl {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_properties: Arc<[vk::QueueFamilyProperties]>,
    pub features: vk::PhysicalDeviceFeatures,
    pub formats: Arc<[(vk::Format, vk::FormatProperties)]>,
    pub info: AdapterInfo,
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
        let info = adapter_info_from_properties(&properties, queue_properties.len());

        Self {
            handle: pdev,
            properties,
            queue_properties: Arc::from(queue_properties),
            features,
            formats: Arc::from(format_properties),
            info,
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

impl Adapter {
    pub fn features(&self) -> AdapterFeatures {
        AdapterFeatures {
            fill_mode_non_solid: self.inner.features.fill_mode_non_solid == vk::TRUE,
        }
    }

    pub fn info(&self) -> &AdapterInfo {
        &self.inner.info
    }

    pub fn limits(&self) -> AdapterLimits {
        self.inner.info.limits
    }

    pub fn default_score(&self) -> u64 {
        self.info().default_score()
    }
}

fn adapter_info_from_properties(
    properties: &vk::PhysicalDeviceProperties,
    queue_family_count: usize,
) -> AdapterInfo {
    AdapterInfo {
        name: physical_device_name(properties),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        device_type: AdapterDeviceType::from_vk(properties.device_type),
        api_version: properties.api_version,
        driver_version: properties.driver_version,
        limits: AdapterLimits {
            max_push_constants_size: properties.limits.max_push_constants_size,
            max_uniform_buffer_range: properties.limits.max_uniform_buffer_range,
            min_uniform_buffer_offset_alignment: properties
                .limits
                .min_uniform_buffer_offset_alignment,
            max_storage_buffer_range: properties.limits.max_storage_buffer_range,
            max_per_stage_resources: properties.limits.max_per_stage_resources,
            max_per_stage_descriptor_sampled_images: properties
                .limits
                .max_per_stage_descriptor_sampled_images,
        },
        queue_family_count,
    }
}

fn physical_device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    let raw_name = properties.device_name.as_ptr();
    unsafe { CStr::from_ptr(raw_name) }
        .to_string_lossy()
        .into_owned()
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
            .map(|physical_device| unsafe { AdapterImpl::new(self, physical_device, formats) })
            .collect::<Vec<_>>();

        Ok(adapters)
    }
}
