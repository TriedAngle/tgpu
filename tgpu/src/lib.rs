extern crate vk_mem as vkm;

use std::fmt;

use ash::vk;

mod adapter;
mod allocations;
mod buffer;
mod command;
mod debug;
mod descriptor;
mod device;
mod image;
mod instance;
mod pipeline;
mod queue;
mod shader;
mod swapchain;
mod sync;
mod descriptor2;
mod freelist;

pub mod raw {
    pub use crate::adapter::{AdapterImpl, RawAdapter};
    pub use crate::buffer::BufferImpl;
    pub use crate::command::{CommandBufferImpl, CommandRecorderImpl};
    pub use crate::device::{DeviceImpl, RawDevice};
    pub use crate::image::{ImageImpl, ImageViewImpl, SamplerImpl};
    pub use crate::instance::{InstanceImpl, RawInstance};
    pub use crate::pipeline::{ComputePipelineImpl, RenderPipelineImpl};
    pub use crate::queue::{QueueImpl, RawQueue};
    pub use crate::swapchain::{SwapchainImpl, SwapchainImplResources};
    pub use crate::sync::SemaphoreImpl;
}

pub use adapter::Adapter;
pub use allocations::Allocation;
pub use ash;
pub use ash::vk::{
    ColorSpaceKHR, CullModeFlags, Format, FrontFace, PolygonMode, PresentModeKHR,
    PrimitiveTopology, QueueFlags, ShaderStageFlags,
};
pub use buffer::{Buffer, BufferInfo, BufferUsage};
pub use command::{
    CommandBuffer, CommandPools, CommandRecorder, RenderInfo, SubmitInfo, ThreadCommandPool,
};
pub use debug::Label;
pub use descriptor::{
    DescriptorBinding, DescriptorPool, DescriptorPoolInfo, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutInfo, DescriptorType, DescriptorWrite,
};
pub use device::{Device, DeviceCreateInfo};
pub use image::{
    BlitImageInfo, CopyImageInfo, Image, ImageCreateInfo, ImageLayout, ImageLayoutTransition,
    ImageTransition, ImageUsage, ImageView, ImageViewCreateInfo, ImageViewOptions, Sampler,
    SamplerCreateInfo, ViewImage, ViewImageCreateInfo,
};
pub use instance::{Instance, InstanceCreateInfo};
pub use pipeline::{ComputePipeline, ComputePipelineInfo, RenderPipeline, RenderPipelineInfo};
pub use queue::{Queue, QueueFamilyInfo, QueueRequest};
pub use shader::{Shader, ShaderEntry, ShaderSource};
pub use swapchain::{Frame, Swapchain, SwapchainCreateInfo};
pub use sync::Semaphore;

pub enum GPUError {
    Vulkan(vk::Result),
}

impl fmt::Debug for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(result) => write!(f, "Vulkan error: {:?}", result),
        }
    }
}

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(result) => write!(f, "Vulkan error: {:?}", result),
        }
    }
}

impl std::error::Error for GPUError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vulkan(_) => None,
        }
    }
}

impl From<vk::Result> for GPUError {
    fn from(value: vk::Result) -> Self {
        Self::Vulkan(value)
    }
}
