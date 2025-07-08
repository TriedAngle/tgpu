extern crate vk_mem as vkm;

use std::fmt;

use ash::vk;

mod adapter;
mod buffer;
mod command;
mod device;
mod instance;
mod pipeline;
mod queue;
mod shader;
mod swapchain;
mod sync;
mod image;
mod allocations;

pub mod raw {
    pub use crate::adapter::{AdapterImpl, RawAdapter};
    pub use crate::buffer::BufferImpl;
    pub use crate::command::{CommandBufferImpl, CommandRecorderImpl};
    pub use crate::device::{DeviceImpl, RawDevice};
    pub use crate::instance::{InstanceImpl, RawInstance};
    pub use crate::queue::{QueueImpl, RawQueue};
    pub use crate::swapchain::{SwapchainImpl, SwapchainImplResources};
    pub use crate::sync::SemaphoreImpl;
    pub use crate::pipeline::{RenderPipelineImpl, ComputePipelineImpl};
}

pub use adapter::Adapter;
pub use ash;
pub use ash::vk::{
    ColorSpaceKHR, CullModeFlags, Format, FrontFace, PolygonMode, PresentModeKHR,
    PrimitiveTopology, QueueFlags,
};
pub use buffer::{Buffer, BufferInfo, BufferUsage};
pub use command::{CommandBuffer, CommandPools, CommandRecorder, ThreadCommandPool};
pub use device::{Device, DeviceCreateInfo};
pub use instance::{Instance, InstanceCreateInfo};
pub use queue::{Queue, QueueFamilyInfo, QueueRequest};
pub use shader::{Shader, ShaderFunction};
pub use swapchain::{Frame, Swapchain, SwapchainCreateInfo};
pub use sync::Semaphore;
pub use pipeline::{RenderPipeline, RenderPipelineInfo, ComputePipeline, ComputePipelineInfo};
pub use allocations::Allocation;

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
