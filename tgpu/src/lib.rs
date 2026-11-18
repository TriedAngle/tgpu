extern crate vk_mem as vkm;

use std::fmt;

use ash::vk;

mod adapter;
mod allocations;
mod bindless;
mod buffer;
mod command;
mod debug;
mod descriptor;
mod device;
#[cfg(feature = "egui")]
pub mod egui;
mod format;
mod geometry;
mod image;
mod instance;
mod pipeline;
mod queue;
mod resource;
mod shader;
mod swapchain;
mod sync;

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

pub use adapter::{
    Adapter, AdapterDescriptorIndexingFeatures, AdapterDeviceType, AdapterFeatures, AdapterInfo,
    AdapterLimits, RankedAdapter,
};
pub use allocations::Allocation;
pub use ash;
pub use ash::vk::{QueueFlags, ShaderStageFlags};
pub use bindless::{
    BINDLESS_READ_BUFFER_BINDING, BINDLESS_RW_BUFFER_BINDING, BINDLESS_SAMPLED_IMAGE_BINDING,
    BINDLESS_SAMPLER_BINDING, BINDLESS_STORAGE_IMAGE_BINDING, BINDLESS_UNIFORM_BUFFER_BINDING,
    BindlessHeap, BindlessInfo, ReadBufferHandle, RwBufferHandle, SampledImageHandle,
    SamplerHandle, StorageImageHandle, UniformBufferHandle,
};
pub use buffer::{
    Buffer, BufferAccessTransition, BufferDesc, BufferTransition, BufferUses, CopyBufferInfo,
};
pub use command::{
    AttachmentLoadOp, AttachmentStoreOp, ClearValue, CommandBuffer, CommandPools, CommandRecorder,
    IndexType, PipelineStageFlags, RenderAttachment, RenderInfo, RenderRecorder, SubmitInfo,
    ThreadCommandPool,
};
pub use debug::Label;
pub use descriptor::{
    DescriptorArena, DescriptorBinding, DescriptorPool, DescriptorPoolInfo, DescriptorSet,
    DescriptorSetLayout, DescriptorSetLayoutInfo, DescriptorType, DescriptorWrite,
};
pub use device::{Device, DeviceCreateInfo, DeviceFeatures};
pub use format::{ColorSpace, Format, SurfaceFormat};
pub use geometry::{Extent2D, Extent3D, Offset, Offset3D, Rect, Viewport};
pub use image::{
    BlitImageInfo, BufferImageCopy, CopyBufferToImageInfo, CopyImageInfo, Filter, Image,
    ImageAspect, ImageDesc, ImageFlags, ImageLayout, ImageLayoutTransition, ImageSubresourceLayers,
    ImageTransition, ImageUses, ImageView, ImageViewCreateInfo, ImageViewOptions, Sampler,
    SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, Texture2DDesc, TextureUses,
    ViewImage, ViewImageDesc,
};
pub use instance::{Instance, InstanceCreateInfo};
pub use pipeline::{
    ColorBlend, CompareOp, ComputePipeline, ComputePipelineInfo, CullMode, FrontFace, PolygonMode,
    PrimitiveTopology, RenderPipeline, RenderPipelineInfo, VertexAttribute, VertexBufferLayout,
    VertexInput, VertexInputRate, push_constant_size,
};
pub use queue::{Queue, QueueFamilyInfo, QueueRequest};
pub use resource::{HostAccess, MemoryPreset};
pub use shader::{Shader, ShaderEntry, ShaderSource};
pub use swapchain::{FormatSelector, Frame, PresentMode, Swapchain, SwapchainCreateInfo};
pub use sync::Semaphore;

/// Compile-time check that the core structures can be shared across threads.
///
/// Individual resources ([`Buffer`], [`Image`], …) and recorders
/// ([`CommandRecorder`], [`RenderRecorder`]) are deliberately *not* thread safe:
/// they are created, recorded and mapped on one thread. What must be shareable
/// is the device/queue layer, so that command buffers can be recorded on
/// several threads and submitted from any of them.
#[allow(dead_code)]
fn assert_core_thread_safety() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Instance>();
    assert_send_sync::<Adapter>();
    assert_send_sync::<Device>();
    assert_send_sync::<Queue>();
    assert_send_sync::<Semaphore>();
    assert_send_sync::<CommandPools>();
    assert_send_sync::<CommandBuffer>();
    assert_send_sync::<BindlessHeap>();
    assert_send_sync::<DescriptorSet>();
    assert_send_sync::<DescriptorPool>();
    assert_send_sync::<DescriptorSetLayout>();
}

pub enum GPUError {
    Vulkan(vk::Result),
    Validation(&'static str),
    Shader(String),
}

impl fmt::Debug for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(result) => write!(f, "Vulkan error: {:?}", result),
            Self::Validation(message) => write!(f, "Validation error: {message}"),
            Self::Shader(message) => write!(f, "Shader error: {message}"),
        }
    }
}

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(result) => write!(f, "Vulkan error: {:?}", result),
            Self::Validation(message) => write!(f, "Validation error: {message}"),
            Self::Shader(message) => write!(f, "Shader error: {message}"),
        }
    }
}

impl std::error::Error for GPUError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vulkan(_) => None,
            Self::Validation(_) => None,
            Self::Shader(_) => None,
        }
    }
}

impl From<vk::Result> for GPUError {
    fn from(value: vk::Result) -> Self {
        Self::Vulkan(value)
    }
}
