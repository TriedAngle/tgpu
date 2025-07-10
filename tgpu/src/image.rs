use std::{fmt, ops, sync::Arc};

use ash::vk;
use vkm::Alloc;

use crate::{Allocation, Device, Label, Queue, raw::RawDevice};

// TODO: support custom stuff
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct ImageUsage: u32 {
        const MAP_READ = 1 << 0;
        const MAP_WRITE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const SAMPLED = 1 << 4;
        const STORAGE = 1 << 5;
        const COLOR = 1 << 6;
        const DEPTH_STENCIL = 1 << 7;
        const TRANSIENT = 1 << 8;
        const INPUT = 1 << 9;
        const SPARSE_BINDING = 1 << 10;
        const SPARSE_RESIDENCY = 1 << 11;
        const SPARSE_ALIASED = 1 << 12;
        const MUTABLE_FORMAT = 1 << 13;
        const CUBE = 1 << 14;
        const DEVICE = 1 << 16;
        const HOST = 1 << 17;
        const LAZY = 1 << 18;
        const HOST_VISIBLE = 1 << 19;
        const COHERENT = 1 << 20;
        const CACHED = 1 << 21;
        const RANDOM_ACCESS = 1 << 22;
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(usage: ImageUsage) -> Self {
        let mut vk_usage = vk::ImageUsageFlags::empty();

        if usage.contains(ImageUsage::COPY_SRC) {
            vk_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if usage.contains(ImageUsage::COPY_DST) {
            vk_usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if usage.contains(ImageUsage::SAMPLED) {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if usage.contains(ImageUsage::STORAGE) {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
        if usage.contains(ImageUsage::COLOR) {
            vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if usage.contains(ImageUsage::DEPTH_STENCIL) {
            vk_usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if usage.contains(ImageUsage::TRANSIENT) {
            vk_usage |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }
        if usage.contains(ImageUsage::INPUT) {
            vk_usage |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
        }

        vk_usage
    }
}

impl From<ImageUsage> for vk::ImageCreateFlags {
    fn from(usage: ImageUsage) -> Self {
        let mut flags = vk::ImageCreateFlags::empty();

        if usage.contains(ImageUsage::SPARSE_BINDING) {
            flags |= vk::ImageCreateFlags::SPARSE_BINDING;
        }
        if usage.contains(ImageUsage::SPARSE_RESIDENCY) {
            flags |= vk::ImageCreateFlags::SPARSE_RESIDENCY;
        }
        if usage.contains(ImageUsage::SPARSE_ALIASED) {
            flags |= vk::ImageCreateFlags::SPARSE_ALIASED;
        }
        if usage.contains(ImageUsage::MUTABLE_FORMAT) {
            flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        if usage.contains(ImageUsage::CUBE) {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }

        flags
    }
}

impl From<ImageUsage> for vkm::MemoryUsage {
    fn from(usage: ImageUsage) -> Self {
        if usage.contains(ImageUsage::DEVICE) {
            vkm::MemoryUsage::AutoPreferDevice
        } else if usage.contains(ImageUsage::HOST) {
            vkm::MemoryUsage::AutoPreferHost
        } else if usage.contains(ImageUsage::LAZY) {
            vkm::MemoryUsage::GpuLazy
        } else {
            unimplemented!("Auto not implemented yet")
            // vkm::MemoryUsage::Auto
        }
    }
}

impl From<ImageUsage> for vk::MemoryPropertyFlags {
    fn from(usage: ImageUsage) -> Self {
        let mut vk_usage = vk::MemoryPropertyFlags::empty();
        if usage.contains(ImageUsage::DEVICE) {
            vk_usage |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
        }
        if usage.contains(ImageUsage::HOST_VISIBLE) {
            vk_usage |= vk::MemoryPropertyFlags::HOST_VISIBLE;
        }
        if usage.contains(ImageUsage::COHERENT) {
            vk_usage |= vk::MemoryPropertyFlags::HOST_COHERENT;
        }
        if usage.contains(ImageUsage::CACHED) {
            vk_usage |= vk::MemoryPropertyFlags::HOST_CACHED;
        }
        if usage.contains(ImageUsage::LAZY) {
            vk_usage |= vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
        }
        vk_usage
    }
}

impl From<ImageUsage> for vkm::AllocationCreateFlags {
    fn from(usage: ImageUsage) -> Self {
        let mut flags = vkm::AllocationCreateFlags::empty();
        if (usage.contains(ImageUsage::HOST_VISIBLE) || usage.contains(ImageUsage::HOST))
            && usage.contains(ImageUsage::MAP_WRITE)
        {
            if usage.contains(ImageUsage::RANDOM_ACCESS) {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            } else {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
            }
        }
        flags
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    pub inner: Arc<ImageImpl>,
    pub format: vk::Format,
}

pub struct ImageImpl {
    pub handle: vk::Image,
    pub device: RawDevice,
    pub allocation: Option<Allocation>,
}

#[derive(Debug, Clone)]
pub struct ImageView {
    pub inner: ImageViewImpl,
    pub sampler: Option<Sampler>,
}

#[derive(Debug, Clone)]
pub struct ImageViewImpl {
    pub handle: vk::ImageView,
    pub device: RawDevice,
    pub image: Arc<ImageImpl>,
}

#[derive(Debug, Clone)]
pub struct Sampler {
    pub inner: Arc<SamplerImpl>,
}

#[derive(Debug, Clone)]
pub struct SamplerImpl {
    pub handle: vk::Sampler,
    pub device: RawDevice,
}

// TODO: maybe have this as a helper?
// pub struct SampledImage {
//     pub image: Image,
//     pub sampler: Sampler,
//     pub view: ImageView,
// }

// TODO: detach from vulkan
pub struct SamplerCreateInfo<'a> {
    pub mag: vk::Filter,
    pub min: vk::Filter,
    pub mipmap: vk::SamplerMipmapMode,
    pub address_u: vk::SamplerAddressMode,
    pub address_v: vk::SamplerAddressMode,
    pub address_w: vk::SamplerAddressMode,
    pub anisotropy: Option<f32>,
    pub compare: Option<vk::CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub label: Option<Label<'a>>,
}

pub struct ImageViewCreateInfo<'a> {
    pub image: &'a Image,
    pub sampler: Option<&'a Sampler>,
    pub ty: vk::ImageViewType,
    pub format: Option<vk::Format>,
    pub aspect: vk::ImageAspectFlags,
    pub swizzle: vk::ComponentMapping,
    pub mips: ops::Range<u32>,
    pub layers: ops::Range<u32>,
    pub label: Option<Label<'a>>,
}

pub struct ImageCreateInfo<'a> {
    pub format: vk::Format,
    pub ty: vk::ImageType,
    pub volume: vk::Extent3D,
    pub mips: u32,
    pub layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: ImageUsage,
    pub sharing: vk::SharingMode,
    pub layout: ImageLayout,
    pub label: Option<Label<'a>>,
}

impl SamplerImpl {
    pub unsafe fn new(device: RawDevice, info: &SamplerCreateInfo<'_>) -> Self {
        let mut create_info = vk::SamplerCreateInfo::default()
            .mag_filter(info.mag)
            .min_filter(info.min)
            .mipmap_mode(info.mipmap)
            .address_mode_u(info.address_u)
            .address_mode_v(info.address_v)
            .address_mode_w(info.address_w)
            .min_lod(info.min_lod)
            .max_lod(info.max_lod);

        if let Some(anisotropy) = info.anisotropy {
            create_info.anisotropy_enable = 1;
            create_info.max_anisotropy = anisotropy;
        }

        let handle = unsafe {
            device
                .handle
                .create_sampler(&create_info, None)
                .expect("Sampler")
        };

        if let Some(label) = &info.label {
            unsafe { device.attach_label(label) };
        }

        Self { handle, device }
    }
}

impl ImageViewImpl {
    pub unsafe fn new(device: RawDevice, info: &ImageViewCreateInfo<'_>) -> Self {
        let mut create_info = vk::ImageViewCreateInfo::default()
            .image(info.image.inner.handle)
            .view_type(info.ty)
            .format(info.image.format)
            .components(info.swizzle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(info.aspect)
                    .base_mip_level(info.mips.start)
                    .level_count(info.mips.len() as u32)
                    .base_array_layer(info.layers.start)
                    .layer_count(info.layers.len() as u32),
            );

        if let Some(format) = info.format {
            create_info.format = format;
        }

        let handle = unsafe {
            device
                .handle
                .create_image_view(&create_info, None)
                .expect("Image View")
        };

        if let Some(label) = &info.label {
            unsafe { device.attach_label(label) };
        }

        Self {
            handle,
            device,
            image: info.image.inner.clone(),
        }
    }
}

impl ImageImpl {
    pub unsafe fn new(device: RawDevice, info: &ImageCreateInfo<'_>) -> Self {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(info.ty)
            .format(info.format)
            .extent(info.volume)
            .mip_levels(info.mips)
            .array_layers(info.layers)
            .samples(info.samples)
            .usage(info.usage.into())
            .sharing_mode(info.sharing)
            .initial_layout(info.layout.into())
            .flags(info.usage.into());

        let create_info = vkm::AllocationCreateInfo {
            usage: info.usage.into(),
            flags: info.usage.into(),
            required_flags: info.usage.into(),
            ..Default::default()
        };

        let (handle, allocation) = unsafe {
            device
                .allocator
                .create_image(&image_info, &create_info)
                .expect("Allocate Image")
        };

        let allocation = Some(Allocation {
            handle: allocation,
            allocator: device.allocator.clone(),
        });

        Self {
            handle,
            device,
            allocation,
        }
    }
}

impl Device {
    pub fn create_sampler(&self, info: &SamplerCreateInfo<'_>) -> Sampler {
        let inner = unsafe { SamplerImpl::new(self.inner.clone(), info) };
        Sampler {
            inner: Arc::new(inner),
        }
    }
    pub fn create_image_view(&self, info: &ImageViewCreateInfo<'_>) -> ImageView {
        let inner = unsafe { ImageViewImpl::new(self.inner.clone(), info) };
        let sampler = info.sampler.cloned();
        ImageView { inner, sampler }
    }
    pub fn create_image(&self, info: &ImageCreateInfo<'_>) -> Image {
        let inner = unsafe { ImageImpl::new(self.inner.clone(), info) };
        Image {
            inner: Arc::new(inner),
            format: info.format,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ImageLayout {
    Undefined,
    Unified,
    General,
    Compute,
    Fragment,
    Color,
    TransferDst,
    Present,
    Custom(vk::ImageLayout),
}

#[derive(Debug, Copy, Clone, Default)]
pub struct ImageLayoutTransition {
    pub layout: ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl Default for ImageLayout {
    fn default() -> Self {
        Self::Undefined
    }
}

#[derive(Debug, Clone)]
pub struct ImageTransition<'a> {
    pub from: ImageLayoutTransition,
    pub to: ImageLayoutTransition,
    pub aspect: vk::ImageAspectFlags,
    pub mips: ops::Range<u32>,
    pub layers: ops::Range<u32>,
    pub queue: Option<(&'a Queue, &'a Queue)>,
    pub dependency: vk::DependencyFlags,
}

impl Default for ImageTransition<'_> {
    fn default() -> Self {
        Self {
            from: ImageLayoutTransition::default(),
            to: ImageLayoutTransition::default(),
            aspect: vk::ImageAspectFlags::empty(),
            mips: 0..1,
            layers: 0..1,
            queue: None,
            dependency: vk::DependencyFlags::empty(),
        }
    }
}

impl ImageLayoutTransition {
    pub const UNDEFINED: Self = Self::new(ImageLayout::Undefined);
    pub const GENERAL: Self = Self::new(ImageLayout::General);
    pub const COMPUTE: Self = Self::new(ImageLayout::Compute);
    pub const PRESENT: Self = Self::new(ImageLayout::Present);

    pub const fn new(layout: ImageLayout) -> Self {
        let (stage, access) = layout.infer_stage_flags();
        Self {
            layout,
            stage,
            access,
        }
    }
    pub fn custom(
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> Self {
        let layout = ImageLayout::Custom(layout);
        Self {
            layout,
            stage,
            access,
        }
    }
}

impl ImageLayout {
    pub const fn infer_stage_flags(&self) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        match self {
            ImageLayout::Undefined => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
            ImageLayout::Unified => (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::NONE,
            ),
            ImageLayout::General => (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::SHADER_READ.as_raw()
                        | vk::AccessFlags2::SHADER_WRITE.as_raw(),
                ),
            ),
            ImageLayout::Compute => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::from_raw(
                    vk::AccessFlags2::SHADER_READ.as_raw()
                        | vk::AccessFlags2::SHADER_WRITE.as_raw(),
                ),
            ),
            ImageLayout::Fragment => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            ImageLayout::Color => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            ImageLayout::TransferDst => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            ImageLayout::Present => (
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            Self::Custom(_) => panic!("Custom cannot infer"),
        }
    }
}

impl From<ImageLayout> for vk::ImageLayout {
    fn from(value: ImageLayout) -> Self {
        match value {
            ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
            ImageLayout::Unified => unimplemented!("Wait for Vulkan ash to implement this"),
            ImageLayout::General => vk::ImageLayout::GENERAL,
            ImageLayout::Compute => vk::ImageLayout::GENERAL,
            ImageLayout::Fragment => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageLayout::Color => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
            ImageLayout::Custom(layout) => layout,
        }
    }
}

impl fmt::Debug for ImageImpl {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl Drop for SamplerImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_sampler(self.handle, None);
        }
    }
}

impl Drop for ImageImpl {
    fn drop(&mut self) {
        unsafe {
            if let Some(allocation) = &mut self.allocation {
                allocation
                    .allocator
                    .destroy_image(self.handle, &mut allocation.handle);
            }
        }
    }
}

impl Drop for ImageViewImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_image_view(self.handle, None);
        }
    }
}
