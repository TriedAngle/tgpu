use std::{fmt, ops, sync::Arc};

use ash::vk;
use vkm::Alloc;

use crate::{
    Allocation, Buffer, Device, GPUError, HostAccess, Label, MemoryPreset, Queue, raw::RawDevice,
};

// TODO: support custom stuff
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default)]
    pub(crate) struct ImageUsage: u32 {
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
        if usage.contains(ImageUsage::HOST_VISIBLE) || usage.contains(ImageUsage::HOST) {
            if usage.contains(ImageUsage::RANDOM_ACCESS) || usage.contains(ImageUsage::MAP_READ) {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            } else if usage.contains(ImageUsage::MAP_WRITE) {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
            }
        }
        flags
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct TextureUses: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        const SAMPLED = 1 << 2;
        const STORAGE = 1 << 3;
        const COLOR_ATTACHMENT = 1 << 4;
        const DEPTH_STENCIL_ATTACHMENT = 1 << 5;
        const INPUT_ATTACHMENT = 1 << 6;
        const TRANSIENT_ATTACHMENT = 1 << 7;
    }
}

pub type ImageUses = TextureUses;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ImageFlags: u32 {
        const SPARSE_BINDING = 1 << 0;
        const SPARSE_RESIDENCY = 1 << 1;
        const SPARSE_ALIASED = 1 << 2;
        const MUTABLE_FORMAT = 1 << 3;
        const CUBE = 1 << 4;
    }
}

#[derive(Debug, Clone)]
pub struct ImageDesc<'a> {
    pub format: vk::Format,
    pub ty: vk::ImageType,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub tiling: vk::ImageTiling,
    pub samples: vk::SampleCountFlags,
    pub usage: ImageUses,
    pub flags: ImageFlags,
    pub memory: MemoryPreset,
    pub host_access: HostAccess,
    pub sharing: vk::SharingMode,
    pub initial_layout: ImageLayout,
    pub label: Option<Label<'a>>,
}

impl Default for ImageDesc<'_> {
    fn default() -> Self {
        Self {
            format: vk::Format::UNDEFINED,
            ty: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D::default(),
            mip_levels: 1,
            array_layers: 1,
            tiling: vk::ImageTiling::OPTIMAL,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: ImageUses::empty(),
            flags: ImageFlags::empty(),
            memory: MemoryPreset::GpuOnly,
            host_access: HostAccess::None,
            sharing: vk::SharingMode::EXCLUSIVE,
            initial_layout: ImageLayout::Undefined,
            label: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViewImageDesc<'a> {
    pub image: ImageDesc<'a>,
    pub sampler: Option<SamplerCreateInfo<'a>>,
    pub view_type: Option<vk::ImageViewType>,
    pub view_format: Option<vk::Format>,
    pub aspect: Option<vk::ImageAspectFlags>,
    pub swizzle: vk::ComponentMapping,
    pub view_mips: Option<ops::Range<u32>>,
    pub view_layers: Option<ops::Range<u32>>,
    pub view_label: Option<Label<'a>>,
}

impl Default for ViewImageDesc<'_> {
    fn default() -> Self {
        Self {
            image: ImageDesc::default(),
            sampler: None,
            view_type: None,
            view_format: None,
            aspect: None,
            swizzle: vk::ComponentMapping::default(),
            view_mips: None,
            view_layers: None,
            view_label: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Texture2DDesc<'a> {
    pub size: [u32; 2],
    pub format: vk::Format,
    pub usage: TextureUses,
    pub memory: MemoryPreset,
    pub sampler: Option<SamplerCreateInfo<'a>>,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub tiling: vk::ImageTiling,
    pub view_format: Option<vk::Format>,
    pub label: Option<Label<'a>>,
    pub view_label: Option<Label<'a>>,
}

impl Default for Texture2DDesc<'_> {
    fn default() -> Self {
        Self {
            size: [0, 0],
            format: vk::Format::UNDEFINED,
            usage: TextureUses::empty(),
            memory: MemoryPreset::GpuOnly,
            sampler: None,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            view_format: None,
            label: None,
            view_label: None,
        }
    }
}

impl From<TextureUses> for ImageUsage {
    fn from(usage: TextureUses) -> Self {
        let mut raw = ImageUsage::empty();
        if usage.contains(TextureUses::COPY_SRC) {
            raw |= ImageUsage::COPY_SRC;
        }
        if usage.contains(TextureUses::COPY_DST) {
            raw |= ImageUsage::COPY_DST;
        }
        if usage.contains(TextureUses::SAMPLED) {
            raw |= ImageUsage::SAMPLED;
        }
        if usage.contains(TextureUses::STORAGE) {
            raw |= ImageUsage::STORAGE;
        }
        if usage.contains(TextureUses::COLOR_ATTACHMENT) {
            raw |= ImageUsage::COLOR;
        }
        if usage.contains(TextureUses::DEPTH_STENCIL_ATTACHMENT) {
            raw |= ImageUsage::DEPTH_STENCIL;
        }
        if usage.contains(TextureUses::INPUT_ATTACHMENT) {
            raw |= ImageUsage::INPUT;
        }
        if usage.contains(TextureUses::TRANSIENT_ATTACHMENT) {
            raw |= ImageUsage::TRANSIENT;
        }
        raw
    }
}

impl From<ImageFlags> for vk::ImageCreateFlags {
    fn from(flags: ImageFlags) -> Self {
        let mut raw = vk::ImageCreateFlags::empty();
        if flags.contains(ImageFlags::SPARSE_BINDING) {
            raw |= vk::ImageCreateFlags::SPARSE_BINDING;
        }
        if flags.contains(ImageFlags::SPARSE_RESIDENCY) {
            raw |= vk::ImageCreateFlags::SPARSE_RESIDENCY;
        }
        if flags.contains(ImageFlags::SPARSE_ALIASED) {
            raw |= vk::ImageCreateFlags::SPARSE_ALIASED;
        }
        if flags.contains(ImageFlags::MUTABLE_FORMAT) {
            raw |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        if flags.contains(ImageFlags::CUBE) {
            raw |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        raw
    }
}

fn validate_texture_2d_desc(desc: &Texture2DDesc<'_>) -> Result<(), GPUError> {
    if desc.size[0] == 0 || desc.size[1] == 0 {
        return Err(GPUError::Validation(
            "2D texture size must be greater than zero",
        ));
    }
    if desc.format == vk::Format::UNDEFINED {
        return Err(GPUError::Validation("2D texture format must be defined"));
    }
    if desc.mip_levels == 0 {
        return Err(GPUError::Validation(
            "2D texture mip_levels must be greater than zero",
        ));
    }
    if desc.array_layers == 0 {
        return Err(GPUError::Validation(
            "2D texture array_layers must be greater than zero",
        ));
    }

    let mut usage = desc.usage;
    if desc.sampler.is_some() {
        usage |= TextureUses::SAMPLED;
    }
    if usage.is_empty() {
        return Err(GPUError::Validation("2D texture usage must not be empty"));
    }

    if usage.contains(TextureUses::TRANSIENT_ATTACHMENT) {
        let attachment_usage = TextureUses::COLOR_ATTACHMENT
            | TextureUses::DEPTH_STENCIL_ATTACHMENT
            | TextureUses::INPUT_ATTACHMENT;

        if !usage.intersects(attachment_usage) {
            return Err(GPUError::Validation(
                "TRANSIENT_ATTACHMENT requires COLOR_ATTACHMENT, DEPTH_STENCIL_ATTACHMENT, or INPUT_ATTACHMENT",
            ));
        }

        let disallowed_usage = TextureUses::COPY_SRC
            | TextureUses::COPY_DST
            | TextureUses::SAMPLED
            | TextureUses::STORAGE;
        if usage.intersects(disallowed_usage) {
            return Err(GPUError::Validation(
                "TRANSIENT_ATTACHMENT cannot be combined with COPY_SRC, COPY_DST, SAMPLED, or STORAGE",
            ));
        }
    }

    if desc.memory == MemoryPreset::TransientAttachment
        && !usage.contains(TextureUses::TRANSIENT_ATTACHMENT)
    {
        return Err(GPUError::Validation(
            "TransientAttachment memory requires TRANSIENT_ATTACHMENT usage",
        ));
    }

    if desc.samples != vk::SampleCountFlags::TYPE_1 && desc.mip_levels != 1 {
        return Err(GPUError::Validation(
            "multisampled 2D textures must use mip_levels = 1",
        ));
    }

    if desc.samples != vk::SampleCountFlags::TYPE_1 && desc.tiling == vk::ImageTiling::LINEAR {
        return Err(GPUError::Validation(
            "multisampled 2D textures cannot use linear tiling",
        ));
    }

    Ok(())
}

fn validate_image_desc(desc: &ImageDesc<'_>) -> Result<(), GPUError> {
    if desc.format == vk::Format::UNDEFINED {
        return Err(GPUError::Validation("image format must be defined"));
    }
    if desc.extent.width == 0 || desc.extent.height == 0 || desc.extent.depth == 0 {
        return Err(GPUError::Validation(
            "image extent components must be greater than zero",
        ));
    }
    if desc.mip_levels == 0 {
        return Err(GPUError::Validation(
            "image mip_levels must be greater than zero",
        ));
    }
    if desc.array_layers == 0 {
        return Err(GPUError::Validation(
            "image array_layers must be greater than zero",
        ));
    }
    if desc.usage.is_empty() {
        return Err(GPUError::Validation("image usage must not be empty"));
    }

    match desc.ty {
        vk::ImageType::TYPE_1D => {
            if desc.extent.height != 1 || desc.extent.depth != 1 {
                return Err(GPUError::Validation(
                    "1D images require extent.height = 1 and extent.depth = 1",
                ));
            }
        }
        vk::ImageType::TYPE_2D => {
            if desc.extent.depth != 1 {
                return Err(GPUError::Validation("2D images require extent.depth = 1"));
            }
        }
        vk::ImageType::TYPE_3D => {
            if desc.array_layers != 1 {
                return Err(GPUError::Validation("3D images require array_layers = 1"));
            }
        }
        _ => {}
    }

    if desc.usage.contains(ImageUses::TRANSIENT_ATTACHMENT) {
        let attachment_usage = ImageUses::COLOR_ATTACHMENT
            | ImageUses::DEPTH_STENCIL_ATTACHMENT
            | ImageUses::INPUT_ATTACHMENT;

        if !desc.usage.intersects(attachment_usage) {
            return Err(GPUError::Validation(
                "TRANSIENT_ATTACHMENT requires COLOR_ATTACHMENT, DEPTH_STENCIL_ATTACHMENT, or INPUT_ATTACHMENT",
            ));
        }

        let disallowed_usage =
            ImageUses::COPY_SRC | ImageUses::COPY_DST | ImageUses::SAMPLED | ImageUses::STORAGE;
        if desc.usage.intersects(disallowed_usage) {
            return Err(GPUError::Validation(
                "TRANSIENT_ATTACHMENT cannot be combined with COPY_SRC, COPY_DST, SAMPLED, or STORAGE",
            ));
        }
    }

    if desc.memory == MemoryPreset::TransientAttachment
        && !desc.usage.contains(ImageUses::TRANSIENT_ATTACHMENT)
    {
        return Err(GPUError::Validation(
            "TransientAttachment memory requires TRANSIENT_ATTACHMENT usage",
        ));
    }

    if desc
        .flags
        .intersects(ImageFlags::SPARSE_RESIDENCY | ImageFlags::SPARSE_ALIASED)
        && !desc.flags.contains(ImageFlags::SPARSE_BINDING)
    {
        return Err(GPUError::Validation(
            "SPARSE_RESIDENCY and SPARSE_ALIASED require SPARSE_BINDING",
        ));
    }

    if desc.flags.contains(ImageFlags::CUBE) {
        if desc.ty != vk::ImageType::TYPE_2D {
            return Err(GPUError::Validation(
                "CUBE images require image type TYPE_2D",
            ));
        }
        if desc.extent.width != desc.extent.height {
            return Err(GPUError::Validation(
                "CUBE images require square width and height",
            ));
        }
        if desc.array_layers < 6 {
            return Err(GPUError::Validation(
                "CUBE images require at least 6 array layers",
            ));
        }
    }

    if desc.samples != vk::SampleCountFlags::TYPE_1 {
        if desc.ty != vk::ImageType::TYPE_2D {
            return Err(GPUError::Validation(
                "multisampled images must use image type TYPE_2D",
            ));
        }
        if desc.mip_levels != 1 {
            return Err(GPUError::Validation(
                "multisampled images must use mip_levels = 1",
            ));
        }
        if desc.flags.contains(ImageFlags::CUBE) {
            return Err(GPUError::Validation(
                "multisampled images cannot use CUBE compatibility",
            ));
        }
        if desc.tiling == vk::ImageTiling::LINEAR {
            return Err(GPUError::Validation(
                "multisampled images cannot use linear tiling",
            ));
        }
    }

    if desc.memory == MemoryPreset::GpuOnly && desc.host_access != HostAccess::None {
        return Err(GPUError::Validation(
            "GpuOnly images cannot request host access; use Upload, Readback, or Dynamic",
        ));
    }

    Ok(())
}

fn validate_view_image_desc(desc: &ViewImageDesc<'_>) -> Result<(), GPUError> {
    validate_image_desc(&desc.image)?;

    if let Some(mips) = &desc.view_mips {
        if mips.start >= mips.end || mips.end > desc.image.mip_levels {
            return Err(GPUError::Validation(
                "view_mips must be a non-empty range within image mip_levels",
            ));
        }
    }

    if let Some(layers) = &desc.view_layers {
        if layers.start >= layers.end || layers.end > desc.image.array_layers {
            return Err(GPUError::Validation(
                "view_layers must be a non-empty range within image array_layers",
            ));
        }
    }

    Ok(())
}

fn infer_image_aspect(format: vk::Format, usage: ImageUses) -> vk::ImageAspectFlags {
    if usage.contains(ImageUses::DEPTH_STENCIL_ATTACHMENT) {
        return depth_stencil_aspect(format);
    }

    let depth_stencil = depth_stencil_aspect(format);
    if !depth_stencil.is_empty() {
        return depth_stencil;
    }

    vk::ImageAspectFlags::COLOR
}

fn infer_image_view_type(desc: &ImageDesc<'_>) -> vk::ImageViewType {
    if desc.flags.contains(ImageFlags::CUBE) {
        if desc.array_layers > 6 {
            return vk::ImageViewType::CUBE_ARRAY;
        }
        return vk::ImageViewType::CUBE;
    }

    match desc.ty {
        vk::ImageType::TYPE_1D if desc.array_layers > 1 => vk::ImageViewType::TYPE_1D_ARRAY,
        vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
        vk::ImageType::TYPE_2D if desc.array_layers > 1 => vk::ImageViewType::TYPE_2D_ARRAY,
        vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
        vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
        _ => vk::ImageViewType::TYPE_2D,
    }
}

fn depth_stencil_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::empty(),
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

#[derive(Debug, Clone)]
pub struct ViewImage {
    pub image: Image,
    pub sampler: Option<Sampler>,
    pub view: ImageView,
}

// TODO: detach from vulkan
#[derive(Debug, Clone)]
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

impl<'a> Default for SamplerCreateInfo<'a> {
    fn default() -> Self {
        Self {
            mag: vk::Filter::LINEAR,
            min: vk::Filter::LINEAR,
            mipmap: vk::SamplerMipmapMode::LINEAR,
            address_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            anisotropy: None,
            compare: None,
            min_lod: 0.0,
            max_lod: 0.0,
            label: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ImageViewOptions<'a> {
    pub sampler: Option<&'a Sampler>,
    pub ty: vk::ImageViewType,
    pub format: Option<vk::Format>,
    pub aspect: vk::ImageAspectFlags,
    pub swizzle: vk::ComponentMapping,
    pub mips: ops::Range<u32>,
    pub layers: ops::Range<u32>,
    pub label: Option<Label<'a>>,
}

#[derive(Debug, Clone)]
pub struct ImageViewCreateInfo<'a> {
    pub image: &'a Image,
    pub options: ImageViewOptions<'a>,
}

#[derive(Debug, Default)]
pub(crate) struct ImageCreateInfo<'a> {
    pub format: vk::Format,
    pub ty: vk::ImageType,
    pub volume: vk::Extent3D,
    pub mips: u32,
    pub layers: u32,
    pub tiling: vk::ImageTiling,
    pub samples: vk::SampleCountFlags,
    pub usage: ImageUsage,
    pub flags: vk::ImageCreateFlags,
    pub sharing: vk::SharingMode,
    pub layout: ImageLayout,
    pub label: Option<Label<'a>>,
}

#[derive(Debug, Copy, Clone)]
pub struct CopyImageInfo<'a> {
    pub src: &'a Image,
    pub src_layout: ImageLayout,
    pub dst: &'a Image,
    pub dst_layout: ImageLayout,
    pub regions: &'a [vk::ImageCopy],
}

#[derive(Debug, Copy, Clone)]
pub struct CopyBufferToImageInfo<'a> {
    pub src: &'a Buffer,
    pub dst: &'a Image,
    pub dst_layout: ImageLayout,
    pub regions: &'a [vk::BufferImageCopy],
}

#[derive(Debug, Copy, Clone)]
pub struct BlitImageInfo<'a> {
    pub src: &'a Image,
    pub src_layout: ImageLayout,
    pub dst: &'a Image,
    pub dst_layout: ImageLayout,
    pub regions: &'a [vk::ImageBlit],
    pub filter: vk::Filter,
}

impl SamplerImpl {
    pub unsafe fn new(device: RawDevice, info: &SamplerCreateInfo<'_>) -> Result<Self, GPUError> {
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

        let handle = unsafe { device.handle.create_sampler(&create_info, None) }?;

        if let Some(label) = &info.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(Self { handle, device })
    }
}

impl ImageViewImpl {
    pub unsafe fn new(device: RawDevice, info: &ImageViewCreateInfo<'_>) -> Result<Self, GPUError> {
        let options = &info.options;
        let mut create_info = vk::ImageViewCreateInfo::default()
            .image(info.image.inner.handle)
            .view_type(options.ty)
            .format(info.image.format)
            .components(options.swizzle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(options.aspect)
                    .base_mip_level(options.mips.start)
                    .level_count(options.mips.len() as u32)
                    .base_array_layer(options.layers.start)
                    .layer_count(options.layers.len() as u32),
            );

        if let Some(format) = options.format {
            create_info.format = format;
        }

        let handle = unsafe { device.handle.create_image_view(&create_info, None) }?;

        if let Some(label) = &options.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(Self {
            handle,
            device,
            image: info.image.inner.clone(),
        })
    }
}

impl ImageImpl {
    pub(crate) unsafe fn new_with_allocation(
        device: RawDevice,
        info: &ImageCreateInfo<'_>,
        create_info: vkm::AllocationCreateInfo,
    ) -> Result<Self, GPUError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(info.ty)
            .format(info.format)
            .extent(info.volume)
            .mip_levels(info.mips)
            .array_layers(info.layers)
            .samples(info.samples)
            .tiling(info.tiling)
            .usage(info.usage.into())
            .sharing_mode(info.sharing)
            .initial_layout(info.layout.into())
            .flags(info.flags | vk::ImageCreateFlags::from(info.usage));

        let (handle, allocation) =
            unsafe { device.allocator.create_image(&image_info, &create_info) }?;

        let allocation = Some(Allocation {
            handle: allocation,
            allocator: device.allocator.clone(),
        });

        if let Some(label) = &info.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(Self {
            handle,
            device,
            allocation,
        })
    }
}

fn allocation_create_info(
    memory: MemoryPreset,
    host_access: HostAccess,
) -> vkm::AllocationCreateInfo {
    let usage = match memory {
        MemoryPreset::GpuOnly => vkm::MemoryUsage::AutoPreferDevice,
        MemoryPreset::Upload | MemoryPreset::Readback => vkm::MemoryUsage::AutoPreferHost,
        MemoryPreset::Dynamic => vkm::MemoryUsage::AutoPreferDevice,
        MemoryPreset::TransientAttachment => vkm::MemoryUsage::GpuLazy,
    };

    let mut flags = vkm::AllocationCreateFlags::empty();
    let mut preferred_flags = vk::MemoryPropertyFlags::empty();

    match host_access {
        HostAccess::None => {}
        HostAccess::WriteSequential => {
            flags |= vkm::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
        }
        HostAccess::ReadRandom | HostAccess::ReadWriteRandom => {
            flags |= vkm::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
        }
    }

    vkm::AllocationCreateInfo {
        usage,
        flags,
        preferred_flags,
        ..Default::default()
    }
}

impl Device {
    pub fn try_create_sampler(&self, info: &SamplerCreateInfo<'_>) -> Result<Sampler, GPUError> {
        let inner = unsafe { SamplerImpl::new(self.inner.clone(), info)? };
        Ok(Sampler {
            inner: Arc::new(inner),
        })
    }

    pub fn create_sampler(&self, info: &SamplerCreateInfo<'_>) -> Sampler {
        self.try_create_sampler(info).expect("Create Sampler")
    }

    pub fn try_create_image_view(
        &self,
        info: &ImageViewCreateInfo<'_>,
    ) -> Result<ImageView, GPUError> {
        let inner = unsafe { ImageViewImpl::new(self.inner.clone(), info)? };
        let sampler = info.options.sampler.cloned();
        Ok(ImageView { inner, sampler })
    }

    pub fn create_image_view(&self, info: &ImageViewCreateInfo<'_>) -> ImageView {
        self.try_create_image_view(info).expect("Create Image View")
    }

    pub fn create_image(&self, desc: &ImageDesc<'_>) -> Result<Image, GPUError> {
        validate_image_desc(desc)?;

        let info = ImageCreateInfo {
            format: desc.format,
            ty: desc.ty,
            volume: desc.extent,
            mips: desc.mip_levels,
            layers: desc.array_layers,
            tiling: desc.tiling,
            samples: desc.samples,
            usage: ImageUsage::from(desc.usage),
            flags: desc.flags.into(),
            sharing: desc.sharing,
            layout: desc.initial_layout,
            label: desc.label.clone(),
        };

        let inner = unsafe {
            ImageImpl::new_with_allocation(
                self.inner.clone(),
                &info,
                allocation_create_info(desc.memory, desc.host_access),
            )?
        };

        Ok(Image {
            inner: Arc::new(inner),
            format: info.format,
        })
    }

    pub fn create_image_with(&self, desc: &ImageDesc<'_>) -> Result<Image, GPUError> {
        self.create_image(desc)
    }

    pub fn create_view_image(&self, desc: &ViewImageDesc<'_>) -> Result<ViewImage, GPUError> {
        validate_view_image_desc(desc)?;

        let image = self.create_image(&desc.image)?;

        let sampler = if let Some(sampler_desc) = desc.sampler.as_ref() {
            Some(self.try_create_sampler(sampler_desc)?)
        } else {
            None
        };

        let view = self.try_create_image_view(&ImageViewCreateInfo {
            image: &image,
            options: ImageViewOptions {
                sampler: sampler.as_ref(),
                ty: desc
                    .view_type
                    .unwrap_or_else(|| infer_image_view_type(&desc.image)),
                format: desc.view_format,
                aspect: desc
                    .aspect
                    .unwrap_or_else(|| infer_image_aspect(desc.image.format, desc.image.usage)),
                swizzle: desc.swizzle,
                mips: desc.view_mips.clone().unwrap_or(0..desc.image.mip_levels),
                layers: desc
                    .view_layers
                    .clone()
                    .unwrap_or(0..desc.image.array_layers),
                label: desc.view_label.clone(),
            },
        })?;

        Ok(ViewImage {
            image,
            sampler,
            view,
        })
    }

    pub fn create_view_image_with(&self, desc: &ViewImageDesc<'_>) -> Result<ViewImage, GPUError> {
        self.create_view_image(desc)
    }

    pub fn create_texture_2d(&self, desc: &Texture2DDesc<'_>) -> Result<ViewImage, GPUError> {
        validate_texture_2d_desc(desc)?;

        if matches!(
            desc.memory,
            MemoryPreset::Upload | MemoryPreset::Readback | MemoryPreset::Dynamic
        ) {
            return Err(GPUError::Validation(
                "Texture2DDesc only supports GpuOnly or TransientAttachment memory; use ImageDesc for explicit image allocation",
            ));
        }

        let mut usage = desc.usage;
        if desc.sampler.is_some() {
            usage |= TextureUses::SAMPLED;
        }

        self.create_view_image(&ViewImageDesc {
            image: ImageDesc {
                format: desc.format,
                ty: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: desc.size[0],
                    height: desc.size[1],
                    depth: 1,
                },
                mip_levels: desc.mip_levels,
                array_layers: desc.array_layers,
                tiling: desc.tiling,
                samples: desc.samples,
                usage,
                memory: desc.memory,
                label: desc.label.clone(),
                ..Default::default()
            },
            sampler: desc.sampler.clone(),
            view_type: None,
            view_format: desc.view_format,
            aspect: Some(infer_image_aspect(desc.format, usage)),
            view_mips: Some(0..desc.mip_levels),
            view_layers: Some(0..desc.array_layers),
            view_label: desc.view_label.clone(),
            ..Default::default()
        })
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
    pub const FRAGMENT: Self = Self::new(ImageLayout::Fragment);
    pub const COMPUTE: Self = Self::new(ImageLayout::Compute);
    pub const PRESENT: Self = Self::new(ImageLayout::Present);
    pub const COLOR: Self = Self::new(ImageLayout::Color);

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
