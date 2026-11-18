use std::ffi::c_void;

use tgpu::{
    BufferAccessTransition, BufferUses, CompareOp, CullMode, FrontFace, HostAccess, ImageLayout,
    ImageUses, MemoryPreset, PrimitiveTopology, VertexInputRate,
};

// ---------------------------------------------------------------------------
// Small POD mirrors
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TgpuExtent2D {
    pub width: u32,
    pub height: u32,
}

impl From<TgpuExtent2D> for tgpu::Extent2D {
    fn from(value: TgpuExtent2D) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

impl From<tgpu::Extent2D> for TgpuExtent2D {
    fn from(value: tgpu::Extent2D) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TgpuExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl From<TgpuExtent3D> for tgpu::Extent3D {
    fn from(value: TgpuExtent3D) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

/// An area, with the same meaning as `VkRect2D`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TgpuRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl From<TgpuRect> for tgpu::Rect {
    fn from(value: TgpuRect) -> Self {
        Self {
            offset: tgpu::Offset {
                x: value.x,
                y: value.y,
            },
            extent: tgpu::Extent2D {
                width: value.width,
                height: value.height,
            },
        }
    }
}

/// A raw window and display, tagged by platform.
///
/// `window` is an `NSView*` (AppKit), `Window` (Xlib), `HWND` (Win32) or
/// `wl_surface*` (Wayland). `display` is `Display*` on Xlib, `wl_display*` on
/// Wayland, and unused (NULL) on AppKit and Win32.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuWindow {
    pub kind: TgpuWindowKind,
    pub display: *mut c_void,
    pub window: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TgpuWindowKind {
    AppKit = 1,
    Xlib = 2,
    Win32 = 3,
    Wayland = 4,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// A `VkFormat` value.
pub type TgpuFormat = i32;
pub const TGPU_FORMAT_UNDEFINED: TgpuFormat = 0;
/// `VK_FORMAT_R8G8B8A8_UNORM`
pub const TGPU_FORMAT_RGBA8_UNORM: TgpuFormat = 37;
/// `VK_FORMAT_R32G32_SFLOAT`
pub const TGPU_FORMAT_RG32_FLOAT: TgpuFormat = 103;
/// `VK_FORMAT_R32G32B32_SFLOAT`
pub const TGPU_FORMAT_RGB32_FLOAT: TgpuFormat = 106;
/// `VK_FORMAT_D32_SFLOAT`
pub const TGPU_FORMAT_DEPTH32_FLOAT: TgpuFormat = 126;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuMemoryPreset {
    #[default]
    GpuOnly = 0,
    Upload = 1,
    Readback = 2,
    Dynamic = 3,
    TransientAttachment = 4,
}

impl From<TgpuMemoryPreset> for MemoryPreset {
    fn from(value: TgpuMemoryPreset) -> Self {
        match value {
            TgpuMemoryPreset::GpuOnly => Self::GpuOnly,
            TgpuMemoryPreset::Upload => Self::Upload,
            TgpuMemoryPreset::Readback => Self::Readback,
            TgpuMemoryPreset::Dynamic => Self::Dynamic,
            TgpuMemoryPreset::TransientAttachment => Self::TransientAttachment,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuHostAccess {
    #[default]
    None = 0,
    WriteSequential = 1,
    ReadRandom = 2,
    ReadWriteRandom = 3,
}

impl From<TgpuHostAccess> for HostAccess {
    fn from(value: TgpuHostAccess) -> Self {
        match value {
            TgpuHostAccess::None => Self::None,
            TgpuHostAccess::WriteSequential => Self::WriteSequential,
            TgpuHostAccess::ReadRandom => Self::ReadRandom,
            TgpuHostAccess::ReadWriteRandom => Self::ReadWriteRandom,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuImageLayout {
    #[default]
    Undefined = 0,
    General = 1,
    Compute = 2,
    Fragment = 3,
    Color = 4,
    DepthStencil = 5,
    TransferDst = 6,
    Present = 7,
}

impl From<TgpuImageLayout> for ImageLayout {
    fn from(value: TgpuImageLayout) -> Self {
        match value {
            TgpuImageLayout::Undefined => Self::Undefined,
            TgpuImageLayout::General => Self::General,
            TgpuImageLayout::Compute => Self::Compute,
            TgpuImageLayout::Fragment => Self::Fragment,
            TgpuImageLayout::Color => Self::Color,
            TgpuImageLayout::DepthStencil => Self::DepthStencil,
            TgpuImageLayout::TransferDst => Self::TransferDst,
            TgpuImageLayout::Present => Self::Present,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuPresentMode {
    #[default]
    Fifo = 0,
    FifoRelaxed = 1,
    Immediate = 2,
    Mailbox = 3,
}

impl From<TgpuPresentMode> for tgpu::PresentMode {
    fn from(value: TgpuPresentMode) -> Self {
        match value {
            TgpuPresentMode::Fifo => Self::Fifo,
            TgpuPresentMode::FifoRelaxed => Self::FifoRelaxed,
            TgpuPresentMode::Immediate => Self::Immediate,
            TgpuPresentMode::Mailbox => Self::Mailbox,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuLoadOp {
    #[default]
    Load = 0,
    Clear = 1,
    DontCare = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuStoreOp {
    #[default]
    Store = 0,
    DontCare = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuCompareOp {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    #[default]
    Always = 7,
}

impl From<TgpuCompareOp> for CompareOp {
    fn from(value: TgpuCompareOp) -> Self {
        match value {
            TgpuCompareOp::Never => Self::Never,
            TgpuCompareOp::Less => Self::Less,
            TgpuCompareOp::Equal => Self::Equal,
            TgpuCompareOp::LessOrEqual => Self::LessOrEqual,
            TgpuCompareOp::Greater => Self::Greater,
            TgpuCompareOp::NotEqual => Self::NotEqual,
            TgpuCompareOp::GreaterOrEqual => Self::GreaterOrEqual,
            TgpuCompareOp::Always => Self::Always,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuCullMode {
    #[default]
    None = 0,
    Front = 1,
    Back = 2,
    FrontAndBack = 3,
}

impl From<TgpuCullMode> for CullMode {
    fn from(value: TgpuCullMode) -> Self {
        match value {
            TgpuCullMode::None => Self::NONE,
            TgpuCullMode::Front => Self::FRONT,
            TgpuCullMode::Back => Self::BACK,
            TgpuCullMode::FrontAndBack => Self::FRONT_AND_BACK,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuFrontFace {
    #[default]
    CounterClockwise = 0,
    Clockwise = 1,
}

impl From<TgpuFrontFace> for FrontFace {
    fn from(value: TgpuFrontFace) -> Self {
        match value {
            TgpuFrontFace::CounterClockwise => Self::CounterClockwise,
            TgpuFrontFace::Clockwise => Self::Clockwise,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuPrimitiveTopology {
    #[default]
    TriangleList = 0,
    PointList = 1,
    LineList = 2,
    LineStrip = 3,
    TriangleStrip = 4,
    TriangleFan = 5,
}

impl From<TgpuPrimitiveTopology> for PrimitiveTopology {
    fn from(value: TgpuPrimitiveTopology) -> Self {
        match value {
            TgpuPrimitiveTopology::TriangleList => Self::TriangleList,
            TgpuPrimitiveTopology::PointList => Self::PointList,
            TgpuPrimitiveTopology::LineList => Self::LineList,
            TgpuPrimitiveTopology::LineStrip => Self::LineStrip,
            TgpuPrimitiveTopology::TriangleStrip => Self::TriangleStrip,
            TgpuPrimitiveTopology::TriangleFan => Self::TriangleFan,
        }
    }
}

/// The stage/access pair for one side of a buffer barrier.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuBufferAccess {
    #[default]
    None = 0,
    TransferSrc = 1,
    TransferDst = 2,
    Indirect = 3,
    ComputeStorageRead = 4,
    ComputeStorageWrite = 5,
    ComputeStorageReadWrite = 6,
    UniformRead = 7,
    VertexInputRead = 8,
}

impl From<TgpuBufferAccess> for BufferAccessTransition {
    fn from(value: TgpuBufferAccess) -> Self {
        match value {
            TgpuBufferAccess::None => Self::NONE,
            TgpuBufferAccess::TransferSrc => Self::TRANSFER_SRC,
            TgpuBufferAccess::TransferDst => Self::TRANSFER_DST,
            TgpuBufferAccess::Indirect => Self::INDIRECT,
            TgpuBufferAccess::ComputeStorageRead => Self::compute_storage_read(),
            TgpuBufferAccess::ComputeStorageWrite => Self::compute_storage_write(),
            TgpuBufferAccess::ComputeStorageReadWrite => Self::compute_storage_read_write(),
            TgpuBufferAccess::UniformRead => Self::uniform_read(),
            TgpuBufferAccess::VertexInputRead => Self::vertex_input_read(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuVertexInputRate {
    #[default]
    Vertex = 0,
    Instance = 1,
}

impl From<TgpuVertexInputRate> for VertexInputRate {
    fn from(value: TgpuVertexInputRate) -> Self {
        match value {
            TgpuVertexInputRate::Vertex => Self::Vertex,
            TgpuVertexInputRate::Instance => Self::Instance,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuBlend {
    #[default]
    Disabled = 0,
    Alpha = 1,
}

impl From<TgpuBlend> for tgpu::ColorBlend {
    fn from(value: TgpuBlend) -> Self {
        match value {
            TgpuBlend::Disabled => Self::Disabled,
            TgpuBlend::Alpha => Self::Alpha,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuFilter {
    Nearest = 0,
    #[default]
    Linear = 1,
}

impl From<TgpuFilter> for tgpu::Filter {
    fn from(value: TgpuFilter) -> Self {
        match value {
            TgpuFilter::Nearest => Self::Nearest,
            TgpuFilter::Linear => Self::Linear,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuSamplerMipmapMode {
    Nearest = 0,
    #[default]
    Linear = 1,
}

impl From<TgpuSamplerMipmapMode> for tgpu::SamplerMipmapMode {
    fn from(value: TgpuSamplerMipmapMode) -> Self {
        match value {
            TgpuSamplerMipmapMode::Nearest => Self::Nearest,
            TgpuSamplerMipmapMode::Linear => Self::Linear,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TgpuSamplerAddressMode {
    Repeat = 0,
    MirroredRepeat = 1,
    #[default]
    ClampToEdge = 2,
    ClampToBorder = 3,
    MirrorClampToEdge = 4,
}

impl From<TgpuSamplerAddressMode> for tgpu::SamplerAddressMode {
    fn from(value: TgpuSamplerAddressMode) -> Self {
        match value {
            TgpuSamplerAddressMode::Repeat => Self::Repeat,
            TgpuSamplerAddressMode::MirroredRepeat => Self::MirroredRepeat,
            TgpuSamplerAddressMode::ClampToEdge => Self::ClampToEdge,
            TgpuSamplerAddressMode::ClampToBorder => Self::ClampToBorder,
            TgpuSamplerAddressMode::MirrorClampToEdge => Self::MirrorClampToEdge,
        }
    }
}

// ---------------------------------------------------------------------------
// Bitflags and capability constants
// ---------------------------------------------------------------------------

pub const TGPU_BUFFER_USAGE_COPY_SRC: u32 = 1 << 0;
pub const TGPU_BUFFER_USAGE_COPY_DST: u32 = 1 << 1;
pub const TGPU_BUFFER_USAGE_INDEX: u32 = 1 << 2;
pub const TGPU_BUFFER_USAGE_VERTEX: u32 = 1 << 3;
pub const TGPU_BUFFER_USAGE_UNIFORM: u32 = 1 << 4;
pub const TGPU_BUFFER_USAGE_STORAGE: u32 = 1 << 5;
pub const TGPU_BUFFER_USAGE_DEVICE_ADDRESS: u32 = 1 << 6;
pub const TGPU_BUFFER_USAGE_INDIRECT: u32 = 1 << 7;

/// Returned by the `tgpu_bindless_add_*` functions on failure.
pub const TGPU_INVALID_HANDLE: u32 = u32::MAX;

pub fn buffer_uses(bits: u32) -> BufferUses {
    BufferUses::from_bits_retain(bits)
}

pub const TGPU_IMAGE_USAGE_COPY_SRC: u32 = 1 << 0;
pub const TGPU_IMAGE_USAGE_COPY_DST: u32 = 1 << 1;
pub const TGPU_IMAGE_USAGE_SAMPLED: u32 = 1 << 2;
pub const TGPU_IMAGE_USAGE_STORAGE: u32 = 1 << 3;
pub const TGPU_IMAGE_USAGE_COLOR_ATTACHMENT: u32 = 1 << 4;
pub const TGPU_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT: u32 = 1 << 5;
pub const TGPU_IMAGE_USAGE_INPUT_ATTACHMENT: u32 = 1 << 6;
pub const TGPU_IMAGE_USAGE_TRANSIENT_ATTACHMENT: u32 = 1 << 7;

pub fn image_uses(bits: u32) -> ImageUses {
    ImageUses::from_bits_retain(bits)
}

pub const TGPU_QUEUE_GRAPHICS: u32 = 1 << 0;
pub const TGPU_QUEUE_COMPUTE: u32 = 1 << 1;
pub const TGPU_QUEUE_TRANSFER: u32 = 1 << 2;
pub const TGPU_QUEUE_SPARSE_BINDING: u32 = 1 << 3;
pub const TGPU_QUEUE_PROTECTED: u32 = 1 << 4;
pub const TGPU_QUEUE_GRAPHICS_COMPUTE: u32 = TGPU_QUEUE_GRAPHICS | TGPU_QUEUE_COMPUTE;

pub const TGPU_DEVICE_FEATURE_FILL_MODE_NON_SOLID: u32 = 1 << 0;
pub const TGPU_DEVICE_FEATURE_MULTI_DRAW_INDIRECT: u32 = 1 << 1;
pub const TGPU_DEVICE_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE: u32 = 1 << 2;
pub const TGPU_DEVICE_FEATURE_DESCRIPTOR_INDEXING: u32 = 1 << 3;
pub const TGPU_DEVICE_FEATURE_BUFFER_DEVICE_ADDRESS: u32 = 1 << 4;
/// The features the rest of this library assumes are available.
pub const TGPU_DEVICE_FEATURES_MODERN: u32 = TGPU_DEVICE_FEATURE_MULTI_DRAW_INDIRECT
    | TGPU_DEVICE_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE
    | TGPU_DEVICE_FEATURE_DESCRIPTOR_INDEXING;

pub const TGPU_ADAPTER_DEVICE_TYPE_OTHER: u32 = 0;
pub const TGPU_ADAPTER_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
pub const TGPU_ADAPTER_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
pub const TGPU_ADAPTER_DEVICE_TYPE_VIRTUAL_GPU: u32 = 3;
pub const TGPU_ADAPTER_DEVICE_TYPE_CPU: u32 = 4;
