use ash::vk;

use crate::macros::vk_value_enum;

vk_value_enum! {
    /// A pixel format.
    ///
    /// Names follow wgpu's `TextureFormat` conventions rather than Vulkan's:
    /// channels are grouped and camel-cased (`R8G8B8A8_SRGB` is
    /// [`Format::Rgba8UnormSrgb`]), floats are `Float`/`Ufloat`, packed formats
    /// carry a `Pack16`/`Pack32` suffix, and depth/stencil formats are
    /// `Depth*`/`Stencil*`.
    ///
    /// Covers the core uncompressed `VkFormat` values. Compressed formats
    /// (BC/ASTC/ETC2) and vendor formats are reachable through
    /// [`Format::Custom`], or simply by converting from [`vk::Format`].
    Format: vk::Format,
    default = UNDEFINED => UNDEFINED,
    {
        // 8-bit color.
        R8Unorm => R8_UNORM,
        R8Snorm => R8_SNORM,
        R8Uint => R8_UINT,
        R8Sint => R8_SINT,
        R8UnormSrgb => R8_SRGB,

        Rg8Unorm => R8G8_UNORM,
        Rg8Snorm => R8G8_SNORM,
        Rg8Uint => R8G8_UINT,
        Rg8Sint => R8G8_SINT,
        Rg8UnormSrgb => R8G8_SRGB,

        Rgb8Unorm => R8G8B8_UNORM,
        Rgb8Snorm => R8G8B8_SNORM,
        Rgb8Uint => R8G8B8_UINT,
        Rgb8Sint => R8G8B8_SINT,
        Rgb8UnormSrgb => R8G8B8_SRGB,

        Bgr8Unorm => B8G8R8_UNORM,
        Bgr8Snorm => B8G8R8_SNORM,
        Bgr8Uint => B8G8R8_UINT,
        Bgr8Sint => B8G8R8_SINT,
        Bgr8UnormSrgb => B8G8R8_SRGB,

        Rgba8Unorm => R8G8B8A8_UNORM,
        Rgba8Snorm => R8G8B8A8_SNORM,
        Rgba8Uint => R8G8B8A8_UINT,
        Rgba8Sint => R8G8B8A8_SINT,
        Rgba8UnormSrgb => R8G8B8A8_SRGB,

        Bgra8Unorm => B8G8R8A8_UNORM,
        Bgra8Snorm => B8G8R8A8_SNORM,
        Bgra8Uint => B8G8R8A8_UINT,
        Bgra8Sint => B8G8R8A8_SINT,
        Bgra8UnormSrgb => B8G8R8A8_SRGB,

        Abgr8UnormPack32 => A8B8G8R8_UNORM_PACK32,
        Abgr8SnormPack32 => A8B8G8R8_SNORM_PACK32,
        Abgr8UintPack32 => A8B8G8R8_UINT_PACK32,
        Abgr8SintPack32 => A8B8G8R8_SINT_PACK32,
        Abgr8UnormSrgbPack32 => A8B8G8R8_SRGB_PACK32,

        // Packed color.
        Rgba4UnormPack16 => R4G4B4A4_UNORM_PACK16,
        Bgra4UnormPack16 => B4G4R4A4_UNORM_PACK16,
        R5g6b5UnormPack16 => R5G6B5_UNORM_PACK16,
        B5g6r5UnormPack16 => B5G6R5_UNORM_PACK16,
        R5g5b5a1UnormPack16 => R5G5B5A1_UNORM_PACK16,
        B5g5r5a1UnormPack16 => B5G5R5A1_UNORM_PACK16,
        A1r5g5b5UnormPack16 => A1R5G5B5_UNORM_PACK16,
        A2r10g10b10UnormPack32 => A2R10G10B10_UNORM_PACK32,
        A2b10g10r10UnormPack32 => A2B10G10R10_UNORM_PACK32,
        B10g11r11UfloatPack32 => B10G11R11_UFLOAT_PACK32,
        E5b9g9r9UfloatPack32 => E5B9G9R9_UFLOAT_PACK32,

        // 16-bit color.
        R16Unorm => R16_UNORM,
        R16Snorm => R16_SNORM,
        R16Uint => R16_UINT,
        R16Sint => R16_SINT,
        R16Float => R16_SFLOAT,

        Rg16Unorm => R16G16_UNORM,
        Rg16Snorm => R16G16_SNORM,
        Rg16Uint => R16G16_UINT,
        Rg16Sint => R16G16_SINT,
        Rg16Float => R16G16_SFLOAT,

        Rgb16Unorm => R16G16B16_UNORM,
        Rgb16Snorm => R16G16B16_SNORM,
        Rgb16Uint => R16G16B16_UINT,
        Rgb16Sint => R16G16B16_SINT,
        Rgb16Float => R16G16B16_SFLOAT,

        Rgba16Unorm => R16G16B16A16_UNORM,
        Rgba16Snorm => R16G16B16A16_SNORM,
        Rgba16Uint => R16G16B16A16_UINT,
        Rgba16Sint => R16G16B16A16_SINT,
        Rgba16Float => R16G16B16A16_SFLOAT,

        // 32-bit color.
        R32Uint => R32_UINT,
        R32Sint => R32_SINT,
        R32Float => R32_SFLOAT,

        Rg32Uint => R32G32_UINT,
        Rg32Sint => R32G32_SINT,
        Rg32Float => R32G32_SFLOAT,

        Rgb32Uint => R32G32B32_UINT,
        Rgb32Sint => R32G32B32_SINT,
        Rgb32Float => R32G32B32_SFLOAT,

        Rgba32Uint => R32G32B32A32_UINT,
        Rgba32Sint => R32G32B32A32_SINT,
        Rgba32Float => R32G32B32A32_SFLOAT,

        // 64-bit color.
        R64Uint => R64_UINT,
        R64Sint => R64_SINT,
        R64Float => R64_SFLOAT,

        Rg64Uint => R64G64_UINT,
        Rg64Sint => R64G64_SINT,
        Rg64Float => R64G64_SFLOAT,

        Rgb64Uint => R64G64B64_UINT,
        Rgb64Sint => R64G64B64_SINT,
        Rgb64Float => R64G64B64_SFLOAT,

        Rgba64Uint => R64G64B64A64_UINT,
        Rgba64Sint => R64G64B64A64_SINT,
        Rgba64Float => R64G64B64A64_SFLOAT,

        // Depth/stencil.
        Depth16Unorm => D16_UNORM,
        Depth24Unorm => X8_D24_UNORM_PACK32,
        Depth32Float => D32_SFLOAT,
        Stencil8 => S8_UINT,
        Depth16UnormStencil8 => D16_UNORM_S8_UINT,
        Depth24UnormStencil8 => D24_UNORM_S8_UINT,
        Depth32FloatStencil8 => D32_SFLOAT_S8_UINT,
    }
}

impl Format {
    /// Returns whether this format is an sRGB-encoded color format.
    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            Self::R8UnormSrgb
                | Self::Rg8UnormSrgb
                | Self::Rgb8UnormSrgb
                | Self::Bgr8UnormSrgb
                | Self::Rgba8UnormSrgb
                | Self::Bgra8UnormSrgb
                | Self::Abgr8UnormSrgbPack32
        )
    }

    /// Returns whether this format has a depth and/or stencil component.
    pub fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth24Unorm
                | Self::Depth32Float
                | Self::Stencil8
                | Self::Depth16UnormStencil8
                | Self::Depth24UnormStencil8
                | Self::Depth32FloatStencil8
        )
    }
}

vk_value_enum! {
    /// A surface color space.
    ColorSpace: vk::ColorSpaceKHR,
    default = SRGB_NONLINEAR => SRGB_NONLINEAR,
    {
        DISPLAY_P3_NONLINEAR_EXT => DISPLAY_P3_NONLINEAR_EXT,
        EXTENDED_SRGB_LINEAR_EXT => EXTENDED_SRGB_LINEAR_EXT,
        EXTENDED_SRGB_NONLINEAR_EXT => EXTENDED_SRGB_NONLINEAR_EXT,
        DISPLAY_P3_LINEAR_EXT => DISPLAY_P3_LINEAR_EXT,
        DCI_P3_NONLINEAR_EXT => DCI_P3_NONLINEAR_EXT,
        BT709_LINEAR_EXT => BT709_LINEAR_EXT,
        BT709_NONLINEAR_EXT => BT709_NONLINEAR_EXT,
        BT2020_LINEAR_EXT => BT2020_LINEAR_EXT,
        HDR10_ST2084_EXT => HDR10_ST2084_EXT,
        HDR10_HLG_EXT => HDR10_HLG_EXT,
        ADOBERGB_LINEAR_EXT => ADOBERGB_LINEAR_EXT,
        ADOBERGB_NONLINEAR_EXT => ADOBERGB_NONLINEAR_EXT,
        PASS_THROUGH_EXT => PASS_THROUGH_EXT,
    }
}

/// A swapchain surface format: a [`Format`] paired with a [`ColorSpace`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct SurfaceFormat {
    pub format: Format,
    pub color_space: ColorSpace,
}

impl SurfaceFormat {
    pub const fn new(format: Format, color_space: ColorSpace) -> Self {
        Self {
            format,
            color_space,
        }
    }
}

impl From<vk::SurfaceFormatKHR> for SurfaceFormat {
    fn from(value: vk::SurfaceFormatKHR) -> Self {
        Self {
            format: value.format.into(),
            color_space: value.color_space.into(),
        }
    }
}

impl From<SurfaceFormat> for vk::SurfaceFormatKHR {
    fn from(value: SurfaceFormat) -> Self {
        Self {
            format: value.format.into(),
            color_space: value.color_space.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_formats_round_trip() {
        for format in Format::ALL {
            assert_eq!(format.as_raw(), vk::Format::from(*format).as_raw());
            assert_eq!(Format::from_raw(format.as_raw()), *format);
            assert_eq!(Format::from(vk::Format::from(*format)), *format);
        }
    }

    #[test]
    fn unmapped_formats_use_custom() {
        let format = Format::from(vk::Format::ASTC_4X4_UNORM_BLOCK);
        assert!(matches!(format, Format::Custom(_)));
        assert_eq!(Format::from_raw(format.as_raw()), format);
        assert_eq!(
            vk::Format::from(format).as_raw(),
            vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw()
        );
    }

    #[test]
    fn named_color_spaces_round_trip() {
        for space in ColorSpace::ALL {
            assert_eq!(ColorSpace::from_raw(space.as_raw()), *space);
            assert_eq!(ColorSpace::from(vk::ColorSpaceKHR::from(*space)), *space);
        }
    }

    #[test]
    fn format_predicates() {
        assert!(Format::Bgra8UnormSrgb.is_srgb());
        assert!(!Format::Bgra8Unorm.is_srgb());
        assert!(Format::Depth32Float.is_depth_stencil());
        assert!(!Format::Rgb32Float.is_depth_stencil());
    }
}
