use ash::vk;

macro_rules! vk_value_enum {
    (
        $(#[$meta:meta])*
        $name:ident : $vk_ty:ty,
        default = $default:ident => $default_vk:ident,
        { $($variant:ident => $vk_value:ident),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[allow(non_camel_case_types)]
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
        pub enum $name {
            #[default]
            $default,
            $($variant,)+
            /// A value with no named variant. Also used for driver- or
            /// extension-provided values this type does not enumerate.
            Custom($vk_ty),
        }

        impl From<$name> for $vk_ty {
            fn from(value: $name) -> Self {
                match value {
                    $name::$default => <$vk_ty>::$default_vk,
                    $($name::$variant => <$vk_ty>::$vk_value,)+
                    $name::Custom(raw) => raw,
                }
            }
        }

        impl From<$vk_ty> for $name {
            fn from(value: $vk_ty) -> Self {
                match value {
                    <$vk_ty>::$default_vk => $name::$default,
                    $(<$vk_ty>::$vk_value => $name::$variant,)+
                    raw => $name::Custom(raw),
                }
            }
        }

        impl $name {
            /// Every named variant, in declaration order.
            pub const ALL: &'static [Self] = &[$name::$default, $($name::$variant,)+];

            /// Returns the raw Khronos value.
            pub fn as_raw(self) -> i32 {
                let raw: $vk_ty = self.into();
                raw.as_raw()
            }

            /// Builds a value from a raw Khronos value, mapping known values to
            /// their named variant.
            pub fn from_raw(raw: i32) -> Self {
                <$vk_ty>::from_raw(raw).into()
            }
        }
    };
}

vk_value_enum! {
    /// A pixel format.
    ///
    /// Covers the core uncompressed `VkFormat` values. Compressed formats
    /// (BC/ASTC/ETC2) and vendor formats are reachable through
    /// [`Format::Custom`], or simply by converting from [`vk::Format`].
    Format: vk::Format,
    default = UNDEFINED => UNDEFINED,
    {
        // 8-bit color.
        R8_UNORM => R8_UNORM,
        R8_SNORM => R8_SNORM,
        R8_UINT => R8_UINT,
        R8_SINT => R8_SINT,
        R8_SRGB => R8_SRGB,

        R8G8_UNORM => R8G8_UNORM,
        R8G8_SNORM => R8G8_SNORM,
        R8G8_UINT => R8G8_UINT,
        R8G8_SINT => R8G8_SINT,
        R8G8_SRGB => R8G8_SRGB,

        R8G8B8_UNORM => R8G8B8_UNORM,
        R8G8B8_SNORM => R8G8B8_SNORM,
        R8G8B8_UINT => R8G8B8_UINT,
        R8G8B8_SINT => R8G8B8_SINT,
        R8G8B8_SRGB => R8G8B8_SRGB,

        B8G8R8_UNORM => B8G8R8_UNORM,
        B8G8R8_SNORM => B8G8R8_SNORM,
        B8G8R8_UINT => B8G8R8_UINT,
        B8G8R8_SINT => B8G8R8_SINT,
        B8G8R8_SRGB => B8G8R8_SRGB,

        R8G8B8A8_UNORM => R8G8B8A8_UNORM,
        R8G8B8A8_SNORM => R8G8B8A8_SNORM,
        R8G8B8A8_UINT => R8G8B8A8_UINT,
        R8G8B8A8_SINT => R8G8B8A8_SINT,
        R8G8B8A8_SRGB => R8G8B8A8_SRGB,

        B8G8R8A8_UNORM => B8G8R8A8_UNORM,
        B8G8R8A8_SNORM => B8G8R8A8_SNORM,
        B8G8R8A8_UINT => B8G8R8A8_UINT,
        B8G8R8A8_SINT => B8G8R8A8_SINT,
        B8G8R8A8_SRGB => B8G8R8A8_SRGB,

        A8B8G8R8_UNORM_PACK32 => A8B8G8R8_UNORM_PACK32,
        A8B8G8R8_SNORM_PACK32 => A8B8G8R8_SNORM_PACK32,
        A8B8G8R8_UINT_PACK32 => A8B8G8R8_UINT_PACK32,
        A8B8G8R8_SINT_PACK32 => A8B8G8R8_SINT_PACK32,
        A8B8G8R8_SRGB_PACK32 => A8B8G8R8_SRGB_PACK32,

        // Packed color.
        R4G4B4A4_UNORM_PACK16 => R4G4B4A4_UNORM_PACK16,
        B4G4R4A4_UNORM_PACK16 => B4G4R4A4_UNORM_PACK16,
        R5G6B5_UNORM_PACK16 => R5G6B5_UNORM_PACK16,
        B5G6R5_UNORM_PACK16 => B5G6R5_UNORM_PACK16,
        R5G5B5A1_UNORM_PACK16 => R5G5B5A1_UNORM_PACK16,
        B5G5R5A1_UNORM_PACK16 => B5G5R5A1_UNORM_PACK16,
        A1R5G5B5_UNORM_PACK16 => A1R5G5B5_UNORM_PACK16,
        A2R10G10B10_UNORM_PACK32 => A2R10G10B10_UNORM_PACK32,
        A2B10G10R10_UNORM_PACK32 => A2B10G10R10_UNORM_PACK32,
        B10G11R11_UFLOAT_PACK32 => B10G11R11_UFLOAT_PACK32,
        E5B9G9R9_UFLOAT_PACK32 => E5B9G9R9_UFLOAT_PACK32,

        // 16-bit color.
        R16_UNORM => R16_UNORM,
        R16_SNORM => R16_SNORM,
        R16_UINT => R16_UINT,
        R16_SINT => R16_SINT,
        R16_SFLOAT => R16_SFLOAT,

        R16G16_UNORM => R16G16_UNORM,
        R16G16_SNORM => R16G16_SNORM,
        R16G16_UINT => R16G16_UINT,
        R16G16_SINT => R16G16_SINT,
        R16G16_SFLOAT => R16G16_SFLOAT,

        R16G16B16_UNORM => R16G16B16_UNORM,
        R16G16B16_SNORM => R16G16B16_SNORM,
        R16G16B16_UINT => R16G16B16_UINT,
        R16G16B16_SINT => R16G16B16_SINT,
        R16G16B16_SFLOAT => R16G16B16_SFLOAT,

        R16G16B16A16_UNORM => R16G16B16A16_UNORM,
        R16G16B16A16_SNORM => R16G16B16A16_SNORM,
        R16G16B16A16_UINT => R16G16B16A16_UINT,
        R16G16B16A16_SINT => R16G16B16A16_SINT,
        R16G16B16A16_SFLOAT => R16G16B16A16_SFLOAT,

        // 32-bit color.
        R32_UINT => R32_UINT,
        R32_SINT => R32_SINT,
        R32_SFLOAT => R32_SFLOAT,

        R32G32_UINT => R32G32_UINT,
        R32G32_SINT => R32G32_SINT,
        R32G32_SFLOAT => R32G32_SFLOAT,

        R32G32B32_UINT => R32G32B32_UINT,
        R32G32B32_SINT => R32G32B32_SINT,
        R32G32B32_SFLOAT => R32G32B32_SFLOAT,

        R32G32B32A32_UINT => R32G32B32A32_UINT,
        R32G32B32A32_SINT => R32G32B32A32_SINT,
        R32G32B32A32_SFLOAT => R32G32B32A32_SFLOAT,

        // 64-bit color.
        R64_UINT => R64_UINT,
        R64_SINT => R64_SINT,
        R64_SFLOAT => R64_SFLOAT,

        R64G64_UINT => R64G64_UINT,
        R64G64_SINT => R64G64_SINT,
        R64G64_SFLOAT => R64G64_SFLOAT,

        R64G64B64_UINT => R64G64B64_UINT,
        R64G64B64_SINT => R64G64B64_SINT,
        R64G64B64_SFLOAT => R64G64B64_SFLOAT,

        R64G64B64A64_UINT => R64G64B64A64_UINT,
        R64G64B64A64_SINT => R64G64B64A64_SINT,
        R64G64B64A64_SFLOAT => R64G64B64A64_SFLOAT,

        // Depth/stencil.
        D16_UNORM => D16_UNORM,
        X8_D24_UNORM_PACK32 => X8_D24_UNORM_PACK32,
        D32_SFLOAT => D32_SFLOAT,
        S8_UINT => S8_UINT,
        D16_UNORM_S8_UINT => D16_UNORM_S8_UINT,
        D24_UNORM_S8_UINT => D24_UNORM_S8_UINT,
        D32_SFLOAT_S8_UINT => D32_SFLOAT_S8_UINT,
    }
}

impl Format {
    /// Returns whether this format is an sRGB-encoded color format.
    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            Self::R8_SRGB
                | Self::R8G8_SRGB
                | Self::R8G8B8_SRGB
                | Self::B8G8R8_SRGB
                | Self::R8G8B8A8_SRGB
                | Self::B8G8R8A8_SRGB
                | Self::A8B8G8R8_SRGB_PACK32
        )
    }

    /// Returns whether this format has a depth and/or stencil component.
    pub fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::D16_UNORM
                | Self::X8_D24_UNORM_PACK32
                | Self::D32_SFLOAT
                | Self::S8_UINT
                | Self::D16_UNORM_S8_UINT
                | Self::D24_UNORM_S8_UINT
                | Self::D32_SFLOAT_S8_UINT
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
        assert!(Format::B8G8R8A8_SRGB.is_srgb());
        assert!(!Format::B8G8R8A8_UNORM.is_srgb());
        assert!(Format::D32_SFLOAT.is_depth_stencil());
        assert!(!Format::R32G32B32_SFLOAT.is_depth_stencil());
    }
}
