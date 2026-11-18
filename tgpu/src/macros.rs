//! Small internal helpers shared by several modules.

/// Declares an enum that mirrors a Khronos value enum, with a `Custom`
/// variant for values this crate does not enumerate.
///
/// Generates the type, both `From` directions, `ALL`, `as_raw` and `from_raw`.
/// Every `#[repr(i32)]` enum in `ash` round-trips through `as_raw`/`from_raw`.
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

pub(crate) use vk_value_enum;
