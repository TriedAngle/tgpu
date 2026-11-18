//! C ABI for [`tgpu`].
//!
//! Include `tgpu.h`. Every exported symbol is prefixed `tgpu_`, opaque handles
//! are `Tgpu*`, and every fallible call reports a [`TgpuStatus`] (or returns
//! `NULL` for constructors). See `tgpu_last_error` for the message.

mod bindless;
mod buffer;
mod command;
mod error;
mod image;
mod instance;
mod pipeline;
mod swapchain;
#[allow(dead_code)]
mod types;

pub use error::TgpuStatus;

/// Version of the C ABI, bumped whenever an exported signature or
/// `#[repr(C)]` type changes incompatibly.
#[unsafe(no_mangle)]
pub extern "C" fn tgpu_api_version() -> u32 {
    1
}
