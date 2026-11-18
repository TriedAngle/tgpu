use std::{
    any::Any,
    cell::RefCell,
    ffi::{CStr, CString, c_char},
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

use tgpu::GPUError;

/// Result of every fallible call in the C API.
///
/// Calls returning an opaque handle return `NULL` instead; the reason is
/// available through [`tgpu_last_error`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TgpuStatus {
    Ok = 0,
    InvalidArgument = 1,
    Validation = 2,
    Vulkan = 3,
    Shader = 4,
    /// A Rust panic was caught at the ABI boundary. The library is still
    /// usable, but the call that panicked had no effect.
    Panic = 5,
}

/// Owned error used inside the binding, so messages can be formatted.
#[derive(Debug)]
pub struct Error {
    pub status: TgpuStatus,
    pub message: String,
}

impl Error {
    pub fn validation(message: impl fmt::Display) -> Self {
        Self {
            status: TgpuStatus::Validation,
            message: message.to_string(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<GPUError> for Error {
    fn from(value: GPUError) -> Self {
        let status = match &value {
            GPUError::Vulkan(_) => TgpuStatus::Vulkan,
            GPUError::Validation(_) => TgpuStatus::Validation,
            GPUError::Shader(_) => TgpuStatus::Shader,
        };
        Self {
            status,
            message: value.to_string(),
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<(TgpuStatus, CString)>> = const { RefCell::new(None) };
}

pub fn set_last_error(status: TgpuStatus, message: impl fmt::Display) {
    let mut message = message.to_string();
    if message.as_bytes().contains(&0) {
        message = message.replace('\0', " ");
    }
    let c_message = CString::new(message).unwrap_or_default();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some((status, c_message)));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Copies the last error message into `buffer` (always NUL terminated) and
/// returns its status. Returns [`TgpuStatus::Ok`] when nothing failed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_last_error(buffer: *mut c_char, capacity: usize) -> TgpuStatus {
    LAST_ERROR.with(|slot| {
        let slot = slot.borrow();
        let Some((status, message)) = slot.as_ref() else {
            if !buffer.is_null() && capacity > 0 {
                unsafe { *buffer = 0 };
            }
            return TgpuStatus::Ok;
        };

        if !buffer.is_null() && capacity > 0 {
            let bytes = message.as_bytes_with_nul();
            let count = bytes.len().min(capacity);
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), buffer, count);
                *buffer.add(count - 1) = 0;
            }
        }

        *status
    })
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "panic in tgpu-native".to_string()
    }
}

/// Runs `f`, converting errors and panics into a status.
pub fn guard<T>(f: impl FnOnce() -> Result<T, Error>) -> Option<T> {
    clear_last_error();

    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(error)) => {
            set_last_error(error.status, &error.message);
            None
        }
        Err(payload) => {
            set_last_error(TgpuStatus::Panic, panic_message(payload));
            None
        }
    }
}

/// [`guard`] for functions that only report success or failure.
pub fn guard_status(f: impl FnOnce() -> Result<(), Error>) -> TgpuStatus {
    match guard(f) {
        Some(()) => TgpuStatus::Ok,
        None => LAST_ERROR.with(|slot| {
            slot.borrow()
                .as_ref()
                .map_or(TgpuStatus::Validation, |error| error.0)
        }),
    }
}

/// [`guard`] for constructors: `NULL` on failure.
pub fn guard_box<T>(f: impl FnOnce() -> Result<Box<T>, Error>) -> *mut T {
    match guard(f) {
        Some(value) => Box::into_raw(value),
        None => ptr::null_mut(),
    }
}

/// Reads a NUL-terminated UTF-8 string from C.
pub unsafe fn cstr<'a>(value: *const c_char) -> Option<&'a str> {
    if value.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(value) }.to_str().ok()
}

pub unsafe fn required_cstr<'a>(value: *const c_char, name: &str) -> Result<&'a str, Error> {
    unsafe { cstr(value) }.ok_or_else(|| Error::validation(format!("{name} must be valid UTF-8")))
}

pub unsafe fn label<'a>(value: *const c_char) -> Option<tgpu::Label<'a>> {
    unsafe { cstr(value) }.map(tgpu::Label::Name)
}

/// Validates the `struct_size` field every desc starts with.
pub fn check_size<T>(size: usize, name: &str) -> Result<(), Error> {
    if size != 0 && size < std::mem::size_of::<T>() {
        return Err(Error::validation(format!(
            "{name}.struct_size is {size}, expected at least {}",
            std::mem::size_of::<T>()
        )));
    }
    Ok(())
}
