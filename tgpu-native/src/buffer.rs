use std::{ffi::c_char, slice};

use tgpu::{Buffer, BufferDesc};

use crate::{
    error::{Error, TgpuStatus, check_size, guard_box, guard_status, label},
    instance::TgpuDevice,
    types::{TgpuHostAccess, TgpuMemoryPreset, buffer_uses},
};

pub struct TgpuBuffer {
    handle: Buffer,
}

impl TgpuBuffer {
    pub fn handle(&self) -> &Buffer {
        &self.handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuBufferDesc {
    pub struct_size: usize,
    pub label: *const c_char,
    pub size: u64,
    /// A combination of `TGPU_BUFFER_USAGE_*`.
    pub usage: u32,
    pub memory: TgpuMemoryPreset,
    pub host_access: TgpuHostAccess,
}

/// Creates a buffer and initialises it from `data` when non-NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_buffer(
    device: *mut TgpuDevice,
    desc: *const TgpuBufferDesc,
    data: *const u8,
    data_len: usize,
) -> *mut TgpuBuffer {
    unsafe { create(device, desc, data, data_len) }
}

unsafe fn create(
    device: *mut TgpuDevice,
    desc: *const TgpuBufferDesc,
    data: *const u8,
    data_len: usize,
) -> *mut TgpuBuffer {
    guard_box(|| {
        let device =
            unsafe { device.as_ref() }.ok_or_else(|| Error::validation("device is NULL"))?;
        let desc =
            unsafe { desc.as_ref() }.ok_or_else(|| Error::validation("TgpuBufferDesc is NULL"))?;
        check_size::<TgpuBufferDesc>(desc.struct_size, "TgpuBufferDesc")?;

        let buffer = device.handle().create_buffer(&BufferDesc {
            label: unsafe { label(desc.label) },
            size: desc.size as usize,
            usage: buffer_uses(desc.usage),
            memory: desc.memory.into(),
            host_access: desc.host_access.into(),
            ..Default::default()
        })?;

        if !data.is_null() && data_len > 0 {
            buffer.write(unsafe { slice::from_raw_parts(data, data_len) }, 0);
        }

        Ok(Box::new(TgpuBuffer { handle: buffer }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_buffer_destroy(buffer: *mut TgpuBuffer) {
    if !buffer.is_null() {
        drop(unsafe { Box::from_raw(buffer) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_buffer_size(buffer: *const TgpuBuffer) -> u64 {
    unsafe { buffer.as_ref() }.map_or(0, |buffer| buffer.handle.size as u64)
}

/// Copies `len` bytes from `data` into the buffer at `offset`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_buffer_write(
    buffer: *mut TgpuBuffer,
    offset: u64,
    data: *const u8,
    len: usize,
) -> TgpuStatus {
    guard_status(|| {
        let buffer =
            unsafe { buffer.as_ref() }.ok_or_else(|| Error::validation("buffer is NULL"))?;
        if len > 0 && data.is_null() {
            return Err(Error::validation("data is NULL"));
        }
        let data = unsafe { slice::from_raw_parts(data, len) };
        buffer.handle.write(data, offset as usize);
        Ok(())
    })
}

/// Copies `len` bytes out of the buffer at `offset` into `data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_buffer_read(
    buffer: *mut TgpuBuffer,
    offset: u64,
    data: *mut u8,
    len: usize,
) -> TgpuStatus {
    guard_status(|| {
        let buffer =
            unsafe { buffer.as_ref() }.ok_or_else(|| Error::validation("buffer is NULL"))?;
        if len > 0 && data.is_null() {
            return Err(Error::validation("data is NULL"));
        }
        let data = unsafe { slice::from_raw_parts_mut(data, len) };
        buffer.handle.read(data, offset as usize, len);
        Ok(())
    })
}
