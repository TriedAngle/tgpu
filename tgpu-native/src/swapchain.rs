use std::{
    ffi::c_void,
    num::NonZeroIsize,
    ptr::NonNull,
};

use raw_window_handle::{
    AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle,
    WindowsDisplayHandle, XlibDisplayHandle, XlibWindowHandle,
};
use tgpu::{Swapchain, SwapchainCreateInfo};

use crate::{
    command::TgpuCommandBuffer,
    error::{check_size, guard_box, guard_status, TgpuStatus, Error},
    image::TgpuImageView,
    instance::{TgpuDevice, TgpuQueue},
    types::{TgpuExtent2D, TgpuPresentMode, TgpuWindow, TgpuWindowKind},
};

pub struct TgpuSwapchain {
    handle: Swapchain,
}

impl TgpuSwapchain {
    pub fn handle(&self) -> &Swapchain {
        &self.handle
    }

    pub fn handle_mut(&mut self) -> &mut Swapchain {
        &mut self.handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TgpuFrame {
    pub index: u32,
    pub flight_index: usize,
    pub suboptimal: bool,
    pub out_of_date: bool,
}

impl From<tgpu::Frame> for TgpuFrame {
    fn from(value: tgpu::Frame) -> Self {
        Self {
            index: value.index,
            flight_index: value.flight_index,
            suboptimal: value.suboptimal,
            out_of_date: value.out_of_date,
        }
    }
}

impl From<TgpuFrame> for tgpu::Frame {
    fn from(value: TgpuFrame) -> Self {
        Self {
            index: value.index,
            flight_index: value.flight_index,
            suboptimal: value.suboptimal,
            out_of_date: value.out_of_date,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuSwapchainDesc {
    pub struct_size: usize,
    pub window: TgpuWindow,
    pub width: u32,
    pub height: u32,
    /// `0` uses the default (3).
    pub image_count: u32,
    pub present_mode: TgpuPresentMode,
}

fn non_null(pointer: *mut c_void, name: &str) -> Result<NonNull<c_void>, Error> {
    NonNull::new(pointer).ok_or_else(|| Error::validation(format!("{name} is NULL")))
}

fn raw_handles(window: &TgpuWindow) -> Result<(RawDisplayHandle, RawWindowHandle), Error> {
    Ok(match window.kind {
        TgpuWindowKind::AppKit => (
            RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
            RawWindowHandle::AppKit(AppKitWindowHandle::new(non_null(
                window.window,
                "window.window (NSView*)",
            )?)),
        ),
        TgpuWindowKind::Xlib => {
            let display = NonNull::new(window.display);
            let window = window.window as usize as u64;
            if window == 0 {
                return Err(Error::validation("window.window is NULL"));
            }
            (
                RawDisplayHandle::Xlib(XlibDisplayHandle::new(display, 0)),
                RawWindowHandle::Xlib(XlibWindowHandle::new(window)),
            )
        }
        TgpuWindowKind::Win32 => {
            let hwnd = NonZeroIsize::new(window.window as isize)
                .ok_or_else(|| Error::validation("window.window (HWND) is NULL"))?;
            (
                RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
                RawWindowHandle::Win32(Win32WindowHandle::new(hwnd)),
            )
        }
        TgpuWindowKind::Wayland => {
            return Err(Error::validation(
                "Wayland surfaces are not wired up yet",
            ));
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_swapchain(
    device: *mut TgpuDevice,
    desc: *const TgpuSwapchainDesc,
) -> *mut TgpuSwapchain {
    guard_box(|| {
        let device = unsafe { device.as_ref() }
            .ok_or_else(|| Error::validation("device is NULL"))?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| Error::validation("TgpuSwapchainDesc is NULL"))?;
        check_size::<TgpuSwapchainDesc>(desc.struct_size, "TgpuSwapchainDesc")?;

        let (display, window) = raw_handles(&desc.window)?;
        let mut info = SwapchainCreateInfo::new(
            display,
            window,
            TgpuExtent2D {
                width: desc.width.max(1),
                height: desc.height.max(1),
            }
            .into(),
        );
        if desc.image_count > 0 {
            info.preferred_image_count = desc.image_count as usize;
        }
        info.preferred_present_mode = desc.present_mode.into();

        let handle = device.handle().create_swapchain(&info)?;
        Ok(Box::new(TgpuSwapchain { handle }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_destroy(swapchain: *mut TgpuSwapchain) {
    if !swapchain.is_null() {
        drop(unsafe { Box::from_raw(swapchain) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_extent(
    swapchain: *const TgpuSwapchain,
    out: *mut TgpuExtent2D,
) -> TgpuStatus {
    guard_status(|| {
        let swapchain = unsafe { swapchain.as_ref() }
            .ok_or_else(|| Error::validation("swapchain is NULL"))?;
        let out = unsafe { out.as_mut() }
            .ok_or_else(|| Error::validation("out is NULL"))?;
        *out = swapchain.handle.extent().into();
        Ok(())
    })
}

/// Returns the surface format as a `VkFormat` value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_format(swapchain: *const TgpuSwapchain) -> i32 {
    unsafe { swapchain.as_ref() }
        .map_or(0, |swapchain| swapchain.handle.format().as_raw())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_max_frames_in_flight(
    swapchain: *const TgpuSwapchain,
) -> usize {
    unsafe { swapchain.as_ref() }.map_or(0, |swapchain| swapchain.handle.max_frames_in_flight())
}

/// Waits for the frame slot, acquires an image, and recreates the swapchain
/// internally when the surface is out of date.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_begin_frame(
    queue: *const TgpuQueue,
    swapchain: *mut TgpuSwapchain,
    out: *mut TgpuFrame,
) -> TgpuStatus {
    guard_status(|| {
        let queue = unsafe { queue.as_ref() }
            .ok_or_else(|| Error::validation("queue is NULL"))?;
        let swapchain = unsafe { swapchain.as_mut() }
            .ok_or_else(|| Error::validation("swapchain is NULL"))?;
        let out = unsafe { out.as_mut() }
            .ok_or_else(|| Error::validation("out is NULL"))?;

        *out = swapchain.handle_mut().begin_frame(queue.handle())?.into();
        Ok(())
    })
}

/// Submits `records`, presents `frame`, and records the timeline value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_end_frame(
    queue: *const TgpuQueue,
    swapchain: *mut TgpuSwapchain,
    frame: TgpuFrame,
    records: *const *mut TgpuCommandBuffer,
    record_count: usize,
) -> TgpuStatus {
    guard_status(|| {
        let queue = unsafe { queue.as_ref() }
            .ok_or_else(|| Error::validation("queue is NULL"))?;
        let swapchain = unsafe { swapchain.as_mut() }
            .ok_or_else(|| Error::validation("swapchain is NULL"))?;
        if record_count > 0 && records.is_null() {
            return Err(Error::validation("records is NULL"));
        }

        let handles = unsafe { std::slice::from_raw_parts(records, record_count) };
        let mut command_buffers = Vec::with_capacity(handles.len());
        for handle in handles {
            let command_buffer = unsafe { handle.as_ref() }
                .ok_or_else(|| Error::validation("command buffer is NULL"))?;
            command_buffers.push(tgpu::CommandBuffer {
                inner: command_buffer.handle().inner.clone(),
            });
        }

        swapchain
            .handle_mut()
            .end_frame(queue.handle(), frame.into(), &command_buffers)?;
        Ok(())
    })
}

/// Borrowed view of the acquired image, valid while the swapchain is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_view(
    swapchain: *const TgpuSwapchain,
    frame: TgpuFrame,
) -> TgpuImageView {
    match unsafe { swapchain.as_ref() } {
        Some(swapchain) => TgpuImageView::new(swapchain.handle.view(frame.into())),
        None => TgpuImageView::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_set_present_mode(
    swapchain: *mut TgpuSwapchain,
    present_mode: TgpuPresentMode,
) {
    if let Some(swapchain) = unsafe { swapchain.as_mut() } {
        swapchain
            .handle_mut()
            .set_preferred_present_mode(present_mode.into());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_set_extent(
    swapchain: *mut TgpuSwapchain,
    extent: TgpuExtent2D,
) {
    if let Some(swapchain) = unsafe { swapchain.as_mut() } {
        swapchain
            .handle_mut()
            .set_preferred_extent(tgpu::Extent2D::new(extent.width, extent.height));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_swapchain_recreate(
    swapchain: *mut TgpuSwapchain,
) -> TgpuStatus {
    guard_status(|| {
        let swapchain = unsafe { swapchain.as_mut() }
            .ok_or_else(|| Error::validation("swapchain is NULL"))?;
        swapchain.handle_mut().recreate()?;
        Ok(())
    })
}
