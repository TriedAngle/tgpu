use tgpu::{BindlessHeap, BindlessInfo, GPUError};

use crate::{
    buffer::TgpuBuffer,
    error::{Error, TgpuStatus, check_size, guard_box, guard_status, set_last_error},
    image::{TgpuImageView, TgpuSampler},
    instance::TgpuDevice,
    types::TgpuImageLayout,
};

pub struct TgpuBindlessHeap {
    handle: BindlessHeap,
}

impl TgpuBindlessHeap {
    pub fn handle(&self) -> &BindlessHeap {
        &self.handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TgpuBindlessDesc {
    pub struct_size: usize,
    pub max_read_buffers: u32,
    pub max_rw_buffers: u32,
    pub max_sampled_images: u32,
    pub max_storage_images: u32,
    pub max_samplers: u32,
    pub max_uniform_buffers: u32,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_bindless(
    device: *mut TgpuDevice,
    desc: *const TgpuBindlessDesc,
) -> *mut TgpuBindlessHeap {
    guard_box(|| {
        let device = unsafe { device.as_ref() }
            .ok_or_else(|| GPUError::Validation("device is NULL".into()))?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| GPUError::Validation("TgpuBindlessDesc is NULL".into()))?;
        check_size::<TgpuBindlessDesc>(desc.struct_size, "TgpuBindlessDesc")?;

        let handle = device.handle().create_bindless_heap(&BindlessInfo {
            max_read_buffers: desc.max_read_buffers,
            max_rw_buffers: desc.max_rw_buffers,
            max_sampled_images: desc.max_sampled_images,
            max_storage_images: desc.max_storage_images,
            max_samplers: desc.max_samplers,
            max_uniform_buffers: desc.max_uniform_buffers,
        });

        Ok(Box::new(TgpuBindlessHeap { handle }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_destroy(heap: *mut TgpuBindlessHeap) {
    if !heap.is_null() {
        drop(unsafe { Box::from_raw(heap) });
    }
}

fn heap<'a>(heap: *const TgpuBindlessHeap) -> Result<&'a TgpuBindlessHeap, Error> {
    unsafe { heap.as_ref() }.ok_or_else(|| Error::validation("bindless heap is NULL"))
}

fn buffer<'a>(buffer: *const TgpuBuffer) -> Result<&'a TgpuBuffer, Error> {
    unsafe { buffer.as_ref() }.ok_or_else(|| Error::validation("buffer is NULL"))
}

fn view(view: &TgpuImageView) -> Result<&tgpu::ImageView, Error> {
    view.as_tgpu()
        .ok_or_else(|| Error::validation("image view handle is NULL"))
}

fn sampler(sampler: &TgpuSampler) -> Result<&tgpu::Sampler, Error> {
    sampler
        .as_tgpu()
        .ok_or_else(|| Error::validation("sampler handle is NULL"))
}

fn failed(error: Error) -> u32 {
    set_last_error(TgpuStatus::InvalidArgument, error);
    crate::types::TGPU_INVALID_HANDLE
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_add_read_buffer(
    heap_ptr: *const TgpuBindlessHeap,
    buffer_ptr: *const TgpuBuffer,
) -> u32 {
    match (heap(heap_ptr), buffer(buffer_ptr)) {
        (Ok(heap), Ok(buffer)) => heap.handle.add_read_buffer(buffer.handle()).0,
        (Err(error), _) | (_, Err(error)) => failed(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_add_rw_buffer(
    heap_ptr: *const TgpuBindlessHeap,
    buffer_ptr: *const TgpuBuffer,
) -> u32 {
    match (heap(heap_ptr), buffer(buffer_ptr)) {
        (Ok(heap), Ok(buffer)) => heap.handle.add_rw_buffer(buffer.handle()).0,
        (Err(error), _) | (_, Err(error)) => failed(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_add_uniform_buffer(
    heap_ptr: *const TgpuBindlessHeap,
    buffer_ptr: *const TgpuBuffer,
    range: u64,
) -> u32 {
    match (heap(heap_ptr), buffer(buffer_ptr)) {
        (Ok(heap), Ok(buffer)) => heap.handle.add_uniform_buffer(buffer.handle(), range).0,
        (Err(error), _) | (_, Err(error)) => failed(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_update_uniform_buffer(
    heap_ptr: *const TgpuBindlessHeap,
    handle: u32,
    buffer_ptr: *const TgpuBuffer,
    range: u64,
) -> TgpuStatus {
    guard_status(|| {
        heap(heap_ptr)?.handle.update_uniform_buffer(
            tgpu::UniformBufferHandle(handle),
            buffer(buffer_ptr)?.handle(),
            range,
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_add_storage_image(
    heap_ptr: *const TgpuBindlessHeap,
    image_view: TgpuImageView,
    layout: TgpuImageLayout,
) -> u32 {
    match (heap(heap_ptr), view(&image_view)) {
        (Ok(heap), Ok(view)) => heap.handle.add_storage_image(view, layout.into()).0,
        (Err(error), _) | (_, Err(error)) => failed(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_add_sampled_image(
    heap_ptr: *const TgpuBindlessHeap,
    image_view: TgpuImageView,
    layout: TgpuImageLayout,
) -> u32 {
    match (heap(heap_ptr), view(&image_view)) {
        (Ok(heap), Ok(view)) => heap.handle.add_sampled_image(view, layout.into()).0,
        (Err(error), _) | (_, Err(error)) => failed(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_add_sampler(
    heap_ptr: *const TgpuBindlessHeap,
    sampler_value: TgpuSampler,
) -> u32 {
    match (heap(heap_ptr), sampler(&sampler_value)) {
        (Ok(heap), Ok(sampler)) => heap.handle.add_sampler(sampler).0,
        (Err(error), _) | (_, Err(error)) => failed(error),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_update_storage_image(
    heap_ptr: *const TgpuBindlessHeap,
    handle: u32,
    image_view: TgpuImageView,
    layout: TgpuImageLayout,
) -> TgpuStatus {
    guard_status(|| {
        heap(heap_ptr)?.handle.update_storage_image(
            tgpu::StorageImageHandle(handle),
            view(&image_view)?,
            layout.into(),
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_update_sampled_image(
    heap_ptr: *const TgpuBindlessHeap,
    handle: u32,
    image_view: TgpuImageView,
    layout: TgpuImageLayout,
) -> TgpuStatus {
    guard_status(|| {
        heap(heap_ptr)?.handle.update_sampled_image(
            tgpu::SampledImageHandle(handle),
            view(&image_view)?,
            layout.into(),
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_bindless_update_sampler(
    heap_ptr: *const TgpuBindlessHeap,
    handle: u32,
    sampler_value: TgpuSampler,
) -> TgpuStatus {
    guard_status(|| {
        heap(heap_ptr)?
            .handle
            .update_sampler(tgpu::SamplerHandle(handle), sampler(&sampler_value)?);
        Ok(())
    })
}
