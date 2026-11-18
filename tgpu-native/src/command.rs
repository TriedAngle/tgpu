use std::slice;

use tgpu::{
    AttachmentLoadOp, AttachmentStoreOp, BufferTransition, ClearValue, CommandBuffer,
    CommandRecorder, ImageLayoutTransition, Rect, RenderAttachment, RenderInfo,
};

use crate::{
    bindless::TgpuBindlessHeap,
    buffer::TgpuBuffer,
    error::{check_size, guard_box, guard_status, Error, TgpuStatus},
    image::{TgpuImage, TgpuImageView},
    instance::TgpuQueue,
    pipeline::{TgpuComputePipeline, TgpuRenderPipeline},
    swapchain::{TgpuFrame, TgpuSwapchain},
    types::{TgpuBufferAccess, TgpuImageLayout, TgpuLoadOp, TgpuRect, TgpuStoreOp},
};

pub struct TgpuCommandRecorder {
    handle: Option<CommandRecorder>,
}

impl TgpuCommandRecorder {
    pub fn handle_mut(&mut self) -> &mut CommandRecorder {
        self.handle.as_mut().expect("command recorder checked by caller")
    }
}

pub struct TgpuCommandBuffer {
    handle: CommandBuffer,
    /// The pool retires a command buffer when its recorder is dropped, using
    /// the submission index stored at that moment. Keep the recorder alive
    /// until after submission, or the pool recycles the buffer too early.
    _recorder: CommandRecorder,
}

impl TgpuCommandBuffer {
    pub fn handle(&self) -> &CommandBuffer {
        &self.handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuColorAttachment {
    pub view: TgpuImageView,
    pub layout: TgpuImageLayout,
    pub load: TgpuLoadOp,
    pub store: TgpuStoreOp,
    /// Used when `load` is `TGPU_LOAD_OP_CLEAR`.
    pub clear_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuDepthAttachment {
    pub view: TgpuImageView,
    pub layout: TgpuImageLayout,
    pub load: TgpuLoadOp,
    pub store: TgpuStoreOp,
    pub clear_depth: f32,
    pub clear_stencil: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuRenderDesc {
    pub struct_size: usize,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub colors: *const TgpuColorAttachment,
    pub color_count: usize,
    pub depth: *const TgpuDepthAttachment,
    pub stencil: *const TgpuDepthAttachment,
}

fn recorder_ref<'a>(recorder: *mut TgpuCommandRecorder) -> Result<&'a mut TgpuCommandRecorder, Error> {
    let recorder = unsafe { recorder.as_mut() }
        .ok_or_else(|| Error::validation("command recorder is NULL"))?;
    if recorder.handle.is_none() {
        return Err(Error::validation("command recorder was already finished"));
    }
    Ok(recorder)
}

fn image_ref(image: &TgpuImage) -> Result<&tgpu::Image, Error> {
    image
        .as_tgpu()
        .ok_or_else(|| Error::validation("image handle is NULL"))
}

fn image_view_ref(view: &TgpuImageView) -> Result<&tgpu::ImageView, Error> {
    view.as_tgpu()
        .ok_or_else(|| Error::validation("image view handle is NULL"))
}

fn depth_attachment(value: &TgpuDepthAttachment) -> Result<RenderAttachment<'_>, Error> {
    Ok(RenderAttachment {
        view: image_view_ref(&value.view)?,
        layout: value.layout.into(),
        load: match value.load {
            TgpuLoadOp::Load => AttachmentLoadOp::Load,
            TgpuLoadOp::Clear => AttachmentLoadOp::Clear,
            TgpuLoadOp::DontCare => AttachmentLoadOp::DontCare,
        },
        store: match value.store {
            TgpuStoreOp::Store => AttachmentStoreOp::Store,
            TgpuStoreOp::DontCare => AttachmentStoreOp::DontCare,
        },
        clear: match value.load {
            TgpuLoadOp::Clear => Some(ClearValue::DepthStencil {
                depth: value.clear_depth,
                stencil: value.clear_stencil,
            }),
            _ => None,
        },
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_queue_begin_commands(
    queue: *const TgpuQueue,
) -> *mut TgpuCommandRecorder {
    guard_box(|| {
        let queue = unsafe { queue.as_ref() }
            .ok_or_else(|| Error::validation("queue is NULL"))?;
        Ok(Box::new(TgpuCommandRecorder {
            handle: Some(queue.handle().record()),
        }))
    })
}

/// Finishes recording and consumes the recorder handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_finish(
    recorder: *mut TgpuCommandRecorder,
) -> *mut TgpuCommandBuffer {
    guard_box(|| {
        if recorder.is_null() {
            return Err(Error::validation("command recorder is NULL"));
        }
        let mut recorder = unsafe { Box::from_raw(recorder) };
        let mut handle = recorder
            .handle
            .take()
            .ok_or_else(|| Error::validation("command recorder was already finished"))?;
        let command_buffer = handle.finish();
        Ok(Box::new(TgpuCommandBuffer {
            handle: command_buffer,
            _recorder: handle,
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_command_buffer_destroy(command_buffer: *mut TgpuCommandBuffer) {
    if !command_buffer.is_null() {
        drop(unsafe { Box::from_raw(command_buffer) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_transition_image(
    recorder: *mut TgpuCommandRecorder,
    image: TgpuImage,
    layout: TgpuImageLayout,
) -> TgpuStatus {
    guard_status(|| {
        recorder_ref(recorder)?
            .handle_mut()
            .image_transition(image_ref(&image)?, ImageLayoutTransition::new(layout.into()));
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_transition_frame(
    recorder: *mut TgpuCommandRecorder,
    swapchain: *const TgpuSwapchain,
    frame: TgpuFrame,
    layout: TgpuImageLayout,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let swapchain = unsafe { swapchain.as_ref() }
            .ok_or_else(|| Error::validation("swapchain is NULL"))?;
        swapchain.handle().transition_frame(
            recorder.handle_mut(),
            frame.into(),
            ImageLayoutTransition::new(layout.into()),
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_buffer_transition(
    recorder: *mut TgpuCommandRecorder,
    buffer: *const TgpuBuffer,
    from: TgpuBufferAccess,
    to: TgpuBufferAccess,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let buffer = unsafe { buffer.as_ref() }
            .ok_or_else(|| Error::validation("buffer is NULL"))?;
        recorder.handle_mut().buffer_transition(
            buffer.handle(),
            BufferTransition {
                from: from.into(),
                to: to.into(),
                ..Default::default()
            },
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_update_buffer(
    recorder: *mut TgpuCommandRecorder,
    buffer: *const TgpuBuffer,
    offset: u64,
    data: *const u8,
    len: usize,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let buffer = unsafe { buffer.as_ref() }
            .ok_or_else(|| Error::validation("buffer is NULL"))?;
        if len > 0 && data.is_null() {
            return Err(Error::validation("data is NULL"));
        }
        recorder
            .handle_mut()
            .update_buffer(buffer.handle(), offset, unsafe {
                slice::from_raw_parts(data, len)
            });
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_bind_vertex_buffer(
    recorder: *mut TgpuCommandRecorder,
    slot: u32,
    buffer: *const TgpuBuffer,
    offset: u64,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let buffer = unsafe { buffer.as_ref() }
            .ok_or_else(|| Error::validation("buffer is NULL"))?;
        recorder
            .handle_mut()
            .bind_vertex_buffer(slot, buffer.handle(), offset);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_bind_compute(
    recorder: *mut TgpuCommandRecorder,
    pipeline: *const TgpuComputePipeline,
    heap: *const TgpuBindlessHeap,
    push_constants: *const u8,
    push_constant_len: usize,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let pipeline = unsafe { pipeline.as_ref() }
            .ok_or_else(|| Error::validation("compute pipeline is NULL"))?;
        let heap = unsafe { heap.as_ref() }
            .ok_or_else(|| Error::validation("bindless heap is NULL"))?;
        if push_constant_len > 0 && push_constants.is_null() {
            return Err(Error::validation("push_constants is NULL"));
        }
        let push_constants = unsafe { slice::from_raw_parts(push_constants, push_constant_len) };
        recorder.handle_mut().bind_compute_bytes(
            pipeline.handle(),
            heap.handle().descriptor_set(),
            push_constants,
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_dispatch(
    recorder: *mut TgpuCommandRecorder,
    x: u32,
    y: u32,
    z: u32,
) -> TgpuStatus {
    guard_status(|| {
        recorder_ref(recorder)?.handle_mut().dispatch(x, y, z);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_bind_render(
    recorder: *mut TgpuCommandRecorder,
    pipeline: *const TgpuRenderPipeline,
    heap: *const TgpuBindlessHeap,
    push_constants: *const u8,
    push_constant_len: usize,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let pipeline = unsafe { pipeline.as_ref() }
            .ok_or_else(|| Error::validation("render pipeline is NULL"))?;
        let heap = unsafe { heap.as_ref() }
            .ok_or_else(|| Error::validation("bindless heap is NULL"))?;
        if push_constant_len > 0 && push_constants.is_null() {
            return Err(Error::validation("push_constants is NULL"));
        }
        let push_constants = unsafe { slice::from_raw_parts(push_constants, push_constant_len) };
        recorder.handle_mut().bind_render_bytes(
            pipeline.handle(),
            heap.handle().descriptor_set(),
            push_constants,
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_begin_render(
    recorder: *mut TgpuCommandRecorder,
    desc: *const TgpuRenderDesc,
) -> TgpuStatus {
    guard_status(|| {
        let recorder = recorder_ref(recorder)?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| Error::validation("TgpuRenderDesc is NULL"))?;
        check_size::<TgpuRenderDesc>(desc.struct_size, "TgpuRenderDesc")?;
        if desc.color_count > 0 && desc.colors.is_null() {
            return Err(Error::validation("colors is NULL"));
        }

        let color_descs = unsafe { slice::from_raw_parts(desc.colors, desc.color_count) };
        let mut colors = Vec::with_capacity(color_descs.len());
        for color in color_descs {
            let clear = match color.load {
                TgpuLoadOp::Clear => Some(ClearValue::Color(color.clear_color)),
                _ => None,
            };
            colors.push(RenderAttachment {
                view: image_view_ref(&color.view)?,
                layout: color.layout.into(),
                load: match color.load {
                    TgpuLoadOp::Load => AttachmentLoadOp::Load,
                    TgpuLoadOp::Clear => AttachmentLoadOp::Clear,
                    TgpuLoadOp::DontCare => AttachmentLoadOp::DontCare,
                },
                store: match color.store {
                    TgpuStoreOp::Store => AttachmentStoreOp::Store,
                    TgpuStoreOp::DontCare => AttachmentStoreOp::DontCare,
                },
                clear,
            });
        }

        let area = Rect::from(TgpuRect {
            x: desc.x,
            y: desc.y,
            width: desc.width,
            height: desc.height,
        });

        let depth = if desc.depth.is_null() {
            None
        } else {
            Some(depth_attachment(unsafe { &*desc.depth })?)
        };
        let stencil = if desc.stencil.is_null() {
            None
        } else {
            Some(depth_attachment(unsafe { &*desc.stencil })?)
        };

        recorder.handle_mut().begin_render_pass(&RenderInfo {
            area,
            colors: &colors,
            depth,
            stencil,
            ..Default::default()
        });
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_end_render(
    recorder: *mut TgpuCommandRecorder,
) -> TgpuStatus {
    guard_status(|| {
        recorder_ref(recorder)?.handle_mut().end_render_pass();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_recorder_draw(
    recorder: *mut TgpuCommandRecorder,
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> TgpuStatus {
    guard_status(|| {
        recorder_ref(recorder)?.handle_mut().draw(
            first_vertex..first_vertex + vertex_count,
            first_instance..first_instance + instance_count,
        );
        Ok(())
    })
}
