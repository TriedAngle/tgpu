use std::ffi::c_char;

use tgpu::{
    ColorBlend, ComputePipeline, ComputePipelineInfo, DescriptorSetLayout, Format, RenderPipeline,
    RenderPipelineInfo, Shader, ShaderEntry, VertexAttribute, VertexBufferLayout, VertexInput,
};

use crate::{
    bindless::TgpuBindlessHeap,
    error::{check_size, guard_box, label, required_cstr, Error},
    instance::TgpuDevice,
    types::{
        TgpuBlend, TgpuCompareOp, TgpuCullMode, TgpuFormat, TgpuFrontFace,
        TgpuPrimitiveTopology, TgpuVertexInputRate,
    },
};

pub struct TgpuShader {
    handle: Shader,
}

pub struct TgpuComputePipeline {
    handle: ComputePipeline,
}

impl TgpuComputePipeline {
    pub fn handle(&self) -> &ComputePipeline {
        &self.handle
    }
}

pub struct TgpuRenderPipeline {
    handle: RenderPipeline,
}

impl TgpuRenderPipeline {
    pub fn handle(&self) -> &RenderPipeline {
        &self.handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuComputePipelineDesc {
    pub struct_size: usize,
    pub label: *const c_char,
    pub shader: *const TgpuShader,
    pub entry: *const c_char,
    /// Optional bindless heap used as descriptor set 0.
    pub bindless: *const TgpuBindlessHeap,
    /// `0` for no push constants.
    pub push_constant_size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuVertexAttribute {
    pub location: u32,
    pub binding: u32,
    /// A `VkFormat` value.
    pub format: i32,
    pub offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuVertexBufferLayout {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: TgpuVertexInputRate,
    pub attributes: *const TgpuVertexAttribute,
    pub attribute_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuRenderPipelineDesc {
    pub struct_size: usize,
    pub label: *const c_char,
    pub vertex_shader: *const TgpuShader,
    pub vertex_entry: *const c_char,
    pub fragment_shader: *const TgpuShader,
    pub fragment_entry: *const c_char,
    pub color_formats: *const TgpuFormat,
    pub color_format_count: usize,
    /// `TGPU_FORMAT_UNDEFINED` for no depth attachment.
    pub depth_format: TgpuFormat,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: TgpuCompareOp,
    /// Optional bindless heap used as descriptor set 0.
    pub bindless: *const TgpuBindlessHeap,
    pub push_constant_size: u32,
    pub cull: TgpuCullMode,
    pub front_face: TgpuFrontFace,
    pub topology: TgpuPrimitiveTopology,
    pub blend: TgpuBlend,
    pub vertex_buffers: *const TgpuVertexBufferLayout,
    pub vertex_buffer_count: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_shader(
    device: *mut TgpuDevice,
    label_ptr: *const c_char,
    words: *const u32,
    word_count: usize,
) -> *mut TgpuShader {
    guard_box(|| {
        let device = unsafe { device.as_ref() }
            .ok_or_else(|| Error::validation("device is NULL"))?;
        if word_count > 0 && words.is_null() {
            return Err(Error::validation("SPIR-V words are NULL"));
        }

        let words = unsafe { std::slice::from_raw_parts(words, word_count) };
        let inner = device
            .handle()
            .create_shader_from_spirv(unsafe { label(label_ptr) }, words);
        Ok(Box::new(TgpuShader { handle: inner }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_shader_destroy(shader: *mut TgpuShader) {
    if !shader.is_null() {
        drop(unsafe { Box::from_raw(shader) });
    }
}

unsafe fn shader_entry<'a>(
    shader: *const TgpuShader,
    entry: *const c_char,
    name: &str,
) -> Result<ShaderEntry<'a>, Error> {
    let shader = unsafe { shader.as_ref() }
        .ok_or_else(|| Error::validation(format!("{name}.shader is NULL")))?;
    let entry = unsafe { required_cstr(entry, name)? };
    Ok(shader.handle.entry(entry))
}

fn layouts<'a>(bindless: *const TgpuBindlessHeap) -> Vec<&'a DescriptorSetLayout> {
    match unsafe { bindless.as_ref() } {
        Some(heap) => vec![heap.handle().layout()],
        None => Vec::new(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_compute_pipeline(
    device: *mut TgpuDevice,
    desc: *const TgpuComputePipelineDesc,
) -> *mut TgpuComputePipeline {
    guard_box(|| {
        let device = unsafe { device.as_ref() }
            .ok_or_else(|| Error::validation("device is NULL"))?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| Error::validation("TgpuComputePipelineDesc is NULL"))?;
        check_size::<TgpuComputePipelineDesc>(desc.struct_size, "TgpuComputePipelineDesc")?;

        let shader = unsafe { shader_entry(desc.shader, desc.entry, "entry")? };
        let layouts = layouts(desc.bindless);

        let inner = device.handle().try_create_compute_pipeline(&ComputePipelineInfo {
            label: unsafe { label(desc.label) },
            shader,
            descriptor_layouts: &layouts,
            push_constant_size: (desc.push_constant_size > 0).then_some(desc.push_constant_size),
            cache: None,
        })?;

        Ok(Box::new(TgpuComputePipeline { handle: inner }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_compute_pipeline_destroy(pipeline: *mut TgpuComputePipeline) {
    if !pipeline.is_null() {
        drop(unsafe { Box::from_raw(pipeline) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_render_pipeline(
    device: *mut TgpuDevice,
    desc: *const TgpuRenderPipelineDesc,
) -> *mut TgpuRenderPipeline {
    guard_box(|| {
        let device = unsafe { device.as_ref() }
            .ok_or_else(|| Error::validation("device is NULL"))?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| Error::validation("TgpuRenderPipelineDesc is NULL"))?;
        check_size::<TgpuRenderPipelineDesc>(desc.struct_size, "TgpuRenderPipelineDesc")?;

        let vertex_shader =
            unsafe { shader_entry(desc.vertex_shader, desc.vertex_entry, "vertex_entry")? };
        let fragment_shader =
            unsafe { shader_entry(desc.fragment_shader, desc.fragment_entry, "fragment_entry")? };

        if desc.color_format_count > 0 && desc.color_formats.is_null() {
            return Err(Error::validation("color_formats is NULL"));
        }
        let color_formats = unsafe {
            std::slice::from_raw_parts(desc.color_formats, desc.color_format_count)
        }
        .iter()
        .map(|format| Format::from_raw(*format))
        .collect::<Vec<_>>();

        let layouts = layouts(desc.bindless);

        let blend_values = if desc.blend == TgpuBlend::Disabled {
            Vec::new()
        } else {
            vec![ColorBlend::from(desc.blend)]
        };

        let mut vertex_buffers = Vec::new();
        let mut attributes = Vec::new();
        if desc.vertex_buffer_count > 0 {
            if desc.vertex_buffers.is_null() {
                return Err(Error::validation("vertex_buffers is NULL"));
            }
            let layouts = unsafe {
                std::slice::from_raw_parts(desc.vertex_buffers, desc.vertex_buffer_count)
            };
            for layout in layouts {
                vertex_buffers.push(VertexBufferLayout {
                    binding: layout.binding,
                    stride: layout.stride,
                    input_rate: layout.input_rate.into(),
                });
                if layout.attribute_count > 0 && layout.attributes.is_null() {
                    return Err(Error::validation("vertex attribute list is NULL"));
                }
                let entries = unsafe {
                    std::slice::from_raw_parts(layout.attributes, layout.attribute_count)
                };
                for attribute in entries {
                    attributes.push(VertexAttribute {
                        location: attribute.location,
                        binding: attribute.binding,
                        format: Format::from_raw(attribute.format),
                        offset: attribute.offset,
                    });
                }
            }
        }
        let vertex_input = (!vertex_buffers.is_empty()).then(|| VertexInput {
            buffers: &vertex_buffers,
            attributes: &attributes,
        });

        let inner = device.handle().try_create_render_pipeline(&RenderPipelineInfo {
            label: unsafe { label(desc.label) },
            vertex_shader,
            fragment_shader,
            color_formats: &color_formats,
            depth_format: (desc.depth_format != 0).then(|| Format::from_raw(desc.depth_format)),
            depth_test: desc.depth_test,
            depth_write: desc.depth_write,
            depth_compare: desc.depth_compare.into(),
            descriptor_layouts: &layouts,
            push_constant_size: (desc.push_constant_size > 0).then_some(desc.push_constant_size),
            blend: (!blend_values.is_empty()).then_some(blend_values.as_slice()),
            blend_states: None,
            vertex_input,
            vertex_input_state: None,
            topology: desc.topology.into(),
            polygon: Default::default(),
            cull: desc.cull.into(),
            front_face: desc.front_face.into(),
            ..Default::default()
        })?;

        Ok(Box::new(TgpuRenderPipeline { handle: inner }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_render_pipeline_destroy(pipeline: *mut TgpuRenderPipeline) {
    if !pipeline.is_null() {
        drop(unsafe { Box::from_raw(pipeline) });
    }
}
