use ash::vk;
use std::sync::Arc;

use crate::{raw::DeviceImpl, Device, ShaderFunction};

pub struct ComputePipelineInfo<'a> {
    pub shader: ShaderFunction<'a>,
    // pub descriptor_layouts: &'a [&'a DescriptorSetLayout],
    pub push_constant_size: Option<u32>,
    pub cache: Option<vk::PipelineCache>,
    pub label: Option<&'a str>,
    pub tag: Option<(u64, &'a [u8])>,
}

impl Default for ComputePipelineInfo<'_> {
    fn default() -> Self {
        Self {
            shader: ShaderFunction::null(),
            // descriptor_layouts: &[],
            push_constant_size: None,
            cache: None,
            label: None,
            tag: None,
        }
    }
}

pub struct RenderPipelineInfo<'a> {
    pub vertex_shader: ShaderFunction<'a>,
    pub fragment_shader: ShaderFunction<'a>,
    pub color_formats: &'a [vk::Format],
    pub depth_format: Option<vk::Format>,
    // pub descriptor_layouts: &'a [&'a DescriptorSetLayout],
    pub push_constant_size: Option<u32>,
    pub blend_states: Option<&'a [vk::PipelineColorBlendAttachmentState]>,
    pub vertex_input_state: Option<vk::PipelineVertexInputStateCreateInfo<'a>>,
    pub topology: vk::PrimitiveTopology,
    pub polygon: vk::PolygonMode,
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub label: Option<&'a str>,
    pub tag: Option<(u64, &'a [u8])>,
}

impl Default for RenderPipelineInfo<'_> {
    fn default() -> Self {
        Self {
            vertex_shader: ShaderFunction::null(),
            fragment_shader: ShaderFunction::null(),
            color_formats: &[],
            depth_format: None,
            // descriptor_layouts: &[],
            push_constant_size: None,
            blend_states: None,
            vertex_input_state: None,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon: vk::PolygonMode::FILL,
            cull: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            label: None,
            tag: None,
        }
    }
}

pub trait Pipeline {
    const BIND_POINT: vk::PipelineBindPoint;
    const SHADER_STAGE_FLAGS: vk::ShaderStageFlags;

    fn handle(&self) -> vk::Pipeline;
    fn layout(&self) -> vk::PipelineLayout;
    fn bind_point(&self) -> vk::PipelineBindPoint {
        Self::BIND_POINT
    }
    fn flags(&self) -> vk::ShaderStageFlags {
        Self::SHADER_STAGE_FLAGS
    }
}

pub struct ComputePipeline {
    pub inner: ComputePipelineImpl,
}

pub struct ComputePipelineImpl {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub shader: vk::ShaderModule,
    device: Arc<DeviceImpl>,
}

pub struct RenderPipeline {
    pub inner: RenderPipelineImpl,
}

pub struct RenderPipelineImpl {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
    pub device: Arc<DeviceImpl>,
}

impl RenderPipelineImpl {
    pub fn new(device: Arc<DeviceImpl>, info: &RenderPipelineInfo) -> RenderPipelineImpl {
        let vertex_shader = device.create_shader_module(info.vertex_shader).unwrap();
        let fragment_shader = device.create_shader_module(info.fragment_shader).unwrap();

        let mut push_constant_ranges = Vec::new();
        if let Some(size) = info.push_constant_size {
            push_constant_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                    .size(size),
            );
        }

        // let layouts = info
        //     .descriptor_layouts
        //     .iter()
        //     .map(|l| l.handle)
        //     .collect::<Vec<_>>();

        let layout_info = vk::PipelineLayoutCreateInfo::default();
            // .set_layouts(&layouts)
            // .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let vertex_stage_name = std::ffi::CString::new(info.vertex_shader.entry_point).unwrap();
        let fragment_stage_name = std::ffi::CString::new(info.fragment_shader.entry_point).unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(&vertex_stage_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(&fragment_stage_name),
        ];

        let vertex_input = info
            .vertex_input_state
            .unwrap_or_else(|| vk::PipelineVertexInputStateCreateInfo::default());

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(info.topology)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .depth_bias_enable(false)
            .polygon_mode(info.polygon)
            .line_width(1.0)
            .cull_mode(info.cull)
            .front_face(info.front_face);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = info.blend_states.as_ref().map_or_else(
            || {
                vec![
                    vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(vk::ColorComponentFlags::RGBA)
                        .blend_enable(false),
                ]
            },
            |&states| states.iter().copied().collect::<Vec<_>>(),
        );

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .attachments(&color_blend_attachment);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&info.color_formats);

        if let Some(format) = info.depth_format {
            rendering_info = rendering_info.depth_attachment_format(format);
        }

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .base_pipeline_handle(vk::Pipeline::null())
            .push_next(&mut rendering_info);


        let handle = unsafe {
            device
                .handle
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .unwrap()[0]
        };

        // device.set_object_debug_info(handle, info.label, info.tag);

        RenderPipelineImpl {
            handle,
            layout,
            vertex_shader,
            fragment_shader,
            device,
        }
    }
}

impl ComputePipelineImpl {
    pub fn new(device: Arc<DeviceImpl>, info: &ComputePipelineInfo<'_>) -> ComputePipelineImpl {
        let module = device.create_shader_module(info.shader).unwrap();
        let mut push_constant_ranges = Vec::new();
        if let Some(size) = info.push_constant_size {
            push_constant_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .size(size),
            );
        }

        // let layouts = info
        //     .descriptor_layouts
        //     .iter()
        //     .map(|l| l.handle)
        //     .collect::<Vec<_>>();

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            // .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let stage_name = std::ffi::CString::new(info.shader.entry_point).unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&stage_name);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let handle = unsafe {
            let cache = info.cache.unwrap_or(vk::PipelineCache::null());
            device
                .handle
                .create_compute_pipelines(cache, &[create_info], None)
                .unwrap()[0]
        };

        // self.set_object_debug_info(handle, info.label, info.tag);

        ComputePipelineImpl {
            handle,
            layout,
            shader: module,
            device,
        }
    }
}

impl Device {
    pub fn create_render_pipeline(&self, info: &RenderPipelineInfo<'_>) -> RenderPipeline {
        let inner = RenderPipelineImpl::new(self.inner.clone(), info);
        RenderPipeline { inner }
    }

    pub fn create_compute_pipeline(&self, info: &ComputePipelineInfo<'_>) -> ComputePipeline {
        let inner = ComputePipelineImpl::new(self.inner.clone(), info);
        ComputePipeline { inner }
    }
}

impl Drop for ComputePipelineImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_shader_module(self.shader, None);
            self.device
                .handle
                .destroy_pipeline_layout(self.layout, None);
            self.device.handle.destroy_pipeline(self.handle, None);
        }
    }
}

impl Drop for RenderPipelineImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
            self.device
                .handle
                .destroy_pipeline_layout(self.layout, None);
            self.device
                .handle
                .destroy_shader_module(self.vertex_shader, None);
            self.device
                .handle
                .destroy_shader_module(self.fragment_shader, None);
        }
    }
}

