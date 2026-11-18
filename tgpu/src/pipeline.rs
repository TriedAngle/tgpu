use ash::vk;
use std::sync::Arc;

use crate::{
    DescriptorSetLayout, Device, Format, Label, ShaderEntry, macros::vk_value_enum, raw::DeviceImpl,
};

vk_value_enum! {
    /// A depth/stencil comparison.
    CompareOp: vk::CompareOp,
    default = Always => ALWAYS,
    {
        Never => NEVER,
        Less => LESS,
        Equal => EQUAL,
        LessOrEqual => LESS_OR_EQUAL,
        Greater => GREATER,
        NotEqual => NOT_EQUAL,
        GreaterOrEqual => GREATER_OR_EQUAL,
    }
}

vk_value_enum! {
    /// How vertices are assembled into primitives.
    PrimitiveTopology: vk::PrimitiveTopology,
    default = TriangleList => TRIANGLE_LIST,
    {
        PointList => POINT_LIST,
        LineList => LINE_LIST,
        LineStrip => LINE_STRIP,
        TriangleStrip => TRIANGLE_STRIP,
        TriangleFan => TRIANGLE_FAN,
        LineListWithAdjacency => LINE_LIST_WITH_ADJACENCY,
        LineStripWithAdjacency => LINE_STRIP_WITH_ADJACENCY,
        TriangleListWithAdjacency => TRIANGLE_LIST_WITH_ADJACENCY,
        TriangleStripWithAdjacency => TRIANGLE_STRIP_WITH_ADJACENCY,
        PatchList => PATCH_LIST,
    }
}

vk_value_enum! {
    /// How polygons are rasterised.
    PolygonMode: vk::PolygonMode,
    default = Fill => FILL,
    {
        Line => LINE,
        Point => POINT,
    }
}

vk_value_enum! {
    /// Which winding order counts as front-facing.
    FrontFace: vk::FrontFace,
    default = CounterClockwise => COUNTER_CLOCKWISE,
    {
        Clockwise => CLOCKWISE,
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct CullMode: u32 {
        const NONE = 0;
        const FRONT = 1 << 0;
        const BACK = 1 << 1;
        const FRONT_AND_BACK = 0b11;
    }
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(value: CullMode) -> Self {
        Self::from_raw(value.bits())
    }
}

impl From<vk::CullModeFlags> for CullMode {
    fn from(value: vk::CullModeFlags) -> Self {
        Self::from_bits_retain(value.as_raw())
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum VertexInputRate {
    #[default]
    Vertex,
    Instance,
    Custom(vk::VertexInputRate),
}

impl From<VertexInputRate> for vk::VertexInputRate {
    fn from(value: VertexInputRate) -> Self {
        match value {
            VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
            VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
            VertexInputRate::Custom(raw) => raw,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VertexBufferLayout {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VertexInputRate,
}

impl From<VertexBufferLayout> for vk::VertexInputBindingDescription {
    fn from(value: VertexBufferLayout) -> Self {
        Self {
            binding: value.binding,
            stride: value.stride,
            input_rate: value.input_rate.into(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

impl From<VertexAttribute> for vk::VertexInputAttributeDescription {
    fn from(value: VertexAttribute) -> Self {
        Self {
            location: value.location,
            binding: value.binding,
            format: value.format.into(),
            offset: value.offset,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VertexInput<'a> {
    pub buffers: &'a [VertexBufferLayout],
    pub attributes: &'a [VertexAttribute],
}

#[derive(Debug, Copy, Clone)]
pub enum ColorBlend {
    Disabled,
    Alpha,
    Custom(vk::PipelineColorBlendAttachmentState),
}

impl ColorBlend {
    fn into_vk(self) -> vk::PipelineColorBlendAttachmentState {
        match self {
            Self::Disabled => vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false),
            Self::Alpha => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            Self::Custom(raw) => raw,
        }
    }
}

/// Push constant byte size for a POD type, for use as
/// `push_constant_size: tgpu::push_constant_size::<PushData>()`.
pub const fn push_constant_size<T>() -> Option<u32> {
    Some(std::mem::size_of::<T>() as u32)
}

pub struct ComputePipelineInfo<'a> {
    pub shader: ShaderEntry<'a>,
    pub descriptor_layouts: &'a [&'a DescriptorSetLayout],
    pub push_constant_size: Option<u32>,
    pub cache: Option<vk::PipelineCache>,
    pub label: Option<Label<'a>>,
}

impl Default for ComputePipelineInfo<'_> {
    fn default() -> Self {
        Self {
            shader: ShaderEntry::null(),
            descriptor_layouts: &[],
            push_constant_size: None,
            cache: None,
            label: None,
        }
    }
}

pub struct RenderPipelineInfo<'a> {
    pub vertex_shader: ShaderEntry<'a>,
    pub fragment_shader: ShaderEntry<'a>,
    pub color_formats: &'a [Format],
    pub depth_format: Option<Format>,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: CompareOp,
    pub descriptor_layouts: &'a [&'a DescriptorSetLayout],
    pub push_constant_size: Option<u32>,
    pub blend: Option<&'a [ColorBlend]>,
    pub blend_states: Option<&'a [vk::PipelineColorBlendAttachmentState]>,
    pub vertex_input: Option<VertexInput<'a>>,
    pub vertex_input_state: Option<vk::PipelineVertexInputStateCreateInfo<'a>>,
    pub topology: PrimitiveTopology,
    pub polygon: PolygonMode,
    pub cull: CullMode,
    pub front_face: FrontFace,
    pub label: Option<Label<'a>>,
}

impl Default for RenderPipelineInfo<'_> {
    fn default() -> Self {
        Self {
            vertex_shader: ShaderEntry::null(),
            fragment_shader: ShaderEntry::null(),
            color_formats: &[],
            depth_format: None,
            depth_test: false,
            depth_write: false,
            depth_compare: CompareOp::Always,
            descriptor_layouts: &[],
            push_constant_size: None,
            blend: None,
            blend_states: None,
            vertex_input: None,
            vertex_input_state: None,
            topology: PrimitiveTopology::TriangleList,
            polygon: PolygonMode::Fill,
            cull: CullMode::NONE,
            front_face: FrontFace::CounterClockwise,
            label: None,
        }
    }
}

pub struct ComputePipeline {
    pub inner: ComputePipelineImpl,
}

pub struct ComputePipelineImpl {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub device: Arc<DeviceImpl>,
}

pub struct RenderPipeline {
    pub inner: RenderPipelineImpl,
}

pub struct RenderPipelineImpl {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub device: Arc<DeviceImpl>,
}

impl RenderPipelineImpl {
    pub fn try_new(
        device: Arc<DeviceImpl>,
        info: &RenderPipelineInfo,
    ) -> Result<RenderPipelineImpl, crate::GPUError> {
        let mut push_constant_ranges = Vec::new();
        if let Some(size) = info.push_constant_size {
            push_constant_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                    .size(size),
            );
        }

        let layouts = info
            .descriptor_layouts
            .iter()
            .map(|l| l.handle)
            .collect::<Vec<_>>();

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&layout_info, None)
                .map_err(crate::GPUError::from)?
        };

        let vertex_stage_name = std::ffi::CString::new(info.vertex_shader.name).unwrap();
        let fragment_stage_name = std::ffi::CString::new(info.fragment_shader.name).unwrap();
        let vertex_shader = info
            .vertex_shader
            .shader
            .expect("RenderPipelineInfo::vertex_shader must be set");
        let fragment_shader = info
            .fragment_shader
            .shader
            .expect("RenderPipelineInfo::fragment_shader must be set");

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader.module.handle)
                .name(&vertex_stage_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader.module.handle)
                .name(&fragment_stage_name),
        ];

        let vertex_bindings: Vec<vk::VertexInputBindingDescription>;
        let vertex_attributes: Vec<vk::VertexInputAttributeDescription>;
        let vertex_input = if let Some(input) = info.vertex_input {
            vertex_bindings = input.buffers.iter().copied().map(Into::into).collect();
            vertex_attributes = input.attributes.iter().copied().map(Into::into).collect();
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_bindings)
                .vertex_attribute_descriptions(&vertex_attributes)
        } else {
            info.vertex_input_state.unwrap_or_default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(info.topology.into())
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .depth_bias_enable(false)
            .polygon_mode(info.polygon.into())
            .line_width(1.0)
            .cull_mode(info.cull.into())
            .front_face(info.front_face.into());

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(info.depth_test)
            .depth_write_enable(info.depth_write)
            .depth_compare_op(info.depth_compare.into())
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = if let Some(blend) = info.blend {
            blend.iter().copied().map(ColorBlend::into_vk).collect()
        } else if let Some(states) = info.blend_states {
            states.to_vec()
        } else {
            vec![ColorBlend::Disabled.into_vk()]
        };

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .attachments(&color_blend_attachment);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = info
            .color_formats
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<vk::Format>>();

        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        if let Some(format) = info.depth_format {
            rendering_info = rendering_info.depth_attachment_format(format.into());
        }

        let mut create_info = vk::GraphicsPipelineCreateInfo::default()
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

        if info.depth_format.is_some() || info.depth_test || info.depth_write {
            create_info = create_info.depth_stencil_state(&depth_stencil);
        }

        let handle = unsafe {
            device
                .handle
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .map_err(|(_, err)| crate::GPUError::from(err))?[0]
        };

        if let Some(label) = &info.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(RenderPipelineImpl {
            handle,
            layout,
            device,
        })
    }

    pub fn new(device: Arc<DeviceImpl>, info: &RenderPipelineInfo) -> RenderPipelineImpl {
        Self::try_new(device, info).expect("Create render pipeline")
    }
}

impl ComputePipelineImpl {
    pub fn new(device: Arc<DeviceImpl>, info: &ComputePipelineInfo<'_>) -> ComputePipelineImpl {
        Self::try_new(device, info).expect("Create compute pipeline")
    }

    pub fn try_new(
        device: Arc<DeviceImpl>,
        info: &ComputePipelineInfo<'_>,
    ) -> Result<ComputePipelineImpl, crate::GPUError> {
        let mut push_constant_ranges = Vec::new();
        if let Some(size) = info.push_constant_size {
            push_constant_ranges.push(
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .size(size),
            );
        }

        let layouts = info
            .descriptor_layouts
            .iter()
            .map(|l| l.handle)
            .collect::<Vec<_>>();

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let stage_name = std::ffi::CString::new(info.shader.name).unwrap();
        let shader = info
            .shader
            .shader
            .expect("ComputePipelineInfo::shader must be set");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module.handle)
            .name(&stage_name);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let handle = unsafe {
            let cache = info.cache.unwrap_or(vk::PipelineCache::null());
            device
                .handle
                .create_compute_pipelines(cache, &[create_info], None)
                .map_err(|(_, err)| crate::GPUError::from(err))?[0]
        };

        if let Some(label) = &info.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(ComputePipelineImpl {
            handle,
            layout,
            device,
        })
    }
}

impl Device {
    pub fn try_create_render_pipeline(
        &self,
        info: &RenderPipelineInfo<'_>,
    ) -> Result<RenderPipeline, crate::GPUError> {
        let inner = RenderPipelineImpl::try_new(self.inner.clone(), info)?;
        Ok(RenderPipeline { inner })
    }

    pub fn create_render_pipeline(&self, info: &RenderPipelineInfo<'_>) -> RenderPipeline {
        self.try_create_render_pipeline(info)
            .expect("Create render pipeline")
    }

    pub fn try_create_compute_pipeline(
        &self,
        info: &ComputePipelineInfo<'_>,
    ) -> Result<ComputePipeline, crate::GPUError> {
        let inner = ComputePipelineImpl::try_new(self.inner.clone(), info)?;
        Ok(ComputePipeline { inner })
    }

    pub fn create_compute_pipeline(&self, info: &ComputePipelineInfo<'_>) -> ComputePipeline {
        self.try_create_compute_pipeline(info)
            .expect("Create compute pipeline")
    }
}

impl Drop for ComputePipelineImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
            self.device
                .handle
                .destroy_pipeline_layout(self.layout, None);
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
        }
    }
}
