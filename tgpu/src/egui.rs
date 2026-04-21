use std::{collections::HashMap, fmt, ops::Range, sync::Arc};

use ::egui::{self as egui_crate, TextureId, epaint};
use ::egui_winit as egui_winit_crate;
use ::egui_winit::winit::{event::WindowEvent, window::Window};
use ash::vk;

use crate::{
    Buffer, BufferAccessTransition, BufferDesc, BufferUses, ColorAttachmentDesc, CommandRecorder,
    CopyBufferToImageInfo, DescriptorBinding, DescriptorPool, DescriptorPoolInfo, DescriptorSet,
    DescriptorSetLayout, DescriptorSetLayoutInfo, DescriptorType, DescriptorWrite, Device,
    GPUError, HostAccess, ImageAccess, ImageDesc, ImageLayout, ImageLayoutTransition,
    ImageTransition, ImageUses, ImportedBufferDesc, ImportedImageDesc, Label, MemoryPreset,
    RenderGraph, RenderPipeline, RenderPipelineInfo, RenderRecorder, SamplerCreateInfo,
    ShaderSource, ShaderStageFlags, Swapchain, ViewImage, ViewImageDesc,
};

pub use ::egui;
pub use ::egui_winit;

const SHADER_WGSL: &str = r#"
struct PushConstants {
    screen_size: vec2<f32>,
};

var<push_constant> pc: PushConstants;

struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) color_gamma: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

fn position_from_screen(screen_pos: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        2.0 * screen_pos.x / pc.screen_size.x - 1.0,
        2.0 * screen_pos.y / pc.screen_size.y - 1.0,
        0.0,
        1.0,
    );
}

fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = uv;
    out.color_gamma = color;
    out.position = position_from_screen(pos);
    return out;
}

@group(0) @binding(0) var ui_texture: texture_2d<f32>;
@group(0) @binding(1) var ui_sampler: sampler;

@fragment
fn fs_main_gamma(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color_gamma * textureSample(ui_texture, ui_sampler, in.tex_coord);
}

@fragment
fn fs_main_linear(in: VertexOutput) -> @location(0) vec4<f32> {
    let color_gamma = in.color_gamma * textureSample(ui_texture, ui_sampler, in.tex_coord);
    return vec4<f32>(linear_from_gamma_rgb(color_gamma.rgb), color_gamma.a);
}
"#;

#[derive(Debug)]
pub enum Error {
    Gpu(GPUError),
    Shader(String),
    Validation(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(err) => write!(f, "{err}"),
            Self::Shader(err) => write!(f, "egui shader error: {err}"),
            Self::Validation(err) => write!(f, "egui validation error: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(err) => Some(err),
            Self::Shader(_) | Self::Validation(_) => None,
        }
    }
}

impl From<GPUError> for Error {
    fn from(value: GPUError) -> Self {
        Self::Gpu(value)
    }
}

#[derive(Debug, Clone)]
pub struct RendererCreateInfo<'a> {
    pub label: Option<Label<'a>>,
    pub color_format: vk::Format,
    pub max_frames_in_flight: usize,
    pub max_textures: u32,
}

impl Default for RendererCreateInfo<'_> {
    fn default() -> Self {
        Self {
            label: Some(Label::Name("egui")),
            color_format: vk::Format::B8G8R8A8_SRGB,
            max_frames_in_flight: 3,
            max_textures: 1024,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    screen_size: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureKind {
    Managed,
    External,
}

struct TextureBinding {
    image: ViewImage,
    descriptor_set: DescriptorSet,
    extent: vk::Extent3D,
    kind: TextureKind,
    initialized: bool,
    options: Option<epaint::textures::TextureOptions>,
}

struct TextureUpload {
    texture_id: TextureId,
    buffer: Buffer,
    regions: Vec<vk::BufferImageCopy>,
}

#[derive(Clone)]
struct DrawBatch {
    texture_id: TextureId,
    clip_rect: egui_crate::Rect,
    index_range: Range<u32>,
}

#[derive(Default)]
struct FrameResources {
    uploads: Vec<TextureUpload>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    draws: Vec<DrawBatch>,
    used_textures: Vec<TextureId>,
}

impl FrameResources {
    fn clear(&mut self) {
        self.uploads.clear();
        self.vertex_buffer = None;
        self.index_buffer = None;
        self.draws.clear();
        self.used_textures.clear();
    }
}

pub struct Renderer {
    device: Device,
    context: egui_crate::Context,
    state: egui_winit_crate::State,
    first_frame: bool,
    pipeline: RenderPipeline,
    texture_layout: DescriptorSetLayout,
    texture_pool: Arc<DescriptorPool>,
    textures: HashMap<TextureId, TextureBinding>,
    frames: Vec<FrameResources>,
    pending_texture_frees: Vec<TextureId>,
    next_user_texture_id: u64,
}

pub struct Frame<'a> {
    renderer: &'a mut Renderer,
    flight_index: usize,
    pixels_per_point: f32,
}

#[derive(Clone, Copy)]
struct GraphUpload<'a> {
    buffer: crate::GraphBuffer,
    image: crate::GraphImage,
    regions: &'a [vk::BufferImageCopy],
}

#[derive(Clone)]
struct PreparedDraw<'a> {
    set: &'a DescriptorSet,
    clip_rect: egui_crate::Rect,
    index_range: Range<u32>,
}

impl Renderer {
    pub fn new(
        window: &Window,
        device: &Device,
        info: &RendererCreateInfo<'_>,
    ) -> Result<Self, Error> {
        if info.max_frames_in_flight == 0 {
            return Err(Error::Validation(
                "max_frames_in_flight must be greater than zero".into(),
            ));
        }
        if info.max_textures == 0 {
            return Err(Error::Validation(
                "max_textures must be greater than zero".into(),
            ));
        }

        let context = egui_crate::Context::default();
        let max_texture_side = Some(
            device
                .adapter
                .inner
                .properties
                .limits
                .max_image_dimension2_d as usize,
        );
        let state = egui_winit_crate::State::new(
            context.clone(),
            egui_crate::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            max_texture_side,
        );

        let texture_binding = [
            DescriptorBinding::unique(0, DescriptorType::SampledImage, ShaderStageFlags::FRAGMENT),
            DescriptorBinding::unique(1, DescriptorType::Sampler, ShaderStageFlags::FRAGMENT),
        ];

        let texture_layout = device.create_descriptor_set_layout(&DescriptorSetLayoutInfo {
            bindings: &texture_binding,
            label: info.label.clone(),
            ..Default::default()
        });

        let texture_pool = device.create_descriptor_pool(&DescriptorPoolInfo {
            max_sets: info.max_textures,
            layouts: &[&texture_layout],
            label: info.label.clone(),
            ..Default::default()
        });

        let shader = device
            .create_shader(info.label.clone(), ShaderSource::Wgsl(SHADER_WGSL))
            .map_err(Error::Shader)?;

        let vertex_binding = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<epaint::Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let vertex_attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(16),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding)
            .vertex_attribute_descriptions(&vertex_attributes);

        let blend_state = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let fragment_entry = if is_srgb_format(info.color_format) {
            "fs_main_linear"
        } else {
            "fs_main_gamma"
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineInfo {
            label: info.label.clone(),
            vertex_shader: shader.entry("vs_main"),
            fragment_shader: shader.entry(fragment_entry),
            color_formats: &[info.color_format],
            descriptor_layouts: &[&texture_layout],
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            blend_states: Some(&blend_state),
            vertex_input_state: Some(vertex_input),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon: vk::PolygonMode::FILL,
            cull: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            ..Default::default()
        });

        let mut frames = Vec::with_capacity(info.max_frames_in_flight);
        for _ in 0..info.max_frames_in_flight {
            frames.push(FrameResources::default());
        }

        Ok(Self {
            device: device.clone(),
            context,
            state,
            first_frame: true,
            pipeline,
            texture_layout,
            texture_pool,
            textures: HashMap::new(),
            frames,
            pending_texture_frees: Vec::new(),
            next_user_texture_id: 0,
        })
    }

    pub fn new_for_swapchain(
        window: &Window,
        device: &Device,
        swapchain: &Swapchain,
    ) -> Result<Self, Error> {
        Self::new(
            window,
            device,
            &RendererCreateInfo {
                color_format: swapchain.format(),
                max_frames_in_flight: swapchain.max_frames_in_flight(),
                ..Default::default()
            },
        )
    }

    pub fn context(&self) -> &egui_crate::Context {
        &self.context
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit_crate::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn texture(&self, id: TextureId) -> Option<&ViewImage> {
        self.textures.get(&id).map(|binding| &binding.image)
    }

    pub fn register_user_texture(
        &mut self,
        texture: &ViewImage,
        extent: vk::Extent2D,
    ) -> Result<TextureId, Error> {
        let id = TextureId::User(self.next_user_texture_id);
        self.next_user_texture_id += 1;
        self.insert_external_texture(id, texture, extent)?;
        Ok(id)
    }

    pub fn update_user_texture(
        &mut self,
        id: TextureId,
        texture: &ViewImage,
        extent: vk::Extent2D,
    ) -> Result<(), Error> {
        let Some(existing) = self.textures.get(&id) else {
            return Err(Error::Validation(format!(
                "cannot update missing egui user texture {id:?}"
            )));
        };
        if existing.kind != TextureKind::External {
            return Err(Error::Validation(format!(
                "texture {id:?} is managed by egui and cannot be updated through update_user_texture"
            )));
        }

        self.device.wait_idle();
        self.insert_external_texture(id, texture, extent)
    }

    pub fn unregister_user_texture(&mut self, id: TextureId) -> Result<(), Error> {
        let Some(existing) = self.textures.get(&id) else {
            return Ok(());
        };
        if existing.kind != TextureKind::External {
            return Err(Error::Validation(format!(
                "texture {id:?} is managed by egui and cannot be unregistered manually"
            )));
        }

        self.device.wait_idle();
        self.textures.remove(&id);
        Ok(())
    }

    pub fn run<'a, F>(
        &'a mut self,
        window: &Window,
        flight_index: usize,
        run_ui: F,
    ) -> Result<Frame<'a>, Error>
    where
        F: FnOnce(&egui_crate::Context),
    {
        if flight_index >= self.frames.len() {
            return Err(Error::Validation(format!(
                "flight index {} is out of range for {} in-flight frames",
                flight_index,
                self.frames.len()
            )));
        }

        self.frames[flight_index].clear();

        let viewport = self
            .state
            .egui_input_mut()
            .viewports
            .entry(egui_crate::ViewportId::ROOT)
            .or_default();
        egui_winit::update_viewport_info(viewport, &self.context, window, self.first_frame);

        let raw_input = self.state.take_egui_input(window);
        self.context.begin_pass(raw_input);
        run_ui(&self.context);
        let egui_crate::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output: _,
        } = self.context.end_pass();
        self.first_frame = false;

        let paint_jobs = self.context.tessellate(shapes, pixels_per_point);
        self.state.handle_platform_output(window, platform_output);

        self.process_texture_deltas(flight_index, &textures_delta.set)?;
        self.build_frame_resources(flight_index, &paint_jobs)?;
        self.pending_texture_frees.extend(textures_delta.free);

        Ok(Frame {
            renderer: self,
            flight_index,
            pixels_per_point,
        })
    }

    fn insert_external_texture(
        &mut self,
        id: TextureId,
        texture: &ViewImage,
        extent: vk::Extent2D,
    ) -> Result<(), Error> {
        let descriptor_set = self.allocate_texture_descriptor_set(texture)?;
        self.textures.insert(
            id,
            TextureBinding {
                image: texture.clone(),
                descriptor_set,
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                kind: TextureKind::External,
                initialized: true,
                options: None,
            },
        );
        Ok(())
    }

    fn process_texture_deltas(
        &mut self,
        flight_index: usize,
        deltas: &[(TextureId, epaint::ImageDelta)],
    ) -> Result<(), Error> {
        if self.pending_texture_frees.is_empty() && deltas.is_empty() {
            return Ok(());
        }

        self.device.wait_idle();
        self.flush_pending_texture_frees();

        for (id, delta) in deltas {
            self.apply_texture_delta(flight_index, *id, delta)?;
        }

        Ok(())
    }

    fn flush_pending_texture_frees(&mut self) {
        for id in self.pending_texture_frees.drain(..) {
            if self
                .textures
                .get(&id)
                .is_some_and(|binding| binding.kind == TextureKind::Managed)
            {
                self.textures.remove(&id);
            }
        }
    }

    fn apply_texture_delta(
        &mut self,
        flight_index: usize,
        id: TextureId,
        delta: &epaint::ImageDelta,
    ) -> Result<(), Error> {
        if matches!(id, TextureId::User(_)) {
            return Err(Error::Validation(format!(
                "egui attempted to upload backend-managed texture {id:?}"
            )));
        }

        let [width, height] = delta.image.size();
        if width == 0 || height == 0 {
            return Err(Error::Validation(format!(
                "egui texture {id:?} has an invalid size of {width}x{height}"
            )));
        }

        let extent = vk::Extent3D {
            width: width as u32,
            height: height as u32,
            depth: 1,
        };

        match delta.pos {
            Some(pos) => {
                let Some(binding) = self.textures.get_mut(&id) else {
                    return Err(Error::Validation(format!(
                        "egui issued a partial update for missing texture {id:?}"
                    )));
                };
                if binding.kind != TextureKind::Managed {
                    return Err(Error::Validation(format!(
                        "egui issued a partial update for non-managed texture {id:?}"
                    )));
                }

                update_texture_sampler(&self.device, binding, delta.options);

                if pos[0] as u32 + extent.width > binding.extent.width
                    || pos[1] as u32 + extent.height > binding.extent.height
                {
                    return Err(Error::Validation(format!(
                        "egui partial update for texture {id:?} exceeds the existing texture bounds"
                    )));
                }
            }
            None => {
                let needs_recreate = self.textures.get(&id).is_none_or(|binding| {
                    binding.kind != TextureKind::Managed || binding.extent != extent
                });

                if needs_recreate {
                    let binding = self.create_managed_texture(extent, delta.options)?;
                    self.textures.insert(id, binding);
                } else if let Some(binding) = self.textures.get_mut(&id) {
                    update_texture_sampler(&self.device, binding, delta.options);
                }
            }
        }

        let upload = self.create_texture_upload(id, delta)?;
        self.frames[flight_index].uploads.push(upload);
        Ok(())
    }

    fn create_managed_texture(
        &self,
        extent: vk::Extent3D,
        options: epaint::textures::TextureOptions,
    ) -> Result<TextureBinding, Error> {
        let image = self.device.create_view_image(&ViewImageDesc {
            image: ImageDesc {
                label: Some(Label::Name("egui texture")),
                format: vk::Format::R8G8B8A8_UNORM,
                extent,
                usage: ImageUses::COPY_DST | ImageUses::SAMPLED,
                ..Default::default()
            },
            sampler: Some(texture_sampler_create_info(options)),
            aspect: Some(vk::ImageAspectFlags::COLOR),
            ..Default::default()
        })?;

        let descriptor_set = self.allocate_texture_descriptor_set(&image)?;

        Ok(TextureBinding {
            image,
            descriptor_set,
            extent,
            kind: TextureKind::Managed,
            initialized: false,
            options: Some(options),
        })
    }

    fn allocate_texture_descriptor_set(&self, image: &ViewImage) -> Result<DescriptorSet, Error> {
        let Some(sampler) = image.sampler.as_ref() else {
            return Err(Error::Validation(
                "egui textures require a sampler-backed ViewImage".into(),
            ));
        };

        let descriptor_set = self
            .device
            .create_descriptor_set(self.texture_pool.clone(), &self.texture_layout);
        descriptor_set.write(&[
            DescriptorWrite::SampledImage {
                binding: 0,
                image_view: &image.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                array_element: None,
            },
            DescriptorWrite::Sampler {
                binding: 1,
                sampler,
                array_element: None,
            },
        ]);
        Ok(descriptor_set)
    }

    fn create_texture_upload(
        &self,
        texture_id: TextureId,
        delta: &epaint::ImageDelta,
    ) -> Result<TextureUpload, Error> {
        let pixels = match &delta.image {
            epaint::ImageData::Color(image) => bytemuck::cast_slice(image.pixels.as_slice()),
        };

        let buffer = self.device.create_buffer(&BufferDesc {
            label: Some(Label::Name("egui texture upload")),
            size: pixels.len(),
            usage: BufferUses::COPY_SRC,
            memory: MemoryPreset::Dynamic,
            host_access: HostAccess::WriteSequential,
            ..Default::default()
        })?;
        buffer.write(pixels, 0);

        let [width, height] = delta.image.size();
        let pos = delta.pos.unwrap_or([0, 0]);
        let regions = vec![
            vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: pos[0] as i32,
                    y: pos[1] as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: width as u32,
                    height: height as u32,
                    depth: 1,
                }),
        ];

        Ok(TextureUpload {
            texture_id,
            buffer,
            regions,
        })
    }

    fn build_frame_resources(
        &mut self,
        flight_index: usize,
        paint_jobs: &[epaint::ClippedPrimitive],
    ) -> Result<(), Error> {
        let mut vertices = Vec::<epaint::Vertex>::new();
        let mut indices = Vec::<u32>::new();
        let frame = &mut self.frames[flight_index];
        let mut skipped_callbacks = 0usize;

        for clipped in paint_jobs {
            match &clipped.primitive {
                epaint::Primitive::Mesh(mesh) => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }

                    if !self.textures.contains_key(&mesh.texture_id) {
                        log::warn!(
                            "Skipping egui mesh for missing texture {:?}",
                            mesh.texture_id
                        );
                        continue;
                    }

                    if !frame.used_textures.contains(&mesh.texture_id) {
                        frame.used_textures.push(mesh.texture_id);
                    }

                    let first_index = indices.len() as u32;
                    let vertex_offset = vertices.len() as u32;
                    vertices.extend_from_slice(&mesh.vertices);
                    indices.extend(mesh.indices.iter().map(|index| index + vertex_offset));
                    frame.draws.push(DrawBatch {
                        texture_id: mesh.texture_id,
                        clip_rect: clipped.clip_rect,
                        index_range: first_index..indices.len() as u32,
                    });
                }
                epaint::Primitive::Callback(_) => skipped_callbacks += 1,
            }
        }

        if skipped_callbacks > 0 {
            log::warn!(
                "Skipping {} egui paint callback(s); tgpu::egui does not support callback rendering yet",
                skipped_callbacks
            );
        }

        if vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        let vertex_buffer = self.device.create_buffer(&BufferDesc {
            label: Some(Label::Name("egui vertices")),
            size: std::mem::size_of_val(vertices.as_slice()),
            usage: BufferUses::VERTEX,
            memory: MemoryPreset::Dynamic,
            host_access: HostAccess::WriteSequential,
            ..Default::default()
        })?;
        vertex_buffer.write_slice(&vertices);

        let index_buffer = self.device.create_buffer(&BufferDesc {
            label: Some(Label::Name("egui indices")),
            size: std::mem::size_of_val(indices.as_slice()),
            usage: BufferUses::INDEX,
            memory: MemoryPreset::Dynamic,
            host_access: HostAccess::WriteSequential,
            ..Default::default()
        })?;
        index_buffer.write_slice(&indices);

        frame.vertex_buffer = Some(vertex_buffer);
        frame.index_buffer = Some(index_buffer);
        Ok(())
    }
}

impl Frame<'_> {
    pub fn pixels_per_point(&self) -> f32 {
        self.pixels_per_point
    }

    pub fn record_uploads(&mut self, cmd: &mut CommandRecorder) {
        let uploads = &self.renderer.frames[self.flight_index].uploads;
        let mut upload_state = HashMap::<TextureId, bool>::new();

        for upload in uploads {
            let Some(texture) = self.renderer.textures.get(&upload.texture_id) else {
                continue;
            };

            let from = upload_state
                .get(&upload.texture_id)
                .copied()
                .unwrap_or(texture.initialized);

            cmd.image_transition(
                &texture.image.image,
                ImageTransition {
                    from: if from {
                        ImageLayoutTransition::FRAGMENT
                    } else {
                        ImageLayoutTransition::UNDEFINED
                    },
                    to: ImageLayoutTransition::new(ImageLayout::TransferDst),
                    aspect: vk::ImageAspectFlags::COLOR,
                    ..Default::default()
                },
            );
            cmd.copy_buffer_to_image(&CopyBufferToImageInfo {
                src: &upload.buffer,
                dst: &texture.image.image,
                dst_layout: ImageLayout::TransferDst,
                regions: &upload.regions,
            });
            cmd.image_transition(
                &texture.image.image,
                ImageTransition {
                    from: ImageLayoutTransition::new(ImageLayout::TransferDst),
                    to: ImageLayoutTransition::FRAGMENT,
                    aspect: vk::ImageAspectFlags::COLOR,
                    ..Default::default()
                },
            );

            upload_state.insert(upload.texture_id, true);
        }

        for texture_id in upload_state.into_keys() {
            if let Some(texture) = self.renderer.textures.get_mut(&texture_id) {
                texture.initialized = true;
            }
        }
    }

    pub fn paint(&mut self, render: &mut RenderRecorder<'_>, extent: vk::Extent2D) {
        let frame = &self.renderer.frames[self.flight_index];
        let Some(vertex_buffer) = frame.vertex_buffer.as_ref() else {
            return;
        };
        let Some(index_buffer) = frame.index_buffer.as_ref() else {
            return;
        };

        let mut draws = Vec::with_capacity(frame.draws.len());
        for draw in &frame.draws {
            let Some(texture) = self.renderer.textures.get(&draw.texture_id) else {
                continue;
            };
            draws.push(PreparedDraw {
                set: &texture.descriptor_set,
                clip_rect: draw.clip_rect,
                index_range: draw.index_range.clone(),
            });
        }

        paint_render_pass(
            &self.renderer.pipeline,
            vertex_buffer,
            index_buffer,
            &draws,
            self.pixels_per_point,
            extent,
            render,
        );
    }

    pub fn add_to_graph<'a>(
        &'a mut self,
        graph: &mut RenderGraph<'a>,
        target: crate::GraphImage,
        target_desc: ColorAttachmentDesc,
        extent: vk::Extent2D,
    ) {
        let touched_textures = {
            let frame = &self.renderer.frames[self.flight_index];
            frame
                .used_textures
                .iter()
                .copied()
                .chain(frame.uploads.iter().map(|upload| upload.texture_id))
                .collect::<Vec<_>>()
        };
        let initial_states = touched_textures
            .iter()
            .filter_map(|texture_id| {
                self.renderer
                    .textures
                    .get(texture_id)
                    .map(|texture| (*texture_id, texture.initialized))
            })
            .collect::<HashMap<_, _>>();

        {
            let frame = &self.renderer.frames[self.flight_index];
            for upload in &frame.uploads {
                if let Some(texture) = self.renderer.textures.get_mut(&upload.texture_id) {
                    texture.initialized = true;
                }
            }
        }

        let renderer: &'a Renderer = &*self.renderer;
        let frame = &renderer.frames[self.flight_index];

        let mut graph_images = HashMap::new();
        for texture_id in touched_textures {
            if graph_images.contains_key(&texture_id) {
                continue;
            }

            let Some(texture) = renderer.textures.get(&texture_id) else {
                continue;
            };

            let image = graph.import_image_view(
                format!("egui_texture_{texture_id:?}"),
                &texture.image.image,
                &texture.image.view,
                ImportedImageDesc {
                    extent: texture.extent,
                    aspect: vk::ImageAspectFlags::COLOR,
                    initial: if initial_states
                        .get(&texture_id)
                        .copied()
                        .unwrap_or(texture.initialized)
                    {
                        ImageLayoutTransition::FRAGMENT
                    } else {
                        ImageLayoutTransition::UNDEFINED
                    },
                    initialized: initial_states
                        .get(&texture_id)
                        .copied()
                        .unwrap_or(texture.initialized),
                },
            );
            graph_images.insert(texture_id, image);
        }

        if !frame.uploads.is_empty() {
            let uploads = frame
                .uploads
                .iter()
                .enumerate()
                .map(|(index, upload)| GraphUpload {
                    buffer: graph.import_buffer(
                        format!("egui_upload_{index}"),
                        &upload.buffer,
                        ImportedBufferDesc {
                            initial: BufferAccessTransition::NONE,
                            initialized: true,
                        },
                    ),
                    image: graph_images[&upload.texture_id],
                    regions: upload.regions.as_slice(),
                })
                .collect::<Vec<_>>();
            let upload_exec = uploads.clone();

            graph.add_copy_pass(
                "egui_uploads",
                |pass| {
                    for upload in &uploads {
                        pass.read_buffer(upload.buffer, crate::BufferAccess::TransferSrc);
                        pass.write_image(upload.image, ImageAccess::TransferDst);
                    }
                },
                move |ctx| {
                    for upload in &upload_exec {
                        let buffer = ctx.buffer(upload.buffer).clone();
                        let image = ctx.image(upload.image).clone();
                        ctx.cmd().copy_buffer_to_image(&CopyBufferToImageInfo {
                            src: &buffer,
                            dst: &image,
                            dst_layout: ImageLayout::TransferDst,
                            regions: upload.regions,
                        });
                    }
                },
            );
        }

        let Some(vertex_buffer) = frame.vertex_buffer.as_ref() else {
            return;
        };
        let Some(index_buffer) = frame.index_buffer.as_ref() else {
            return;
        };

        let graph_vertex = graph.import_buffer(
            "egui_vertices",
            vertex_buffer,
            ImportedBufferDesc {
                initial: BufferAccessTransition::NONE,
                initialized: true,
            },
        );
        let graph_index = graph.import_buffer(
            "egui_indices",
            index_buffer,
            ImportedBufferDesc {
                initial: BufferAccessTransition::NONE,
                initialized: true,
            },
        );

        let draws = frame
            .draws
            .iter()
            .filter_map(|draw| {
                renderer
                    .textures
                    .get(&draw.texture_id)
                    .map(|texture| PreparedDraw {
                        set: &texture.descriptor_set,
                        clip_rect: draw.clip_rect,
                        index_range: draw.index_range.clone(),
                    })
            })
            .collect::<Vec<_>>();
        let used_images = frame
            .used_textures
            .iter()
            .filter_map(|texture_id| graph_images.get(texture_id).copied())
            .collect::<Vec<_>>();
        let pipeline = &renderer.pipeline;
        let pixels_per_point = self.pixels_per_point;

        graph.add_render_pass(
            "egui",
            |pass| {
                pass.read_buffer(graph_vertex, crate::BufferAccess::Vertex);
                pass.read_buffer(graph_index, crate::BufferAccess::Index);
                for image in &used_images {
                    pass.read_image(*image, ImageAccess::SampledFragment);
                }
                pass.write_color(target, target_desc);
            },
            move |ctx| {
                let vertex_buffer = ctx.buffer(graph_vertex).clone();
                let index_buffer = ctx.buffer(graph_index).clone();
                ctx.begin_render(|render| {
                    paint_render_pass(
                        pipeline,
                        &vertex_buffer,
                        &index_buffer,
                        &draws,
                        pixels_per_point,
                        extent,
                        render,
                    );
                });
            },
        );
    }
}

fn paint_render_pass(
    pipeline: &RenderPipeline,
    vertex_buffer: &Buffer,
    index_buffer: &Buffer,
    draws: &[PreparedDraw<'_>],
    pixels_per_point: f32,
    extent: vk::Extent2D,
    render: &mut RenderRecorder<'_>,
) {
    if draws.is_empty() {
        return;
    }

    render.bind_render_pipeline(pipeline);
    render.bind_vertex_buffer(0, vertex_buffer, 0);
    render.bind_index_buffer(index_buffer, 0, vk::IndexType::UINT32);
    render.viewport(vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    render.push_render_constants(
        pipeline,
        PushConstants {
            screen_size: [
                extent.width as f32 / pixels_per_point,
                extent.height as f32 / pixels_per_point,
            ],
        },
    );

    for draw in draws {
        let Some(scissor) = clip_rect_to_scissor(draw.clip_rect, pixels_per_point, extent) else {
            continue;
        };

        render.bind_render_descriptor_set(draw.set, pipeline, 0, &[]);
        render.scissor(scissor);
        render.draw_indexed(draw.index_range.clone(), 0, 0..1);
    }

    render.scissor(vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    });
}

fn clip_rect_to_scissor(
    clip_rect: egui_crate::Rect,
    pixels_per_point: f32,
    extent: vk::Extent2D,
) -> Option<vk::Rect2D> {
    let min_x = (clip_rect.min.x * pixels_per_point)
        .floor()
        .clamp(0.0, extent.width as f32);
    let min_y = (clip_rect.min.y * pixels_per_point)
        .floor()
        .clamp(0.0, extent.height as f32);
    let max_x = (clip_rect.max.x * pixels_per_point)
        .ceil()
        .clamp(min_x, extent.width as f32);
    let max_y = (clip_rect.max.y * pixels_per_point)
        .ceil()
        .clamp(min_y, extent.height as f32);

    if max_x <= min_x || max_y <= min_y {
        return None;
    }

    Some(vk::Rect2D {
        offset: vk::Offset2D {
            x: min_x as i32,
            y: min_y as i32,
        },
        extent: vk::Extent2D {
            width: (max_x - min_x) as u32,
            height: (max_y - min_y) as u32,
        },
    })
}

fn update_texture_sampler(
    device: &Device,
    binding: &mut TextureBinding,
    options: epaint::textures::TextureOptions,
) {
    if binding.kind == TextureKind::External || binding.options == Some(options) {
        return;
    }

    let sampler = device.create_sampler(&texture_sampler_create_info(options));
    binding.image.sampler = Some(sampler.clone());
    binding.image.view.sampler = Some(sampler.clone());
    binding.descriptor_set.write(&[
        DescriptorWrite::SampledImage {
            binding: 0,
            image_view: &binding.image.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            array_element: None,
        },
        DescriptorWrite::Sampler {
            binding: 1,
            sampler: &sampler,
            array_element: None,
        },
    ]);
    binding.options = Some(options);
}

fn texture_sampler_create_info(
    options: epaint::textures::TextureOptions,
) -> SamplerCreateInfo<'static> {
    SamplerCreateInfo {
        mag: texture_filter(options.magnification),
        min: texture_filter(options.minification),
        mipmap: texture_mipmap_mode(options.mipmap_mode),
        address_u: texture_wrap(options.wrap_mode),
        address_v: texture_wrap(options.wrap_mode),
        address_w: texture_wrap(options.wrap_mode),
        label: Some(Label::Name("egui sampler")),
        ..Default::default()
    }
}

fn texture_filter(filter: epaint::textures::TextureFilter) -> vk::Filter {
    match filter {
        epaint::textures::TextureFilter::Nearest => vk::Filter::NEAREST,
        epaint::textures::TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

fn texture_mipmap_mode(filter: Option<epaint::textures::TextureFilter>) -> vk::SamplerMipmapMode {
    match filter.unwrap_or(epaint::textures::TextureFilter::Nearest) {
        epaint::textures::TextureFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
        epaint::textures::TextureFilter::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

fn texture_wrap(wrap: epaint::textures::TextureWrapMode) -> vk::SamplerAddressMode {
    match wrap {
        epaint::textures::TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        epaint::textures::TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
        epaint::textures::TextureWrapMode::MirroredRepeat => {
            vk::SamplerAddressMode::MIRRORED_REPEAT
        }
    }
}

fn is_srgb_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_SRGB
            | vk::Format::R8G8_SRGB
            | vk::Format::R8G8B8_SRGB
            | vk::Format::B8G8R8_SRGB
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}
