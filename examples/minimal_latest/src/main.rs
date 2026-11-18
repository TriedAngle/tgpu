use std::{error::Error, path::Path, time::Instant};

use raw_window_handle::HasDisplayHandle;
use tgpu::egui::egui;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const SHADER: &str = include_str!("shader.slang");
const START_SIZE: tgpu::Extent2D = tgpu::Extent2D::new(800, 600);

const VERTICES: [Vertex; 6] = [
    Vertex {
        position: [0.0, -0.5, 0.5],
        color: [1.0, 0.0, 0.0],
        tex_coord: [0.5, 0.5],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        color: [0.0, 0.0, 1.0],
        tex_coord: [0.5, 0.5],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        color: [0.0, 1.0, 0.0],
        tex_coord: [0.5, 0.5],
    },
    Vertex {
        position: [0.1, -0.4, 0.75],
        color: [0.3, 0.3, 0.3],
        tex_coord: [0.5, 1.0],
    },
    Vertex {
        position: [-0.4, 0.6, 0.25],
        color: [1.0, 1.0, 1.0],
        tex_coord: [1.0, 0.0],
    },
    Vertex {
        position: [0.6, 0.6, 0.75],
        color: [0.7, 0.7, 0.7],
        tex_coord: [0.0, 0.0],
    },
];

const POINTS: [[f32; 2]; 4] = [[0.05, 0.0], [-0.05, 0.0], [0.0, -0.05], [0.0, 0.05]];

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneInfo {
    resolution: [f32; 2],
    anim_value: f32,
    num_data: u32,
    tex_id: u32,
    _pad0: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushData {
    vertex_buffer: tgpu::RwBufferHandle,
    rotation_angle: f32,
    num_vertex: u32,
    _pad0: u32,
    scene_info: tgpu::UniformBufferHandle,
    points: tgpu::ReadBufferHandle,
    texture_image: tgpu::SampledImageHandle,
    texture_sampler: tgpu::SamplerHandle,
    color: [f32; 4],
}

struct FrameData {
    scene_buffer: tgpu::Buffer,
    scene_handle: tgpu::UniformBufferHandle,
    last_signal_value: u64,
}

struct GBuffer {
    color: tgpu::ViewImage,
    depth: tgpu::ViewImage,
    extent: tgpu::Extent2D,
    color_format: tgpu::Format,
    depth_format: tgpu::Format,
}

#[allow(dead_code)]
pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    egui: tgpu::egui::Renderer,
    gbuffer_texture: egui::TextureId,
    gbuffer: GBuffer,
    bindless: tgpu::BindlessHeap,
    vertex_buffer: tgpu::Buffer,
    vertex_handle: tgpu::RwBufferHandle,
    points_buffer: tgpu::Buffer,
    points_handle: tgpu::ReadBufferHandle,
    frame_data: Vec<FrameData>,
    sampler: tgpu::Sampler,
    sampler_handle: tgpu::SamplerHandle,
    textures: Vec<tgpu::ViewImage>,
    texture_handles: Vec<tgpu::SampledImageHandle>,
    compute_pipeline: tgpu::ComputePipeline,
    graphics_pipeline_with_texture: tgpu::RenderPipeline,
    graphics_pipeline_without_texture: tgpu::RenderPipeline,
    frame_ring_current: usize,
    swapchain_needs_rebuild: bool,
    gbuffer_needs_rebuild: bool,
    vertex_was_read: bool,
    v_sync: bool,
    exit_requested: bool,
    image_id: usize,
    viewport_size: tgpu::Extent2D,
    clear_color: [f32; 4],
    start_time: Instant,
    last_frame_time: Instant,
    delta_time: f32,
    frame_count: u64,
}

impl Render {
    pub fn new(window: Window) -> Result<Self, Box<dyn Error>> {
        let display = window.display_handle()?.as_raw();
        let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
            app_name: "tgpu minimal latest",
            engine_name: "tgpu example",
            display: Some(display),
        })?;

        let adapter = instance.default_adapter()?;

        let (device, queue) = adapter.request_device(&tgpu::DeviceCreateInfo::default())?;

        let size = window.inner_size();
        let preferred_extent = tgpu::Extent2D::new(size.width.max(1), size.height.max(1));
        let swapchain = device.create_swapchain_for_window(&window, preferred_extent)?;

        initialize_swapchain_images(&queue, &swapchain);

        let viewport_size = swapchain.extent();
        let gbuffer = create_gbuffer(&device, viewport_size)?;
        let mut egui = tgpu::egui::Renderer::new_for_swapchain(&window, &device, &swapchain)?;
        let gbuffer_texture = egui.register_user_texture(&gbuffer.color, gbuffer.extent)?;

        let bindless = device.create_bindless_heap(&tgpu::BindlessInfo {
            max_read_buffers: 1,
            max_rw_buffers: 1,
            max_sampled_images: 2,
            max_samplers: 1,
            max_uniform_buffers: 16,
            ..Default::default()
        });

        let vertex_buffer = device.create_buffer_from_slice(
            &VERTICES,
            &tgpu::BufferDesc {
                label: Some(tgpu::Label::Name("minimal latest vertices")),
                usage: tgpu::BufferUses::VERTEX | tgpu::BufferUses::STORAGE,
                memory: tgpu::MemoryPreset::Dynamic,
                host_access: tgpu::HostAccess::WriteSequential,
                ..Default::default()
            },
        )?;
        let vertex_handle = bindless.add_rw_buffer(&vertex_buffer);

        let points_buffer = device.create_buffer_from_slice(
            &POINTS,
            &tgpu::BufferDesc {
                label: Some(tgpu::Label::Name("minimal latest points")),
                usage: tgpu::BufferUses::STORAGE,
                memory: tgpu::MemoryPreset::Dynamic,
                host_access: tgpu::HostAccess::WriteSequential,
                ..Default::default()
            },
        )?;
        let points_handle = bindless.add_read_buffer(&points_buffer);

        let sampler = device.create_sampler(&tgpu::SamplerCreateInfo {
            address_u: tgpu::SamplerAddressMode::Repeat,
            address_v: tgpu::SamplerAddressMode::Repeat,
            address_w: tgpu::SamplerAddressMode::Repeat,
            ..Default::default()
        });
        let sampler_handle = bindless.add_sampler(&sampler);

        let mut uploads = Vec::new();
        let mut init = queue.record();
        let texture_a = create_texture(&device, &mut init, &mut uploads, 0)?;
        let texture_b = create_texture(&device, &mut init, &mut uploads, 1)?;
        let texture_handles = vec![
            bindless.add_sampled_image(&texture_a.view, tgpu::ImageLayout::Fragment),
            bindless.add_sampled_image(&texture_b.view, tgpu::ImageLayout::Fragment),
        ];
        let init_signal = queue.submit(tgpu::SubmitInfo {
            records: &[init.finish()],
            ..Default::default()
        });
        queue.timeline.wait(init_signal, None);
        drop(uploads);

        let mut frame_data = Vec::new();
        ensure_frame_data(
            &device,
            &bindless,
            &mut frame_data,
            swapchain.max_frames_in_flight(),
        )?;

        let shader = device.create_shader(
            Some(tgpu::Label::Name("minimal latest shader")),
            tgpu::ShaderSource::Slang(SHADER.as_bytes()),
        )?;

        let compute_pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
            label: Some(tgpu::Label::Name("minimal latest compute")),
            shader: shader.entry("computeMain"),
            descriptor_layouts: &[bindless.layout()],
            push_constant_size: tgpu::push_constant_size::<PushData>(),
            cache: None,
        });

        let graphics_pipeline_without_texture = create_graphics_pipeline(
            &device,
            &bindless,
            &gbuffer,
            &shader,
            "fragmentMainColor",
            "minimal latest graphics color",
        );
        let graphics_pipeline_with_texture = create_graphics_pipeline(
            &device,
            &bindless,
            &gbuffer,
            &shader,
            "fragmentMainTextured",
            "minimal latest graphics textured",
        );

        let now = Instant::now();
        Ok(Self {
            window,
            instance,
            device,
            queue,
            swapchain,
            egui,
            gbuffer_texture,
            gbuffer,
            bindless,
            vertex_buffer,
            vertex_handle,
            points_buffer,
            points_handle,
            frame_data,
            sampler,
            sampler_handle,
            textures: vec![texture_a, texture_b],
            texture_handles,
            compute_pipeline,
            graphics_pipeline_with_texture,
            graphics_pipeline_without_texture,
            frame_ring_current: 0,
            swapchain_needs_rebuild: false,
            gbuffer_needs_rebuild: false,
            vertex_was_read: false,
            v_sync: false,
            exit_requested: false,
            image_id: 0,
            viewport_size,
            clear_color: [0.2, 0.2, 0.3, 1.0],
            start_time: now,
            last_frame_time: now,
            delta_time: 1.0 / 60.0,
            frame_count: 0,
        })
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        if self.swapchain_needs_rebuild {
            self.recreate_swapchain()?;
        }
        if self.gbuffer_needs_rebuild {
            self.recreate_gbuffer()?;
        }

        let now = Instant::now();
        self.delta_time = (now - self.last_frame_time).as_secs_f32().max(1.0 / 1000.0);
        self.last_frame_time = now;

        let ring = self.frame_ring_current;
        self.queue
            .timeline
            .wait(self.frame_data[ring].last_signal_value, None);

        let mut image_id = self.image_id;
        let mut clear_color = self.clear_color;
        let mut v_sync = self.v_sync;
        let mut exit_requested = false;
        let mut request_swapchain_rebuild = false;
        let mut requested_viewport = self.viewport_size;
        let fps = 1.0 / self.delta_time;
        let gbuffer_texture = self.gbuffer_texture;

        let mut egui_frame = self.egui.run(&self.window, ring, |ctx| {
            build_ui(
                ctx,
                gbuffer_texture,
                &mut image_id,
                &mut clear_color,
                &mut v_sync,
                &mut request_swapchain_rebuild,
                &mut exit_requested,
                &mut requested_viewport,
                fps,
            );
        })?;

        self.exit_requested = exit_requested;
        self.image_id = image_id.min(self.texture_handles.len().saturating_sub(1));
        self.clear_color = clear_color;
        if self.v_sync != v_sync || request_swapchain_rebuild {
            self.v_sync = v_sync;
            self.swapchain_needs_rebuild = true;
        }
        if requested_viewport.width > 0
            && requested_viewport.height > 0
            && requested_viewport != self.viewport_size
        {
            self.viewport_size = requested_viewport;
            self.gbuffer_needs_rebuild = true;
        }

        if self.swapchain_needs_rebuild || self.gbuffer_needs_rebuild {
            let mut recorder = self.queue.record();
            egui_frame.record_uploads(&mut recorder);
            self.frame_data[ring].last_signal_value = self.queue.submit(tgpu::SubmitInfo {
                records: &[recorder.finish()],
                ..Default::default()
            });
            return Ok(());
        }

        let frame = self.swapchain.acquire_next(None)?;
        if frame.out_of_date {
            self.swapchain_needs_rebuild = true;
            return Ok(());
        }

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let scene = SceneInfo {
            resolution: [
                self.gbuffer.extent.width as f32,
                self.gbuffer.extent.height as f32,
            ],
            anim_value: 0.5 * ((5.0 * elapsed).sin() + 1.0) + 0.5,
            num_data: POINTS.len() as u32,
            tex_id: self.image_id as u32,
            _pad0: [0; 3],
        };
        let mut recorder = self.queue.record();

        // Update the scene info from the GPU instead of writing mapped memory.
        recorder.buffer_transition(
            &self.frame_data[ring].scene_buffer,
            tgpu::BufferTransition {
                from: tgpu::BufferAccessTransition::uniform_read(),
                to: tgpu::BufferAccessTransition::TRANSFER_DST,
                ..Default::default()
            },
        );
        recorder.update_buffer_slice(&self.frame_data[ring].scene_buffer, &[scene]);
        recorder.buffer_transition(
            &self.frame_data[ring].scene_buffer,
            tgpu::BufferTransition {
                from: tgpu::BufferAccessTransition::TRANSFER_DST,
                to: tgpu::BufferAccessTransition::uniform_read(),
                ..Default::default()
            },
        );
        egui_frame.record_uploads(&mut recorder);

        let vertex_from = if self.vertex_was_read {
            vertex_input_access()
        } else {
            tgpu::BufferAccessTransition::NONE
        };
        recorder.buffer_transition(
            &self.vertex_buffer,
            tgpu::BufferTransition {
                from: vertex_from,
                to: tgpu::BufferAccessTransition::compute_storage_write(),
                ..Default::default()
            },
        );
        recorder.bind_compute(
            &self.compute_pipeline,
            self.bindless.descriptor_set(),
            PushData {
                vertex_buffer: self.vertex_handle,
                rotation_angle: 1.2 * self.delta_time,
                num_vertex: 3,
                _pad0: 0,
                scene_info: self.frame_data[ring].scene_handle,
                points: self.points_handle,
                texture_image: self.texture_handles[self.image_id],
                texture_sampler: self.sampler_handle,
                color: [0.0; 4],
            },
        );
        recorder.dispatch(1, 1, 1);
        recorder.buffer_transition(
            &self.vertex_buffer,
            tgpu::BufferTransition {
                from: tgpu::BufferAccessTransition::compute_storage_write(),
                to: vertex_input_access(),
                ..Default::default()
            },
        );

        recorder.image_transition(
            &self.gbuffer.color.image,
            tgpu::ImageLayoutTransition::COLOR,
        );
        recorder.image_transition(
            &self.gbuffer.depth.image,
            tgpu::ImageLayoutTransition::DEPTH_STENCIL,
        );

        let gbuffer_color =
            tgpu::RenderAttachment::color(&self.gbuffer.color.view).clear_color(self.clear_color);
        let gbuffer_depth =
            tgpu::RenderAttachment::depth(&self.gbuffer.depth.view).clear_depth(1.0);
        let gbuffer_extent = self.gbuffer.extent;
        recorder.begin_render(
            &tgpu::RenderInfo {
                colors: &[gbuffer_color],
                depth: Some(gbuffer_depth),
                area: tgpu::Rect::from_extent(gbuffer_extent),
                ..Default::default()
            },
            |render| {
                render.viewport(tgpu::Viewport::from_extent(gbuffer_extent));
                render.scissor(tgpu::Rect::from_extent(gbuffer_extent));
                render.bind_vertex_buffer(0, &self.vertex_buffer, 0);

                let base_push = PushData {
                    vertex_buffer: self.vertex_handle,
                    rotation_angle: 0.0,
                    num_vertex: 0,
                    _pad0: 0,
                    scene_info: self.frame_data[ring].scene_handle,
                    points: self.points_handle,
                    texture_image: self.texture_handles[self.image_id],
                    texture_sampler: self.sampler_handle,
                    color: [1.0, 0.0, 0.0, 1.0],
                };

                render.bind_render(
                    &self.graphics_pipeline_without_texture,
                    self.bindless.descriptor_set(),
                    base_push,
                );
                render.draw(0..3, 0..1);

                let textured_push = PushData {
                    color: [0.0, 1.0, 0.0, 1.0],
                    ..base_push
                };
                render.bind_render(
                    &self.graphics_pipeline_with_texture,
                    self.bindless.descriptor_set(),
                    textured_push,
                );
                render.draw(3..6, 0..1);
            },
        );
        recorder.image_transition(
            &self.gbuffer.color.image,
            tgpu::ImageLayoutTransition::FRAGMENT,
        );

        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageLayoutTransition::COLOR,
        );
        let swapchain_extent = self.swapchain.extent();
        let swapchain_color = tgpu::RenderAttachment::color(self.swapchain.view(frame))
            .clear_color([0.0, 0.0, 0.0, 1.0]);
        recorder.begin_render(
            &tgpu::RenderInfo {
                colors: &[swapchain_color],
                area: tgpu::Rect::from_extent(swapchain_extent),
                ..Default::default()
            },
            |render| {
                egui_frame.paint(render, swapchain_extent);
            },
        );
        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageLayoutTransition::PRESENT,
        );

        let available_semaphore = self.swapchain.available_semaphore(frame);
        let finished_semaphore = self.swapchain.finished_semaphore(frame);
        let signal_value = self.queue.submit(tgpu::SubmitInfo {
            records: &[recorder.finish()],
            wait_binary: &[(
                available_semaphore,
                tgpu::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )],
            signal_binary: &[finished_semaphore],
            ..Default::default()
        });
        self.frame_data[ring].last_signal_value = signal_value;

        if self.swapchain.present(&self.queue, frame)? {
            self.swapchain_needs_rebuild = true;
        }

        self.frame_ring_current =
            (self.frame_ring_current + 1) % self.swapchain.max_frames_in_flight();
        self.vertex_was_read = true;
        self.frame_count += 1;

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<(), Box<dyn Error>> {
        self.device.wait_idle();
        let size = self.window.inner_size();
        self.swapchain
            .set_preferred_extent(tgpu::Extent2D::new(size.width.max(1), size.height.max(1)));
        self.swapchain.set_preferred_present_mode(if self.v_sync {
            tgpu::PresentMode::Fifo
        } else {
            tgpu::PresentMode::Mailbox
        });
        self.swapchain.recreate()?;
        initialize_swapchain_images(&self.queue, &self.swapchain);
        self.egui
            .resize_frames(self.swapchain.max_frames_in_flight());
        ensure_frame_data(
            &self.device,
            &self.bindless,
            &mut self.frame_data,
            self.swapchain.max_frames_in_flight(),
        )?;
        for frame in &mut self.frame_data {
            frame.last_signal_value = 0;
        }
        self.frame_ring_current = 0;
        self.swapchain_needs_rebuild = false;
        Ok(())
    }

    fn recreate_gbuffer(&mut self) -> Result<(), Box<dyn Error>> {
        self.device.wait_idle();
        let gbuffer = create_gbuffer(&self.device, self.viewport_size)?;
        self.egui
            .update_user_texture(self.gbuffer_texture, &gbuffer.color, gbuffer.extent)?;
        self.gbuffer = gbuffer;
        self.gbuffer_needs_rebuild = false;
        Ok(())
    }
}

fn initialize_swapchain_images(queue: &tgpu::Queue, swapchain: &tgpu::Swapchain) {
    let mut recorder = queue.record();
    for image in swapchain.images() {
        recorder.image_transition(image, tgpu::ImageLayoutTransition::PRESENT);
    }
    let signal = queue.submit(tgpu::SubmitInfo {
        records: &[recorder.finish()],
        ..Default::default()
    });
    queue.timeline.wait(signal, None);
}

fn ensure_frame_data(
    device: &tgpu::Device,
    bindless: &tgpu::BindlessHeap,
    frame_data: &mut Vec<FrameData>,
    count: usize,
) -> Result<(), tgpu::GPUError> {
    while frame_data.len() < count {
        let scene_buffer = device.create_buffer(&tgpu::BufferDesc {
            label: Some(tgpu::Label::Name("minimal latest scene info")),
            size: std::mem::size_of::<SceneInfo>(),
            usage: tgpu::BufferUses::UNIFORM | tgpu::BufferUses::COPY_DST,
            memory: tgpu::MemoryPreset::GpuOnly,
            ..Default::default()
        })?;
        let scene_handle =
            bindless.add_uniform_buffer(&scene_buffer, std::mem::size_of::<SceneInfo>() as u64);
        frame_data.push(FrameData {
            scene_buffer,
            scene_handle,
            last_signal_value: 0,
        });
    }
    Ok(())
}

fn create_gbuffer(
    device: &tgpu::Device,
    extent: tgpu::Extent2D,
) -> Result<GBuffer, tgpu::GPUError> {
    let extent = tgpu::Extent2D::new(extent.width.max(1), extent.height.max(1));
    let color_format = tgpu::Format::R8G8B8A8_UNORM;
    let depth_format = tgpu::Format::D32_SFLOAT;
    let color = device.create_view_image(&tgpu::ViewImageDesc {
        image: tgpu::ImageDesc {
            label: Some(tgpu::Label::Name("minimal latest gbuffer color")),
            format: color_format,
            extent: extent.to_3d(1),
            usage: tgpu::ImageUses::COLOR_ATTACHMENT | tgpu::ImageUses::SAMPLED,
            ..Default::default()
        },
        sampler: Some(tgpu::SamplerCreateInfo::default()),
        aspect: Some(tgpu::ImageAspect::COLOR),
        ..Default::default()
    })?;
    let depth = device.create_view_image(&tgpu::ViewImageDesc {
        image: tgpu::ImageDesc {
            label: Some(tgpu::Label::Name("minimal latest gbuffer depth")),
            format: depth_format,
            extent: extent.to_3d(1),
            usage: tgpu::ImageUses::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        aspect: Some(tgpu::ImageAspect::DEPTH),
        ..Default::default()
    })?;

    Ok(GBuffer {
        color,
        depth,
        extent,
        color_format,
        depth_format,
    })
}

fn create_texture(
    device: &tgpu::Device,
    recorder: &mut tgpu::CommandRecorder,
    uploads: &mut Vec<tgpu::Buffer>,
    variant: u32,
) -> Result<tgpu::ViewImage, tgpu::GPUError> {
    let (size, pixels) = load_texture_pixels(variant);
    let upload = device.create_buffer(&tgpu::BufferDesc {
        label: Some(tgpu::Label::Name("minimal latest texture upload")),
        size: pixels.len(),
        usage: tgpu::BufferUses::COPY_SRC,
        memory: tgpu::MemoryPreset::Upload,
        host_access: tgpu::HostAccess::WriteSequential,
        ..Default::default()
    })?;
    upload.write(&pixels, 0);

    let texture = device.create_view_image(&tgpu::ViewImageDesc {
        image: tgpu::ImageDesc {
            label: Some(tgpu::Label::Name("minimal latest texture")),
            format: tgpu::Format::R8G8B8A8_UNORM,
            extent: size.to_3d(1),
            usage: tgpu::ImageUses::COPY_DST | tgpu::ImageUses::SAMPLED,
            ..Default::default()
        },
        aspect: Some(tgpu::ImageAspect::COLOR),
        ..Default::default()
    })?;

    recorder.image_transition(
        &texture.image,
        tgpu::ImageLayoutTransition::new(tgpu::ImageLayout::TransferDst),
    );
    let region = tgpu::BufferImageCopy::image_2d(tgpu::ImageAspect::COLOR, size);
    recorder.copy_buffer_to_image(&tgpu::CopyBufferToImageInfo {
        src: &upload,
        dst: &texture.image,
        dst_layout: tgpu::ImageLayout::TransferDst,
        regions: &[region],
    });
    recorder.image_transition(&texture.image, tgpu::ImageLayoutTransition::FRAGMENT);

    uploads.push(upload);
    Ok(texture)
}

fn load_texture_pixels(variant: u32) -> (tgpu::Extent2D, Vec<u8>) {
    let filename = if variant == 0 {
        "image1.jpg"
    } else {
        "image2.jpg"
    };
    let search_paths = [
        "examples/minimal_latest/resources",
        "resources",
        "../vk_minimal_latest/resources",
        "../../vk_minimal_latest/resources",
        "../../../vk_minimal_latest/resources",
    ];

    for root in search_paths {
        let path = Path::new(root).join(filename);
        if !path.exists() {
            continue;
        }

        match image::open(&path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let size = tgpu::Extent2D::new(rgba.width(), rgba.height());
                return (size, rgba.into_raw());
            }
            Err(err) => log::warn!("failed to load {}: {err}", path.display()),
        }
    }

    let size = tgpu::Extent2D::new(256, 256);
    (size, procedural_texture_pixels(size, variant))
}

fn procedural_texture_pixels(size: tgpu::Extent2D, variant: u32) -> Vec<u8> {
    let mut pixels = vec![0; (size.width * size.height * 4) as usize];
    for y in 0..size.height {
        for x in 0..size.width {
            let offset = ((y * size.width + x) * 4) as usize;
            let fx = x as f32 / (size.width - 1) as f32;
            let fy = y as f32 / (size.height - 1) as f32;
            let checker = ((x / 32) + (y / 32)) % 2 == 0;
            let (r, g, b) = if variant == 0 {
                let pulse = if checker { 1.0 } else { 0.35 };
                (255.0 * fx, 180.0 * pulse, 255.0 * (1.0 - fy))
            } else {
                let dx = fx - 0.5;
                let dy = fy - 0.5;
                let ring = ((dx * dx + dy * dy).sqrt() * 18.0).sin().abs();
                (255.0 * ring, 255.0 * fy, 80.0 + 120.0 * fx)
            };
            pixels[offset] = r as u8;
            pixels[offset + 1] = g as u8;
            pixels[offset + 2] = b as u8;
            pixels[offset + 3] = 255;
        }
    }
    pixels
}

fn create_graphics_pipeline(
    device: &tgpu::Device,
    bindless: &tgpu::BindlessHeap,
    gbuffer: &GBuffer,
    shader: &tgpu::Shader,
    fragment_entry: &str,
    label: &'static str,
) -> tgpu::RenderPipeline {
    let vertex_bindings = [tgpu::VertexBufferLayout {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: tgpu::VertexInputRate::Vertex,
    }];
    let vertex_attributes = [
        tgpu::VertexAttribute {
            location: 0,
            binding: 0,
            format: tgpu::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, position) as u32,
        },
        tgpu::VertexAttribute {
            location: 1,
            binding: 0,
            format: tgpu::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, color) as u32,
        },
        tgpu::VertexAttribute {
            location: 2,
            binding: 0,
            format: tgpu::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, tex_coord) as u32,
        },
    ];

    device.create_render_pipeline(&tgpu::RenderPipelineInfo {
        label: Some(tgpu::Label::Name(label)),
        vertex_shader: shader.entry("vertexMain"),
        fragment_shader: shader.entry(fragment_entry),
        color_formats: &[gbuffer.color_format],
        depth_format: Some(gbuffer.depth_format),
        depth_test: true,
        depth_write: true,
        depth_compare: tgpu::CompareOp::LessOrEqual,
        descriptor_layouts: &[bindless.layout()],
        push_constant_size: tgpu::push_constant_size::<PushData>(),
        blend: Some(&[tgpu::ColorBlend::Alpha]),
        vertex_input: Some(tgpu::VertexInput {
            buffers: &vertex_bindings,
            attributes: &vertex_attributes,
        }),
        topology: tgpu::PrimitiveTopology::TriangleList,
        polygon: tgpu::PolygonMode::Fill,
        cull: tgpu::CullMode::NONE,
        front_face: tgpu::FrontFace::CounterClockwise,
        ..Default::default()
    })
}

fn vertex_input_access() -> tgpu::BufferAccessTransition {
    tgpu::BufferAccessTransition::vertex_input_read()
}

fn build_ui(
    ctx: &egui::Context,
    texture: egui::TextureId,
    image_id: &mut usize,
    clear_color: &mut [f32; 4],
    v_sync: &mut bool,
    request_swapchain_rebuild: &mut bool,
    exit_requested: &mut bool,
    viewport_size: &mut tgpu::Extent2D,
    fps: f32,
) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.checkbox(v_sync, "vSync").changed() {
                    *request_swapchain_rebuild = true;
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    *exit_requested = true;
                    ui.close();
                }
            });
        });
    });

    egui::SidePanel::left("Settings")
        .default_width(230.0)
        .show(ctx, |ui| {
            ui.radio_value(image_id, 0, "Image 1");
            ui.radio_value(image_id, 1, "Image 2");
            ui.separator();

            let mut color = egui::Color32::from_rgb(
                (clear_color[0] * 255.0).round() as u8,
                (clear_color[1] * 255.0).round() as u8,
                (clear_color[2] * 255.0).round() as u8,
            );
            ui.label("Clear Color");
            if egui::color_picker::color_picker_color32(
                ui,
                &mut color,
                egui::color_picker::Alpha::Opaque,
            ) {
                clear_color[0] = color.r() as f32 / 255.0;
                clear_color[1] = color.g() as f32 / 255.0;
                clear_color[2] = color.b() as f32 / 255.0;
            }
        });

    egui::CentralPanel::default().show(ctx, |ui| {
        let available = ui.available_size().max(egui::vec2(1.0, 1.0));
        *viewport_size = tgpu::Extent2D::new(
            available.x.round().max(1.0) as u32,
            available.y.round().max(1.0) as u32,
        );

        let response = ui.add(egui::Image::new((texture, available)));
        let painter = ui.painter_at(response.rect);
        painter.text(
            response.rect.left_top(),
            egui::Align2::LEFT_TOP,
            format!("FPS: {fps:.1}"),
            egui::FontId::proportional(14.0),
            egui::Color32::WHITE,
        );
    });
}

#[derive(Default)]
pub struct App {
    render: Option<Render>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("tgpu minimal latest")
                    .with_inner_size(LogicalSize::new(
                        START_SIZE.width as f64,
                        START_SIZE.height as f64,
                    )),
            )
            .expect("Create window");
        window.request_redraw();
        self.render = Some(Render::new(window).expect("Create renderer"));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(render) = &mut self.render {
            let response = render.egui.on_window_event(&render.window, &event);
            if response.repaint {
                render.window.request_redraw();
            }
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(render) = &mut self.render
                    && size.width > 0
                    && size.height > 0
                {
                    render.swapchain_needs_rebuild = true;
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(render) = &mut self.render {
                    if let Err(err) = render.render_frame() {
                        log::error!("render frame failed: {err}");
                    }
                    if render.exit_requested {
                        event_loop.exit();
                        return;
                    }
                    render.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::builder()
        .filter_module("naga", log::LevelFilter::Warn)
        .init();

    let event_loop = EventLoop::new().expect("Create event loop");
    let mut app = App::default();
    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("run app failed: {err}");
    }
}
