use rand::Rng;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tgpu::ash::vk;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    output_image: tgpu::StorageImageHandle,
    particles: tgpu::RwBufferHandle,
    present_texture: tgpu::SampledImageHandle,
    present_sampler: tgpu::SamplerHandle,
    window: [u32; 2],
    mouse: [f32; 2],
    dt: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    color: [f32; 4],
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

const PARTICLE_COUNT: usize = 10000;
const SHADER: &str = include_str!("./shader.slang");

#[allow(unused)]
pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    graphics_queue: tgpu::Queue,
    async_compute_queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    particles: Vec<Particle>,
    particle_buffer: tgpu::Buffer,
    present_image: tgpu::ViewImage,
    bindless: tgpu::BindlessHeap,
    present_pipeline: tgpu::RenderPipeline,
    compute_pipeline: tgpu::ComputePipeline,
    clear_pipeline: tgpu::ComputePipeline,
    graph_cache: tgpu::RenderGraphCache,
    pc: PushConstants,
    frame_count: usize,
}

impl Render {
    pub fn new(window: Window) -> Result<Render, tgpu::GPUError> {
        let display = window.display_handle().unwrap().as_raw();

        let instance = tgpu::Instance::new_with_display(
            &tgpu::InstanceCreateInfo {
                app_name: "Particles GPU Graph",
                engine_name: "Example Engine",
            },
            display,
        )?;

        let adapters = instance.adapters(&[])?.collect::<Vec<_>>();
        let adapter = adapters[0].clone();

        let (device, mut queues) = instance.request_device(
            &tgpu::DeviceCreateInfo::default(),
            adapter,
            &[
                tgpu::QueueRequest {
                    required_flags: tgpu::QueueFlags::GRAPHICS,
                    exclude_flags: tgpu::QueueFlags::empty(),
                    strict: false,
                    allow_fallback_share: true,
                },
                tgpu::QueueRequest {
                    required_flags: tgpu::QueueFlags::COMPUTE,
                    exclude_flags: tgpu::QueueFlags::GRAPHICS,
                    strict: false,
                    allow_fallback_share: true,
                },
            ],
        )?;

        let graphics_queue = queues.next().unwrap();
        let async_compute_queue = queues.next().unwrap();

        let size = window.inner_size();

        let swapchain = device.create_swapchain(&tgpu::SwapchainCreateInfo {
            display: window.display_handle().unwrap().as_raw(),
            window: window.window_handle().unwrap().as_raw(),
            preferred_extent: vk::Extent2D {
                width: size.width,
                height: size.height,
            },
            preferred_image_count: 3,
            preferred_present_mode: tgpu::PresentModeKHR::MAILBOX,
            format_selector: Box::new(|formats| {
                formats
                    .iter()
                    .find(|f| {
                        f.format == tgpu::Format::B8G8R8A8_SRGB
                            && f.color_space == tgpu::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .copied()
                    .unwrap_or(formats[0])
            }),
        })?;

        let mut particles = vec![Particle::default(); PARTICLE_COUNT];
        let rng = &mut rand::rng();

        for particle in &mut particles {
            particle.position = [
                rng.random_range(0.0..size.width as f32),
                rng.random_range(0.0..size.height as f32),
            ];
            particle.velocity = [rng.random_range(-30.0..30.0), rng.random_range(-30.0..30.0)];
            particle.color = [
                rng.random_range(0.5..1.0),
                rng.random_range(0.5..1.0),
                rng.random_range(0.5..1.0),
                1.0,
            ];
        }

        let particle_buffer = device.create_buffer(&tgpu::BufferDesc {
            label: Some(tgpu::Label::Name("particle buffer")),
            size: std::mem::size_of::<Particle>() * PARTICLE_COUNT,
            usage: tgpu::BufferUses::STORAGE | tgpu::BufferUses::COPY_DST,
            memory: tgpu::MemoryPreset::Dynamic,
            host_access: tgpu::HostAccess::ReadWriteRandom,
            ..Default::default()
        })?;
        particle_buffer.write_slice(&particles);

        let present_image = create_present_image(&device, &swapchain)?;

        let bindless = device.create_bindless_heap(&tgpu::BindlessInfo {
            max_rw_buffers: 1,
            max_sampled_images: 1,
            max_storage_images: 1,
            max_samplers: 1,
            ..Default::default()
        });

        let particle_buffer_handle = bindless.add_rw_buffer(&particle_buffer);
        let present_storage_image_handle =
            bindless.add_storage_image(&present_image.view, vk::ImageLayout::GENERAL);
        let present_texture_handle = bindless.add_sampled_image(
            &present_image.view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let present_sampler_handle = bindless.add_sampler(present_image.sampler.as_ref().unwrap());

        let shader = device
            .create_shader(None, tgpu::ShaderSource::Slang(SHADER.as_bytes()))
            .expect("Particle Shader");

        let compute_pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
            label: Some(tgpu::Label::Name("Compute Pipeline")),
            shader: shader.entry("computeMain"),
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            descriptor_layouts: &[bindless.layout()],
            cache: None,
        });

        let clear_pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
            label: Some(tgpu::Label::Name("Clear Pipeline")),
            shader: shader.entry("computeClear"),
            descriptor_layouts: &[bindless.layout()],
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            cache: None,
        });

        let present_pipeline = device.create_render_pipeline(&tgpu::RenderPipelineInfo {
            label: Some(tgpu::Label::Name("Present Pipeline")),
            vertex_shader: shader.entry("vertexMain"),
            fragment_shader: shader.entry("fragmentMain"),
            color_formats: &[swapchain.format()],
            depth_format: None,
            descriptor_layouts: &[bindless.layout()],
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            blend_states: None,
            vertex_input_state: None,
            topology: tgpu::PrimitiveTopology::TRIANGLE_LIST,
            polygon: tgpu::PolygonMode::FILL,
            cull: tgpu::CullModeFlags::BACK,
            front_face: tgpu::FrontFace::COUNTER_CLOCKWISE,
        });

        let pc = PushConstants {
            output_image: present_storage_image_handle,
            particles: particle_buffer_handle,
            present_texture: present_texture_handle,
            present_sampler: present_sampler_handle,
            window: [size.width, size.height],
            mouse: [0.0; 2],
            dt: 0.0,
            _pad: 0.0,
        };

        Ok(Self {
            window,
            instance,
            device,
            graphics_queue,
            async_compute_queue,
            swapchain,
            particles,
            particle_buffer,
            present_image,
            bindless,
            present_pipeline,
            compute_pipeline,
            clear_pipeline,
            graph_cache: tgpu::RenderGraphCache::new(),
            pc,
            frame_count: 0,
        })
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.swapchain.acquire_next(None)?;
        log::trace!("Start Frame {:?}", frame.index);
        if frame.suboptimal {
            self.handle_resize();
            return Ok(());
        }

        let extent = self.swapchain.extent();
        let particle_groups = PARTICLE_COUNT.div_ceil(256) as u32;
        let width_groups = extent.width.div_ceil(16);
        let height_groups = extent.height.div_ceil(16);
        let frame_number = self.frame_count as f32;
        let color = (frame_number / 120.0).sin().abs();
        let invert_color = 1.0 - color;

        let mut graph = tgpu::RenderGraph::new(tgpu::RenderGraphInfo {
            device: &self.device,
            graphics: &self.graphics_queue,
            async_compute: Some(&self.async_compute_queue),
            copy: None,
            bindless: Some(&self.bindless),
        });

        let particle_buffer = graph.import_buffer(
            "particles",
            &self.particle_buffer,
            tgpu::ImportedBufferDesc {
                initial: tgpu::BufferAccessTransition::NONE,
                initialized: true,
            },
        );
        let present_image = graph.import_image(
            "present_image",
            &self.present_image,
            tgpu::ImportedImageDesc {
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                aspect: vk::ImageAspectFlags::COLOR,
                initial: tgpu::ImageLayoutTransition::UNDEFINED,
                initialized: false,
            },
        );
        let backbuffer = graph.import_swapchain_image("backbuffer", &mut self.swapchain, frame);

        let particle_pc = self.pc;
        let particle_pipeline = &self.compute_pipeline;
        graph.add_compute_pass(
            "simulate_particles",
            |pass| {
                pass.queue(tgpu::PassQueue::AsyncCompute);
                pass.read_write_buffer(
                    particle_buffer,
                    tgpu::BufferAccess::StorageComputeReadWrite,
                );
                pass.write_image(present_image, tgpu::ImageAccess::StorageComputeWrite);
            },
            move |ctx| {
                let bindless = ctx.bindless().expect("bindless heap");
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(particle_pipeline);
                cmd.bind_compute_descriptor_set(
                    bindless.descriptor_set(),
                    particle_pipeline,
                    0,
                    &[],
                );
                cmd.push_compute_constants(particle_pipeline, particle_pc);
                cmd.dispatch(particle_groups, 1, 1);
            },
        );

        let clear_pc = self.pc;
        let clear_pipeline = &self.clear_pipeline;
        graph.add_compute_pass(
            "fade_present_image",
            |pass| {
                pass.queue(tgpu::PassQueue::AsyncCompute);
                pass.read_write_image(present_image, tgpu::ImageAccess::StorageComputeReadWrite);
            },
            move |ctx| {
                let bindless = ctx.bindless().expect("bindless heap");
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(clear_pipeline);
                cmd.bind_compute_descriptor_set(bindless.descriptor_set(), clear_pipeline, 0, &[]);
                cmd.push_compute_constants(clear_pipeline, clear_pc);
                cmd.dispatch(width_groups, height_groups, 1);
            },
        );

        let present_pc = self.pc;
        let present_pipeline = &self.present_pipeline;
        graph.add_render_pass(
            "present",
            |pass| {
                pass.queue(tgpu::PassQueue::Graphics);
                pass.read_image(present_image, tgpu::ImageAccess::SampledFragment);
                pass.write_color(
                    backbuffer,
                    tgpu::ColorAttachmentDesc::clear([0.0, color, invert_color, 1.0]),
                );
            },
            move |ctx| {
                {
                    let bindless = ctx.bindless().expect("bindless heap");
                    let cmd = ctx.cmd();
                    cmd.bind_render_pipeline(present_pipeline);
                    cmd.bind_render_descriptor_set(
                        bindless.descriptor_set(),
                        present_pipeline,
                        0,
                        &[],
                    );
                    cmd.push_render_constants(present_pipeline, present_pc);
                }

                ctx.begin_render(|recorder| {
                    recorder.viewport(vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: extent.width as f32,
                        height: extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    });
                    recorder.scissor(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    });
                    recorder.draw(0..6, 0..1);
                });
            },
        );

        let outcome = graph.execute_cached(&mut self.graph_cache)?;
        if outcome.needs_swapchain_recreation {
            self.handle_resize();
            return Ok(());
        }

        self.frame_count += 1;
        log::trace!(
            "Finish Frame {:?} (cache_hit={})",
            frame.index,
            outcome.cache_hit
        );

        Ok(())
    }

    fn handle_resize(&mut self) {
        log::debug!("recreate swapchain");
        let size = self.window.inner_size();
        self.swapchain.set_preferred_extent(vk::Extent2D {
            width: size.width,
            height: size.height,
        });
        let _ = self.swapchain.recreate();
        self.pc.window = [size.width, size.height];

        let present_image =
            create_present_image(&self.device, &self.swapchain).expect("Create Present Image");

        self.bindless.update_storage_image(
            self.pc.output_image,
            &present_image.view,
            vk::ImageLayout::GENERAL,
        );
        self.bindless.update_sampled_image(
            self.pc.present_texture,
            &present_image.view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        self.bindless.update_sampler(
            self.pc.present_sampler,
            present_image.sampler.as_ref().unwrap(),
        );

        self.present_image = present_image;
        self.graph_cache.clear();
    }
}

fn create_present_image(
    device: &tgpu::Device,
    swapchain: &tgpu::Swapchain,
) -> Result<tgpu::ViewImage, tgpu::GPUError> {
    device.create_view_image(&tgpu::ViewImageDesc {
        image: tgpu::ImageDesc {
            label: Some(tgpu::Label::Name("present image")),
            format: swapchain.format(),
            extent: vk::Extent3D {
                width: swapchain.extent().width,
                height: swapchain.extent().height,
                depth: 1,
            },
            usage: tgpu::ImageUses::COPY_SRC
                | tgpu::ImageUses::COPY_DST
                | tgpu::ImageUses::STORAGE
                | tgpu::ImageUses::COLOR_ATTACHMENT
                | tgpu::ImageUses::SAMPLED,
            ..Default::default()
        },
        sampler: Some(tgpu::SamplerCreateInfo {
            label: Some(tgpu::Label::Name("Present Sampler")),
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub struct App {
    render: Option<Render>,
    last_frame_instant: Option<std::time::Instant>,
    smoothed_dt: f32,
    exit_after_frames: Option<usize>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            render: None,
            last_frame_instant: None,
            smoothed_dt: 0.0,
            exit_after_frames: std::env::var("TGPU_EXIT_AFTER_FRAMES")
                .ok()
                .and_then(|value| value.parse().ok()),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .expect("Acquire Window");

        window.request_redraw();
        let render = Render::new(window).expect("Create Render");
        self.render = Some(render);
        self.last_frame_instant = Some(std::time::Instant::now());
        self.smoothed_dt = 1.0 / 60.0;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
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
            WindowEvent::RedrawRequested => {
                if let Some(render) = &mut self.render {
                    let now = std::time::Instant::now();
                    let measured_dt = if let Some(last) = self.last_frame_instant {
                        (now - last).as_secs_f32()
                    } else {
                        1.0 / 60.0
                    };
                    self.last_frame_instant = Some(now);

                    let clamped_dt = measured_dt.clamp(1.0 / 240.0, 1.0 / 15.0);
                    let alpha = 0.1_f32;
                    let dt = if self.smoothed_dt == 0.0 {
                        clamped_dt
                    } else {
                        self.smoothed_dt * (1.0 - alpha) + clamped_dt * alpha
                    };
                    self.smoothed_dt = dt;

                    render.pc.dt = dt * 6.0;
                    let _ = render.render_frame();

                    if self
                        .exit_after_frames
                        .is_some_and(|limit| render.frame_count >= limit)
                    {
                        event_loop.exit();
                        return;
                    }

                    render.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(render) = &mut self.render {
                    let height = render.window.inner_size().height as f32;
                    render.pc.mouse = [position.x as f32, height - position.y as f32];
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

    let event_loop = EventLoop::new().expect("acquire event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}
