use std::sync::Arc;

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
    window: [u32; 2],
    mouse: [f32; 2],
    dt: f32,
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

#[allow(unused)]
pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    particles: Vec<Particle>,
    particle_buffer: tgpu::Buffer,
    present_image: tgpu::ViewImage,
    layout: tgpu::DescriptorSetLayout,
    pool: Arc<tgpu::DescriptorPool>,
    descriptor_set: tgpu::DescriptorSet,

    present_pipeline: tgpu::RenderPipeline,
    compute_pipeline: tgpu::ComputePipeline,
    clear_pipeline: tgpu::ComputePipeline,
    pc: PushConstants,

    frame_count: usize,
}

impl Render {
    pub fn new(window: Window) -> Result<Render, tgpu::GPUError> {
        let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
            app_name: "Particles",
            engine_name: "Example Engine",
        })?;

        let adapters = instance.adapters(&[])?.collect::<Vec<_>>();
        let adapter = adapters[0].clone();

        let (device, mut queues) = instance.request_device(
            &tgpu::DeviceCreateInfo {},
            adapter,
            &[tgpu::QueueRequest {
                required_flags: tgpu::QueueFlags::GRAPHICS,
                exclude_flags: tgpu::QueueFlags::empty(),
                strict: false,
                allow_fallback_share: true,
            }],
        )?;

        let queue = queues.next().unwrap();

        let swapchain = device.create_swapchain(&tgpu::SwapchainCreateInfo {
            display: window.display_handle().unwrap().as_raw(),
            window: window.window_handle().unwrap().as_raw(),
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

        let size = window.inner_size();

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

        let particle_buffer = device.create_buffer(&tgpu::BufferInfo {
            label: Some(tgpu::Label::Name("particle buffer")),
            size: std::mem::size_of::<Particle>() * PARTICLE_COUNT,
            usage: tgpu::BufferUsage::STORAGE
                | tgpu::BufferUsage::COPY_DST
                | tgpu::BufferUsage::MAP_WRITE
                | tgpu::BufferUsage::DEVICE
                | tgpu::BufferUsage::COHERENT
                | tgpu::BufferUsage::HOST_VISIBLE
                | tgpu::BufferUsage::MAP_READ
                | tgpu::BufferUsage::MAP_WRITE,
        })?;

        particle_buffer.write(bytemuck::cast_slice(&particles), 0);

        let layout = device.create_descriptor_set_layout(&tgpu::DescriptorSetLayoutInfo {
            label: Some(tgpu::Label::Name("Descriptor Set")),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            bindings: &[
                tgpu::DescriptorBinding::unique(
                    0,
                    tgpu::DescriptorType::StorageImage,
                    tgpu::ShaderStageFlags::COMPUTE,
                ),
                tgpu::DescriptorBinding::unique(
                    1,
                    tgpu::DescriptorType::StorageBuffer,
                    tgpu::ShaderStageFlags::COMPUTE,
                ),
                tgpu::DescriptorBinding::unique(
                    2,
                    tgpu::DescriptorType::SampledImage,
                    tgpu::ShaderStageFlags::FRAGMENT,
                ),
                tgpu::DescriptorBinding::unique(
                    3,
                    tgpu::DescriptorType::Sampler,
                    tgpu::ShaderStageFlags::FRAGMENT,
                ),
            ],
        });

        // TODO: I don't think should be an Arc<>, we should hide the smart pointer !
        let pool = device.create_descriptor_pool(&tgpu::DescriptorPoolInfo {
            label: Some(tgpu::Label::Name("Descriptor Pool")),
            max_sets: 1,
            layouts: &[&layout],
            flags: vk::DescriptorPoolCreateFlags::empty(),
        });

        let descriptor_set = device.create_descriptor_set(pool.clone(), &layout);

        // TODO: this api is a mess right now
        let present_image = device.create_sampled_image(&tgpu::ViewImageCreateInfo {
            image: &tgpu::ImageCreateInfo {
                format: swapchain.format(),
                ty: vk::ImageType::TYPE_2D,
                volume: vk::Extent3D {
                    width: swapchain.extent().width,
                    height: swapchain.extent().height,
                    depth: 1,
                },
                mips: 1,
                layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: tgpu::ImageUsage::COPY_SRC
                    | tgpu::ImageUsage::COPY_DST
                    | tgpu::ImageUsage::STORAGE
                    | tgpu::ImageUsage::COLOR
                    | tgpu::ImageUsage::DEVICE
                    | tgpu::ImageUsage::SAMPLED, // TODO: sampled should be automated
                ..Default::default()
            },
            sampler: Some(&tgpu::SamplerCreateInfo {
                label: Some(tgpu::Label::Name("Present Sampler")),
                ..Default::default()
            }),
            view: tgpu::ImageViewOptions {
                ty: vk::ImageViewType::TYPE_2D, // TODO: this should probably be automatic,
                // investigate this
                format: Some(swapchain.format()),
                mips: 0..1,
                layers: 0..1,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        });

        // TODO: maybe use impl trait here, so writing &image would be enough
        descriptor_set.write(&[
            tgpu::DescriptorWrite::StorageImage {
                binding: 0,
                image_view: &present_image.view,
                image_layout: vk::ImageLayout::GENERAL,
                array_element: None,
            },
            tgpu::DescriptorWrite::StorageBuffer {
                binding: 1,
                buffer: &particle_buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
                array_element: None,
            },
            tgpu::DescriptorWrite::SampledImage {
                binding: 2,
                image_view: &present_image.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                array_element: None,
            },
            tgpu::DescriptorWrite::Sampler {
                binding: 3,
                sampler: present_image.sampler.as_ref().unwrap(),
                array_element: None,
            },
        ]);

        const SHADER: &str = include_str!("./shader.slang");

        let shader = device
            .create_shader(None, tgpu::ShaderSource::Slang(SHADER.as_bytes()))
            .expect("Compute Shader");

        let compute_pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
            label: Some(tgpu::Label::Name("Compute Pipeline")),
            shader: shader.entry("computeMain"),
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            descriptor_layouts: &[&layout],
            cache: None,
        });

        let clear_pipeline = device.create_compute_pipeline(&tgpu::ComputePipelineInfo {
            label: Some(tgpu::Label::Name("Clear Pipeline")),
            shader: shader.entry("computeClear"),
            descriptor_layouts: &[&layout],
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            cache: None,
        });

        let present_pipeline = device.create_render_pipeline(&tgpu::RenderPipelineInfo {
            label: Some(tgpu::Label::Name("Present Pipeline")),
            vertex_shader: shader.entry("vertexMain"),
            fragment_shader: shader.entry("fragmentMain"),
            color_formats: &[swapchain.format()],
            depth_format: None,
            descriptor_layouts: &[&layout],
            push_constant_size: None,
            blend_states: None,
            vertex_input_state: None,
            topology: tgpu::PrimitiveTopology::TRIANGLE_LIST,
            polygon: tgpu::PolygonMode::FILL,
            cull: tgpu::CullModeFlags::BACK,
            front_face: tgpu::FrontFace::COUNTER_CLOCKWISE,
        });

        let pc = PushConstants {
            window: [size.width, size.height],
            mouse: [0.; 2],
            dt: 0.,
        };

        let new = Self {
            window,
            instance,
            device,
            queue,
            swapchain,
            particles,
            particle_buffer,
            present_image,
            layout,
            pool,
            descriptor_set,
            present_pipeline,
            compute_pipeline,
            clear_pipeline,
            pc,
            frame_count: 0,
        };

        Ok(new)
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.swapchain.acquire_next(None)?;
        log::trace!("Start Frame {:?}", frame.index);
        if frame.suboptimal {
            self.handle_resize();
            return Ok(());
        }
        let mut recorder = self.queue.record();

        recorder.image_transition(
            &self.present_image.image,
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::UNDEFINED,
                to: tgpu::ImageLayoutTransition::COMPUTE,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        );

        let particle_groups = (PARTICLE_COUNT + 255) / 256;
        recorder.bind_compute_pipeline(&self.compute_pipeline);
        recorder.bind_compute_descriptor_set(&self.descriptor_set, &self.compute_pipeline, 0, &[]);
        recorder.push_compute_constants(&self.compute_pipeline, self.pc);
        recorder.dispatch(particle_groups as u32, 1, 1);

        let width = (self.swapchain.extent().width + 15) / 16;
        let height = (self.swapchain.extent().height + 15) / 16;
        recorder.bind_compute_pipeline(&self.clear_pipeline);
        recorder.bind_compute_descriptor_set(&self.descriptor_set, &self.clear_pipeline, 0, &[]);
        recorder.push_compute_constants(&self.clear_pipeline, self.pc);
        recorder.dispatch(width, height, 1);

        recorder.image_transition(
            &self.present_image.image,
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::COMPUTE,
                to: tgpu::ImageLayoutTransition::FRAGMENT,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        );

        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::UNDEFINED,
                to: tgpu::ImageLayoutTransition::COLOR,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        );

        let frame_number = self.frame_count as f32;
        let color = (frame_number / 120.0).sin().abs();
        let invert_color = 1.0 - color;

        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.view(frame).inner.handle)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, color, invert_color, 1.0],
                },
            });

        recorder.bind_render_pipeline(&self.present_pipeline);
        recorder.bind_render_descriptor_set(&self.descriptor_set, &self.present_pipeline, 0, &[]);

        recorder.begin_render(
            &tgpu::RenderInfo {
                colors: &[attachment],
                area: vk::Rect2D {
                    extent: self.swapchain.extent(),
                    ..Default::default()
                },
                ..Default::default()
            },
            |recorder| {
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.swapchain.extent().width as f32,
                    height: self.swapchain.extent().height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };

                let scissor = vk::Rect2D {
                    extent: self.swapchain.extent(),
                    ..Default::default()
                };

                recorder.viewport(viewport);
                recorder.scissor(scissor);

                recorder.draw(0..6, 0..1);
            },
        );

        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::COLOR,
                to: tgpu::ImageLayoutTransition::PRESENT,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        );

        let available_semaphore = self.swapchain.inner.available_semaphore(frame);
        let finished_semaphore = self.swapchain.inner.finished_semaphore(frame);

        self.queue.submit(tgpu::SubmitInfo {
            records: &[recorder.finish()],
            wait_binary: &[(
                available_semaphore,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )],
            signal_binary: &[finished_semaphore],
            fence: Some(self.swapchain.inner.fence(frame)),
            ..Default::default()
        });

        match self.swapchain.present(&self.queue, frame) {
            Ok(true) | Err(_) => {
                self.handle_resize();
                return Ok(());
            }
            _ => {}
        }

        self.frame_count += 1;

        log::trace!("Finish Frame {:?}", frame.index);

        Ok(())
    }

    fn handle_resize(&mut self) {
        log::debug!("recreate swapchain");
        let _ = self.swapchain.recreate();
        let size = self.window.inner_size();
        self.pc.window = [size.width, size.height];

        let present_image = self
            .device
            .create_sampled_image(&tgpu::ViewImageCreateInfo {
                image: &tgpu::ImageCreateInfo {
                    format: self.swapchain.format(),
                    ty: vk::ImageType::TYPE_2D,
                    volume: vk::Extent3D {
                        width: self.swapchain.extent().width,
                        height: self.swapchain.extent().height,
                        depth: 1,
                    },
                    mips: 1,
                    layers: 1,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: tgpu::ImageUsage::COPY_SRC
                        | tgpu::ImageUsage::COPY_DST
                        | tgpu::ImageUsage::STORAGE
                        | tgpu::ImageUsage::COLOR
                        | tgpu::ImageUsage::DEVICE
                        | tgpu::ImageUsage::SAMPLED, // TODO: sampled should be automated
                    ..Default::default()
                },
                sampler: Some(&tgpu::SamplerCreateInfo {
                    label: Some(tgpu::Label::Name("Present Sampler")),
                    ..Default::default()
                }),
                view: tgpu::ImageViewOptions {
                    ty: vk::ImageViewType::TYPE_2D, // TODO: this should probably be automatic,
                    // investigate this
                    format: Some(self.swapchain.format()),
                    mips: 0..1,
                    layers: 0..1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    ..Default::default()
                },
            });

        self.descriptor_set.write(&[
            tgpu::DescriptorWrite::StorageImage {
                binding: 0,
                image_view: &present_image.view,
                image_layout: vk::ImageLayout::GENERAL,
                array_element: None,
            },
            tgpu::DescriptorWrite::SampledImage {
                binding: 2,
                image_view: &present_image.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                array_element: None,
            },
            tgpu::DescriptorWrite::Sampler {
                binding: 3,
                sampler: present_image.sampler.as_ref().unwrap(),
                array_element: None,
            },
        ]);

        self.present_image = present_image;
    }
}

#[derive(Default)]
pub struct App {
    render: Option<Render>,
    last_frame_instant: Option<std::time::Instant>,
    smoothed_dt: f32,
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

                    let clamped_dt = measured_dt.clamp(1.0 / 240.0, 1.0 / 15.0); // between ~4.17ms and ~66.7ms

                    let alpha = 0.1_f32; // smaller = smoother, larger = more responsive
                    let dt = if self.smoothed_dt == 0.0 {
                        clamped_dt
                    } else {
                        self.smoothed_dt * (1.0 - alpha) + clamped_dt * alpha
                    };
                    self.smoothed_dt = dt;

                    render.pc.dt = dt * 6.0;
                    let _ = render.render_frame();
                    render.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(render) = &mut self.render {
                    let height = render.window.inner_size().height as f32;
                    render.pc.mouse = [position.x as f32, height - position.y as f32];
                }
            }
            _ => (),
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
