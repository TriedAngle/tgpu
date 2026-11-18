use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const TRIANGLE_SHADER_SLANG: &str = r#"
struct Vertex {
    float2 position;
    float2 _pad0;
    float4 color;
};

struct PushConstants {
    uint vertex_buffer;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Vertex> vertex_buffers[];

[[vk::push_constant]]
cbuffer PC {
    PushConstants pc;
};

struct VSOutput {
    float4 position : SV_Position;
    float4 color;
};

[shader("vertex")]
VSOutput vmain(uint vertexId : SV_VertexID) {
    Vertex vertex = vertex_buffers[pc.vertex_buffer][vertexId];
    VSOutput o;
    o.position = float4(vertex.position, 0.0, 1.0);
    o.color = vertex.color;
    return o;
}

[shader("fragment")]
float4 fmain(VSOutput input) : SV_Target0 {
    return input.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    _pad0: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    vertex_buffer: tgpu::ReadBufferHandle,
}

#[allow(unused)]
pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    vertex_buffer: tgpu::Buffer,
    bindless: tgpu::BindlessHeap,
    pipeline: tgpu::RenderPipeline,
    pc: PushConstants,
    frame_count: usize,
    frame_values: Vec<u64>,
}

impl Render {
    pub fn new(window: Window) -> Result<Render, tgpu::GPUError> {
        let display = window.display_handle().unwrap().as_raw();

        let instance = tgpu::Instance::new_with_display(
            &tgpu::InstanceCreateInfo {
                app_name: "Simple Buffer Triangle",
                engine_name: "Example Engine",
            },
            display,
        )?;

        let adapters = instance.adapters(&[])?.collect::<Vec<_>>();
        let adapter = adapters[0].clone();

        let (device, mut queues) = instance.request_device(
            &tgpu::DeviceCreateInfo::default(),
            adapter,
            &[tgpu::QueueRequest {
                required_flags: tgpu::QueueFlags::GRAPHICS,
                exclude_flags: tgpu::QueueFlags::empty(),
                strict: false,
                allow_fallback_share: true,
            }],
        )?;

        let queue = queues.next().unwrap();

        let vertices = [
            Vertex {
                position: [0.5, 0.5],
                _pad0: [0.0, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5],
                _pad0: [0.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.0, -0.5],
                _pad0: [0.0, 0.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];

        let vertex_buffer = device.create_buffer(&tgpu::BufferDesc {
            label: Some(tgpu::Label::Name("triangle vertices")),
            size: std::mem::size_of_val(&vertices),
            usage: tgpu::BufferUses::STORAGE,
            memory: tgpu::MemoryPreset::Dynamic,
            host_access: tgpu::HostAccess::WriteSequential,
            ..Default::default()
        })?;
        vertex_buffer.write_slice(&vertices);

        let bindless = device.create_bindless_heap(&tgpu::BindlessInfo {
            max_read_buffers: 1,
            ..Default::default()
        });
        let vertex_buffer_handle = bindless.add_read_buffer(&vertex_buffer);

        let size = window.inner_size();

        let swapchain = device.create_swapchain(&tgpu::SwapchainCreateInfo {
            display: window.display_handle().unwrap().as_raw(),
            window: window.window_handle().unwrap().as_raw(),
            preferred_extent: tgpu::Extent2D::new(size.width, size.height),
            preferred_image_count: 3,
            preferred_present_mode: tgpu::PresentMode::Mailbox,
            format_selector: Box::new(|formats| {
                formats
                    .iter()
                    .find(|f| {
                        f.format == tgpu::Format::B8G8R8A8_SRGB
                            && f.color_space == tgpu::ColorSpace::SRGB_NONLINEAR
                    })
                    .copied()
                    .unwrap_or(formats[0])
            }),
        })?;

        let shader = device
            .create_shader(
                Some(tgpu::Label::Name("shader")),
                tgpu::ShaderSource::Slang(TRIANGLE_SHADER_SLANG.as_bytes()),
            )
            .expect("Shader");

        let pipeline = device.create_render_pipeline(&tgpu::RenderPipelineInfo {
            label: Some(tgpu::Label::Name("Present Pipeline")),
            vertex_shader: shader.entry("vmain"),
            fragment_shader: shader.entry("fmain"),
            descriptor_layouts: &[bindless.layout()],
            push_constant_size: Some(std::mem::size_of::<PushConstants>() as u32),
            cull: tgpu::CullMode::BACK,
            topology: tgpu::PrimitiveTopology::TriangleList,
            polygon: tgpu::PolygonMode::Fill,
            front_face: tgpu::FrontFace::Clockwise,
            color_formats: &[swapchain.format()],
            ..Default::default()
        });
        let frame_values = vec![0; swapchain.max_frames_in_flight()];

        let new = Self {
            window,
            instance,
            device,
            queue,
            swapchain,
            vertex_buffer,
            bindless,
            pipeline,
            pc: PushConstants {
                vertex_buffer: vertex_buffer_handle,
            },
            frame_count: 0,
            frame_values,
        };

        Ok(new)
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let ring = self.swapchain.flight_index();
        self.queue.timeline.wait(self.frame_values[ring], None);

        let frame = self.swapchain.acquire_next(None)?;
        log::trace!("Start Frame {:?}", frame.index);
        if frame.out_of_date {
            log::debug!("recreate swapchain");
            let size = self.window.inner_size();
            self.swapchain
                .set_preferred_extent(tgpu::Extent2D::new(size.width, size.height));
            let _ = self.swapchain.recreate();
            self.frame_values = vec![0; self.swapchain.max_frames_in_flight()];
            return Ok(());
        }
        let mut recorder = self.queue.record();

        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::UNDEFINED,
                to: tgpu::ImageLayoutTransition::COLOR,
                aspect: tgpu::ImageAspect::COLOR,
                ..Default::default()
            },
        );

        let frame_number = self.frame_count as f32;

        let color = (frame_number / 120.0).sin().abs();
        let invert_color = 1.0 - color;

        let attachment = tgpu::RenderAttachment::color(self.swapchain.view(frame)).clear_color([
            0.0,
            invert_color,
            color,
            0.5,
        ]);

        recorder.bind_render_pipeline(&self.pipeline);
        recorder.bind_render_descriptor_set(self.bindless.descriptor_set(), &self.pipeline, 0, &[]);
        recorder.push_render_constants(&self.pipeline, self.pc);

        recorder.begin_render(
            &tgpu::RenderInfo {
                colors: &[attachment],
                area: tgpu::Rect::from_extent(self.swapchain.extent()),
                ..Default::default()
            },
            |recorder| {
                let viewport = tgpu::Viewport::from_extent(self.swapchain.extent());
                let scissor = tgpu::Rect::from_extent(self.swapchain.extent());

                recorder.viewport(viewport);
                recorder.scissor(scissor);

                recorder.draw(0..3, 0..1);
            },
        );

        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::COLOR,
                to: tgpu::ImageLayoutTransition::PRESENT,
                aspect: tgpu::ImageAspect::COLOR,
                ..Default::default()
            },
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
        self.frame_values[ring] = signal_value;

        match self.swapchain.present(&self.queue, frame) {
            Ok(true) | Err(_) => {
                log::debug!("recreate swapchain");
                let size = self.window.inner_size();
                self.swapchain
                    .set_preferred_extent(tgpu::Extent2D::new(size.width, size.height));
                let _ = self.swapchain.recreate();
                self.frame_values = vec![0; self.swapchain.max_frames_in_flight()];
                return Ok(());
            }
            _ => {}
        }

        self.frame_count += 1;

        log::trace!("Finish Frame {:?}", frame.index);

        Ok(())
    }
}

#[derive(Default)]
pub struct App {
    render: Option<Render>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .expect("Acquire Window");

        window.request_redraw();
        let render = Render::new(window).expect("Create Render");
        self.render = Some(render);
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
                    let _ = render.render_frame();
                    render.window.request_redraw();
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
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}
