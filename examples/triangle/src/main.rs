use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tgpu::ash::vk;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    buffer: tgpu::Buffer,
    pipeline: tgpu::RenderPipeline,
}

impl Render {
    pub fn new(window: Window) -> Result<Render, tgpu::GPUError> {
        let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
            app_name: "Triangle",
            engine_name: "Example Engine",
        })?;

        let adapters = instance.adapters(&[])?.collect::<Vec<_>>();
        println!("adapters: {:#?}", adapters);

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

        let buffer = device.create_buffer(&tgpu::BufferInfo {
            label: Some("test"),
            size: 420,
            usage: tgpu::BufferUsage::COPY_DST
                | tgpu::BufferUsage::MAP_WRITE
                | tgpu::BufferUsage::DEVICE,
        })?;

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

        let shader = device
            .create_shader(
                r#"
const positions = array<vec2f, 3>(
  vec2f(0.0, -0.5),
  vec2f(0.5, 0.5),
  vec2f(-0.5, 0.5)
);

const colors = array<vec3f, 3>(
  vec3f(1.0, 0.0, 0.0),
  vec3f(0.0, 1.0, 0.0),
  vec3f(0.0, 0.0, 1.0)
);

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) fragColor: vec3f,
};

@vertex
fn vmain(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var output: VertexOutput;
  output.position = vec4f(positions[vertex_index], 0.0, 1.0);
  output.fragColor = colors[vertex_index];
  return output;
}

@fragment
fn fmain(input: VertexOutput) -> @location(0) vec4f {
  return vec4f(input.fragColor, 1.0);
}
        "#,
            )
            .expect("Shader");

        let pipeline = device.create_render_pipeline(&tgpu::RenderPipelineInfo {
            label: Some("Present Pipeline"),
            vertex_shader: shader.entry("vmain"),
            fragment_shader: shader.entry("fmain"),
            cull: tgpu::CullModeFlags::BACK,
            topology: tgpu::PrimitiveTopology::TRIANGLE_LIST,
            polygon: tgpu::PolygonMode::FILL,
            front_face: vk::FrontFace::CLOCKWISE,
            ..Default::default()
        });

        let new = Self {
            window,
            instance,
            device,
            queue,
            swapchain,
            buffer,
            pipeline,
        };

        Ok(new)
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.swapchain.acquire_next(None)?;
        if frame.suboptimal {
            // TODO swapchain recreation
        }
        // TODO resize
        let mut recorder = self.queue.record();

        {
            // TODO: finish this
            let _attachment = vk::RenderingAttachmentInfo::default();

            let _ = recorder.begin_render(&tgpu::RenderInfo {
                colors: &[],
                ..Default::default()
            });

            let viewport = vk::Viewport {
                width: self.swapchain.extent.width as f32,
                height: self.swapchain.extent.height as f32,
                max_depth: 1.0,
                ..Default::default()
            };

            let scissor = vk::Rect2D {
                extent: self.swapchain.extent,
                ..Default::default()
            };

            recorder.viewport(viewport);
            recorder.scissor(scissor);

            recorder.bind_render_pipeline(&self.pipeline);
            recorder.draw(0..3, 0..1);

            unsafe {
                let inner_recorder = &mut *recorder.inner.get();
                inner_recorder.end_rendering();
            }
        }

        recorder.image_transition(
            self.swapchain.image(frame),
            tgpu::ImageTransition {
                from: tgpu::ImageLayoutTransition::UNDEFINED,
                to: tgpu::ImageLayoutTransition::PRESENT,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        );

        self.queue.submit(tgpu::SubmitInfo {
            records: &[recorder.finish()],
            ..Default::default()
        });

        match self.swapchain.present(&self.queue, frame) {
            Ok(true) | Err(_) => {
                // TODO swapchain recreation
            }
            _ => {}
        }

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
        window_id: winit::window::WindowId,
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
