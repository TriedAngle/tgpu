use raw_window_handle::HasDisplayHandle;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const TRIANGLE_SHADER_SLANG: &str = r#"
struct VSOutput {
    float4 position : SV_Position;
    float3 color;
};

static const float2 positions[3] = {
    float2( 0.5,  0.5),
    float2(-0.5,  0.5), 
    float2( 0.0, -0.5),
};

static const float3 colors[3] = { 
    float3(1.0, 0.0, 0.0),   // red
    float3(0.0, 1.0, 0.0),   // green
    float3(0.0, 0.0, 1.0),   // blue
};

[shader("vertex")]
VSOutput vmain(uint vertexId : SV_VertexID) {
    VSOutput o;
    o.position = float4(positions[vertexId], 0.0, 1.0);
    o.color    = colors[vertexId];
    return o;
}

[shader("fragment")]
float4 fmain(VSOutput input) : SV_Target0 {
    return float4(input.color, 1.0);
}
"#;

#[allow(unused)]
pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    buffer: tgpu::Buffer,
    pipeline: tgpu::RenderPipeline,
    frame_count: usize,
}

impl Render {
    pub fn new(window: Window) -> Result<Render, tgpu::GPUError> {
        let display = window.display_handle().unwrap().as_raw();

        let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
            app_name: "Triangle",
            engine_name: "Example Engine",
            display: Some(display),
        })?;

        let adapter = instance.default_adapter()?;

        let (device, queue) = adapter.request_device(&tgpu::DeviceCreateInfo::default())?;

        let size = window.inner_size();

        // TODO: use this buffer
        let buffer = device.create_buffer(&tgpu::BufferDesc {
            label: Some(tgpu::Label::Name("test")),
            size: 420,
            usage: tgpu::BufferUses::COPY_DST,
            memory: tgpu::MemoryPreset::Dynamic,
            host_access: tgpu::HostAccess::WriteSequential,
            ..Default::default()
        })?;

        let swapchain = device
            .create_swapchain_for_window(&window, tgpu::Extent2D::new(size.width, size.height))?;

        let shader = device.create_shader(
            Some(tgpu::Label::Name("shader")),
            tgpu::ShaderSource::Slang(TRIANGLE_SHADER_SLANG.as_bytes()),
        )?;

        let pipeline = device.create_render_pipeline(&tgpu::RenderPipelineInfo {
            label: Some(tgpu::Label::Name("Present Pipeline")),
            vertex_shader: shader.entry("vmain"),
            fragment_shader: shader.entry("fmain"),
            cull: tgpu::CullMode::BACK,
            topology: tgpu::PrimitiveTopology::TriangleList,
            polygon: tgpu::PolygonMode::Fill,
            front_face: tgpu::FrontFace::Clockwise,
            color_formats: &[swapchain.format()],
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
            frame_count: 0,
        };

        Ok(new)
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.swapchain.begin_frame(&self.queue)?;
        log::trace!("Start Frame {:?}", frame.index);
        let mut recorder = self.queue.record();

        self.swapchain
            .transition_frame(&mut recorder, frame, tgpu::ImageLayoutTransition::COLOR);

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

        recorder.begin_render(
            &tgpu::RenderInfo {
                colors: &[attachment],
                area: tgpu::Rect::from_extent(self.swapchain.extent()),
                ..Default::default()
            },
            |recorder| {
                recorder.draw(0..3, 0..1);
            },
        );

        self.swapchain
            .transition_frame(&mut recorder, frame, tgpu::ImageLayoutTransition::PRESENT);

        self.swapchain
            .end_frame(&self.queue, frame, &[recorder.finish()])?;

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
    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("run app failed: {err}");
    }
}
