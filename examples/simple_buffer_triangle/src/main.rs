use raw_window_handle::HasDisplayHandle;

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
}

impl Render {
    pub fn new(window: Window) -> Result<Render, tgpu::GPUError> {
        let display = window.display_handle().unwrap().as_raw();

        let instance = tgpu::Instance::new(&tgpu::InstanceCreateInfo {
            app_name: "Simple Buffer Triangle",
            engine_name: "Example Engine",
            display: Some(display),
        })?;

        let adapter = instance.default_adapter()?;

        let (device, queue) = adapter.request_device(&tgpu::DeviceCreateInfo::default())?;

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

        let vertex_buffer = device.create_buffer_from_slice(
            &vertices,
            &tgpu::BufferDesc {
                label: Some(tgpu::Label::Name("triangle vertices")),
                usage: tgpu::BufferUses::STORAGE,
                memory: tgpu::MemoryPreset::Dynamic,
                host_access: tgpu::HostAccess::WriteSequential,
                ..Default::default()
            },
        )?;

        let bindless = device.create_bindless_heap(&tgpu::BindlessInfo {
            max_read_buffers: 1,
            ..Default::default()
        });
        let vertex_buffer_handle = bindless.add_read_buffer(&vertex_buffer);

        let size = window.inner_size();

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
            descriptor_layouts: &[bindless.layout()],
            push_constant_size: tgpu::push_constant_size::<PushConstants>(),
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
            vertex_buffer,
            bindless,
            pipeline,
            pc: PushConstants {
                vertex_buffer: vertex_buffer_handle,
            },
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

        recorder.bind_render(&self.pipeline, self.bindless.descriptor_set(), self.pc);

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
    event_loop.run_app(&mut app).expect("run app");
}
