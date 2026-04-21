use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tgpu::ash::vk;
use tgpu::egui::egui;

use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const TRIANGLE_SHADER_WGSL: &str = r#"
struct VSOut {
    @location(0) color: vec4<f32>,
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> VSOut {
    var out: VSOut;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TriangleVertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[allow(unused)]
pub struct Render {
    window: Window,
    instance: tgpu::Instance,
    device: tgpu::Device,
    queue: tgpu::Queue,
    swapchain: tgpu::Swapchain,
    triangle_vertices: tgpu::Buffer,
    triangle_pipeline: tgpu::RenderPipeline,
    graph_cache: tgpu::RenderGraphCache,
    ui: tgpu::egui::Renderer,
    clear_color: [f32; 3],
    counter: u32,
    frame_count: usize,
}

impl Render {
    pub fn new(window: Window) -> Result<Self, Box<dyn std::error::Error>> {
        let display = window.display_handle().unwrap().as_raw();

        let instance = tgpu::Instance::new_with_display(
            &tgpu::InstanceCreateInfo {
                app_name: "egui RenderGraph",
                engine_name: "Example Engine",
            },
            display,
        )?;

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

        let triangle_shader = device
            .create_shader(
                Some(tgpu::Label::Name("triangle shader")),
                tgpu::ShaderSource::Wgsl(TRIANGLE_SHADER_WGSL),
            )
            .map_err(std::io::Error::other)?;

        let triangle_vertex_layout = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<TriangleVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let triangle_vertex_attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(TriangleVertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(std::mem::offset_of!(TriangleVertex, color) as u32),
        ];
        let triangle_vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&triangle_vertex_layout)
            .vertex_attribute_descriptions(&triangle_vertex_attributes);

        let triangle_blend = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];

        let triangle_pipeline = device.create_render_pipeline(&tgpu::RenderPipelineInfo {
            label: Some(tgpu::Label::Name("triangle pipeline")),
            vertex_shader: triangle_shader.entry("vs_main"),
            fragment_shader: triangle_shader.entry("fs_main"),
            color_formats: &[swapchain.format()],
            vertex_input_state: Some(triangle_vertex_input),
            blend_states: Some(&triangle_blend),
            topology: tgpu::PrimitiveTopology::TRIANGLE_LIST,
            polygon: tgpu::PolygonMode::FILL,
            cull: tgpu::CullModeFlags::NONE,
            front_face: tgpu::FrontFace::COUNTER_CLOCKWISE,
            ..Default::default()
        });

        let triangle_vertices_data = [
            TriangleVertex {
                position: [-0.82, 0.86],
                color: [1.0, 0.35, 0.28, 1.0],
            },
            TriangleVertex {
                position: [-0.28, 0.86],
                color: [0.98, 0.82, 0.32, 1.0],
            },
            TriangleVertex {
                position: [-0.55, 0.12],
                color: [0.26, 0.72, 1.0, 1.0],
            },
        ];
        let triangle_vertices = device.create_buffer(&tgpu::BufferDesc {
            label: Some(tgpu::Label::Name("triangle vertices")),
            size: std::mem::size_of_val(&triangle_vertices_data),
            usage: tgpu::BufferUses::VERTEX,
            memory: tgpu::MemoryPreset::Dynamic,
            host_access: tgpu::HostAccess::WriteSequential,
            ..Default::default()
        })?;
        triangle_vertices.write_slice(&triangle_vertices_data);

        let ui = tgpu::egui::Renderer::new_for_swapchain(&window, &device, &swapchain)?;

        Ok(Self {
            window,
            instance,
            device,
            queue,
            swapchain,
            triangle_vertices,
            triangle_pipeline,
            graph_cache: tgpu::RenderGraphCache::new(),
            ui,
            clear_color: [0.09, 0.12, 0.16],
            counter: 0,
            frame_count: 0,
        })
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.swapchain.acquire_next(None)?;
        if frame.suboptimal {
            self.handle_resize();
            return Ok(());
        }

        let flight_index = self.swapchain.flight_index();
        let frame_count = self.frame_count;
        let counter = &mut self.counter;
        let clear_color = &mut self.clear_color;

        let mut ui_frame = self.ui.run(&self.window, flight_index, |ctx| {
            egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                ui.heading("tgpu + egui + RenderGraph");
                ui.label("This frame uses the render graph path. The same renderer also supports manual command recording.");
            });

            egui::Window::new("Controls")
                .default_width(320.0)
                .show(ctx, |ui| {
                    ui.label(format!("Frame {}", frame_count));
                    ui.label(format!("Counter {}", *counter));
                    if ui.button("Increment").clicked() {
                        *counter += 1;
                    }
                    if ui.button("Reset").clicked() {
                        *counter = 0;
                    }

                    ui.separator();
                    ui.label("Background");
                    ui.add(egui::Slider::new(&mut clear_color[0], 0.0..=1.0).text("red"));
                    ui.add(egui::Slider::new(&mut clear_color[1], 0.0..=1.0).text("green"));
                    ui.add(egui::Slider::new(&mut clear_color[2], 0.0..=1.0).text("blue"));

                    ui.separator();
                    ui.label("Resize the window and keep interacting with the UI to exercise swapchain recreation and the graph cache.");
                    ui.label("A colored triangle is rendered from a vertex buffer underneath this UI.");
                });
        })?;

        let mut graph = tgpu::RenderGraph::new(tgpu::RenderGraphInfo {
            device: &self.device,
            graphics: &self.queue,
            async_compute: None,
            copy: None,
            bindless: None,
        });

        let extent = self.swapchain.extent();
        let backbuffer = graph.import_swapchain_image("backbuffer", &mut self.swapchain, frame);
        let triangle_vertices = graph.import_buffer(
            "triangle_vertices",
            &self.triangle_vertices,
            tgpu::ImportedBufferDesc {
                initial: tgpu::BufferAccessTransition::NONE,
                initialized: true,
            },
        );
        let clear = self.clear_color;
        graph.add_render_pass(
            "clear",
            |pass| {
                pass.write_color(
                    backbuffer,
                    tgpu::ColorAttachmentDesc::clear([clear[0], clear[1], clear[2], 1.0]),
                );
            },
            |ctx| {
                ctx.begin_render(|_| {});
            },
        );

        let triangle_pipeline = &self.triangle_pipeline;
        graph.add_render_pass(
            "triangle",
            |pass| {
                pass.read_buffer(triangle_vertices, tgpu::BufferAccess::Vertex);
                pass.write_color(backbuffer, tgpu::ColorAttachmentDesc::load());
            },
            move |ctx| {
                let vertex_buffer = ctx.buffer(triangle_vertices).clone();
                ctx.begin_render(|render| {
                    render.bind_render_pipeline(triangle_pipeline);
                    render.bind_vertex_buffer(0, &vertex_buffer, 0);
                    render.viewport(vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: extent.width as f32,
                        height: extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    });
                    render.scissor(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    });
                    render.draw(0..3, 0..1);
                });
            },
        );

        ui_frame.add_to_graph(
            &mut graph,
            backbuffer,
            tgpu::ColorAttachmentDesc::load(),
            extent,
        );

        let outcome = graph.execute_cached(&mut self.graph_cache)?;
        if outcome.needs_swapchain_recreation {
            self.handle_resize();
            return Ok(());
        }

        self.frame_count += 1;
        Ok(())
    }

    fn handle_resize(&mut self) {
        let size = self.window.inner_size();
        self.swapchain.set_preferred_extent(vk::Extent2D {
            width: size.width,
            height: size.height,
        });
        let _ = self.swapchain.recreate();
        self.graph_cache.clear();
    }
}

#[derive(Default)]
pub struct App {
    render: Option<Render>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes().with_title("tgpu egui render graph example"),
            )
            .expect("Acquire Window");

        window.request_redraw();
        self.render = Some(Render::new(window).expect("Create Render"));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(render) = &mut self.render {
            let _ = render.ui.on_window_event(&render.window, &event);
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
            WindowEvent::RedrawRequested => {
                if let Some(render) = &mut self.render {
                    let _ = render.render_frame();
                    render.window.request_redraw();
                }
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(render) = &mut self.render {
                    render.handle_resize();
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
