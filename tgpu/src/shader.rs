use std::sync::Arc;

use ash::vk;

use crate::{Device, Label, raw::RawDevice};

pub enum ShaderSource<'a> {
    Slang(&'a u8),
    Glsl(&'a u8),
    Wgsl(&'a str),
    SpirV(&'a [u32]),
}

pub struct Shader {
    pub module: ShaderModule,
}

pub struct ShaderEntry<'a> {
    pub shader: &'a Shader,
    pub name: &'a str,
}

pub struct ShaderModule {
    pub device: RawDevice,
    pub handle: vk::ShaderModule,
}

impl<'a> ShaderEntry<'a> {
    pub fn null() -> Self {
        unsafe {
            Self {
                shader: std::mem::transmute(std::ptr::null::<Shader>()),
                name: "",
            }
        }
    }
}

impl Shader {
    pub fn entry<'a>(&'a self, name: &'a str) -> ShaderEntry<'a> {
        ShaderEntry {
            shader: &self,
            name,
        }
    }
}

impl Device {
    pub fn create_shader<'a>(
        &self,
        label: Option<Label<'a>>,
        source: &ShaderSource<'a>,
    ) -> Result<Shader, String> {
        match source {
            ShaderSource::Slang(_code) => unimplemented!(),
            ShaderSource::Glsl(_code) => unimplemented!(),
            ShaderSource::Wgsl(code) => {
                let wgsl_shader = WgslShader::new(code)?;
                let spirv = wgsl_shader.compile().map_err(|e| e.to_string())?;
                Ok(self.create_shader_from_spirv(label, &spirv))
            }
            ShaderSource::SpirV(spirv) => Ok(self.create_shader_from_spirv(label, spirv)),
        }
    }

    pub fn create_shader_from_spirv<'a>(
        &self,
        label: Option<Label<'a>>,
        spirv: &'a [u32],
    ) -> Shader {
        let handle = unsafe { self.inner.create_shader_module_from_spirv(label, spirv) };
        let module = ShaderModule {
            device: self.inner.clone(),
            handle,
        };
        Shader { module }
    }

    // fn compile_glsl_shader<'a>(&self, code: &[u8]) -> Result<Arc<u32>, String> {
    //
    // }
}

pub struct WgslShader {
    module: naga::Module,
    info: naga::valid::ModuleInfo,
}

impl WgslShader {
    pub fn new(source: &str) -> Result<Self, String> {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(m) => m,
            Err(err) => {
                let detailed_msg = err.emit_to_string(source);
                return Result::Err(detailed_msg);
            }
        };

        let flags = naga::valid::ValidationFlags::all();
        let capabilities = naga::valid::Capabilities::all();
        let info = match naga::valid::Validator::new(flags, capabilities).validate(&module) {
            Ok(info) => info,
            Err(err) => {
                let detailed_msg = err.emit_to_string(source);
                return Result::Err(detailed_msg);
            }
        };

        let shader = WgslShader { module, info };
        Ok(shader)
    }

    pub fn compile(&self) -> Result<Arc<[u32]>, naga::back::spv::Error> {
        use naga::back::spv;
        let flags = spv::WriterFlags::empty();
        let options = spv::Options {
            flags,
            lang_version: (1, 3),
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            debug_info: None,
            ..Default::default()
        };

        let mut writer = spv::Writer::new(&options)?;
        let mut compiled = Vec::new();

        writer.write(&self.module, &self.info, None, &None, &mut compiled)?;
        let compiled = Arc::from(compiled);

        Ok(compiled)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_shader_module(self.handle, None) };
    }
}

#[cfg(test)]
mod tests {

    //     #[test]
    // fn create_wgsl_shader() {
    //         let source = ShaderSource::Wgsl(WGSL_TRIANGLE.as_bytes().into());
    //
    //     }

    const WGSL_TRIANGLE: &str = r#"
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
        "#;

    fn device() -> (crate::Device, crate::Queue) {
        let instance = crate::Instance::new(&crate::InstanceCreateInfo {
            app_name: "Triangle",
            engine_name: "Example Engine",
        })
        .unwrap();

        let adapters = instance.adapters(&[]).unwrap().collect::<Vec<_>>();

        let adapter = adapters[0].clone();

        let (device, mut queues) = instance
            .request_device(
                &crate::DeviceCreateInfo {},
                adapter,
                &[crate::QueueRequest {
                    required_flags: crate::QueueFlags::GRAPHICS,
                    exclude_flags: crate::QueueFlags::empty(),
                    strict: false,
                    allow_fallback_share: true,
                }],
            )
            .unwrap();

        (device, queues.next().unwrap())
    }
}
