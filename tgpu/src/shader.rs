use std::{fs, path::Path, process::Command, sync::Arc};

use ash::vk;

use crate::{Device, Label, raw::RawDevice};

pub enum ShaderSource<'a> {
    Slang(&'a [u8]),
    Glsl(&'a [u8]),
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
        source: ShaderSource<'a>,
    ) -> Result<Shader, String> {
        match source {
            ShaderSource::Slang(code) => {
                let spirv = compile_slang_from_bytes(code)?;
                Ok(self.create_shader_from_spirv(label, &spirv))
            }
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

/// File-based API: compile `<input>.slang` to `<output>.spv` using `slangc`.
///
/// Equivalent to:
/// slangc <input>.slang \
///     -target spirv \
///     -profile spirv_1_4 \
///     -fvk-use-entrypoint-name \
///     -emit-spirv-directly \
///     -o <output>.spv
pub fn compile_slang_to_spirv<I, O>(input: I, output: O) -> Result<(), String>
where
    I: AsRef<Path>,
    O: AsRef<Path>,
{
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    if !input_path.exists() {
        return Err(format!(
            "Input file does not exist: {}",
            input_path.display()
        ));
    }

    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create output directory {}: {e}",
                parent.display()
            )
        })?;
    }

    let output = Command::new("slangc")
        .arg(input_path)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("spirv_1_4")
        .arg("-fvk-use-entrypoint-name")
        .arg("-emit-spirv-directly")
        .arg("-o")
        .arg(output_path)
        .output()
        .map_err(|e| format!("Failed to execute `slangc`: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`slangc` failed with exit code {:?}:\n{}",
            output.status.code(),
            stderr
        ));
    }

    Ok(())
}

pub fn compile_slang_from_bytes(source: &[u8]) -> Result<Arc<[u32]>, String> {
    let dir =
        tempfile::tempdir().map_err(|e| format!("Failed to create temporary directory: {e}"))?;

    let input_path = dir.path().join("input.slang");
    let output_path = dir.path().join("output.spv");

    fs::write(&input_path, source).map_err(|e| {
        format!(
            "Failed to write temp input file {}: {e}",
            input_path.display()
        )
    })?;

    compile_slang_to_spirv(&input_path, &output_path)?;

    let spv_bytes = fs::read(&output_path).map_err(|e| {
        format!(
            "Failed to read SPIR-V output {}: {e}",
            output_path.display()
        )
    })?;

    if spv_bytes.len() % 4 != 0 {
        return Err(format!(
            "SPIR-V output length ({}) is not 4-byte aligned",
            spv_bytes.len()
        ));
    }

    let words: Vec<u32> = {
        let u32_slice: &[u32] = bytemuck::cast_slice(&spv_bytes);
        u32_slice.to_vec()
    };

    Ok(Arc::from(words.into_boxed_slice()))
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
