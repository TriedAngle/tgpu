use ash::vk;
use naga::back::spv;

use crate::{Device, raw::DeviceImpl};

pub struct Shader {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
    pub source: String,
}

impl Shader {
    pub fn entry<'a>(&'a self, name: &'a str) -> ShaderFunction<'a> {
        ShaderFunction {
            shader: self,
            entry_point: name,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ShaderFunction<'a> {
    pub shader: &'a Shader,
    pub entry_point: &'a str,
}

impl ShaderFunction<'_> {
    pub fn null() -> Self {
        unsafe {
            Self {
                shader: std::mem::transmute(std::ptr::null::<Shader>()),
                entry_point: "",
            }
        }
    }

    pub fn entry_point_idx(&self) -> Option<usize> {
        self.shader
            .module
            .entry_points
            .iter()
            .position(|ep| ep.name == self.entry_point)
    }

    pub fn to_spirv(&self) -> Result<Vec<u32>, spv::Error> {
        let entry_point_idx = match self.entry_point_idx() {
            Some(idx) => idx,
            None => return Result::Err(spv::Error::EntryPointNotFound),
        };

        let entry = &self.shader.module.entry_points[entry_point_idx];

        let flags = spv::WriterFlags::empty();
        let options = spv::Options {
            flags,
            lang_version: (1, 3),
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            debug_info: None,
            ..Default::default()
        };

        let pipeline_options = spv::PipelineOptions {
            entry_point: entry.name.clone(),
            shader_stage: entry.stage,
        };
        let mut writer = spv::Writer::new(&options)?;

        let mut compiled = Vec::new();
        writer.write(
            &self.shader.module,
            &self.shader.info,
            Some(&pipeline_options),
            &None,
            &mut compiled,
        )?;

        Ok(compiled)
    }
}

impl Shader {
    pub fn new(source: &str) -> Result<Shader, String> {
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

        Ok(Shader {
            module,
            info,
            source: source.to_owned(),
        })
    }
}

impl DeviceImpl {
    pub fn create_shader_module(
        &self,
        shader: ShaderFunction<'_>,
    ) -> Result<vk::ShaderModule, spv::Error> {
        let spirv = shader.to_spirv()?;

        let info = vk::ShaderModuleCreateInfo::default().code(&spirv);

        let module = unsafe {
            self.handle
                .create_shader_module(&info, None)
                .expect("Create shader module")
        };

        Ok(module)
    }
}

impl Device {
    pub fn create_shader(&self, source: &str) -> Result<Shader, String> {
        Shader::new(source)
    }
}
