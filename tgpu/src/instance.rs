use ash::vk;
use std::borrow::Cow;
use std::ffi;
use std::sync::Arc;

use crate::{Adapter, GPUError};

pub struct Instance {
    pub(crate) inner: RawInstance,
}

pub type RawInstance = Arc<InstanceImpl>;

pub struct InstanceImpl {
    pub entry: ash::Entry,
    pub handle: ash::Instance,
}

#[derive(Default)]
pub struct InstanceCreateInfo<'a> {
    pub app_name: &'a str,
    pub engine_name: &'a str,
}

impl Instance {
    pub fn new(info: &InstanceCreateInfo<'_>) -> Result<Self, GPUError> {
        let app_name = ffi::CString::new(info.app_name).expect("Convert to cstring");
        let engine_name = ffi::CString::new(info.engine_name).expect("Convert to cstring");

        let (extensions, flags) = Self::get_required_extensions_and_flags();

        let validation_layer = ffi::CString::new("VK_LAYER_KHRONOS_validation")
            .expect("Create Validation Layer String");

        let layers = vec![validation_layer.as_ptr()];

        let instance =
            unsafe { InstanceImpl::new(&app_name, &engine_name, &extensions, &layers, flags)? };

        let instance = Arc::new(instance);

        Ok(Self { inner: instance })
    }

    pub fn adapters(
        &self,
        formats: &[vk::Format],
    ) -> Result<impl Iterator<Item = Adapter> + use<>, GPUError> {
        let adapters = unsafe { self.inner.adapters(formats)? };
        Ok(adapters.into_iter().map(|inner| Adapter {
            inner: Arc::new(inner),
        }))
    }

    fn get_required_extensions_and_flags() -> (Vec<*const i8>, vk::InstanceCreateFlags) {
        let mut extensions = vec![ash::khr::surface::NAME.as_ptr()];
        let mut flags = vk::InstanceCreateFlags::empty();

        // TODO: make this toggable
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());

        #[cfg(target_os = "windows")]
        extensions.push(ash::khr::win32_surface::NAME.as_ptr());

        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        extensions.push(ash::khr::xlib_surface::NAME.as_ptr());

        #[cfg(target_os = "macos")]
        {
            extensions.push(ash::ext::metal_surface::NAME.as_ptr());
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        #[cfg(target_os = "android")]
        extensions.push(ash::khr::android_surface::NAME.as_ptr());

        #[cfg(target_os = "ios")]
        extensions.push(ash::mvk::ios_surface::NAME.as_ptr());

        (extensions, flags)
    }

    pub fn raw(&self) -> RawInstance {
        self.inner.clone()
    }
}

impl InstanceImpl {
    pub unsafe fn new(
        app_name: &ffi::CStr,
        engine_name: &ffi::CStr,
        extensions: &[*const i8],
        layers: &[*const i8],
        flags: vk::InstanceCreateFlags,
    ) -> Result<Self, GPUError> {
        let entry = match unsafe { Self::load_entry() } {
            Ok(entry) => entry,
            Err(e) => {
                panic!("Error loading instance: {:?}", e);
            }
        };

        let version = unsafe { Self::get_vulkan_instance_version(&entry).map_err(GPUError::from)? };

        log::debug!(
            "Entry Created: Vulkan {}.{}.{}",
            vk::api_version_major(version),
            vk::api_version_minor(version),
            vk::api_version_patch(version),
        );

        let afo = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut dfo = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let ifo = vk::InstanceCreateInfo::default()
            .application_info(&afo)
            .enabled_extension_names(extensions)
            .enabled_layer_names(layers)
            .flags(flags)
            .push_next(&mut dfo);

        let handle = match unsafe { entry.create_instance(&ifo, None) } {
            Ok(handle) => handle,
            Err(e) => return Err(GPUError::Vulkan(e)),
        };

        Ok(Self { entry, handle })
    }

    pub unsafe fn load_entry() -> Result<ash::Entry, ash::LoadingError> {
        match unsafe { ash::Entry::load() } {
            Ok(entry) => Ok(entry),
            Err(e) => {
                log::error!("Failed to load Vulkan loader: {}", e);
                if cfg!(target_os = "macos") {
                    log::error!(
                        "Ensure the Vulkan SDK is installed and VULKAN_SDK, DYLD_LIBRARY_PATH, VK_ICD_FILENAMES environment variables are set correctly."
                    );
                    log::error!(
                        "See: https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html"
                    );
                }
                return Err(e);
            }
        }
    }

    pub unsafe fn get_vulkan_instance_version(entry: &ash::Entry) -> Result<u32, vk::Result> {
        unsafe {
            match entry.try_enumerate_instance_version() {
                Ok(Some(version)) => Ok(version),
                Ok(None) => Ok(vk::make_api_version(0, 1, 0, 0)),
                Err(e) => Err(e),
            }
        }
    }

    pub unsafe fn properties(&self, pdev: vk::PhysicalDevice) -> vk::PhysicalDeviceProperties {
        unsafe { self.handle.get_physical_device_properties(pdev) }
    }

    pub unsafe fn features(&self, pdev: vk::PhysicalDevice) -> vk::PhysicalDeviceFeatures {
        unsafe { self.handle.get_physical_device_features(pdev) }
    }

    pub unsafe fn queue_properties(
        &self,
        pdev: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.handle
                .get_physical_device_queue_family_properties(pdev)
        }
    }
    pub unsafe fn format_properties(
        &self,
        pdev: vk::PhysicalDevice,
        formats: &[vk::Format],
    ) -> Vec<(vk::Format, vk::FormatProperties)> {
        formats
            .iter()
            .map(|&format| unsafe {
                let props = self
                    .handle
                    .get_physical_device_format_properties(pdev, format);
                (format, props)
            })
            .collect::<Vec<_>>()
    }

    pub unsafe fn queue_family_properties(
        &self,
        pdev: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        unsafe {
            self.handle
                .get_physical_device_queue_family_properties(pdev)
        }
    }

    pub unsafe fn create_device_handle(
        &self,
        info: &vk::DeviceCreateInfo,
        pdev: vk::PhysicalDevice,
    ) -> ash::Device {
        unsafe {
            self.handle
                .create_device(pdev, info, None)
                .expect("Create Device")
        }
    }
}

impl Drop for InstanceImpl {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}

pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message_id_number = callback_data.message_id_number;
    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    };
    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!(
                "{:?} [{} ({})] : {}",
                message_type,
                message_id_name,
                message_id_number,
                message,
            )
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!(
                "{:?} [{} ({})] : {}",
                message_type,
                message_id_name,
                message_id_number,
                message,
            )
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!(
                "{:?} [{} ({})] : {}",
                message_type,
                message_id_name,
                message_id_number,
                message,
            )
        }

        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            log::trace!(
                "{:?} [{} ({})] : {}",
                message_type,
                message_id_name,
                message_id_number,
                message,
            )
        }
        _ => {
            log::warn!(
                "Unknown Severity: {:?}\n{:?} [{} ({})] : {}",
                message_severity,
                message_type,
                message_id_name,
                message_id_number,
                message,
            )
        }
    }

    vk::FALSE
}
