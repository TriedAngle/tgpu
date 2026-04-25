use ash::vk;
use raw_window_handle::RawDisplayHandle;
use std::borrow::Cow;
use std::ffi;
use std::sync::Arc;

use crate::{Adapter, AdapterDescriptorIndexingFeatures, AdapterFeatures, GPUError, RankedAdapter};

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

        let (extensions, flags) = Self::get_required_extensions_and_flags(None)?;

        let validation_layer = ffi::CString::new("VK_LAYER_KHRONOS_validation")
            .expect("Create Validation Layer String");

        let layers = vec![validation_layer.as_ptr()];

        let instance =
            unsafe { InstanceImpl::new(&app_name, &engine_name, &extensions, &layers, flags)? };

        let instance = Arc::new(instance);

        Ok(Self { inner: instance })
    }

    pub fn new_with_display(
        info: &InstanceCreateInfo<'_>,
        display: RawDisplayHandle,
    ) -> Result<Self, GPUError> {
        let app_name = ffi::CString::new(info.app_name).expect("Convert to cstring");
        let engine_name = ffi::CString::new(info.engine_name).expect("Convert to cstring");

        let (extensions, flags) = Self::get_required_extensions_and_flags(Some(display))?;

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

    pub fn rank_adapters(&self, formats: &[vk::Format]) -> Result<Vec<RankedAdapter>, GPUError> {
        let mut adapters = self.adapters(formats)?.collect::<Vec<_>>();
        adapters.sort_by(|left, right| {
            right
                .default_score()
                .cmp(&left.default_score())
                .then_with(|| left.info().name.cmp(&right.info().name))
        });

        Ok(adapters
            .into_iter()
            .map(|adapter| RankedAdapter {
                score: adapter.default_score(),
                adapter,
            })
            .collect())
    }

    pub fn default_adapter(
        &self,
        formats: &[vk::Format],
    ) -> Result<Option<RankedAdapter>, GPUError> {
        Ok(self.rank_adapters(formats)?.into_iter().next())
    }

    fn get_required_extensions_and_flags(
        display: Option<RawDisplayHandle>,
    ) -> Result<(Vec<*const i8>, vk::InstanceCreateFlags), GPUError> {
        let mut extensions = if let Some(display) = display {
            ash_window::enumerate_required_extensions(display)?.to_vec()
        } else {
            vec![ash::khr::surface::NAME.as_ptr()]
        };
        let flags = vk::InstanceCreateFlags::empty();

        #[cfg(target_os = "macos")]
        let mut flags = flags;

        // TODO: make this toggable
        push_unique(&mut extensions, ash::ext::debug_utils::NAME.as_ptr());

        #[cfg(target_os = "windows")]
        {
            push_unique(&mut extensions, ash::khr::win32_surface::NAME.as_ptr());
        }

        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        {
            push_unique(&mut extensions, ash::khr::xlib_surface::NAME.as_ptr());
            push_unique(&mut extensions, ash::khr::wayland_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "macos")]
        {
            push_unique(&mut extensions, ash::ext::metal_surface::NAME.as_ptr());
            push_unique(
                &mut extensions,
                ash::khr::portability_enumeration::NAME.as_ptr(),
            );
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        #[cfg(target_os = "android")]
        {
            push_unique(&mut extensions, ash::khr::android_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "ios")]
        {
            push_unique(&mut extensions, ash::mvk::ios_surface::NAME.as_ptr());
        }

        Ok((extensions, flags))
    }

    pub fn raw(&self) -> RawInstance {
        self.inner.clone()
    }
}

fn push_unique(extensions: &mut Vec<*const i8>, extension: *const i8) {
    if !extensions.contains(&extension) {
        extensions.push(extension);
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
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
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
                Err(e)
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

    pub unsafe fn features(&self, pdev: vk::PhysicalDevice) -> AdapterFeatures {
        let (fill_mode_non_solid, descriptor_indexing, buffer_device_address, shader_int64) = {
            let mut descriptor_indexing_features =
                vk::PhysicalDeviceDescriptorIndexingFeatures::default();
            let mut buffer_device_address_features =
                vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut descriptor_indexing_features)
                .push_next(&mut buffer_device_address_features);

            unsafe { self.handle.get_physical_device_features2(pdev, &mut features2) };

            let base_features = features2.features;
            let fill_mode_non_solid = base_features.fill_mode_non_solid == vk::TRUE;
            let uniform_buffer_dynamic_indexing =
                base_features.shader_uniform_buffer_array_dynamic_indexing == vk::TRUE;
            let sampled_image_dynamic_indexing =
                base_features.shader_sampled_image_array_dynamic_indexing == vk::TRUE;
            let storage_image_dynamic_indexing =
                base_features.shader_storage_image_array_dynamic_indexing == vk::TRUE;
            let storage_buffer_dynamic_indexing =
                base_features.shader_storage_buffer_array_dynamic_indexing == vk::TRUE;
            let shader_int64 = base_features.shader_int64 == vk::TRUE;
            let _ = features2;

            let descriptor_indexing = AdapterDescriptorIndexingFeatures {
                uniform_buffer_dynamic_indexing,
                sampled_image_dynamic_indexing,
                storage_image_dynamic_indexing,
                storage_buffer_dynamic_indexing,
                partially_bound: descriptor_indexing_features.descriptor_binding_partially_bound
                    == vk::TRUE,
                update_unused_while_pending: descriptor_indexing_features
                    .descriptor_binding_update_unused_while_pending
                    == vk::TRUE,
                uniform_buffer_update_after_bind: descriptor_indexing_features
                    .descriptor_binding_uniform_buffer_update_after_bind
                    == vk::TRUE,
                sampled_image_update_after_bind: descriptor_indexing_features
                    .descriptor_binding_sampled_image_update_after_bind
                    == vk::TRUE,
                storage_image_update_after_bind: descriptor_indexing_features
                    .descriptor_binding_storage_image_update_after_bind
                    == vk::TRUE,
                storage_buffer_update_after_bind: descriptor_indexing_features
                    .descriptor_binding_storage_buffer_update_after_bind
                    == vk::TRUE,
                runtime_descriptor_array: descriptor_indexing_features.runtime_descriptor_array
                    == vk::TRUE,
                uniform_buffer_non_uniform_indexing: descriptor_indexing_features
                    .shader_uniform_buffer_array_non_uniform_indexing
                    == vk::TRUE,
                sampled_image_non_uniform_indexing: descriptor_indexing_features
                    .shader_sampled_image_array_non_uniform_indexing
                    == vk::TRUE,
                storage_image_non_uniform_indexing: descriptor_indexing_features
                    .shader_storage_image_array_non_uniform_indexing
                    == vk::TRUE,
                storage_buffer_non_uniform_indexing: descriptor_indexing_features
                    .shader_storage_buffer_array_non_uniform_indexing
                    == vk::TRUE,
            };

            let buffer_device_address = buffer_device_address_features.buffer_device_address == vk::TRUE;

            (
                fill_mode_non_solid,
                descriptor_indexing,
                buffer_device_address,
                shader_int64,
            )
        };

        AdapterFeatures {
            fill_mode_non_solid,
            descriptor_indexing,
            buffer_device_address,
            shader_int64,
        }
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
