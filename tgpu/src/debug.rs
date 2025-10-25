use ash::vk;

use crate::raw::DeviceImpl;

#[derive(Debug, Clone)]
pub enum Label<'a> {
    Name(&'a str),
    Tag((u64, &'a [u8])),
    Both((&'a str, (u64, &'a [u8]))),
}

impl DeviceImpl {
    pub unsafe fn set_object_name<T: vk::Handle>(&self, handle: T, name: &str) {
        let name = std::ffi::CString::new(name).unwrap();
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name);

        unsafe {
            self.ext.debug.set_debug_utils_object_name(&info).unwrap();
        }
    }

    pub unsafe fn set_object_tag<T: vk::Handle>(&self, handle: T, tag_name: u64, tag_data: &[u8]) {
        let info = vk::DebugUtilsObjectTagInfoEXT::default()
            .object_handle(handle)
            .tag_name(tag_name)
            .tag(tag_data);

        unsafe {
            self.ext.debug.set_debug_utils_object_tag(&info).unwrap();
        }
    }

    pub unsafe fn attach_label<T: vk::Handle>(&self, handle: T, label: &Label<'_>) {
        match label {
            Label::Name(name) => unsafe { self.set_object_name(handle, name) },
            Label::Tag((name, data)) => unsafe { self.set_object_tag(handle, *name, data) },
            Label::Both((name, (tag_name, data))) => unsafe {
                let raw = handle.as_raw();
                self.set_object_name(T::from_raw(raw), name);
                self.set_object_tag(T::from_raw(raw), *tag_name, data);
            },
        }
        // TODO
    }
}
