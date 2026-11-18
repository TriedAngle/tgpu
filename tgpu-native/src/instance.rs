use std::{ffi::c_char, ptr};

use tgpu::{
    AdapterDeviceType, Device, DeviceCreateInfo, DeviceFeatures, Instance, InstanceCreateInfo,
    Queue, QueueFlags, QueueRequest,
};

use crate::{
    error::{Error, TgpuStatus, check_size, cstr, guard_box, guard_status},
    types::{
        TGPU_ADAPTER_DEVICE_TYPE_CPU, TGPU_ADAPTER_DEVICE_TYPE_DISCRETE_GPU,
        TGPU_ADAPTER_DEVICE_TYPE_INTEGRATED_GPU, TGPU_ADAPTER_DEVICE_TYPE_OTHER,
        TGPU_ADAPTER_DEVICE_TYPE_VIRTUAL_GPU, TGPU_DEVICE_FEATURE_BUFFER_DEVICE_ADDRESS,
        TGPU_DEVICE_FEATURE_DESCRIPTOR_INDEXING, TGPU_DEVICE_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE,
        TGPU_DEVICE_FEATURE_FILL_MODE_NON_SOLID, TGPU_DEVICE_FEATURE_MULTI_DRAW_INDIRECT,
        TGPU_QUEUE_COMPUTE, TGPU_QUEUE_GRAPHICS, TGPU_QUEUE_PROTECTED, TGPU_QUEUE_SPARSE_BINDING,
        TGPU_QUEUE_TRANSFER,
    },
};

pub struct TgpuInstance {
    handle: Instance,
}

/// Owns the Vulkan device and every queue created with it.
///
/// Queues are borrowed handles obtained from [`tgpu_device_queue`]; they live
/// as long as the device.
pub struct TgpuDevice {
    handle: Device,
}

impl TgpuDevice {
    pub fn handle(&self) -> &Device {
        &self.handle
    }
}

pub struct TgpuQueue {
    handle: Queue,
}

impl TgpuQueue {
    pub fn handle(&self) -> &Queue {
        &self.handle
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuInstanceDesc {
    pub struct_size: usize,
    pub app_name: *const c_char,
    pub engine_name: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuQueueDesc {
    pub struct_size: usize,
    /// A combination of `TGPU_QUEUE_*`.
    pub flags: u32,
    /// Fail instead of accepting a family with extra capabilities.
    pub strict: bool,
    /// Allow several requests to share one queue.
    pub allow_fallback_share: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuDeviceDesc {
    pub struct_size: usize,
    /// A combination of `TGPU_DEVICE_FEATURE_*`.
    pub features: u32,
    pub queue_count: usize,
    /// `queue_count` entries, or NULL for one graphics+compute queue.
    pub queues: *const TgpuQueueDesc,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuAdapterInfo {
    /// NUL-terminated UTF-8, truncated to 256 bytes.
    pub name: [c_char; 256],
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    /// Raw `VkPhysicalDeviceProperties::apiVersion`.
    pub api_version: u32,
    pub driver_version: u32,
}

impl TgpuAdapterInfo {
    fn from_tgpu(info: &tgpu::AdapterInfo) -> Self {
        let mut name = [0 as c_char; 256];
        for (slot, byte) in name.iter_mut().zip(info.name.as_bytes()) {
            *slot = *byte as c_char;
        }
        name[255] = 0;

        let device_type = match info.device_type {
            AdapterDeviceType::IntegratedGpu => TGPU_ADAPTER_DEVICE_TYPE_INTEGRATED_GPU,
            AdapterDeviceType::DiscreteGpu => TGPU_ADAPTER_DEVICE_TYPE_DISCRETE_GPU,
            AdapterDeviceType::VirtualGpu => TGPU_ADAPTER_DEVICE_TYPE_VIRTUAL_GPU,
            AdapterDeviceType::Cpu => TGPU_ADAPTER_DEVICE_TYPE_CPU,
            AdapterDeviceType::Other => TGPU_ADAPTER_DEVICE_TYPE_OTHER,
        };

        Self {
            name,
            vendor_id: info.vendor_id,
            device_id: info.device_id,
            device_type,
            api_version: info.api_version,
            driver_version: info.driver_version,
        }
    }
}

fn queue_flags(bits: u32) -> QueueFlags {
    let mut flags = QueueFlags::empty();
    if bits & TGPU_QUEUE_GRAPHICS != 0 {
        flags |= QueueFlags::GRAPHICS;
    }
    if bits & TGPU_QUEUE_COMPUTE != 0 {
        flags |= QueueFlags::COMPUTE;
    }
    if bits & TGPU_QUEUE_TRANSFER != 0 {
        flags |= QueueFlags::TRANSFER;
    }
    if bits & TGPU_QUEUE_SPARSE_BINDING != 0 {
        flags |= QueueFlags::SPARSE_BINDING;
    }
    if bits & TGPU_QUEUE_PROTECTED != 0 {
        flags |= QueueFlags::PROTECTED;
    }
    flags
}

fn device_features(bits: u32) -> DeviceFeatures {
    DeviceFeatures {
        fill_mode_non_solid: bits & TGPU_DEVICE_FEATURE_FILL_MODE_NON_SOLID != 0,
        multi_draw_indirect: bits & TGPU_DEVICE_FEATURE_MULTI_DRAW_INDIRECT != 0,
        draw_indirect_first_instance: bits & TGPU_DEVICE_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE != 0,
        descriptor_indexing: bits & TGPU_DEVICE_FEATURE_DESCRIPTOR_INDEXING != 0,
        buffer_device_address: bits & TGPU_DEVICE_FEATURE_BUFFER_DEVICE_ADDRESS != 0,
    }
}

/// Creates a Vulkan instance.
///
/// Pass `NULL` for `desc` to use the defaults. Returns `NULL` on failure; call
/// [`crate::tgpu_last_error`] for the reason.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_instance_create(desc: *const TgpuInstanceDesc) -> *mut TgpuInstance {
    guard_box(|| {
        let mut app_name = "tgpu-native";
        let mut engine_name = "tgpu-native";

        if let Some(desc) = unsafe { desc.as_ref() } {
            check_size::<TgpuInstanceDesc>(desc.struct_size, "TgpuInstanceDesc")?;
            app_name = unsafe { cstr(desc.app_name) }.unwrap_or(app_name);
            engine_name = unsafe { cstr(desc.engine_name) }.unwrap_or(engine_name);
        }

        let inner = Instance::new(&InstanceCreateInfo {
            app_name,
            engine_name,
            display: None,
        })?;
        Ok(Box::new(TgpuInstance { handle: inner }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_instance_destroy(instance: *mut TgpuInstance) {
    if !instance.is_null() {
        drop(unsafe { Box::from_raw(instance) });
    }
}

/// Creates a device with the requested queues.
///
/// `desc` may be NULL, in which case one graphics+compute queue is created.
/// Creates a device and writes the requested queue handles into `out_queues`.
///
/// `out_queues` must have room for `desc.queue_count` entries (or one when
/// `desc` is NULL). Each queue is owned by the caller; destroy it with
/// [`tgpu_queue_destroy`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create(
    instance: *mut TgpuInstance,
    desc: *const TgpuDeviceDesc,
    out_queues: *mut *mut TgpuQueue,
    out_capacity: usize,
) -> *mut TgpuDevice {
    guard_box(|| {
        let instance =
            unsafe { instance.as_ref() }.ok_or_else(|| Error::validation("instance is NULL"))?;

        let (features, requests) = if let Some(desc) = unsafe { desc.as_ref() } {
            check_size::<TgpuDeviceDesc>(desc.struct_size, "TgpuDeviceDesc")?;

            let requests = if desc.queue_count == 0 || desc.queues.is_null() {
                vec![QueueRequest::graphics_compute()]
            } else {
                let queues = unsafe { std::slice::from_raw_parts(desc.queues, desc.queue_count) };
                queues
                    .iter()
                    .map(|queue| {
                        check_size::<TgpuQueueDesc>(queue.struct_size, "TgpuQueueDesc")?;
                        Ok(QueueRequest {
                            required_flags: queue_flags(queue.flags),
                            exclude_flags: QueueFlags::empty(),
                            strict: queue.strict,
                            allow_fallback_share: queue.allow_fallback_share,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?
            };

            (device_features(desc.features), requests)
        } else {
            (
                DeviceFeatures::modern_bindless_defaults(),
                vec![QueueRequest::graphics_compute()],
            )
        };

        let adapter = instance.handle.default_adapter()?;
        let (device, queues) = instance.handle.request_device_with_queues(
            &DeviceCreateInfo { features },
            adapter,
            &requests,
        )?;

        let queues = queues
            .map(|inner| Box::new(TgpuQueue { handle: inner }))
            .collect::<Vec<_>>();

        if out_queues.is_null() || out_capacity < queues.len() {
            return Err(Error::validation(format!(
                "out_queues must hold {} queue handles",
                queues.len()
            )));
        }

        for (index, queue) in queues.into_iter().enumerate() {
            unsafe { *out_queues.add(index) = Box::into_raw(queue) };
        }

        Ok(Box::new(TgpuDevice { handle: device }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_destroy(device: *mut TgpuDevice) {
    if !device.is_null() {
        drop(unsafe { Box::from_raw(device) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_wait_idle(device: *mut TgpuDevice) {
    if let Some(device) = unsafe { device.as_ref() } {
        device.handle.wait_idle();
    }
}

/// Fills `out` with information about the selected adapter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_adapter_info(
    device: *mut TgpuDevice,
    out: *mut TgpuAdapterInfo,
) -> TgpuStatus {
    guard_status(|| {
        let device =
            unsafe { device.as_ref() }.ok_or_else(|| Error::validation("device is NULL"))?;
        let out = unsafe { out.as_mut() }.ok_or_else(|| Error::validation("out is NULL"))?;
        *out = TgpuAdapterInfo::from_tgpu(device.handle.adapter.info());
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_queue_destroy(queue: *mut TgpuQueue) {
    if !queue.is_null() {
        drop(unsafe { Box::from_raw(queue) });
    }
}

/// Copies the adapter name into `buffer`. Returns the number of bytes written
/// (excluding the NUL) or 0 when there is nothing to copy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_name(
    device: *mut TgpuDevice,
    buffer: *mut c_char,
    capacity: usize,
) -> usize {
    let Some(device) = (unsafe { device.as_ref() }) else {
        return 0;
    };
    if buffer.is_null() || capacity == 0 {
        return 0;
    }

    let name = device.handle.adapter.info().name.as_bytes();
    let count = name.len().min(capacity - 1);
    unsafe {
        ptr::copy_nonoverlapping(name.as_ptr().cast::<c_char>(), buffer, count);
        *buffer.add(count) = 0;
    }
    count
}
