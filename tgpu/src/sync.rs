use std::{sync::Arc, time::Duration};

use ash::vk;

use crate::{
    Device,
    raw::{DeviceImpl, RawDevice},
};

#[derive(Debug, Clone)]
pub struct Semaphore {
    pub inner: RawSemaphore,
}

pub type RawSemaphore = Arc<SemaphoreImpl>;

// pub struct Fence {
// }

#[derive(Debug)]
pub struct SemaphoreImpl {
    pub handle: vk::Semaphore,
    pub device: RawDevice,
}

impl Semaphore {
    pub fn get(&self) -> u64 {
        unsafe { self.inner.get() }
    }

    pub fn signal(&self, value: u64) {
        unsafe { self.inner.signal(value) };
    }

    pub fn wait(&self, value: u64, timeout: Option<Duration>) {
        unsafe { self.inner.wait(value, timeout) };
    }
}

impl SemaphoreImpl {
    pub unsafe fn get(&self) -> u64 {
        unsafe { self.device.get_semaphore_value(self.handle) }
    }

    pub unsafe fn signal(&self, value: u64) {
        unsafe { self.device.signal_semaphore(self.handle, value) };
    }

    pub unsafe fn wait(&self, value: u64, timeout: Option<Duration>) {
        unsafe { self.device.wait_semaphore(self.handle, value, timeout) };
    }

    pub unsafe fn new_signal(device: RawDevice) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe {
            device
                .handle
                .create_semaphore(&semaphore_info, None)
                .expect("Create Signal Semaphore")
        };
        Self { handle, device }
    }
    pub unsafe fn new_timeline(device: Arc<DeviceImpl>, value: u64) -> Self {
        let mut semaphore_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(value);

        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type_info);

        let handle = unsafe {
            device
                .handle
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        };

        SemaphoreImpl {
            handle,
            device: device.clone(),
        }
    }
}

impl Device {
    pub fn create_semaphore(&self, value: u64) -> Semaphore {
        let inner = unsafe { SemaphoreImpl::new_timeline(self.inner.clone(), value) };
        Semaphore { inner: Arc::new(inner) }
    }
    pub fn create_signal_semaphore(&self) -> Semaphore {
        let inner = unsafe { SemaphoreImpl::new_signal(self.inner.clone()) };
        Semaphore { inner: Arc::new(inner) }
    }
}

impl Drop for SemaphoreImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_semaphore(self.handle, None);
        }
    }
}
