use std::{sync::Arc, time::Duration};

use ash::vk;

use crate::{
    Device,
    raw::{DeviceImpl, RawDevice},
};

#[derive(Debug, Clone)]
pub struct Semaphore {
    pub inner: SemaphoreImpl,
}

// pub struct Fence {
// }

#[derive(Debug, Clone)]
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

    pub unsafe fn new_timeline(device: Arc<DeviceImpl>, value: u64) -> Self {
        let handle = unsafe { device.create_timeline_semaphore(value) };
        SemaphoreImpl {
            handle,
            device: device.clone(),
        }
    }
}

impl Device {
    pub fn create_semaphore(&self, value: u64) -> Semaphore {
        let inner = unsafe { SemaphoreImpl::new_timeline(self.inner.clone(), value) };
        Semaphore { inner }
    }
}

impl Drop for SemaphoreImpl {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_semaphore(self.handle, None);
        }
    }
}
