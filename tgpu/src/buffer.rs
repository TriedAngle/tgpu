use ash::vk;
use std::{cell::UnsafeCell, ptr, sync::Arc};
use vkm::Alloc;

use bitflags;

use crate::{Device, GPUError, Label, raw::RawDevice};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct BufferUsage: u32 {
        const MAP_READ = 1 << 0;
        const MAP_WRITE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const INDEX = 1 << 4;
        const VERTEX = 1 << 5;
        const UNIFORM = 1 << 6;
        const STORAGE = 1 << 7;
        const QUERY = 1 << 8;
        const SHARE = 1 << 12;

        const DEVICE = 1 << 16;
        const HOST = 1 << 17;
        const LAZY = 1 << 18;
        const HOST_VISIBLE = 1 << 19;
        const COHERENT = 1 << 20;
        const CACHED = 1 << 21;
        const RANDOM_ACCESS = 1 << 22;
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(usage: BufferUsage) -> Self {
        let mut vk_usage = vk::BufferUsageFlags::empty();
        if usage.contains(BufferUsage::COPY_SRC) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if usage.contains(BufferUsage::COPY_DST) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if usage.contains(BufferUsage::INDEX) {
            vk_usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if usage.contains(BufferUsage::VERTEX) {
            vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if usage.contains(BufferUsage::UNIFORM) {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if usage.contains(BufferUsage::STORAGE) {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        vk_usage
    }
}

impl From<BufferUsage> for vkm::MemoryUsage {
    fn from(usage: BufferUsage) -> Self {
        if usage.contains(BufferUsage::DEVICE) {
            vkm::MemoryUsage::AutoPreferDevice
        } else if usage.contains(BufferUsage::HOST) {
            vkm::MemoryUsage::AutoPreferHost
        } else if usage.contains(BufferUsage::LAZY) {
            vkm::MemoryUsage::GpuLazy
        } else {
            unimplemented!("Auto not implemented yet")
            // vkm::MemoryUsage::Auto
        }
    }
}

impl From<BufferUsage> for vk::MemoryPropertyFlags {
    fn from(usage: BufferUsage) -> Self {
        let mut vk_usage = vk::MemoryPropertyFlags::empty();
        if usage.contains(BufferUsage::DEVICE) {
            vk_usage |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
        }
        if usage.contains(BufferUsage::HOST_VISIBLE) {
            vk_usage |= vk::MemoryPropertyFlags::HOST_VISIBLE;
        }
        if usage.contains(BufferUsage::COHERENT) {
            vk_usage |= vk::MemoryPropertyFlags::HOST_COHERENT;
        }
        if usage.contains(BufferUsage::CACHED) {
            vk_usage |= vk::MemoryPropertyFlags::HOST_CACHED;
        }
        if usage.contains(BufferUsage::LAZY) {
            vk_usage |= vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
        }
        vk_usage
    }
}

impl From<BufferUsage> for vkm::AllocationCreateFlags {
    fn from(usage: BufferUsage) -> Self {
        let mut flags = vkm::AllocationCreateFlags::empty();
        if (usage.contains(BufferUsage::HOST_VISIBLE) || usage.contains(BufferUsage::HOST))
            && usage.contains(BufferUsage::MAP_WRITE)
        {
            if usage.contains(BufferUsage::RANDOM_ACCESS) {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            } else {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
            }
        }
        flags
    }
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Default)]
pub struct BufferInfo<'a> {
    pub size: u64,
    pub usage: BufferUsage,
    pub label: Option<Label<'a>>,
}

#[derive(Debug)]
pub struct BufferImpl {
    pub handle: vk::Buffer,
    pub allocation: UnsafeCell<vkm::Allocation>,
    pub size: u64,
    pub usage: BufferUsage,
    pub device: RawDevice,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub inner: Arc<BufferImpl>,
    pub size: u64,
    pub usage: BufferUsage,
}

impl Device {
    pub fn create_buffer(&self, info: &BufferInfo<'_>) -> Result<Buffer, GPUError> {
        let (size, usage) = (info.size, info.usage);
        let inner = BufferImpl::new(self.inner.clone(), info)?;
        let buffer = Buffer {
            inner: Arc::new(inner),
            size,
            usage,
        };
        Ok(buffer)
    }
}

impl Buffer {
    pub fn map(&self, offset: usize) -> *mut u8 {
        unsafe { self.inner.map(offset) }
    }

    pub fn unmap(&self) {
        unsafe { self.inner.unmap() };
    }

    pub fn write(&self, data: &[u8], offset: usize) {
        let size = data.len();
        unsafe {
            let mapping = self.map(offset);

            ptr::copy(data.as_ptr(), mapping, size);

            self.unmap();
        }
    }

    pub fn read(&self, buffer: &mut [u8], offset: usize, size: usize) {
        unsafe {
            let mapping = self.map(offset);
            std::ptr::copy_nonoverlapping(mapping, buffer.as_mut_ptr(), size);
            self.unmap();
        }
    }
}

impl BufferImpl {
    pub fn new(device: RawDevice, info: &BufferInfo<'_>) -> Result<BufferImpl, GPUError> {
        let sharing = if info.usage.contains(BufferUsage::SHARE) {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let buffer_info = vk::BufferCreateInfo::default()
            .size(info.size)
            .sharing_mode(sharing)
            .usage(info.usage.into());

        let create_info = vkm::AllocationCreateInfo {
            usage: info.usage.into(),
            flags: info.usage.into(),
            required_flags: info.usage.into(),
            ..Default::default()
        };

        let (handle, allocation) =
            unsafe { device.allocator.create_buffer(&buffer_info, &create_info)? };

        if let Some(label) = &info.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(BufferImpl {
            handle,
            allocation: UnsafeCell::new(allocation),
            size: info.size,
            usage: info.usage,
            device,
        })
    }

    pub unsafe fn map(&self, offset: usize) -> *mut u8 {
        let allocation = unsafe { self.allocation.get().as_mut().unwrap() };
        unsafe {
            self.device
                .allocator
                .map_memory(allocation)
                .unwrap()
                .add(offset)
        }
    }

    pub unsafe fn unmap(&self) {
        let allocation = unsafe { self.allocation.get().as_mut().unwrap() };
        unsafe { self.device.allocator.unmap_memory(allocation) };
    }
}

impl Drop for BufferImpl {
    fn drop(&mut self) {
        let allocation = self.allocation.get_mut();
        unsafe {
            self.device
                .allocator
                .destroy_buffer(self.handle, allocation);
        }
    }
}
