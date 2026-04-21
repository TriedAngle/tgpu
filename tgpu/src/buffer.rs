use ash::vk;
use std::{cell::UnsafeCell, ptr, sync::Arc};
use vkm::Alloc;

use crate::{Device, GPUError, HostAccess, Label, MemoryPreset, raw::RawDevice};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct BufferUses: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        const INDEX = 1 << 2;
        const VERTEX = 1 << 3;
        const UNIFORM = 1 << 4;
        const STORAGE = 1 << 5;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct BufferUsage: u32 {
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
        if usage.contains(BufferUsage::HOST_VISIBLE) || usage.contains(BufferUsage::HOST) {
            if usage.contains(BufferUsage::RANDOM_ACCESS) || usage.contains(BufferUsage::MAP_READ) {
                flags |= vkm::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            } else if usage.contains(BufferUsage::MAP_WRITE) {
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

#[derive(Debug, Clone)]
pub struct BufferDesc<'a> {
    pub size: usize,
    pub usage: BufferUses,
    pub memory: MemoryPreset,
    pub host_access: HostAccess,
    pub sharing: vk::SharingMode,
    pub label: Option<Label<'a>>,
}

impl Default for BufferDesc<'_> {
    fn default() -> Self {
        Self {
            size: 0,
            usage: BufferUses::empty(),
            memory: MemoryPreset::GpuOnly,
            host_access: HostAccess::None,
            sharing: vk::SharingMode::EXCLUSIVE,
            label: None,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BufferInfo<'a> {
    pub size: usize,
    pub usage: BufferUsage,
    pub label: Option<Label<'a>>,
}

#[derive(Debug)]
pub struct BufferImpl {
    pub handle: vk::Buffer,
    pub(crate) allocation: UnsafeCell<vkm::Allocation>,
    pub(crate) usage: BufferUsage,
    pub(crate) device: RawDevice,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub inner: Arc<BufferImpl>,
    pub size: usize,
    pub uses: BufferUses,
    pub memory: MemoryPreset,
    pub host_access: HostAccess,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct BufferAccessTransition {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl BufferAccessTransition {
    pub const NONE: Self = Self::new(vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
    pub const TRANSFER_SRC: Self = Self::new(
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_READ,
    );
    pub const TRANSFER_DST: Self = Self::new(
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
    );
    pub const INDIRECT: Self = Self::new(
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::INDIRECT_COMMAND_READ,
    );

    pub const fn new(stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Self {
        Self { stage, access }
    }

    pub fn compute_storage_read() -> Self {
        Self::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )
    }

    pub fn compute_storage_write() -> Self {
        Self::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
        )
    }

    pub fn compute_storage_read_write() -> Self {
        Self::new(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BufferTransition<'a> {
    pub from: BufferAccessTransition,
    pub to: BufferAccessTransition,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub queue: Option<(&'a crate::Queue, &'a crate::Queue)>,
    pub dependency: vk::DependencyFlags,
}

impl Default for BufferTransition<'_> {
    fn default() -> Self {
        Self {
            from: BufferAccessTransition::default(),
            to: BufferAccessTransition::default(),
            offset: 0,
            size: vk::WHOLE_SIZE,
            queue: None,
            dependency: vk::DependencyFlags::empty(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CopyBufferInfo<'a> {
    pub src: &'a Buffer,
    pub dst: &'a Buffer,
    pub regions: &'a [vk::BufferCopy],
}

impl Device {
    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<Buffer, GPUError> {
        if desc.size == 0 {
            return Err(GPUError::Validation(
                "buffer size must be greater than zero",
            ));
        }
        if desc.usage.is_empty() {
            return Err(GPUError::Validation("buffer usage must not be empty"));
        }
        if desc.memory == MemoryPreset::TransientAttachment {
            return Err(GPUError::Validation(
                "TransientAttachment memory is only valid for images",
            ));
        }

        let host_access = match (desc.memory, desc.host_access) {
            (MemoryPreset::Upload, HostAccess::None) => HostAccess::WriteSequential,
            (MemoryPreset::Readback, HostAccess::None) => HostAccess::ReadRandom,
            (MemoryPreset::Dynamic, HostAccess::None) => HostAccess::WriteSequential,
            (_, host_access) => host_access,
        };

        if desc.memory == MemoryPreset::GpuOnly && host_access != HostAccess::None {
            return Err(GPUError::Validation(
                "GpuOnly buffers cannot request host access; use Upload, Readback, or Dynamic",
            ));
        }

        let mut usage: BufferUsage = desc.usage.into();

        match desc.memory {
            MemoryPreset::GpuOnly => usage |= BufferUsage::DEVICE,
            MemoryPreset::Upload => usage |= BufferUsage::HOST | BufferUsage::HOST_VISIBLE,
            MemoryPreset::Readback => {
                usage |= BufferUsage::HOST | BufferUsage::HOST_VISIBLE | BufferUsage::CACHED
            }
            MemoryPreset::Dynamic => usage |= BufferUsage::DEVICE | BufferUsage::HOST_VISIBLE,
            MemoryPreset::TransientAttachment => unreachable!(),
        }

        match host_access {
            HostAccess::None => {}
            HostAccess::WriteSequential => usage |= BufferUsage::MAP_WRITE,
            HostAccess::ReadRandom => usage |= BufferUsage::MAP_READ | BufferUsage::RANDOM_ACCESS,
            HostAccess::ReadWriteRandom => {
                usage |= BufferUsage::MAP_READ | BufferUsage::MAP_WRITE | BufferUsage::RANDOM_ACCESS
            }
        }

        if desc.sharing == vk::SharingMode::CONCURRENT {
            usage |= BufferUsage::SHARE;
        }

        let info = BufferInfo {
            size: desc.size,
            usage,
            label: desc.label.clone(),
        };
        let inner = BufferImpl::new_with_allocation(
            self.inner.clone(),
            &info,
            allocation_create_info(desc.memory, host_access),
        )?;

        Ok(Buffer {
            inner: Arc::new(inner),
            size: info.size,
            uses: desc.usage,
            memory: desc.memory,
            host_access,
        })
    }

    pub fn create_buffer_with(&self, desc: &BufferDesc<'_>) -> Result<Buffer, GPUError> {
        self.create_buffer(desc)
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
        debug_assert!(
            self.inner.usage.contains(BufferUsage::MAP_WRITE),
            "Writing requires MAP_WRITE"
        );
        let size = data.len();
        unsafe {
            let mapping = self.map(offset);

            ptr::copy(data.as_ptr(), mapping, size);
            self.inner.flush(offset, size);

            self.unmap();
        }
    }

    pub fn read(&self, buffer: &mut [u8], offset: usize, size: usize) {
        debug_assert!(
            self.inner.usage.contains(BufferUsage::MAP_READ),
            "Reading requires MAP_READ"
        );
        unsafe {
            let mapping = self.map(offset);
            self.inner.invalidate(offset, size);
            std::ptr::copy_nonoverlapping(mapping, buffer.as_mut_ptr(), size);
            self.unmap();
        }
    }

    pub fn write_slice<T: bytemuck::Pod>(&self, data: &[T]) {
        self.write(bytemuck::cast_slice(data), 0);
    }

    pub fn read_slice<T: bytemuck::Pod>(&self, data: &mut [T]) {
        let size = std::mem::size_of_val(data);
        self.read(bytemuck::cast_slice_mut(data), 0, size);
    }
}

impl From<BufferUses> for BufferUsage {
    fn from(usage: BufferUses) -> Self {
        let mut raw = BufferUsage::empty();
        if usage.contains(BufferUses::COPY_SRC) {
            raw |= BufferUsage::COPY_SRC;
        }
        if usage.contains(BufferUses::COPY_DST) {
            raw |= BufferUsage::COPY_DST;
        }
        if usage.contains(BufferUses::INDEX) {
            raw |= BufferUsage::INDEX;
        }
        if usage.contains(BufferUses::VERTEX) {
            raw |= BufferUsage::VERTEX;
        }
        if usage.contains(BufferUses::UNIFORM) {
            raw |= BufferUsage::UNIFORM;
        }
        if usage.contains(BufferUses::STORAGE) {
            raw |= BufferUsage::STORAGE;
        }
        raw
    }
}

fn allocation_create_info(
    memory: MemoryPreset,
    host_access: HostAccess,
) -> vkm::AllocationCreateInfo {
    let usage = match memory {
        MemoryPreset::GpuOnly => vkm::MemoryUsage::AutoPreferDevice,
        MemoryPreset::Upload | MemoryPreset::Readback => vkm::MemoryUsage::AutoPreferHost,
        MemoryPreset::Dynamic => vkm::MemoryUsage::AutoPreferDevice,
        MemoryPreset::TransientAttachment => vkm::MemoryUsage::GpuLazy,
    };

    let mut flags = vkm::AllocationCreateFlags::empty();
    let mut preferred_flags = vk::MemoryPropertyFlags::empty();

    match host_access {
        HostAccess::None => {}
        HostAccess::WriteSequential => {
            flags |= vkm::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
        }
        HostAccess::ReadRandom | HostAccess::ReadWriteRandom => {
            flags |= vkm::AllocationCreateFlags::HOST_ACCESS_RANDOM;
            preferred_flags |= vk::MemoryPropertyFlags::HOST_CACHED;
        }
    }

    vkm::AllocationCreateInfo {
        usage,
        flags,
        preferred_flags,
        ..Default::default()
    }
}

impl BufferImpl {
    pub(crate) fn new_with_allocation(
        device: RawDevice,
        info: &BufferInfo<'_>,
        create_info: vkm::AllocationCreateInfo,
    ) -> Result<BufferImpl, GPUError> {
        let sharing = if info.usage.contains(BufferUsage::SHARE) {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let buffer_info = vk::BufferCreateInfo::default()
            .size(info.size as u64)
            .sharing_mode(sharing)
            .usage(info.usage.into());

        let (handle, allocation) =
            unsafe { device.allocator.create_buffer(&buffer_info, &create_info)? };

        if let Some(label) = &info.label {
            unsafe { device.attach_label(handle, label) };
        }

        Ok(BufferImpl {
            handle,
            allocation: UnsafeCell::new(allocation),
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

    pub unsafe fn flush(&self, offset: usize, size: usize) {
        let allocation = unsafe { &*self.allocation.get() };
        self.device
            .allocator
            .flush_allocation(allocation, offset as vk::DeviceSize, size as vk::DeviceSize)
            .expect("Flush Buffer Allocation");
    }

    pub unsafe fn invalidate(&self, offset: usize, size: usize) {
        let allocation = unsafe { &*self.allocation.get() };
        self.device
            .allocator
            .invalidate_allocation(allocation, offset as vk::DeviceSize, size as vk::DeviceSize)
            .expect("Invalidate Buffer Allocation");
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
