use parking_lot::Mutex;

use ash::vk;

use crate::{
    Buffer, DescriptorBinding, DescriptorPoolInfo, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutInfo, DescriptorType, DescriptorWrite, Device, ImageView, Sampler,
    ShaderStageFlags,
};

pub const BINDLESS_READ_BUFFER_BINDING: u32 = 0;
pub const BINDLESS_RW_BUFFER_BINDING: u32 = 1;
pub const BINDLESS_SAMPLED_IMAGE_BINDING: u32 = 2;
pub const BINDLESS_STORAGE_IMAGE_BINDING: u32 = 3;
pub const BINDLESS_SAMPLER_BINDING: u32 = 4;
pub const BINDLESS_UNIFORM_BUFFER_BINDING: u32 = 5;

#[derive(Debug, Default, Clone, Copy)]
pub struct BindlessInfo {
    pub max_read_buffers: u32,
    pub max_rw_buffers: u32,
    pub max_sampled_images: u32,
    pub max_storage_images: u32,
    pub max_samplers: u32,
    pub max_uniform_buffers: u32,
}

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReadBufferHandle(pub u32);

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RwBufferHandle(pub u32);

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SampledImageHandle(pub u32);

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StorageImageHandle(pub u32);

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SamplerHandle(pub u32);

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformBufferHandle(pub u32);

pub struct BindlessHeap {
    set: DescriptorSet,
    layout: DescriptorSetLayout,
    read_buffers: SlotAllocator,
    rw_buffers: SlotAllocator,
    sampled_images: SlotAllocator,
    storage_images: SlotAllocator,
    samplers: SlotAllocator,
    uniform_buffers: SlotAllocator,
}

#[derive(Debug)]
struct SlotAllocator {
    kind: &'static str,
    capacity: u32,
    inner: Mutex<SlotAllocatorInner>,
}

#[derive(Debug)]
struct SlotAllocatorInner {
    next: u32,
    free: Vec<u32>,
    used: Vec<bool>,
}

impl SlotAllocator {
    fn new(kind: &'static str, capacity: u32) -> Self {
        Self {
            kind,
            capacity,
            inner: Mutex::new(SlotAllocatorInner {
                next: 0,
                free: Vec::new(),
                used: vec![false; capacity as usize],
            }),
        }
    }

    fn allocate(&self) -> u32 {
        let mut inner = self.inner.lock();

        if let Some(slot) = inner.free.pop() {
            inner.used[slot as usize] = true;
            return slot;
        }

        assert!(
            inner.next < self.capacity,
            "Bindless {} heap exhausted (capacity: {})",
            self.kind,
            self.capacity
        );

        let slot = inner.next;
        inner.next += 1;
        inner.used[slot as usize] = true;
        slot
    }

    fn free(&self, slot: u32) {
        let mut inner = self.inner.lock();
        assert!(
            slot < self.capacity,
            "Bindless {} handle {} is out of range",
            self.kind,
            slot
        );
        assert!(
            inner.used[slot as usize],
            "Bindless {} handle {} is not allocated",
            self.kind, slot
        );

        inner.used[slot as usize] = false;
        inner.free.push(slot);
    }

    fn assert_allocated(&self, slot: u32) {
        let inner = self.inner.lock();
        assert!(
            slot < self.capacity,
            "Bindless {} handle {} is out of range",
            self.kind,
            slot
        );
        assert!(
            inner.used[slot as usize],
            "Bindless {} handle {} is not allocated",
            self.kind, slot
        );
    }
}

fn bindless_binding(binding: u32, ty: DescriptorType, count: u32) -> DescriptorBinding {
    DescriptorBinding {
        binding,
        ty,
        count: count.max(1),
        stages: ShaderStageFlags::ALL,
        flags: Some(
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        ),
    }
}

impl Device {
    pub fn create_bindless_heap(&self, info: &BindlessInfo) -> BindlessHeap {
        let bindings = [
            bindless_binding(
                BINDLESS_READ_BUFFER_BINDING,
                DescriptorType::StorageBuffer,
                info.max_read_buffers,
            ),
            bindless_binding(
                BINDLESS_RW_BUFFER_BINDING,
                DescriptorType::StorageBuffer,
                info.max_rw_buffers,
            ),
            bindless_binding(
                BINDLESS_SAMPLED_IMAGE_BINDING,
                DescriptorType::SampledImage,
                info.max_sampled_images,
            ),
            bindless_binding(
                BINDLESS_STORAGE_IMAGE_BINDING,
                DescriptorType::StorageImage,
                info.max_storage_images,
            ),
            bindless_binding(
                BINDLESS_SAMPLER_BINDING,
                DescriptorType::Sampler,
                info.max_samplers,
            ),
            bindless_binding(
                BINDLESS_UNIFORM_BUFFER_BINDING,
                DescriptorType::UniformBuffer,
                info.max_uniform_buffers,
            ),
        ];

        let layout = self.create_descriptor_set_layout(&DescriptorSetLayoutInfo {
            bindings: &bindings,
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            label: None,
        });

        let pool = self.create_descriptor_pool(&DescriptorPoolInfo {
            max_sets: 1,
            layouts: &[&layout],
            flags: vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            label: None,
        });

        let set = self.create_descriptor_set(pool, &layout);

        BindlessHeap {
            set,
            layout,
            read_buffers: SlotAllocator::new("read buffer", info.max_read_buffers),
            rw_buffers: SlotAllocator::new("rw buffer", info.max_rw_buffers),
            sampled_images: SlotAllocator::new("sampled image", info.max_sampled_images),
            storage_images: SlotAllocator::new("storage image", info.max_storage_images),
            samplers: SlotAllocator::new("sampler", info.max_samplers),
            uniform_buffers: SlotAllocator::new("uniform buffer", info.max_uniform_buffers),
        }
    }
}

impl BindlessHeap {
    pub fn layout(&self) -> &DescriptorSetLayout {
        &self.layout
    }

    pub fn descriptor_set(&self) -> &DescriptorSet {
        &self.set
    }

    pub fn add_read_buffer(&self, buffer: &Buffer) -> ReadBufferHandle {
        let handle = ReadBufferHandle(self.read_buffers.allocate());
        self.update_read_buffer(handle, buffer);
        handle
    }

    pub fn update_read_buffer(&self, handle: ReadBufferHandle, buffer: &Buffer) {
        self.read_buffers.assert_allocated(handle.0);
        self.set.write(&[DescriptorWrite::StorageBuffer {
            binding: BINDLESS_READ_BUFFER_BINDING,
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
            array_element: Some(handle.0),
        }]);
    }

    pub fn free_read_buffer(&self, handle: ReadBufferHandle) {
        self.read_buffers.free(handle.0);
    }

    pub fn add_rw_buffer(&self, buffer: &Buffer) -> RwBufferHandle {
        let handle = RwBufferHandle(self.rw_buffers.allocate());
        self.update_rw_buffer(handle, buffer);
        handle
    }

    pub fn update_rw_buffer(&self, handle: RwBufferHandle, buffer: &Buffer) {
        self.rw_buffers.assert_allocated(handle.0);
        self.set.write(&[DescriptorWrite::StorageBuffer {
            binding: BINDLESS_RW_BUFFER_BINDING,
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
            array_element: Some(handle.0),
        }]);
    }

    pub fn free_rw_buffer(&self, handle: RwBufferHandle) {
        self.rw_buffers.free(handle.0);
    }

    pub fn add_sampled_image(
        &self,
        view: &ImageView,
        layout: vk::ImageLayout,
    ) -> SampledImageHandle {
        let handle = SampledImageHandle(self.sampled_images.allocate());
        self.update_sampled_image(handle, view, layout);
        handle
    }

    pub fn update_sampled_image(
        &self,
        handle: SampledImageHandle,
        view: &ImageView,
        layout: vk::ImageLayout,
    ) {
        self.sampled_images.assert_allocated(handle.0);
        self.set.write(&[DescriptorWrite::SampledImage {
            binding: BINDLESS_SAMPLED_IMAGE_BINDING,
            image_view: view,
            image_layout: layout,
            array_element: Some(handle.0),
        }]);
    }

    pub fn free_sampled_image(&self, handle: SampledImageHandle) {
        self.sampled_images.free(handle.0);
    }

    pub fn add_storage_image(
        &self,
        view: &ImageView,
        layout: vk::ImageLayout,
    ) -> StorageImageHandle {
        let handle = StorageImageHandle(self.storage_images.allocate());
        self.update_storage_image(handle, view, layout);
        handle
    }

    pub fn update_storage_image(
        &self,
        handle: StorageImageHandle,
        view: &ImageView,
        layout: vk::ImageLayout,
    ) {
        self.storage_images.assert_allocated(handle.0);
        self.set.write(&[DescriptorWrite::StorageImage {
            binding: BINDLESS_STORAGE_IMAGE_BINDING,
            image_view: view,
            image_layout: layout,
            array_element: Some(handle.0),
        }]);
    }

    pub fn free_storage_image(&self, handle: StorageImageHandle) {
        self.storage_images.free(handle.0);
    }

    pub fn add_sampler(&self, sampler: &Sampler) -> SamplerHandle {
        let handle = SamplerHandle(self.samplers.allocate());
        self.update_sampler(handle, sampler);
        handle
    }

    pub fn update_sampler(&self, handle: SamplerHandle, sampler: &Sampler) {
        self.samplers.assert_allocated(handle.0);
        self.set.write(&[DescriptorWrite::Sampler {
            binding: BINDLESS_SAMPLER_BINDING,
            sampler,
            array_element: Some(handle.0),
        }]);
    }

    pub fn free_sampler(&self, handle: SamplerHandle) {
        self.samplers.free(handle.0);
    }

    pub fn add_uniform_buffer(
        &self,
        buffer: &Buffer,
        range: vk::DeviceSize,
    ) -> UniformBufferHandle {
        let handle = UniformBufferHandle(self.uniform_buffers.allocate());
        self.update_uniform_buffer(handle, buffer, range);
        handle
    }

    pub fn update_uniform_buffer(
        &self,
        handle: UniformBufferHandle,
        buffer: &Buffer,
        range: vk::DeviceSize,
    ) {
        self.uniform_buffers.assert_allocated(handle.0);
        self.set.write(&[DescriptorWrite::UniformBuffer {
            binding: BINDLESS_UNIFORM_BUFFER_BINDING,
            buffer,
            offset: 0,
            range,
            array_element: Some(handle.0),
        }]);
    }

    pub fn free_uniform_buffer(&self, handle: UniformBufferHandle) {
        self.uniform_buffers.free(handle.0);
    }
}
