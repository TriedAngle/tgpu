use std::{collections::HashMap, sync::Arc};

use ash::vk;

use crate::{freelist::FreeList, raw::RawDevice};

pub struct DescriptorHandleRaw {
    index: u32,
    ty: DescriptorType,
}

pub struct BindlessPool {
    pub inner: Arc<BindlessPoolImpl>,
}

// TODO: we should probably use a normal "pool" abstraction and just implement bindless pool on top
// of that.
pub struct BindlessPoolImpl {
    pub device: RawDevice,
    pub handle: vk::DescriptorPool,
    pub layout: Layout,
    pub set: vk::DescriptorSet,

    pub allocations: HashMap<DescriptorType, FreeList<()>>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum DescriptorType {
    Sampler,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    UniformDynamicBuffer,
    StorageDynamicBuffer,
    InputAttachment,
    Raw(i32),
}

#[derive(Debug, Copy, Clone)]
pub struct DescriptorBinding {
    pub binding: u32,
    pub ty: DescriptorType,
    pub count: u32,
    pub stages: vk::ShaderStageFlags,
    pub flags: Option<vk::DescriptorBindingFlags>,
}

pub struct Layout {
    pub handle: vk::DescriptorSetLayout,
    pub bindings: Vec<DescriptorBinding>,
    pub device: RawDevice,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BindlessPoolCreateInfo {
    pub max_samplers: usize,
    pub max_sampler_image_combined: usize,
    pub max_sampled_images: usize,
    pub max_storage_images: usize,
    pub max_uniform_texel_buffers: usize,
    pub max_storage_texel_buffers: usize,
    pub max_uniform_buffers: usize,
    pub max_storage_buffers: usize,
    pub max_dynamic_uniform_buffers: usize,
    pub max_dynamic_storage_buffers: usize,
    pub max_input_attachments: usize,
}

impl BindlessPoolImpl {
    pub fn new(device: RawDevice, info: BindlessPoolCreateInfo) -> Self {
        let binding_types_count = [
            (vk::DescriptorType::SAMPLER, info.max_samplers),
            (
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                info.max_sampler_image_combined,
            ),
            (vk::DescriptorType::SAMPLED_IMAGE, info.max_sampled_images),
            (vk::DescriptorType::STORAGE_IMAGE, info.max_storage_images),
            (
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                info.max_uniform_texel_buffers,
            ),
            (
                vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                info.max_storage_texel_buffers,
            ),
            (vk::DescriptorType::UNIFORM_BUFFER, info.max_uniform_buffers),
            (vk::DescriptorType::STORAGE_BUFFER, info.max_storage_buffers),
            (
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                info.max_dynamic_uniform_buffers,
            ),
            (
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                info.max_dynamic_storage_buffers,
            ),
            (
                vk::DescriptorType::INPUT_ATTACHMENT,
                info.max_input_attachments,
            ),
        ];

        let mut binding_idx = 0;
        let bindings = binding_types_count.map(|(ty, max)| {
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(binding_idx)
                .descriptor_type(ty)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .descriptor_count(max as u32);

            binding_idx += 1;
            binding
        });

        let binding_flags = bindings.map(|_| {
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
        });

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags_info);

        let layout = unsafe {
            device
                .handle
                .create_descriptor_set_layout(&layout_info, None)
                .expect("Create Descriptor Layout")
        };

        let pool_sizes = binding_types_count.map(|(ty, max)| {
            vk::DescriptorPoolSize::default()
                .ty(ty)
                .descriptor_count(max as u32)
        });

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let pool_handle =
            unsafe { device.handle.create_descriptor_pool(&pool_info, None) }.expect("Create Pool");

        let layouts = [layout];
        let set_allocation_info = vk::DescriptorSetAllocateInfo::default()
            .set_layouts(&layouts)
            .descriptor_pool(pool_handle);

        let set = unsafe { device.handle.allocate_descriptor_sets(&set_allocation_info) }
            .expect("Create Set")[0];

        let mut allocations = HashMap::new();

        binding_types_count.iter().for_each(|&(ty, _count)| {
            let freelist = FreeList::new();
            allocations.insert(ty.into(), freelist);
        });

        let layout_bindings = bindings
            .iter()
            .zip(&binding_flags)
            .map(|(binding, flags)| DescriptorBinding {
                binding: binding.binding,
                ty: binding.descriptor_type.into(),
                count: binding.descriptor_count,
                stages: binding.stage_flags,
                flags: Some(*flags),
            })
            .collect();

        let layout = Layout {
            handle: layout,
            bindings: layout_bindings,
            device: device.clone(),
        };

        Self {
            device,
            handle: pool_handle,
            layout,
            set,
            allocations,
        }
    }

    pub fn layout<'a>(&'a self) -> &'a Layout {
        &self.layout
    }

    pub fn allocate(&self, ty: DescriptorType) -> DescriptorHandleRaw {
        let list = self.allocations.get(&ty).expect("allocations");
        let index = list.insert(());
        let handle = DescriptorHandleRaw {
            ty: ty,
            index: index as u32,
        };
        handle
    }

    pub fn free(&self, handle: DescriptorHandleRaw) {
        let ty = handle.ty;
        let index = handle.index as usize;
        let list = self.allocations.get(&ty).expect("allocations");
        let _data = list.remove(index);
    }
}
impl From<DescriptorType> for vk::DescriptorType {
    fn from(value: DescriptorType) -> Self {
        match value {
            DescriptorType::Sampler => vk::DescriptorType::SAMPLER,
            DescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            DescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            DescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            DescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            DescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            DescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            DescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorType::UniformDynamicBuffer => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            DescriptorType::StorageDynamicBuffer => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            DescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
            DescriptorType::Raw(raw) => vk::DescriptorType::from_raw(raw),
        }
    }
}

impl From<vk::DescriptorType> for DescriptorType {
    fn from(value: vk::DescriptorType) -> Self {
        match value {
            vk::DescriptorType::SAMPLER => DescriptorType::Sampler,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER => DescriptorType::CombinedImageSampler,
            vk::DescriptorType::SAMPLED_IMAGE => DescriptorType::SampledImage,
            vk::DescriptorType::STORAGE_IMAGE => DescriptorType::StorageImage,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER => DescriptorType::UniformTexelBuffer,
            vk::DescriptorType::STORAGE_TEXEL_BUFFER => DescriptorType::StorageTexelBuffer,
            vk::DescriptorType::UNIFORM_BUFFER => DescriptorType::UniformBuffer,
            vk::DescriptorType::STORAGE_BUFFER => DescriptorType::StorageBuffer,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => DescriptorType::UniformDynamicBuffer,
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => DescriptorType::StorageDynamicBuffer,
            vk::DescriptorType::INPUT_ATTACHMENT => DescriptorType::InputAttachment,
            other => DescriptorType::Raw(other.as_raw()),
        }
    }
}
