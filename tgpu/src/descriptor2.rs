use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use ash::vk;

use crate::{Buffer, freelist::FreeList, raw::RawDevice};

pub struct DescriptorHandle {
    index: u32,
    ty: i32,
}

pub struct BindlessPool {
    pub inner: Arc<BindlessPoolImpl>,
}

pub struct BindlessPoolImpl {
    pub device: RawDevice,
    pub handle: vk::DescriptorPool,
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,

    pub allocations: HashMap<vk::DescriptorType, FreeList<()>>,
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
            allocations.insert(ty, freelist);
        });

        Self {
            device,
            handle: pool_handle,
            layout,
            set,
            allocations,
        }
    }

    // pub fn layout(&self) -> 

    pub fn allocate(&self, ty: vk::DescriptorType) -> DescriptorHandle {
        let list = self.allocations.get(&ty).expect("allocations");
        let index = list.insert(());
        let handle = DescriptorHandle {
            ty: ty.as_raw() as i32,
            index: index as u32,
        };
        handle
    }

    pub fn free(&self, handle: DescriptorHandle) {
        let ty = vk::DescriptorType::from_raw(handle.ty);
        let index = handle.index as usize;
        let list = self.allocations.get(&ty).expect("allocations");
        let _data = list.remove(index);
    }
}
