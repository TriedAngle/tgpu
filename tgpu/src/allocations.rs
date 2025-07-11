use std::{mem::ManuallyDrop, sync::Arc};

pub struct Allocation {
    pub handle: vkm::Allocation,
    pub allocator: Arc<ManuallyDrop<vkm::Allocator>>,
}
