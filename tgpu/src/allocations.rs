use std::sync::Arc;

pub struct Allocation {
    pub handle: vkm::Allocation,
    pub allocator: Arc<vkm::Allocator>,
}
