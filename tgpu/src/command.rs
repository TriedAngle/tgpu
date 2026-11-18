use ash::vk;
use parking_lot::Mutex;
use std::{
    cell::{Cell, RefCell, UnsafeCell},
    collections::HashMap,
    ops,
    rc::Rc,
    thread::{self, ThreadId},
};

use crate::{
    BlitImageInfo, Buffer, BufferTransition, ComputePipeline, CopyBufferInfo,
    CopyBufferToImageInfo, CopyImageInfo, DescriptorSet, GPUError, Image, ImageLayout,
    ImageTransition, ImageView, Queue, Rect, RenderPipeline, Semaphore, Viewport,
    raw::{ComputePipelineImpl, QueueImpl, RawDevice, RenderPipelineImpl},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IndexType {
    U16,
    U32,
}

impl From<IndexType> for vk::IndexType {
    fn from(value: IndexType) -> Self {
        match value {
            IndexType::U16 => Self::UINT16,
            IndexType::U32 => Self::UINT32,
        }
    }
}

bitflags::bitflags! {
    /// A subset of the legacy `VkPipelineStageFlags` bits used for submission
    /// dependencies. Use [`PipelineStageFlags::from_bits_retain`] for anything
    /// not named here.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct PipelineStageFlags: u32 {
        const TOP_OF_PIPE = 1 << 0;
        const DRAW_INDIRECT = 1 << 1;
        const VERTEX_INPUT = 1 << 2;
        const EARLY_FRAGMENT_TESTS = 1 << 3;
        const LATE_FRAGMENT_TESTS = 1 << 4;
        const FRAGMENT_SHADER = 1 << 7;
        const COLOR_ATTACHMENT_OUTPUT = 1 << 8;
        const COMPUTE_SHADER = 1 << 11;
        const TRANSFER = 1 << 12;
        const BOTTOM_OF_PIPE = 1 << 13;
        const ALL_GRAPHICS = 1 << 15;
        const ALL_COMMANDS = 1 << 16;
    }
}

impl From<PipelineStageFlags> for vk::PipelineStageFlags {
    fn from(value: PipelineStageFlags) -> Self {
        Self::from_raw(value.bits())
    }
}

impl From<vk::PipelineStageFlags> for PipelineStageFlags {
    fn from(value: vk::PipelineStageFlags) -> Self {
        Self::from_bits_retain(value.as_raw())
    }
}

#[derive(Debug)]
pub struct CommandPools {
    pub device: RawDevice,
    pub pools: Mutex<HashMap<ThreadId, Rc<ThreadCommandPool>>>,
}

unsafe impl Send for CommandPools {}
unsafe impl Sync for CommandPools {}

#[derive(Debug)]
pub struct ThreadCommandPool {
    pub handle: vk::CommandPool,
    pub device: RawDevice,
    pub ready: RefCell<Vec<CommandBufferImpl>>,
    pub dropped: RefCell<Vec<DroppedCommandBuffer>>,
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub inner: CommandBufferImpl,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommandBufferImpl {
    pub handle: vk::CommandBuffer,
    pub submission: Rc<Cell<u64>>,
}

#[derive(Debug, Copy, Clone)]
pub struct DroppedCommandBuffer {
    pub handle: vk::CommandBuffer,
    pub submission: u64,
}

#[derive(Debug)]
pub struct CommandRecorder {
    pub inner: Rc<UnsafeCell<CommandRecorderImpl>>,
}

#[derive(Debug, Clone)]
pub struct CommandRecorderImpl {
    pub buffer: CommandBufferImpl,
    pub pool: Rc<ThreadCommandPool>,
    pub device: RawDevice,
}

#[derive(Debug, Copy, Clone)]
pub enum AttachmentLoadOp {
    Load,
    Clear,
    DontCare,
}

impl From<AttachmentLoadOp> for vk::AttachmentLoadOp {
    fn from(value: AttachmentLoadOp) -> Self {
        match value {
            AttachmentLoadOp::Load => vk::AttachmentLoadOp::LOAD,
            AttachmentLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            AttachmentLoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum AttachmentStoreOp {
    Store,
    DontCare,
}

impl From<AttachmentStoreOp> for vk::AttachmentStoreOp {
    fn from(value: AttachmentStoreOp) -> Self {
        match value {
            AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
            AttachmentStoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ClearValue {
    Color([f32; 4]),
    IntColor([i32; 4]),
    UintColor([u32; 4]),
    DepthStencil { depth: f32, stencil: u32 },
}

impl From<ClearValue> for vk::ClearValue {
    fn from(value: ClearValue) -> Self {
        match value {
            ClearValue::Color(float32) => vk::ClearValue {
                color: vk::ClearColorValue { float32 },
            },
            ClearValue::IntColor(int32) => vk::ClearValue {
                color: vk::ClearColorValue { int32 },
            },
            ClearValue::UintColor(uint32) => vk::ClearValue {
                color: vk::ClearColorValue { uint32 },
            },
            ClearValue::DepthStencil { depth, stencil } => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RenderAttachment<'a> {
    pub view: &'a ImageView,
    pub layout: ImageLayout,
    pub load: AttachmentLoadOp,
    pub store: AttachmentStoreOp,
    pub clear: Option<ClearValue>,
}

impl<'a> RenderAttachment<'a> {
    pub fn new(view: &'a ImageView, layout: ImageLayout) -> Self {
        Self {
            view,
            layout,
            load: AttachmentLoadOp::Load,
            store: AttachmentStoreOp::Store,
            clear: None,
        }
    }

    pub fn color(view: &'a ImageView) -> Self {
        Self::new(view, ImageLayout::Color)
    }

    pub fn depth(view: &'a ImageView) -> Self {
        Self::new(view, ImageLayout::DepthStencil)
    }

    pub fn layout(mut self, layout: ImageLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn load_op(mut self, load: AttachmentLoadOp) -> Self {
        self.load = load;
        self
    }

    pub fn store_op(mut self, store: AttachmentStoreOp) -> Self {
        self.store = store;
        self
    }

    pub fn load(mut self) -> Self {
        self.load = AttachmentLoadOp::Load;
        self.clear = None;
        self
    }

    pub fn clear_value(mut self, clear: ClearValue) -> Self {
        self.load = AttachmentLoadOp::Clear;
        self.clear = Some(clear);
        self
    }

    pub fn clear_color(self, color: [f32; 4]) -> Self {
        self.clear_value(ClearValue::Color(color))
    }

    pub fn clear_depth(self, depth: f32) -> Self {
        self.clear_depth_stencil(depth, 0)
    }

    pub fn clear_depth_stencil(self, depth: f32, stencil: u32) -> Self {
        self.clear_value(ClearValue::DepthStencil { depth, stencil })
    }

    pub fn dont_care_load(mut self) -> Self {
        self.load = AttachmentLoadOp::DontCare;
        self.clear = None;
        self
    }

    pub fn store(mut self) -> Self {
        self.store = AttachmentStoreOp::Store;
        self
    }

    pub fn dont_care_store(mut self) -> Self {
        self.store = AttachmentStoreOp::DontCare;
        self
    }

    fn as_vk(&self) -> vk::RenderingAttachmentInfo<'static> {
        let mut info = vk::RenderingAttachmentInfo::default()
            .image_view(self.view.inner.handle)
            .image_layout(self.layout.into())
            .load_op(self.load.into())
            .store_op(self.store.into());

        if let Some(clear) = self.clear {
            info = info.clear_value(clear.into());
        }

        info
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RenderInfo<'a> {
    pub area: Rect,
    pub layers: u32,
    pub colors: &'a [RenderAttachment<'a>],
    pub depth: Option<RenderAttachment<'a>>,
    pub stencil: Option<RenderAttachment<'a>>,
}

pub struct RenderRecorder<'a> {
    pub command_recorder: &'a mut CommandRecorder,
}

impl Default for RenderInfo<'_> {
    fn default() -> Self {
        Self {
            area: Rect::default(),
            layers: 1,
            colors: &[],
            depth: None,
            stencil: None,
        }
    }
}

impl CommandRecorder {
    pub fn finish(&mut self) -> CommandBuffer {
        let inner = unsafe { &mut *self.inner.get() };
        let buffer = unsafe { inner.finish() };
        CommandBuffer { inner: buffer }
    }

    pub fn image_transition(&mut self, image: &Image, transition: ImageTransition) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.image_transition(image.inner.handle, transition) };
    }

    pub fn buffer_transition(&mut self, buffer: &Buffer, transition: BufferTransition) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.buffer_transition(buffer.inner.handle, transition) };
    }

    pub fn bind_render_pipeline(&mut self, pipeline: &RenderPipeline) {
        let inner = unsafe { &mut *self.inner.get() };
        let inner_pipeline = &pipeline.inner;
        unsafe { inner.bind_render_pipeline(inner_pipeline) };
    }

    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let inner = unsafe { &mut *self.inner.get() };
        let inner_pipeline = &pipeline.inner;
        unsafe { inner.bind_compute_pipeline(inner_pipeline) };
    }

    pub fn begin_render<'a, F>(&'a mut self, info: &RenderInfo, render: F)
    where
        F: FnOnce(&mut RenderRecorder<'a>),
    {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.begin_render(info) };

        {
            let mut rec = RenderRecorder {
                command_recorder: self,
            };
            render(&mut rec)
        }
        unsafe { inner.end_rendering() };
    }

    pub fn copy_image(&mut self, info: &CopyImageInfo<'_>) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.copy_image(info) };
    }

    pub fn copy_buffer(&mut self, info: &CopyBufferInfo<'_>) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.copy_buffer(info) };
    }

    pub fn copy_buffer_to_image(&mut self, info: &CopyBufferToImageInfo<'_>) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.copy_buffer_to_image(info) };
    }

    pub fn blit_image(&mut self, info: &BlitImageInfo<'_>) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.blit_image(info) };
    }

    pub fn bind_compute_descriptor_set(
        &self,
        set: &DescriptorSet,
        pipeline: &ComputePipeline,
        index: u32,
        offsets: &[u32],
    ) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe {
            inner.bind_compute_descriptor_set(set, &pipeline.inner, index, offsets);
        }
    }

    pub fn push_compute_constants<T: bytemuck::Pod>(&mut self, pipeline: &ComputePipeline, pc: T) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe {
            inner.push_compute_constants(&pipeline.inner, pc);
        }
    }

    pub fn bind_render_descriptor_set(
        &self,
        set: &DescriptorSet,
        pipeline: &RenderPipeline,
        index: u32,
        offsets: &[u32],
    ) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe {
            inner.bind_render_descriptor_set(set, &pipeline.inner, index, offsets);
        }
    }

    pub fn push_render_constants<T: bytemuck::Pod>(&mut self, pipeline: &RenderPipeline, pc: T) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe {
            inner.push_render_constants(&pipeline.inner, pc);
        }
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let inner = unsafe { &mut *self.inner.get() };
        unsafe { inner.dispatch(x, y, z) };
    }
}

impl<'a> RenderRecorder<'a> {
    pub fn viewport(&mut self, viewport: Viewport) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.viewport(viewport) };
    }

    pub fn scissor(&mut self, scissor: Rect) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.scissor(scissor) };
    }

    pub fn draw(&mut self, vertex: ops::Range<u32>, instance: ops::Range<u32>) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.draw(vertex, instance) };
    }

    pub fn draw_indexed(
        &mut self,
        index: ops::Range<u32>,
        vertex_offset: i32,
        instance: ops::Range<u32>,
    ) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.draw_indexed(index, vertex_offset, instance) };
    }

    pub fn image_transition(&mut self, image: &Image, transition: ImageTransition) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.image_transition(image.inner.handle, transition) };
    }

    pub fn bind_render_pipeline(&mut self, pipeline: &RenderPipeline) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        let inner_pipeline = &pipeline.inner;
        unsafe { inner.bind_render_pipeline(inner_pipeline) };
    }

    pub fn bind_render_descriptor_set(
        &mut self,
        set: &DescriptorSet,
        pipeline: &RenderPipeline,
        index: u32,
        offsets: &[u32],
    ) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe {
            inner.bind_render_descriptor_set(set, &pipeline.inner, index, offsets);
        }
    }

    pub fn push_render_constants<T: bytemuck::Pod>(&mut self, pipeline: &RenderPipeline, pc: T) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe {
            inner.push_render_constants(&pipeline.inner, pc);
        }
    }

    pub fn bind_vertex_buffer(&mut self, slot: u32, buffer: &Buffer, offset: vk::DeviceSize) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.bind_vertex_buffer(slot, buffer, offset) };
    }

    pub fn bind_index_buffer(
        &mut self,
        buffer: &Buffer,
        offset: vk::DeviceSize,
        index_type: IndexType,
    ) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        unsafe { inner.bind_index_buffer(buffer, offset, index_type) };
    }

    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let inner = unsafe { &mut *self.command_recorder.inner.get() };
        let inner_pipeline = &pipeline.inner;
        unsafe { inner.bind_compute_pipeline(inner_pipeline) };
    }
}

impl CommandRecorderImpl {
    pub unsafe fn finish(&mut self) -> CommandBufferImpl {
        unsafe {
            let _ = self
                .pool
                .device
                .handle
                .end_command_buffer(self.buffer.handle);
        }
        self.buffer.clone()
    }

    pub unsafe fn bind_render_pipeline(&self, pipeline: &RenderPipelineImpl) {
        unsafe {
            self.device.handle.cmd_bind_pipeline(
                self.buffer.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
        }
    }

    pub unsafe fn bind_compute_pipeline(&self, pipeline: &ComputePipelineImpl) {
        unsafe {
            self.device.handle.cmd_bind_pipeline(
                self.buffer.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle,
            );
        }
    }

    pub unsafe fn viewport(&self, viewport: Viewport) {
        let viewport = viewport.into();
        unsafe {
            self.device
                .handle
                .cmd_set_viewport(self.buffer.handle, 0, &[viewport]);
        }
    }

    pub unsafe fn scissor(&self, scissor: Rect) {
        let scissor = scissor.into();
        unsafe {
            self.device
                .handle
                .cmd_set_scissor(self.buffer.handle, 0, &[scissor]);
        }
    }

    pub unsafe fn begin_render(&self, info: &RenderInfo<'_>) {
        let colors = info
            .colors
            .iter()
            .map(RenderAttachment::as_vk)
            .collect::<Vec<_>>();
        let depth = info.depth.as_ref().map(RenderAttachment::as_vk);
        let stencil = info.stencil.as_ref().map(RenderAttachment::as_vk);

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(info.area.into())
            .layer_count(info.layers)
            .color_attachments(&colors);

        if let Some(depth) = &depth {
            rendering_info = rendering_info.depth_attachment(depth)
        }
        if let Some(stencil) = &stencil {
            rendering_info = rendering_info.stencil_attachment(stencil)
        }
        unsafe {
            self.device
                .ext
                .dynamic
                .cmd_begin_rendering(self.buffer.handle, &rendering_info);
        }
    }

    pub unsafe fn end_rendering(&self) {
        unsafe {
            self.device
                .ext
                .dynamic
                .cmd_end_rendering(self.buffer.handle);
        }
    }

    pub unsafe fn draw(&self, vertex: ops::Range<u32>, instance: ops::Range<u32>) {
        unsafe {
            self.device.handle.cmd_draw(
                self.buffer.handle,
                vertex.len() as u32,
                instance.len() as u32,
                vertex.start,
                instance.start,
            );
        }
    }

    pub unsafe fn draw_indexed(
        &self,
        index: ops::Range<u32>,
        vertex_offset: i32,
        instance: ops::Range<u32>,
    ) {
        unsafe {
            self.device.handle.cmd_draw_indexed(
                self.buffer.handle,
                index.len() as u32,
                instance.len() as u32,
                index.start,
                vertex_offset,
                instance.start,
            );
        }
    }

    pub unsafe fn bind_vertex_buffer(&self, slot: u32, buffer: &Buffer, offset: vk::DeviceSize) {
        unsafe {
            self.device.handle.cmd_bind_vertex_buffers(
                self.buffer.handle,
                slot,
                &[buffer.inner.handle],
                &[offset],
            );
        }
    }

    pub unsafe fn bind_index_buffer(
        &self,
        buffer: &Buffer,
        offset: vk::DeviceSize,
        index_type: IndexType,
    ) {
        unsafe {
            self.device.handle.cmd_bind_index_buffer(
                self.buffer.handle,
                buffer.inner.handle,
                offset,
                index_type.into(),
            );
        }
    }

    pub unsafe fn image_transition(&self, image: vk::Image, transition: ImageTransition) {
        let old_layout = transition.from.layout.into();
        let (src_stage, src_access) = (transition.from.stage, transition.from.access);

        let new_layout = transition.to.layout.into();
        let (dst_stage, dst_access) = (transition.to.stage, transition.to.access);

        let mut barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(old_layout)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .new_layout(new_layout)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(transition.aspect.into())
                    .base_mip_level(transition.mips.start)
                    .level_count(transition.mips.len() as u32)
                    .base_array_layer(transition.layers.start)
                    .layer_count(transition.layers.len() as u32),
            );

        if let Some((src, dst)) = transition.queue {
            barrier.src_queue_family_index = src.inner.info.family_index;
            barrier.dst_queue_family_index = dst.inner.info.family_index;
        }

        let image_memory_barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default()
            .dependency_flags(transition.dependency)
            .image_memory_barriers(&image_memory_barriers);

        unsafe {
            self.device
                .ext
                .sync2
                .cmd_pipeline_barrier2(self.buffer.handle, &dependency_info);
        }
    }

    pub unsafe fn buffer_transition(&self, buffer: vk::Buffer, transition: BufferTransition) {
        let (src_stage, src_access) = (transition.from.stage, transition.from.access);
        let (dst_stage, dst_access) = (transition.to.stage, transition.to.access);

        let mut barrier = vk::BufferMemoryBarrier2::default()
            .buffer(buffer)
            .offset(transition.offset)
            .size(transition.size)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);

        if let Some((src, dst)) = transition.queue {
            barrier.src_queue_family_index = src.inner.info.family_index;
            barrier.dst_queue_family_index = dst.inner.info.family_index;
        }

        let buffer_memory_barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default()
            .dependency_flags(transition.dependency)
            .buffer_memory_barriers(&buffer_memory_barriers);

        unsafe {
            self.device
                .ext
                .sync2
                .cmd_pipeline_barrier2(self.buffer.handle, &dependency_info);
        }
    }

    pub unsafe fn copy_image(&self, info: &CopyImageInfo<'_>) {
        if info.regions.is_empty() {
            return;
        }

        if info.regions.len() == 1 {
            let r = &info.regions[0];
            let single = [vk::ImageCopy2::default()
                .src_subresource(r.src_subresource)
                .src_offset(r.src_offset)
                .dst_subresource(r.dst_subresource)
                .dst_offset(r.dst_offset)
                .extent(r.extent)];

            let copy_info2 = vk::CopyImageInfo2::default()
                .src_image(info.src.inner.handle)
                .src_image_layout(info.src_layout.into())
                .dst_image(info.dst.inner.handle)
                .dst_image_layout(info.dst_layout.into())
                .regions(&single);

            unsafe {
                self.device
                    .handle
                    .cmd_copy_image2(self.buffer.handle, &copy_info2);
            }
        } else {
            let regions2: Vec<vk::ImageCopy2> = info
                .regions
                .iter()
                .map(|r| {
                    vk::ImageCopy2::default()
                        .src_subresource(r.src_subresource)
                        .src_offset(r.src_offset)
                        .dst_subresource(r.dst_subresource)
                        .dst_offset(r.dst_offset)
                        .extent(r.extent)
                })
                .collect();

            let copy_info2 = vk::CopyImageInfo2::default()
                .src_image(info.src.inner.handle)
                .src_image_layout(info.src_layout.into())
                .dst_image(info.dst.inner.handle)
                .dst_image_layout(info.dst_layout.into())
                .regions(&regions2);

            unsafe {
                self.device
                    .handle
                    .cmd_copy_image2(self.buffer.handle, &copy_info2)
            };
        }
    }

    pub unsafe fn blit_image(&self, info: &BlitImageInfo<'_>) {
        if info.regions.is_empty() {
            return;
        }

        if info.regions.len() == 1 {
            let r = &info.regions[0];
            let single = [vk::ImageBlit2::default()
                .src_subresource(r.src_subresource)
                .src_offsets(r.src_offsets)
                .dst_subresource(r.dst_subresource)
                .dst_offsets(r.dst_offsets)];

            let blit_info2 = vk::BlitImageInfo2::default()
                .src_image(info.src.inner.handle)
                .src_image_layout(info.src_layout.into())
                .dst_image(info.dst.inner.handle)
                .dst_image_layout(info.dst_layout.into())
                .regions(&single)
                .filter(info.filter.into());

            unsafe {
                self.device
                    .handle
                    .cmd_blit_image2(self.buffer.handle, &blit_info2);
            }
        } else {
            let regions2: Vec<vk::ImageBlit2> = info
                .regions
                .iter()
                .map(|r| {
                    vk::ImageBlit2::default()
                        .src_subresource(r.src_subresource)
                        .src_offsets(r.src_offsets)
                        .dst_subresource(r.dst_subresource)
                        .dst_offsets(r.dst_offsets)
                })
                .collect();

            let blit_info2 = vk::BlitImageInfo2::default()
                .src_image(info.src.inner.handle)
                .src_image_layout(info.src_layout.into())
                .dst_image(info.dst.inner.handle)
                .dst_image_layout(info.dst_layout.into())
                .regions(&regions2)
                .filter(info.filter.into());

            unsafe {
                self.device
                    .handle
                    .cmd_blit_image2(self.buffer.handle, &blit_info2);
            }
        }
    }

    pub unsafe fn copy_buffer(&self, info: &CopyBufferInfo<'_>) {
        if info.regions.is_empty() {
            return;
        }

        unsafe {
            self.device.handle.cmd_copy_buffer(
                self.buffer.handle,
                info.src.inner.handle,
                info.dst.inner.handle,
                info.regions,
            );
        }
    }

    pub unsafe fn copy_buffer_to_image(&self, info: &CopyBufferToImageInfo<'_>) {
        if info.regions.is_empty() {
            return;
        }

        let regions = info
            .regions
            .iter()
            .copied()
            .map(Into::into)
            .collect::<Vec<vk::BufferImageCopy>>();

        unsafe {
            self.device.handle.cmd_copy_buffer_to_image(
                self.buffer.handle,
                info.src.inner.handle,
                info.dst.inner.handle,
                info.dst_layout.into(),
                &regions,
            );
        }
    }

    pub unsafe fn bind_compute_descriptor_set(
        &self,
        set: &DescriptorSet,
        pipeline: &ComputePipelineImpl,
        index: u32,
        offsets: &[u32],
    ) {
        unsafe {
            self.device.handle.cmd_bind_descriptor_sets(
                self.buffer.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                index,
                &[set.handle],
                offsets,
            );
        }
    }

    pub unsafe fn push_compute_constants<T: bytemuck::Pod>(
        &mut self,
        pipeline: &ComputePipelineImpl,
        pc: T,
    ) {
        unsafe {
            self.device.handle.cmd_push_constants(
                self.buffer.handle,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&[pc]),
            );
        }
    }

    pub unsafe fn bind_render_descriptor_set(
        &self,
        set: &DescriptorSet,
        pipeline: &RenderPipelineImpl,
        index: u32,
        offsets: &[u32],
    ) {
        unsafe {
            self.device.handle.cmd_bind_descriptor_sets(
                self.buffer.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                index,
                &[set.handle],
                offsets,
            );
        }
    }

    pub unsafe fn push_render_constants<T: bytemuck::Pod>(
        &mut self,
        pipeline: &RenderPipelineImpl,
        pc: T,
    ) {
        unsafe {
            self.device.handle.cmd_push_constants(
                self.buffer.handle,
                pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::cast_slice(&[pc]),
            );
        }
    }

    pub unsafe fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.device.handle.cmd_dispatch(self.buffer.handle, x, y, z);
        }
    }
}

#[derive(Debug, Default)]
pub struct SubmitInfo<'a> {
    pub records: &'a [CommandBuffer],
    pub wait_binary: &'a [(&'a Semaphore, PipelineStageFlags)],
    pub wait_timeline: &'a [(&'a Semaphore, u64, PipelineStageFlags)],
    pub signal_binary: &'a [&'a Semaphore],
    pub signal_timeline: &'a [(&'a Semaphore, u64)],
}

impl Queue {
    pub fn record(&self) -> CommandRecorder {
        let tid = thread::current().id();
        let pool = self.pools.get(tid, &self.inner);

        let buffer = pool.get();

        unsafe {
            let _ = self.inner.device.handle.begin_command_buffer(
                buffer.handle,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            );
        }

        let inner = CommandRecorderImpl {
            buffer,
            pool: pool.clone(),
            device: pool.device.clone(),
        };

        CommandRecorder {
            inner: Rc::new(UnsafeCell::new(inner)),
        }
    }

    // TODO: we can merge here already, do that maybe
    pub fn submit(&self, info: SubmitInfo<'_>) -> u64 {
        let _lock = self.lock();
        let submission_index = self
            .submission_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let timeline = self.timeline.clone();
        let pools = &self.pools;

        let command_buffers = info
            .records
            .iter()
            .map(|b| b.inner.clone())
            .collect::<Vec<_>>();

        let wait_binary = info
            .wait_binary
            .iter()
            .map(|(s, f)| (s.inner.handle, (*f).into()))
            .collect::<Vec<_>>();

        let wait_timeline = info
            .wait_timeline
            .iter()
            .map(|(s, v, f)| (s.inner.handle, *v, (*f).into()))
            .collect::<Vec<_>>();

        let signal_binary = info
            .signal_binary
            .iter()
            .map(|s| s.inner.handle)
            .collect::<Vec<_>>();

        let signal_timeline = info
            .signal_timeline
            .iter()
            .map(|(s, v)| (s.inner.handle, *v))
            .collect::<Vec<_>>();

        self.inner
            .submit(
                submission_index,
                timeline,
                pools,
                &command_buffers,
                &wait_binary,
                &wait_timeline,
                &signal_binary,
                &signal_timeline,
            )
            .unwrap()
    }
}

impl QueueImpl {
    pub fn submit(
        &self,
        submission_index: u64,
        timeline: Semaphore,
        pools: &CommandPools,
        command_buffers: &[CommandBufferImpl],
        wait_binary: &[(vk::Semaphore, vk::PipelineStageFlags)],
        wait_timeline: &[(vk::Semaphore, u64, vk::PipelineStageFlags)],
        signal_binary: &[vk::Semaphore],
        signal_timeline: &[(vk::Semaphore, u64)],
    ) -> Result<u64, GPUError> {
        let timeline_index = timeline.get();

        pools.try_cleanup(timeline_index);

        let submit_buffers = command_buffers
            .iter()
            .map(|b| {
                b.submission.set(submission_index);
                b.handle
            })
            .collect::<Vec<_>>();

        let mut signal_timeline = signal_timeline.to_vec();
        signal_timeline.push((timeline.inner.handle, submission_index));

        let mut wait_semaphores = Vec::with_capacity(wait_binary.len() + wait_timeline.len());
        let mut wait_stages = Vec::with_capacity(wait_binary.len() + wait_timeline.len());
        let mut wait_values = Vec::with_capacity(wait_timeline.len());

        for &(sem, stage) in wait_binary {
            wait_semaphores.push(sem);
            wait_stages.push(stage);
            wait_values.push(0);
        }

        for &(sem, value, stage) in wait_timeline {
            wait_semaphores.push(sem);
            wait_stages.push(stage);
            wait_values.push(value);
        }

        let mut signal_semaphores = Vec::with_capacity(signal_binary.len() + signal_timeline.len());
        let mut signal_values = Vec::with_capacity(signal_timeline.len());

        for &sem in signal_binary {
            signal_semaphores.push(sem);
            signal_values.push(0);
        }

        for &(sem, value) in &signal_timeline {
            signal_semaphores.push(sem);
            signal_values.push(value);
        }

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&submit_buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_info);

        unsafe {
            self.device
                .handle
                .queue_submit(self.handle, &[submit_info], vk::Fence::null())?;
        }

        Ok(submission_index)
    }
}

impl CommandPools {
    pub fn new(device: RawDevice) -> Self {
        Self {
            device,
            pools: Mutex::new(HashMap::new()),
        }
    }
    fn get(&self, tid: thread::ThreadId, queue: &QueueImpl) -> Rc<ThreadCommandPool> {
        let mut pools = self.pools.lock();
        if let Some(pool) = pools.get(&tid) {
            return pool.clone();
        }

        let handle = queue.create_command_pool().expect("Create Command Pool");
        let pool = ThreadCommandPool {
            handle,
            device: self.device.clone(),
            ready: RefCell::new(Vec::new()),
            dropped: RefCell::new(Vec::new()),
        };

        let pool = Rc::new(pool);
        pools.insert(tid, pool.clone());
        pool
    }

    pub fn try_cleanup(&self, completed_index: u64) {
        let pools = self.pools.lock();

        let thread_id = thread::current().id();
        let pool = pools.get(&thread_id).unwrap();

        pool.try_cleanup(completed_index);
    }
}

impl ThreadCommandPool {
    pub fn get(&self) -> CommandBufferImpl {
        let mut ready = self.ready.borrow_mut();
        if let Some(buffer) = ready.pop() {
            return buffer;
        }

        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(5);

        let buffer_handles = unsafe { self.device.handle.allocate_command_buffers(&info).unwrap() };

        let mut buffers = buffer_handles
            .into_iter()
            .map(|buffer| CommandBufferImpl {
                handle: buffer,
                submission: Rc::new(Cell::new(0)),
            })
            .collect::<Vec<_>>();

        let buffer = buffers.pop().unwrap();

        ready.extend(buffers);

        buffer
    }

    pub fn retire(&self, buffer: DroppedCommandBuffer) {
        let mut dropped = self.dropped.borrow_mut();
        dropped.push(buffer);
    }

    pub fn try_cleanup(&self, completed_index: u64) {
        let mut retired = self.dropped.borrow_mut();

        let freeable = {
            let mut freeable = Vec::new();
            retired.retain(|b| {
                if b.submission <= completed_index {
                    freeable.push(b.handle);
                    false
                } else {
                    true
                }
            });
            freeable
        };

        if !freeable.is_empty() {
            let mut ready = self.ready.borrow_mut();
            if ready.len() < 10 {
                let new_ready_buffers = freeable.into_iter().map(|b| {
                    unsafe {
                        let _ = self.device.handle.reset_command_buffer(
                            b,
                            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                        );
                    }
                    CommandBufferImpl {
                        handle: b,
                        submission: Rc::new(Cell::new(0)),
                    }
                });
                ready.extend(new_ready_buffers);
            } else {
                unsafe {
                    self.device
                        .handle
                        .free_command_buffers(self.handle, &freeable);
                }
            }
        }
    }
}

impl Drop for CommandRecorderImpl {
    fn drop(&mut self) {
        let buffer = DroppedCommandBuffer {
            handle: self.buffer.handle,
            submission: self.buffer.submission.get(),
        };
        self.pool.retire(buffer);
    }
}

impl Drop for CommandPools {
    fn drop(&mut self) {
        for pool in self.pools.get_mut().values_mut() {
            unsafe { self.device.wait_idle() };

            let ready = pool.ready.borrow();
            let ready = ready.iter().map(|b| b.handle).collect::<Vec<_>>();

            let dropped = pool.dropped.borrow();
            let dropped = dropped.iter().map(|b| b.handle).collect::<Vec<_>>();

            unsafe {
                if !ready.is_empty() {
                    self.device.handle.free_command_buffers(pool.handle, &ready);
                }
                if !dropped.is_empty() {
                    self.device
                        .handle
                        .free_command_buffers(pool.handle, &dropped);
                }
                self.device.handle.destroy_command_pool(pool.handle, None);
            }
        }
    }
}
