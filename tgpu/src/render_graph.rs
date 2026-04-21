use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    fmt::Write as _,
    marker::PhantomData,
    ptr::NonNull,
};

use ash::vk;

use crate::{
    BindlessHeap, Buffer, BufferAccessTransition, BufferDesc, BufferTransition, BufferUses,
    CommandRecorder, CopyBufferInfo, Device, Frame, GPUError, HostAccess, Image, ImageDesc,
    ImageLayout, ImageLayoutTransition, ImageTransition, ImageUses, MemoryPreset, Queue,
    RenderInfo, Swapchain, ViewImage,
};

#[derive(Debug)]
pub enum RenderGraphError {
    Validation(String),
    Gpu(GPUError),
}

impl fmt::Display for RenderGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(message) => write!(f, "Render graph validation error: {message}"),
            Self::Gpu(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RenderGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(_) => None,
            Self::Gpu(err) => Some(err),
        }
    }
}

impl From<GPUError> for RenderGraphError {
    fn from(value: GPUError) -> Self {
        Self::Gpu(value)
    }
}

#[derive(Clone, Copy)]
pub struct RenderGraphInfo<'a> {
    pub device: &'a Device,
    pub graphics: &'a Queue,
    pub async_compute: Option<&'a Queue>,
    pub copy: Option<&'a Queue>,
    pub bindless: Option<&'a BindlessHeap>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RenderGraphExecution {
    pub cache_hit: bool,
    pub needs_swapchain_recreation: bool,
}

#[derive(Debug, Default)]
pub struct RenderGraphCache {
    plans: HashMap<GraphSignature, CompiledPlan>,
}

impl RenderGraphCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.plans.clear();
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassQueue {
    Graphics,
    AsyncCompute,
    Copy,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferAccess {
    StorageComputeRead,
    StorageComputeWrite,
    StorageComputeReadWrite,
    TransferSrc,
    TransferDst,
    Vertex,
    Index,
    Indirect,
}

impl BufferAccess {
    fn transition(self) -> BufferAccessTransition {
        match self {
            Self::StorageComputeRead => BufferAccessTransition::compute_storage_read(),
            Self::StorageComputeWrite => BufferAccessTransition::compute_storage_write(),
            Self::StorageComputeReadWrite => BufferAccessTransition::compute_storage_read_write(),
            Self::TransferSrc => BufferAccessTransition::TRANSFER_SRC,
            Self::TransferDst => BufferAccessTransition::TRANSFER_DST,
            Self::Vertex => BufferAccessTransition::new(
                vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            ),
            Self::Index => BufferAccessTransition::new(
                vk::PipelineStageFlags2::INDEX_INPUT,
                vk::AccessFlags2::INDEX_READ,
            ),
            Self::Indirect => BufferAccessTransition::INDIRECT,
        }
    }

    fn is_write(self) -> bool {
        matches!(
            self,
            Self::StorageComputeWrite | Self::StorageComputeReadWrite | Self::TransferDst
        )
    }

    fn requires_initialized(self) -> bool {
        matches!(
            self,
            Self::StorageComputeRead
                | Self::StorageComputeReadWrite
                | Self::TransferSrc
                | Self::Vertex
                | Self::Index
                | Self::Indirect
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageAccess {
    SampledFragment,
    SampledCompute,
    StorageComputeRead,
    StorageComputeWrite,
    StorageComputeReadWrite,
    TransferSrc,
    TransferDst,
}

impl ImageAccess {
    fn transition(self) -> ImageLayoutTransition {
        match self {
            Self::SampledFragment => ImageLayoutTransition::FRAGMENT,
            Self::SampledCompute => ImageLayoutTransition::custom(
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            Self::StorageComputeRead => ImageLayoutTransition::custom(
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            Self::StorageComputeWrite => ImageLayoutTransition::custom(
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
            ),
            Self::StorageComputeReadWrite => ImageLayoutTransition::custom(
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            ),
            Self::TransferSrc => ImageLayoutTransition::custom(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            Self::TransferDst => ImageLayoutTransition::custom(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        }
    }

    fn is_write(self) -> bool {
        matches!(
            self,
            Self::StorageComputeWrite | Self::StorageComputeReadWrite | Self::TransferDst
        )
    }

    fn requires_initialized(self) -> bool {
        matches!(
            self,
            Self::SampledFragment
                | Self::SampledCompute
                | Self::StorageComputeRead
                | Self::StorageComputeReadWrite
                | Self::TransferSrc
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImportedBufferDesc {
    pub initial: BufferAccessTransition,
    pub initialized: bool,
}

impl Default for ImportedBufferDesc {
    fn default() -> Self {
        Self {
            initial: BufferAccessTransition::NONE,
            initialized: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImportedImageDesc {
    pub extent: vk::Extent3D,
    pub aspect: vk::ImageAspectFlags,
    pub initial: ImageLayoutTransition,
    pub initialized: bool,
}

impl Default for ImportedImageDesc {
    fn default() -> Self {
        Self {
            extent: vk::Extent3D::default(),
            aspect: vk::ImageAspectFlags::COLOR,
            initial: ImageLayoutTransition::UNDEFINED,
            initialized: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransientBufferDesc {
    pub size: usize,
    pub usage: BufferUses,
    pub memory: MemoryPreset,
    pub host_access: HostAccess,
}

impl Default for TransientBufferDesc {
    fn default() -> Self {
        Self {
            size: 0,
            usage: BufferUses::empty(),
            memory: MemoryPreset::GpuOnly,
            host_access: HostAccess::None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransientImageDesc {
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: ImageUses,
    pub aspect: vk::ImageAspectFlags,
    pub memory: MemoryPreset,
}

impl Default for TransientImageDesc {
    fn default() -> Self {
        Self {
            format: vk::Format::UNDEFINED,
            extent: vk::Extent3D::default(),
            usage: ImageUses::empty(),
            aspect: vk::ImageAspectFlags::empty(),
            memory: MemoryPreset::GpuOnly,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ColorAttachmentDesc {
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear: vk::ClearColorValue,
}

impl fmt::Debug for ColorAttachmentDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColorAttachmentDesc")
            .field("load_op", &self.load_op)
            .field("store_op", &self.store_op)
            .finish_non_exhaustive()
    }
}

impl ColorAttachmentDesc {
    pub fn clear(color: [f32; 4]) -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: vk::ClearColorValue { float32: color },
        }
    }

    pub fn load() -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: vk::ClearColorValue { float32: [0.0; 4] },
        }
    }

    pub fn dont_care() -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: vk::ClearColorValue { float32: [0.0; 4] },
        }
    }

    fn requires_initialized(self) -> bool {
        self.load_op == vk::AttachmentLoadOp::LOAD
    }
}

#[derive(Clone, Copy)]
pub struct DepthAttachmentDesc {
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear: vk::ClearDepthStencilValue,
}

impl fmt::Debug for DepthAttachmentDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DepthAttachmentDesc")
            .field("load_op", &self.load_op)
            .field("store_op", &self.store_op)
            .field("clear_depth", &self.clear.depth)
            .field("clear_stencil", &self.clear.stencil)
            .finish()
    }
}

impl DepthAttachmentDesc {
    pub fn clear(depth: f32) -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: vk::ClearDepthStencilValue { depth, stencil: 0 },
        }
    }

    pub fn load() -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            clear: vk::ClearDepthStencilValue::default(),
        }
    }

    fn requires_initialized(self) -> bool {
        self.load_op == vk::AttachmentLoadOp::LOAD
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphBuffer(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphImage(u32);

pub struct RenderGraph<'a> {
    info: RenderGraphInfo<'a>,
    buffers: Vec<BufferBinding<'a>>,
    images: Vec<ImageBinding<'a>>,
    passes: Vec<PassNode<'a>>,
}

impl<'a> RenderGraph<'a> {
    pub fn new(info: RenderGraphInfo<'a>) -> Self {
        Self {
            info,
            buffers: Vec::new(),
            images: Vec::new(),
            passes: Vec::new(),
        }
    }

    fn build_metadata(&self) -> GraphMetadata {
        GraphMetadata {
            buffers: self.buffers.iter().map(|b| b.metadata.clone()).collect(),
            images: self.images.iter().map(|i| i.metadata.clone()).collect(),
            passes: self.passes.iter().map(|p| p.metadata.clone()).collect(),
        }
    }

    pub fn import_buffer(
        &mut self,
        name: impl Into<String>,
        buffer: &'a Buffer,
        desc: ImportedBufferDesc,
    ) -> GraphBuffer {
        let handle = GraphBuffer(self.buffers.len() as u32);
        self.buffers.push(BufferBinding {
            metadata: BufferMetadata {
                name: name.into(),
                initial: desc.initial,
                initialized: desc.initialized,
                kind: BufferKind::Imported,
            },
            kind: BufferBindingKind::Imported { buffer },
        });
        handle
    }

    pub fn create_transient_buffer(
        &mut self,
        name: impl Into<String>,
        desc: TransientBufferDesc,
    ) -> GraphBuffer {
        let handle = GraphBuffer(self.buffers.len() as u32);
        self.buffers.push(BufferBinding {
            metadata: BufferMetadata {
                name: name.into(),
                initial: BufferAccessTransition::NONE,
                initialized: false,
                kind: BufferKind::Transient,
            },
            kind: BufferBindingKind::Transient { desc },
        });
        handle
    }

    pub fn import_image(
        &mut self,
        name: impl Into<String>,
        image: &'a ViewImage,
        desc: ImportedImageDesc,
    ) -> GraphImage {
        self.import_image_view(name, &image.image, &image.view, desc)
    }

    pub fn import_image_view(
        &mut self,
        name: impl Into<String>,
        image: &'a Image,
        view: &'a crate::ImageView,
        desc: ImportedImageDesc,
    ) -> GraphImage {
        let handle = GraphImage(self.images.len() as u32);
        self.images.push(ImageBinding {
            metadata: ImageMetadata {
                name: name.into(),
                extent: desc.extent,
                aspect: if desc.aspect.is_empty() {
                    infer_image_aspect(image.format)
                } else {
                    desc.aspect
                },
                initial: desc.initial,
                initialized: desc.initialized,
                kind: ImageKind::Imported,
            },
            kind: ImageBindingKind::Imported { image, view },
        });
        handle
    }

    pub fn import_swapchain_image(
        &mut self,
        name: impl Into<String>,
        swapchain: &'a mut Swapchain,
        frame: Frame,
    ) -> GraphImage {
        let extent = swapchain.extent();
        let handle = GraphImage(self.images.len() as u32);
        self.images.push(ImageBinding {
            metadata: ImageMetadata {
                name: name.into(),
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                aspect: vk::ImageAspectFlags::COLOR,
                initial: ImageLayoutTransition::UNDEFINED,
                initialized: false,
                kind: ImageKind::Swapchain,
            },
            kind: ImageBindingKind::Swapchain {
                swapchain: NonNull::from(swapchain),
                frame,
                marker: PhantomData,
            },
        });
        handle
    }

    pub fn create_transient_image(
        &mut self,
        name: impl Into<String>,
        desc: TransientImageDesc,
    ) -> GraphImage {
        let handle = GraphImage(self.images.len() as u32);
        self.images.push(ImageBinding {
            metadata: ImageMetadata {
                name: name.into(),
                extent: desc.extent,
                aspect: if desc.aspect.is_empty() {
                    infer_image_aspect(desc.format)
                } else {
                    desc.aspect
                },
                initial: ImageLayoutTransition::UNDEFINED,
                initialized: false,
                kind: ImageKind::Transient,
            },
            kind: ImageBindingKind::Transient { desc },
        });
        handle
    }

    pub fn add_compute_pass<B, E>(&mut self, name: impl Into<String>, build: B, exec: E)
    where
        B: FnOnce(&mut PassBuilder),
        E: for<'pass> FnMut(&mut PassContext<'pass, 'a>) + 'a,
    {
        self.add_pass(
            name.into(),
            PassKind::Compute,
            build,
            PassCallback::Compute(Box::new(exec)),
        );
    }

    pub fn add_copy_pass<B, E>(&mut self, name: impl Into<String>, build: B, exec: E)
    where
        B: FnOnce(&mut PassBuilder),
        E: for<'pass> FnMut(&mut PassContext<'pass, 'a>) + 'a,
    {
        self.add_pass(
            name.into(),
            PassKind::Copy,
            build,
            PassCallback::Copy(Box::new(exec)),
        );
    }

    pub fn add_render_pass<B, E>(&mut self, name: impl Into<String>, build: B, exec: E)
    where
        B: FnOnce(&mut RenderPassBuilder),
        E: for<'pass> FnMut(&mut RenderPassContext<'pass, 'a>) + 'a,
    {
        let mut metadata = PassMetadata::new(name.into(), PassKind::Render);
        build(&mut RenderPassBuilder {
            inner: PassBuilder {
                metadata: &mut metadata,
            },
        });
        self.passes.push(PassNode {
            metadata,
            callback: PassCallback::Render(Box::new(exec)),
        });
    }

    fn add_pass<B>(&mut self, name: String, kind: PassKind, build: B, callback: PassCallback<'a>)
    where
        B: FnOnce(&mut PassBuilder),
    {
        let mut metadata = PassMetadata::new(name, kind);
        build(&mut PassBuilder {
            metadata: &mut metadata,
        });
        self.passes.push(PassNode { metadata, callback });
    }

    pub fn dump_text(&self) -> Result<String, RenderGraphError> {
        let metadata = self.build_metadata();
        let queue_caps = QueueAvailability::from_info(&self.info);
        let prepared = prepare_metadata(&metadata, queue_caps)?;
        log_validation_warnings(&prepared.warnings);
        let plan = compile_metadata(&prepared.metadata, queue_caps)?;
        Ok(format_graph_text(
            &prepared.metadata,
            &plan,
            &prepared.warnings,
        ))
    }

    pub fn to_dot(&self) -> Result<String, RenderGraphError> {
        let metadata = self.build_metadata();
        let queue_caps = QueueAvailability::from_info(&self.info);
        let prepared = prepare_metadata(&metadata, queue_caps)?;
        log_validation_warnings(&prepared.warnings);
        let plan = compile_metadata(&prepared.metadata, queue_caps)?;
        Ok(format_graph_dot(&prepared.metadata, &plan))
    }

    pub fn execute(self) -> Result<RenderGraphExecution, RenderGraphError> {
        self.execute_inner(None)
    }

    pub fn execute_cached(
        self,
        cache: &mut RenderGraphCache,
    ) -> Result<RenderGraphExecution, RenderGraphError> {
        self.execute_inner(Some(cache))
    }

    fn execute_inner(
        mut self,
        cache: Option<&mut RenderGraphCache>,
    ) -> Result<RenderGraphExecution, RenderGraphError> {
        let metadata = self.build_metadata();
        let queue_caps = QueueAvailability::from_info(&self.info);
        let prepared = prepare_metadata(&metadata, queue_caps)?;
        log_validation_warnings(&prepared.warnings);
        let metadata = prepared.metadata;

        let (plan, cache_hit) = if let Some(cache) = cache {
            if let Some(plan) = cache.plans.get(&prepared.signature) {
                (plan.clone(), true)
            } else {
                let plan = compile_metadata(&metadata, queue_caps)?;
                cache.plans.insert(prepared.signature, plan.clone());
                (plan, false)
            }
        } else {
            (compile_metadata(&metadata, queue_caps)?, false)
        };
        let mut plan = plan;

        for (compiled, pass) in plan.passes.iter_mut().zip(&metadata.passes) {
            if pass.kind == PassKind::Render {
                compiled.render = Some(build_render_targets(pass, &metadata)?);
            }
        }

        let realized_buffers = realize_buffers(self.info.device, &self.buffers)?;
        let realized_images = realize_images(self.info.device, &self.images)?;

        let mut submission_values = vec![0u64; self.passes.len()];
        let mut needs_swapchain_recreation = false;
        let mut swapchain_to_present: Option<(NonNull<Swapchain>, Frame)> = None;

        for (index, (pass, compiled)) in self.passes.iter_mut().zip(&plan.passes).enumerate() {
            let queue = queue_ref(&self.info, compiled.queue);
            let mut recorder = queue.record();

            for barrier in &compiled.buffer_barriers {
                recorder.buffer_transition(
                    realized_buffers[barrier.buffer.0 as usize].buffer(),
                    BufferTransition {
                        from: barrier.from,
                        to: barrier.to,
                        ..Default::default()
                    },
                );
            }

            for barrier in &compiled.image_barriers {
                recorder.image_transition(
                    realized_images[barrier.image.0 as usize].image(),
                    ImageTransition {
                        from: barrier.from,
                        to: barrier.to,
                        aspect: metadata.images[barrier.image.0 as usize].aspect,
                        ..Default::default()
                    },
                );
            }

            match &mut pass.callback {
                PassCallback::Compute(cb) | PassCallback::Copy(cb) => {
                    let mut ctx = PassContext {
                        recorder: &mut recorder,
                        bindless: self.info.bindless,
                        buffers: &realized_buffers,
                        images: &realized_images,
                    };
                    cb(&mut ctx);
                }
                PassCallback::Render(cb) => {
                    let mut ctx = RenderPassContext {
                        pass: PassContext {
                            recorder: &mut recorder,
                            bindless: self.info.bindless,
                            buffers: &realized_buffers,
                            images: &realized_images,
                        },
                        render: compiled.render.clone().expect("render pass info"),
                    };
                    cb(&mut ctx);
                }
            }

            for barrier in &compiled.post_image_barriers {
                recorder.image_transition(
                    realized_images[barrier.image.0 as usize].image(),
                    ImageTransition {
                        from: barrier.from,
                        to: barrier.to,
                        aspect: metadata.images[barrier.image.0 as usize].aspect,
                        ..Default::default()
                    },
                );
            }

            let mut wait_timeline = Vec::new();
            let mut wait_by_queue = BTreeMap::<QueueKind, u64>::new();
            for &wait_pass in &compiled.wait_passes {
                let value = submission_values[wait_pass];
                let source_queue = plan.passes[wait_pass].queue;
                wait_by_queue
                    .entry(source_queue)
                    .and_modify(|current| *current = (*current).max(value))
                    .or_insert(value);
            }

            for (queue_kind, value) in wait_by_queue {
                wait_timeline.push((
                    &queue_ref(&self.info, queue_kind).timeline,
                    value,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                ));
            }

            let mut wait_binary = Vec::new();
            let mut signal_binary = Vec::new();
            let mut fence = None;

            if compiled.waits_on_swapchain_acquire {
                let swapchain = compiled
                    .swapchain_resource
                    .and_then(|resource| match &self.images[resource.0 as usize].kind {
                        ImageBindingKind::Swapchain { swapchain, .. } => Some(*swapchain),
                        _ => None,
                    })
                    .expect("swapchain acquire target");
                let frame = match &self.images[compiled.swapchain_resource.unwrap().0 as usize].kind
                {
                    ImageBindingKind::Swapchain { frame, .. } => *frame,
                    _ => unreachable!(),
                };
                // SAFETY: the graph owns the swapchain's unique mutable borrow for the duration of
                // execution, so taking a shared reference here is sound.
                wait_binary.push((
                    unsafe { swapchain.as_ref() }
                        .inner
                        .available_semaphore(frame),
                    vk::PipelineStageFlags::ALL_COMMANDS,
                ));
            }

            if compiled.signals_swapchain_present {
                let resource = compiled
                    .swapchain_resource
                    .expect("swapchain signal target");
                let (swapchain, frame) = match &self.images[resource.0 as usize].kind {
                    ImageBindingKind::Swapchain {
                        swapchain, frame, ..
                    } => (*swapchain, *frame),
                    _ => unreachable!(),
                };
                // SAFETY: the graph owns the swapchain's unique mutable borrow for the duration of
                // execution, so taking a shared reference here is sound.
                signal_binary.push(
                    unsafe { swapchain.as_ref() }
                        .inner
                        .finished_semaphore(frame),
                );
                fence = Some(unsafe { swapchain.as_ref() }.inner.fence(frame));
                swapchain_to_present = Some((swapchain, frame));
            }

            let wait_timeline_refs = wait_timeline
                .iter()
                .map(|(semaphore, value, stage)| (*semaphore, *value, *stage))
                .collect::<Vec<_>>();
            let wait_binary_refs = wait_binary
                .iter()
                .map(|(semaphore, stage)| (*semaphore, *stage))
                .collect::<Vec<_>>();
            let signal_binary_refs = signal_binary.iter().copied().collect::<Vec<_>>();

            submission_values[index] = queue.submit(crate::SubmitInfo {
                records: &[recorder.finish()],
                wait_binary: &wait_binary_refs,
                wait_timeline: &wait_timeline_refs,
                signal_binary: &signal_binary_refs,
                fence,
                ..Default::default()
            });
        }

        if let Some((mut swapchain, frame)) = swapchain_to_present {
            // SAFETY: `import_swapchain_image` takes `&mut Swapchain`, so the graph has exclusive
            // access while executing and can present through the stored raw pointer here.
            needs_swapchain_recreation =
                unsafe { swapchain.as_mut() }.present(self.info.graphics, frame)?;
        }

        Ok(RenderGraphExecution {
            cache_hit,
            needs_swapchain_recreation,
        })
    }
}

pub struct PassBuilder<'a> {
    metadata: &'a mut PassMetadata,
}

impl PassBuilder<'_> {
    pub fn queue(&mut self, queue: PassQueue) {
        self.metadata.requested_queue = queue;
    }

    pub fn read_buffer(&mut self, buffer: GraphBuffer, access: BufferAccess) {
        self.metadata.buffer_uses.push(BufferUse { buffer, access });
    }

    pub fn write_buffer(&mut self, buffer: GraphBuffer, access: BufferAccess) {
        self.metadata.buffer_uses.push(BufferUse { buffer, access });
    }

    pub fn read_write_buffer(&mut self, buffer: GraphBuffer, access: BufferAccess) {
        self.metadata.buffer_uses.push(BufferUse { buffer, access });
    }

    pub fn read_image(&mut self, image: GraphImage, access: ImageAccess) {
        self.metadata
            .image_uses
            .push(ImageUse::General { image, access });
    }

    pub fn write_image(&mut self, image: GraphImage, access: ImageAccess) {
        self.metadata
            .image_uses
            .push(ImageUse::General { image, access });
    }

    pub fn read_write_image(&mut self, image: GraphImage, access: ImageAccess) {
        self.metadata
            .image_uses
            .push(ImageUse::General { image, access });
    }
}

pub struct RenderPassBuilder<'a> {
    inner: PassBuilder<'a>,
}

impl RenderPassBuilder<'_> {
    pub fn queue(&mut self, queue: PassQueue) {
        self.inner.queue(queue);
    }

    pub fn read_buffer(&mut self, buffer: GraphBuffer, access: BufferAccess) {
        self.inner.read_buffer(buffer, access);
    }

    pub fn read_image(&mut self, image: GraphImage, access: ImageAccess) {
        self.inner.read_image(image, access);
    }

    pub fn write_color(&mut self, image: GraphImage, desc: ColorAttachmentDesc) {
        self.inner
            .metadata
            .image_uses
            .push(ImageUse::ColorAttachment { image, desc });
    }

    pub fn write_depth(&mut self, image: GraphImage, desc: DepthAttachmentDesc) {
        self.inner
            .metadata
            .image_uses
            .push(ImageUse::DepthAttachment { image, desc });
    }
}

pub struct PassContext<'pass, 'graph> {
    recorder: &'pass mut CommandRecorder,
    bindless: Option<&'graph BindlessHeap>,
    buffers: &'pass [RealizedBuffer<'graph>],
    images: &'pass [RealizedImage<'graph>],
}

impl<'pass, 'graph> PassContext<'pass, 'graph> {
    pub fn cmd(&mut self) -> &mut CommandRecorder {
        self.recorder
    }

    pub fn bindless(&self) -> Option<&'graph BindlessHeap> {
        self.bindless
    }

    pub fn buffer(&self, handle: GraphBuffer) -> &Buffer {
        self.buffers[handle.0 as usize].buffer()
    }

    pub fn image(&self, handle: GraphImage) -> &Image {
        self.images[handle.0 as usize].image()
    }

    pub fn image_view(&self, handle: GraphImage) -> &crate::ImageView {
        self.images[handle.0 as usize].view()
    }

    pub fn copy_buffer(&mut self, src: GraphBuffer, dst: GraphBuffer, regions: &[vk::BufferCopy]) {
        let src = self.buffers[src.0 as usize].buffer();
        let dst = self.buffers[dst.0 as usize].buffer();
        self.recorder
            .copy_buffer(&CopyBufferInfo { src, dst, regions });
    }
}

pub struct RenderPassContext<'pass, 'graph> {
    pass: PassContext<'pass, 'graph>,
    render: CompiledRenderTargets,
}

impl<'pass, 'graph> RenderPassContext<'pass, 'graph> {
    pub fn cmd(&mut self) -> &mut CommandRecorder {
        self.pass.cmd()
    }

    pub fn bindless(&self) -> Option<&'graph BindlessHeap> {
        self.pass.bindless()
    }

    pub fn buffer(&self, handle: GraphBuffer) -> &Buffer {
        self.pass.buffer(handle)
    }

    pub fn image(&self, handle: GraphImage) -> &Image {
        self.pass.image(handle)
    }

    pub fn image_view(&self, handle: GraphImage) -> &crate::ImageView {
        self.pass.image_view(handle)
    }

    pub fn begin_render<F>(&mut self, render: F)
    where
        F: FnOnce(&mut crate::command::RenderRecorder<'_>),
    {
        let colors = self
            .render
            .colors
            .iter()
            .map(|target| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(self.pass.image_view(target.image).inner.handle)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(target.desc.load_op)
                    .store_op(target.desc.store_op)
                    .clear_value(vk::ClearValue {
                        color: target.desc.clear,
                    })
            })
            .collect::<Vec<_>>();

        let depth = self.render.depth.as_ref().map(|target| {
            vk::RenderingAttachmentInfo::default()
                .image_view(self.pass.image_view(target.image).inner.handle)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(target.desc.load_op)
                .store_op(target.desc.store_op)
                .clear_value(vk::ClearValue {
                    depth_stencil: target.desc.clear,
                })
        });

        self.pass.recorder.begin_render(
            &RenderInfo {
                area: self.render.area,
                colors: &colors,
                depth,
                ..Default::default()
            },
            render,
        );
    }
}

type ComputeCallback<'a> = dyn for<'pass> FnMut(&mut PassContext<'pass, 'a>) + 'a;
type RenderCallback<'a> = dyn for<'pass> FnMut(&mut RenderPassContext<'pass, 'a>) + 'a;

enum PassCallback<'a> {
    Compute(Box<ComputeCallback<'a>>),
    Copy(Box<ComputeCallback<'a>>),
    Render(Box<RenderCallback<'a>>),
}

struct PassNode<'a> {
    metadata: PassMetadata,
    callback: PassCallback<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassKind {
    Render,
    Compute,
    Copy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QueueKind {
    Graphics,
    AsyncCompute,
    Copy,
}

#[derive(Debug, Clone)]
struct PassMetadata {
    name: String,
    kind: PassKind,
    requested_queue: PassQueue,
    buffer_uses: Vec<BufferUse>,
    image_uses: Vec<ImageUse>,
}

impl PassMetadata {
    fn new(name: String, kind: PassKind) -> Self {
        Self {
            name,
            kind,
            requested_queue: PassQueue::Auto,
            buffer_uses: Vec::new(),
            image_uses: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct GraphMetadata {
    buffers: Vec<BufferMetadata>,
    images: Vec<ImageMetadata>,
    passes: Vec<PassMetadata>,
}

#[derive(Debug, Clone)]
struct BufferMetadata {
    name: String,
    initial: BufferAccessTransition,
    initialized: bool,
    kind: BufferKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferKind {
    Imported,
    Transient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageKind {
    Imported,
    Transient,
    Swapchain,
}

#[derive(Debug, Clone)]
struct ImageMetadata {
    name: String,
    extent: vk::Extent3D,
    aspect: vk::ImageAspectFlags,
    initial: ImageLayoutTransition,
    initialized: bool,
    kind: ImageKind,
}

#[derive(Debug)]
struct PreparedGraph {
    metadata: GraphMetadata,
    warnings: Vec<String>,
    signature: GraphSignature,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphSignature(String);

#[derive(Debug, Clone, Copy)]
struct BufferUse {
    buffer: GraphBuffer,
    access: BufferAccess,
}

#[derive(Debug, Clone, Copy)]
enum ImageUse {
    General {
        image: GraphImage,
        access: ImageAccess,
    },
    ColorAttachment {
        image: GraphImage,
        desc: ColorAttachmentDesc,
    },
    DepthAttachment {
        image: GraphImage,
        desc: DepthAttachmentDesc,
    },
}

impl ImageUse {
    fn image(self) -> GraphImage {
        match self {
            Self::General { image, .. }
            | Self::ColorAttachment { image, .. }
            | Self::DepthAttachment { image, .. } => image,
        }
    }

    fn transition(self) -> ImageLayoutTransition {
        match self {
            Self::General { access, .. } => access.transition(),
            Self::ColorAttachment { .. } => ImageLayoutTransition::COLOR,
            Self::DepthAttachment { .. } => depth_attachment_transition(),
        }
    }

    fn is_write(self) -> bool {
        match self {
            Self::General { access, .. } => access.is_write(),
            Self::ColorAttachment { .. } | Self::DepthAttachment { .. } => true,
        }
    }

    fn requires_initialized(self) -> bool {
        match self {
            Self::General { access, .. } => access.requires_initialized(),
            Self::ColorAttachment { desc, .. } => desc.requires_initialized(),
            Self::DepthAttachment { desc, .. } => desc.requires_initialized(),
        }
    }
}

struct BufferBinding<'a> {
    metadata: BufferMetadata,
    kind: BufferBindingKind<'a>,
}

enum BufferBindingKind<'a> {
    Imported { buffer: &'a Buffer },
    Transient { desc: TransientBufferDesc },
}

struct ImageBinding<'a> {
    metadata: ImageMetadata,
    kind: ImageBindingKind<'a>,
}

enum ImageBindingKind<'a> {
    Imported {
        image: &'a Image,
        view: &'a crate::ImageView,
    },
    Swapchain {
        swapchain: NonNull<Swapchain>,
        frame: Frame,
        marker: PhantomData<&'a mut Swapchain>,
    },
    Transient {
        desc: TransientImageDesc,
    },
}

enum RealizedBuffer<'a> {
    Imported(&'a Buffer),
    Transient(Buffer),
}

impl<'a> RealizedBuffer<'a> {
    fn buffer(&self) -> &Buffer {
        match self {
            Self::Imported(buffer) => buffer,
            Self::Transient(buffer) => buffer,
        }
    }
}

enum RealizedImage<'a> {
    Imported {
        image: &'a Image,
        view: &'a crate::ImageView,
    },
    Swapchain {
        swapchain: NonNull<Swapchain>,
        frame: Frame,
        marker: PhantomData<&'a mut Swapchain>,
    },
    Transient(ViewImage),
}

impl<'a> RealizedImage<'a> {
    fn image(&self) -> &Image {
        match self {
            Self::Imported { image, .. } => image,
            Self::Swapchain {
                swapchain, frame, ..
            } => {
                // SAFETY: the graph owns the swapchain's unique mutable borrow while executing.
                unsafe { swapchain.as_ref() }.image(*frame)
            }
            Self::Transient(image) => &image.image,
        }
    }

    fn view(&self) -> &crate::ImageView {
        match self {
            Self::Imported { view, .. } => view,
            Self::Swapchain {
                swapchain, frame, ..
            } => {
                // SAFETY: the graph owns the swapchain's unique mutable borrow while executing.
                unsafe { swapchain.as_ref() }.view(*frame)
            }
            Self::Transient(image) => &image.view,
        }
    }
}

#[derive(Debug, Clone)]
struct CompiledPlan {
    passes: Vec<CompiledPass>,
}

#[derive(Debug, Clone)]
struct CompiledPass {
    queue: QueueKind,
    buffer_barriers: Vec<CompiledBufferBarrier>,
    image_barriers: Vec<CompiledImageBarrier>,
    post_image_barriers: Vec<CompiledImageBarrier>,
    wait_passes: Vec<usize>,
    render: Option<CompiledRenderTargets>,
    waits_on_swapchain_acquire: bool,
    signals_swapchain_present: bool,
    swapchain_resource: Option<GraphImage>,
}

#[derive(Debug, Clone)]
struct CompiledRenderTargets {
    colors: Vec<CompiledColorAttachment>,
    depth: Option<CompiledDepthAttachment>,
    area: vk::Rect2D,
}

#[derive(Debug, Clone)]
struct CompiledColorAttachment {
    image: GraphImage,
    desc: ColorAttachmentDesc,
}

#[derive(Debug, Clone)]
struct CompiledDepthAttachment {
    image: GraphImage,
    desc: DepthAttachmentDesc,
}

#[derive(Debug, Clone, Copy)]
struct CompiledBufferBarrier {
    buffer: GraphBuffer,
    from: BufferAccessTransition,
    to: BufferAccessTransition,
}

#[derive(Debug, Clone, Copy)]
struct CompiledImageBarrier {
    image: GraphImage,
    from: ImageLayoutTransition,
    to: ImageLayoutTransition,
}

#[derive(Debug, Clone, Copy)]
struct QueueAvailability {
    has_async: bool,
    async_same_family: bool,
    has_copy: bool,
    copy_same_family: bool,
}

impl QueueAvailability {
    fn from_info(info: &RenderGraphInfo<'_>) -> Self {
        let graphics_family = info.graphics.inner.info.family_index;
        Self {
            has_async: info.async_compute.is_some(),
            async_same_family: info
                .async_compute
                .is_some_and(|queue| queue.inner.info.family_index == graphics_family),
            has_copy: info.copy.is_some(),
            copy_same_family: info
                .copy
                .is_some_and(|queue| queue.inner.info.family_index == graphics_family),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TrackedBufferState {
    transition: BufferAccessTransition,
    initialized: bool,
    queue: Option<QueueKind>,
    write: bool,
    last_pass: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct TrackedImageState {
    transition: ImageLayoutTransition,
    initialized: bool,
    queue: Option<QueueKind>,
    write: bool,
    last_pass: Option<usize>,
}

fn realize_buffers<'a>(
    device: &Device,
    bindings: &[BufferBinding<'a>],
) -> Result<Vec<RealizedBuffer<'a>>, RenderGraphError> {
    let mut realized = Vec::with_capacity(bindings.len());
    for binding in bindings {
        match binding.kind {
            BufferBindingKind::Imported { buffer } => {
                realized.push(RealizedBuffer::Imported(buffer))
            }
            BufferBindingKind::Transient { desc } => {
                let buffer = device.create_buffer(&BufferDesc {
                    label: None,
                    size: desc.size,
                    usage: desc.usage,
                    memory: desc.memory,
                    host_access: desc.host_access,
                    ..Default::default()
                })?;
                realized.push(RealizedBuffer::Transient(buffer));
            }
        }
    }
    Ok(realized)
}

fn realize_images<'a>(
    device: &Device,
    bindings: &[ImageBinding<'a>],
) -> Result<Vec<RealizedImage<'a>>, RenderGraphError> {
    let mut realized = Vec::with_capacity(bindings.len());
    for binding in bindings {
        match binding.kind {
            ImageBindingKind::Imported { image, view } => {
                realized.push(RealizedImage::Imported { image, view })
            }
            ImageBindingKind::Swapchain {
                swapchain,
                frame,
                marker,
            } => realized.push(RealizedImage::Swapchain {
                swapchain,
                frame,
                marker,
            }),
            ImageBindingKind::Transient { desc } => {
                let image = device.create_view_image(&crate::ViewImageDesc {
                    image: ImageDesc {
                        label: None,
                        format: desc.format,
                        extent: desc.extent,
                        usage: desc.usage,
                        memory: desc.memory,
                        ..Default::default()
                    },
                    sampler: None,
                    aspect: Some(if desc.aspect.is_empty() {
                        infer_image_aspect(desc.format)
                    } else {
                        desc.aspect
                    }),
                    ..Default::default()
                })?;
                realized.push(RealizedImage::Transient(image));
            }
        }
    }
    Ok(realized)
}

fn queue_ref<'a>(info: &'a RenderGraphInfo<'a>, queue: QueueKind) -> &'a Queue {
    match queue {
        QueueKind::Graphics => info.graphics,
        QueueKind::AsyncCompute => info.async_compute.expect("async compute queue"),
        QueueKind::Copy => info.copy.expect("copy queue"),
    }
}

fn prepare_metadata(
    metadata: &GraphMetadata,
    queues: QueueAvailability,
) -> Result<PreparedGraph, RenderGraphError> {
    let mut metadata = metadata.clone();
    let buffers = metadata.buffers.clone();
    let images = metadata.images.clone();
    let mut warnings = Vec::new();

    for pass in &mut metadata.passes {
        sanitize_buffer_uses(pass, &buffers, &mut warnings)?;
        sanitize_image_uses(pass, &images, &mut warnings)?;
    }

    warnings.extend(find_unused_warnings(&metadata));

    Ok(PreparedGraph {
        signature: build_signature(&metadata, queues),
        metadata,
        warnings,
    })
}

fn log_validation_warnings(warnings: &[String]) {
    for warning in warnings {
        log::warn!("{warning}");
    }
}

fn sanitize_buffer_uses(
    pass: &mut PassMetadata,
    buffers: &[BufferMetadata],
    warnings: &mut Vec<String>,
) -> Result<(), RenderGraphError> {
    let mut seen = BTreeMap::<u32, BufferAccess>::new();
    let mut deduped = Vec::with_capacity(pass.buffer_uses.len());

    for use_info in pass.buffer_uses.drain(..) {
        let entry = seen.get(&use_info.buffer.0).copied();
        if let Some(existing) = entry {
            let buffer_name = &buffers[use_info.buffer.0 as usize].name;
            if existing == use_info.access {
                warnings.push(format!(
                    "pass '{}' declares buffer '{}' multiple times with the same access; ignoring the duplicate declaration",
                    pass.name, buffer_name
                ));
                continue;
            }

            return Err(RenderGraphError::Validation(format!(
                "pass '{}' declares buffer '{}' with conflicting accesses ({:?} vs {:?}); use a single declaration that matches the real usage",
                pass.name, buffer_name, existing, use_info.access
            )));
        }

        seen.insert(use_info.buffer.0, use_info.access);
        deduped.push(use_info);
    }

    pass.buffer_uses = deduped;
    Ok(())
}

fn sanitize_image_uses(
    pass: &mut PassMetadata,
    images: &[ImageMetadata],
    warnings: &mut Vec<String>,
) -> Result<(), RenderGraphError> {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SeenImageUse {
        General(ImageAccess),
        ColorAttachment,
        DepthAttachment,
    }

    let mut seen = BTreeMap::<u32, SeenImageUse>::new();
    let mut deduped = Vec::with_capacity(pass.image_uses.len());

    for use_info in pass.image_uses.drain(..) {
        let (image, seen_use) = match use_info {
            ImageUse::General { image, access } => (image, SeenImageUse::General(access)),
            ImageUse::ColorAttachment { image, .. } => (image, SeenImageUse::ColorAttachment),
            ImageUse::DepthAttachment { image, .. } => (image, SeenImageUse::DepthAttachment),
        };

        if let Some(existing) = seen.get(&image.0).copied() {
            let image_name = &images[image.0 as usize].name;
            if existing == seen_use && matches!(seen_use, SeenImageUse::General(_)) {
                warnings.push(format!(
                    "pass '{}' declares image '{}' multiple times with the same access; ignoring the duplicate declaration",
                    pass.name, image_name
                ));
                continue;
            }

            return Err(RenderGraphError::Validation(format!(
                "pass '{}' declares image '{}' with conflicting usages ({existing:?} vs {seen_use:?})",
                pass.name, image_name
            )));
        }

        seen.insert(image.0, seen_use);
        deduped.push(use_info);
    }

    pass.image_uses = deduped;
    Ok(())
}

fn find_unused_warnings(metadata: &GraphMetadata) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut buffer_use_counts = vec![0usize; metadata.buffers.len()];
    let mut image_use_counts = vec![0usize; metadata.images.len()];

    for pass in &metadata.passes {
        for use_info in &pass.buffer_uses {
            buffer_use_counts[use_info.buffer.0 as usize] += 1;
        }
        for use_info in &pass.image_uses {
            image_use_counts[use_info.image().0 as usize] += 1;
        }
    }

    for (index, count) in buffer_use_counts.iter().enumerate() {
        if *count == 0 {
            warnings.push(format!(
                "buffer node '{}' is unused",
                metadata.buffers[index].name
            ));
        }
    }

    for (index, count) in image_use_counts.iter().enumerate() {
        if *count == 0 {
            warnings.push(format!(
                "image node '{}' is unused",
                metadata.images[index].name
            ));
        }
    }

    for (pass_index, pass) in metadata.passes.iter().enumerate() {
        if pass.buffer_uses.is_empty() && pass.image_uses.is_empty() {
            warnings.push(format!(
                "pass node '{}' is unused because it declares no resource accesses",
                pass.name
            ));
            continue;
        }

        let mut has_writes = false;
        let mut writes_escape = false;

        for use_info in &pass.buffer_uses {
            if use_info.access.is_write() {
                has_writes = true;
                let buffer = &metadata.buffers[use_info.buffer.0 as usize];
                if buffer.kind == BufferKind::Imported
                    || buffer_used_later(metadata, use_info.buffer, pass_index)
                {
                    writes_escape = true;
                }
            }
        }

        for use_info in &pass.image_uses {
            if use_info.is_write() {
                has_writes = true;
                let image = &metadata.images[use_info.image().0 as usize];
                if matches!(image.kind, ImageKind::Imported | ImageKind::Swapchain)
                    || image_used_later(metadata, use_info.image(), pass_index)
                {
                    writes_escape = true;
                }
            }
        }

        if !has_writes {
            warnings.push(format!(
                "pass node '{}' is unused because it only reads resources",
                pass.name
            ));
        } else if !writes_escape {
            warnings.push(format!(
                "pass node '{}' is unused because all of its outputs write to transient resources that are never consumed",
                pass.name
            ));
        }
    }

    warnings
}

fn buffer_used_later(metadata: &GraphMetadata, buffer: GraphBuffer, pass_index: usize) -> bool {
    metadata.passes.iter().skip(pass_index + 1).any(|pass| {
        pass.buffer_uses
            .iter()
            .any(|use_info| use_info.buffer == buffer)
    })
}

fn image_used_later(metadata: &GraphMetadata, image: GraphImage, pass_index: usize) -> bool {
    metadata.passes.iter().skip(pass_index + 1).any(|pass| {
        pass.image_uses
            .iter()
            .any(|use_info| use_info.image() == image)
    })
}

fn build_signature(metadata: &GraphMetadata, queues: QueueAvailability) -> GraphSignature {
    let mut signature = String::new();
    let _ = write!(
        signature,
        "queues:{}:{}:{}:{};",
        queues.has_async as u8,
        queues.async_same_family as u8,
        queues.has_copy as u8,
        queues.copy_same_family as u8,
    );

    for buffer in &metadata.buffers {
        let _ = write!(
            signature,
            "B:{}:{}:",
            buffer.kind as u8, buffer.initialized as u8
        );
        push_buffer_transition_signature(&mut signature, buffer.initial);
        signature.push(';');
    }

    for image in &metadata.images {
        let _ = write!(
            signature,
            "I:{}:{}:{}:{}:{}:",
            image.kind as u8,
            image.extent.width,
            image.extent.height,
            image.extent.depth,
            image.aspect.as_raw(),
        );
        let _ = write!(signature, "{}:", image.initialized as u8);
        push_image_transition_signature(&mut signature, image.initial);
        signature.push(';');
    }

    for pass in &metadata.passes {
        let _ = write!(
            signature,
            "P:{}:{}:{}:{}:",
            pass.kind as u8,
            pass.requested_queue as u8,
            pass.buffer_uses.len(),
            pass.image_uses.len(),
        );

        for use_info in &pass.buffer_uses {
            let _ = write!(signature, "b{}:{:?};", use_info.buffer.0, use_info.access);
        }

        for use_info in &pass.image_uses {
            push_image_use_signature(&mut signature, *use_info);
            signature.push(';');
        }

        signature.push('|');
    }

    GraphSignature(signature)
}

fn push_buffer_transition_signature(signature: &mut String, transition: BufferAccessTransition) {
    let _ = write!(
        signature,
        "{}:{}",
        transition.stage.as_raw(),
        transition.access.as_raw(),
    );
}

fn push_image_transition_signature(signature: &mut String, transition: ImageLayoutTransition) {
    push_image_layout_signature(signature, transition.layout);
    let _ = write!(
        signature,
        ":{}:{}",
        transition.stage.as_raw(),
        transition.access.as_raw(),
    );
}

fn push_image_layout_signature(signature: &mut String, layout: ImageLayout) {
    match layout {
        ImageLayout::Undefined => signature.push_str("undefined"),
        ImageLayout::Unified => signature.push_str("unified"),
        ImageLayout::General => signature.push_str("general"),
        ImageLayout::Compute => signature.push_str("compute"),
        ImageLayout::Fragment => signature.push_str("fragment"),
        ImageLayout::Color => signature.push_str("color"),
        ImageLayout::TransferDst => signature.push_str("transfer-dst"),
        ImageLayout::Present => signature.push_str("present"),
        ImageLayout::Custom(layout) => {
            let _ = write!(signature, "custom:{}", layout.as_raw());
        }
    }
}

fn push_image_use_signature(signature: &mut String, use_info: ImageUse) {
    match use_info {
        ImageUse::General { image, access } => {
            let _ = write!(signature, "g{}:{access:?}", image.0);
        }
        ImageUse::ColorAttachment { image, desc } => {
            let _ = write!(
                signature,
                "c{}:{}:{}",
                image.0,
                desc.load_op.as_raw(),
                desc.store_op.as_raw(),
            );
        }
        ImageUse::DepthAttachment { image, desc } => {
            let _ = write!(
                signature,
                "d{}:{}:{}",
                image.0,
                desc.load_op.as_raw(),
                desc.store_op.as_raw(),
            );
        }
    }
}

fn compile_metadata(
    metadata: &GraphMetadata,
    queues: QueueAvailability,
) -> Result<CompiledPlan, RenderGraphError> {
    let mut compiled = metadata
        .passes
        .iter()
        .map(|_| CompiledPass {
            queue: QueueKind::Graphics,
            buffer_barriers: Vec::new(),
            image_barriers: Vec::new(),
            post_image_barriers: Vec::new(),
            wait_passes: Vec::new(),
            render: None,
            waits_on_swapchain_acquire: false,
            signals_swapchain_present: false,
            swapchain_resource: None,
        })
        .collect::<Vec<_>>();

    let mut buffer_states = metadata
        .buffers
        .iter()
        .map(|buffer| TrackedBufferState {
            transition: buffer.initial,
            initialized: buffer.initialized,
            queue: None,
            write: false,
            last_pass: None,
        })
        .collect::<Vec<_>>();

    let mut image_states = metadata
        .images
        .iter()
        .map(|image| TrackedImageState {
            transition: image.initial,
            initialized: image.initialized,
            queue: None,
            write: false,
            last_pass: None,
        })
        .collect::<Vec<_>>();

    let mut swapchain_resource: Option<GraphImage> = None;
    let mut first_swapchain_use: Option<usize> = None;
    let mut last_swapchain_use: Option<usize> = None;

    for (index, pass) in metadata.passes.iter().enumerate() {
        let uses_swapchain = pass.image_uses.iter().any(|use_info| {
            metadata.images[use_info.image().0 as usize].kind == ImageKind::Swapchain
        });
        let resolved_queue = resolve_queue(pass.kind, pass.requested_queue, uses_swapchain, queues);
        compiled[index].queue = resolved_queue;

        if pass.kind == PassKind::Render {
            compiled[index].render = Some(build_render_targets(pass, metadata)?);
        }

        for buffer_use in &pass.buffer_uses {
            let resource = &metadata.buffers[buffer_use.buffer.0 as usize];
            let state = &mut buffer_states[buffer_use.buffer.0 as usize];
            let transition = buffer_use.access.transition();

            if !state.initialized && buffer_use.access.requires_initialized() {
                return Err(RenderGraphError::Validation(format!(
                    "pass '{}' reads transient or uninitialized buffer '{}' before it is written",
                    pass.name, resource.name
                )));
            }

            if let Some(last_pass) = state.last_pass
                && state.queue != Some(resolved_queue)
                && (state.write || buffer_use.access.is_write())
            {
                push_unique_wait(&mut compiled[index].wait_passes, last_pass);
            }

            if buffer_barrier_needed(*state, transition, buffer_use.access.is_write()) {
                compiled[index].buffer_barriers.push(CompiledBufferBarrier {
                    buffer: buffer_use.buffer,
                    from: state.transition,
                    to: transition,
                });
            }

            *state = TrackedBufferState {
                transition,
                initialized: true,
                queue: Some(resolved_queue),
                write: buffer_use.access.is_write(),
                last_pass: Some(index),
            };
        }

        for image_use in &pass.image_uses {
            let image = image_use.image();
            let resource = &metadata.images[image.0 as usize];
            let state = &mut image_states[image.0 as usize];
            let transition = image_use.transition();

            if resource.kind == ImageKind::Swapchain {
                if let Some(existing) = swapchain_resource {
                    if existing != image {
                        return Err(RenderGraphError::Validation(
                            "V1 render graph currently supports at most one imported swapchain image"
                                .into(),
                        ));
                    }
                } else {
                    swapchain_resource = Some(image);
                }
                first_swapchain_use.get_or_insert(index);
                last_swapchain_use = Some(index);
                compiled[index].swapchain_resource = Some(image);
            }

            if !state.initialized && image_use.requires_initialized() {
                return Err(RenderGraphError::Validation(format!(
                    "pass '{}' reads transient or uninitialized image '{}' before it is written",
                    pass.name, resource.name
                )));
            }

            if let Some(last_pass) = state.last_pass
                && state.queue != Some(resolved_queue)
                && (state.write || image_use.is_write())
            {
                push_unique_wait(&mut compiled[index].wait_passes, last_pass);
            }

            if image_barrier_needed(*state, transition, image_use.is_write()) {
                compiled[index].image_barriers.push(CompiledImageBarrier {
                    image,
                    from: state.transition,
                    to: transition,
                });
            }

            *state = TrackedImageState {
                transition,
                initialized: true,
                queue: Some(resolved_queue),
                write: image_use.is_write(),
                last_pass: Some(index),
            };
        }
    }

    if let (Some(resource), Some(first_use), Some(last_use)) =
        (swapchain_resource, first_swapchain_use, last_swapchain_use)
    {
        compiled[first_use].waits_on_swapchain_acquire = true;
        compiled[last_use].signals_swapchain_present = true;
        compiled[last_use].swapchain_resource = Some(resource);

        let final_state = image_states[resource.0 as usize];
        if final_state.initialized {
            compiled[last_use]
                .post_image_barriers
                .push(CompiledImageBarrier {
                    image: resource,
                    from: final_state.transition,
                    to: ImageLayoutTransition::PRESENT,
                });
        }
    }

    Ok(CompiledPlan { passes: compiled })
}

fn build_render_targets(
    pass: &PassMetadata,
    metadata: &GraphMetadata,
) -> Result<CompiledRenderTargets, RenderGraphError> {
    let mut colors = Vec::new();
    let mut depth = None;

    for image_use in &pass.image_uses {
        match image_use {
            ImageUse::ColorAttachment { image, desc } => colors.push(CompiledColorAttachment {
                image: *image,
                desc: *desc,
            }),
            ImageUse::DepthAttachment { image, desc } => {
                if depth.is_some() {
                    return Err(RenderGraphError::Validation(format!(
                        "render pass '{}' declares more than one depth attachment",
                        pass.name
                    )));
                }
                depth = Some(CompiledDepthAttachment {
                    image: *image,
                    desc: *desc,
                });
            }
            ImageUse::General { .. } => {}
        }
    }

    if colors.is_empty() && depth.is_none() {
        return Err(RenderGraphError::Validation(format!(
            "render pass '{}' must declare at least one color or depth attachment",
            pass.name
        )));
    }

    let extent = if let Some(color) = colors.first() {
        metadata.images[color.image.0 as usize].extent
    } else {
        metadata.images[depth.as_ref().unwrap().image.0 as usize].extent
    };

    Ok(CompiledRenderTargets {
        colors,
        depth,
        area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
        },
    })
}

fn format_graph_text(metadata: &GraphMetadata, plan: &CompiledPlan, warnings: &[String]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "RenderGraph");

    if !warnings.is_empty() {
        let _ = writeln!(out, "Warnings:");
        for warning in warnings {
            let _ = writeln!(out, "  - {warning}");
        }
    }

    let _ = writeln!(out, "Buffers:");
    for (index, buffer) in metadata.buffers.iter().enumerate() {
        let _ = writeln!(
            out,
            "  B{index} {} kind={} initialized={} initial={}",
            buffer.name,
            buffer_kind_name(buffer.kind),
            buffer.initialized,
            format_buffer_transition(buffer.initial),
        );
    }

    let _ = writeln!(out, "Images:");
    for (index, image) in metadata.images.iter().enumerate() {
        let _ = writeln!(
            out,
            "  I{index} {} kind={} extent={}x{}x{} initialized={} initial={}",
            image.name,
            image_kind_name(image.kind),
            image.extent.width,
            image.extent.height,
            image.extent.depth,
            image.initialized,
            format_image_transition(image.initial),
        );
    }

    let _ = writeln!(out, "Passes:");
    for (index, (pass, compiled)) in metadata.passes.iter().zip(&plan.passes).enumerate() {
        let _ = writeln!(
            out,
            "  P{index} {} kind={} queue={}",
            pass.name,
            pass_kind_name(pass.kind),
            queue_kind_name(compiled.queue),
        );

        if pass.buffer_uses.is_empty() && pass.image_uses.is_empty() {
            let _ = writeln!(out, "    uses: none");
        } else {
            for use_info in &pass.buffer_uses {
                let _ = writeln!(
                    out,
                    "    buffer B{} {} access={}",
                    use_info.buffer.0,
                    metadata.buffers[use_info.buffer.0 as usize].name,
                    buffer_access_name(use_info.access),
                );
            }

            for use_info in &pass.image_uses {
                let image = use_info.image();
                let _ = writeln!(
                    out,
                    "    image I{} {} usage={}",
                    image.0,
                    metadata.images[image.0 as usize].name,
                    image_use_name(*use_info),
                );
            }
        }

        if !compiled.wait_passes.is_empty() {
            let _ = writeln!(out, "    waits: {:?}", compiled.wait_passes);
        }

        for barrier in &compiled.buffer_barriers {
            let _ = writeln!(
                out,
                "    buffer barrier B{} {} -> {}",
                barrier.buffer.0,
                format_buffer_transition(barrier.from),
                format_buffer_transition(barrier.to),
            );
        }

        for barrier in &compiled.image_barriers {
            let _ = writeln!(
                out,
                "    image barrier I{} {} -> {}",
                barrier.image.0,
                format_image_transition(barrier.from),
                format_image_transition(barrier.to),
            );
        }

        for barrier in &compiled.post_image_barriers {
            let _ = writeln!(
                out,
                "    post image barrier I{} {} -> {}",
                barrier.image.0,
                format_image_transition(barrier.from),
                format_image_transition(barrier.to),
            );
        }

        if compiled.waits_on_swapchain_acquire {
            let _ = writeln!(out, "    waits on swapchain acquire");
        }
        if compiled.signals_swapchain_present {
            let _ = writeln!(out, "    signals swapchain present");
        }
    }

    out
}

fn format_graph_dot(metadata: &GraphMetadata, plan: &CompiledPlan) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "digraph RenderGraph {{");
    let _ = writeln!(out, "  rankdir=LR;");
    let _ = writeln!(out, "  node [fontname=\"Helvetica\"];");
    let _ = writeln!(out, "  edge [fontname=\"Helvetica\"];");

    for (index, buffer) in metadata.buffers.iter().enumerate() {
        let _ = writeln!(
            out,
            "  b{index} [shape=ellipse, label=\"B{index}: {}\\n{} buffer\"];",
            escape_dot(&buffer.name),
            buffer_kind_name(buffer.kind),
        );
    }

    for (index, image) in metadata.images.iter().enumerate() {
        let _ = writeln!(
            out,
            "  i{index} [shape=ellipse, label=\"I{index}: {}\\n{} image\"];",
            escape_dot(&image.name),
            image_kind_name(image.kind),
        );
    }

    for queue in [
        QueueKind::Graphics,
        QueueKind::AsyncCompute,
        QueueKind::Copy,
    ] {
        let pass_indices = plan
            .passes
            .iter()
            .enumerate()
            .filter_map(|(index, pass)| (pass.queue == queue).then_some(index))
            .collect::<Vec<_>>();

        if pass_indices.is_empty() {
            continue;
        }

        let _ = writeln!(out, "  subgraph cluster_{} {{", queue_cluster_name(queue));
        let _ = writeln!(out, "    label=\"{}\";", queue_kind_name(queue));
        for index in pass_indices {
            let _ = writeln!(
                out,
                "    p{index} [shape=box, label=\"P{index}: {}\\n{}\"];",
                escape_dot(&metadata.passes[index].name),
                pass_kind_name(metadata.passes[index].kind),
            );
        }
        let _ = writeln!(out, "  }}");
    }

    for (pass_index, pass) in metadata.passes.iter().enumerate() {
        for use_info in &pass.buffer_uses {
            let label = buffer_access_name(use_info.access);
            if matches!(use_info.access, BufferAccess::StorageComputeReadWrite) {
                let _ = writeln!(
                    out,
                    "  b{} -> p{pass_index} [style=dashed, label=\"read {label}\"];",
                    use_info.buffer.0,
                );
                let _ = writeln!(
                    out,
                    "  p{pass_index} -> b{} [label=\"write {label}\"];",
                    use_info.buffer.0,
                );
            } else if use_info.access.is_write() {
                let _ = writeln!(
                    out,
                    "  p{pass_index} -> b{} [label=\"{label}\"];",
                    use_info.buffer.0,
                );
            } else {
                let _ = writeln!(
                    out,
                    "  b{} -> p{pass_index} [label=\"{label}\"];",
                    use_info.buffer.0,
                );
            }
        }

        for use_info in &pass.image_uses {
            let image = use_info.image().0;
            match use_info {
                ImageUse::General { access, .. }
                    if *access == ImageAccess::StorageComputeReadWrite =>
                {
                    let _ = writeln!(
                        out,
                        "  i{image} -> p{pass_index} [style=dashed, label=\"read {}\"];",
                        image_use_name(*use_info),
                    );
                    let _ = writeln!(
                        out,
                        "  p{pass_index} -> i{image} [label=\"write {}\"];",
                        image_use_name(*use_info),
                    );
                }
                _ if use_info.is_write() => {
                    let _ = writeln!(
                        out,
                        "  p{pass_index} -> i{image} [label=\"{}\"];",
                        image_use_name(*use_info),
                    );
                }
                _ => {
                    let _ = writeln!(
                        out,
                        "  i{image} -> p{pass_index} [label=\"{}\"];",
                        image_use_name(*use_info),
                    );
                }
            }
        }
    }

    let _ = writeln!(out, "}}");
    out
}

fn escape_dot(text: &str) -> String {
    text.replace('"', "\\\"")
}

fn queue_cluster_name(queue: QueueKind) -> &'static str {
    match queue {
        QueueKind::Graphics => "graphics",
        QueueKind::AsyncCompute => "async_compute",
        QueueKind::Copy => "copy",
    }
}

fn pass_kind_name(kind: PassKind) -> &'static str {
    match kind {
        PassKind::Render => "render",
        PassKind::Compute => "compute",
        PassKind::Copy => "copy",
    }
}

fn queue_kind_name(queue: QueueKind) -> &'static str {
    match queue {
        QueueKind::Graphics => "graphics",
        QueueKind::AsyncCompute => "async_compute",
        QueueKind::Copy => "copy",
    }
}

fn buffer_kind_name(kind: BufferKind) -> &'static str {
    match kind {
        BufferKind::Imported => "imported",
        BufferKind::Transient => "transient",
    }
}

fn image_kind_name(kind: ImageKind) -> &'static str {
    match kind {
        ImageKind::Imported => "imported",
        ImageKind::Transient => "transient",
        ImageKind::Swapchain => "swapchain",
    }
}

fn buffer_access_name(access: BufferAccess) -> &'static str {
    match access {
        BufferAccess::StorageComputeRead => "storage-compute-read",
        BufferAccess::StorageComputeWrite => "storage-compute-write",
        BufferAccess::StorageComputeReadWrite => "storage-compute-read-write",
        BufferAccess::TransferSrc => "transfer-src",
        BufferAccess::TransferDst => "transfer-dst",
        BufferAccess::Vertex => "vertex",
        BufferAccess::Index => "index",
        BufferAccess::Indirect => "indirect",
    }
}

fn image_use_name(use_info: ImageUse) -> &'static str {
    match use_info {
        ImageUse::General { access, .. } => match access {
            ImageAccess::SampledFragment => "sampled-fragment",
            ImageAccess::SampledCompute => "sampled-compute",
            ImageAccess::StorageComputeRead => "storage-compute-read",
            ImageAccess::StorageComputeWrite => "storage-compute-write",
            ImageAccess::StorageComputeReadWrite => "storage-compute-read-write",
            ImageAccess::TransferSrc => "transfer-src",
            ImageAccess::TransferDst => "transfer-dst",
        },
        ImageUse::ColorAttachment { .. } => "color-attachment",
        ImageUse::DepthAttachment { .. } => "depth-attachment",
    }
}

fn format_buffer_transition(transition: BufferAccessTransition) -> String {
    format!(
        "stage={} access={}",
        transition.stage.as_raw(),
        transition.access.as_raw()
    )
}

fn format_image_transition(transition: ImageLayoutTransition) -> String {
    let mut layout = String::new();
    push_image_layout_signature(&mut layout, transition.layout);
    format!(
        "layout={layout} stage={} access={}",
        transition.stage.as_raw(),
        transition.access.as_raw()
    )
}

fn resolve_queue(
    pass_kind: PassKind,
    requested: PassQueue,
    uses_swapchain: bool,
    queues: QueueAvailability,
) -> QueueKind {
    let mut resolved = match (pass_kind, requested) {
        (PassKind::Render, PassQueue::Graphics | PassQueue::Auto) => QueueKind::Graphics,
        (PassKind::Render, PassQueue::AsyncCompute | PassQueue::Copy) => {
            warn_queue_fallback("render passes only support the graphics queue in V1");
            QueueKind::Graphics
        }
        (PassKind::Compute, PassQueue::Graphics) => QueueKind::Graphics,
        (PassKind::Compute, PassQueue::AsyncCompute) => {
            if queues.has_async && queues.async_same_family {
                QueueKind::AsyncCompute
            } else {
                warn_queue_fallback(
                    "async compute requested but no same-family async queue is available; falling back to graphics",
                );
                QueueKind::Graphics
            }
        }
        (PassKind::Compute, PassQueue::Copy) => {
            warn_queue_fallback("compute passes cannot run on the copy queue in V1");
            QueueKind::Graphics
        }
        (PassKind::Compute, PassQueue::Auto) => {
            if queues.has_async && queues.async_same_family {
                QueueKind::AsyncCompute
            } else {
                QueueKind::Graphics
            }
        }
        (PassKind::Copy, PassQueue::Copy) => {
            if queues.has_copy && queues.copy_same_family {
                QueueKind::Copy
            } else {
                warn_queue_fallback(
                    "copy queue requested but no same-family copy queue is available; falling back to graphics",
                );
                QueueKind::Graphics
            }
        }
        (PassKind::Copy, PassQueue::Graphics) => QueueKind::Graphics,
        (PassKind::Copy, PassQueue::AsyncCompute) => {
            warn_queue_fallback("copy passes cannot run on the async compute queue in V1");
            QueueKind::Graphics
        }
        (PassKind::Copy, PassQueue::Auto) => {
            if queues.has_copy && queues.copy_same_family {
                QueueKind::Copy
            } else {
                QueueKind::Graphics
            }
        }
    };

    if uses_swapchain && resolved != QueueKind::Graphics {
        warn_queue_fallback(
            "swapchain images are restricted to the graphics queue in V1; falling back to graphics",
        );
        resolved = QueueKind::Graphics;
    }

    resolved
}

fn warn_queue_fallback(message: &str) {
    if cfg!(debug_assertions) {
        log::warn!("{message}");
    }
}

fn push_unique_wait(wait_passes: &mut Vec<usize>, pass: usize) {
    if !wait_passes.contains(&pass) {
        wait_passes.push(pass);
    }
}

fn buffer_barrier_needed(
    state: TrackedBufferState,
    next: BufferAccessTransition,
    next_is_write: bool,
) -> bool {
    !state.initialized
        || state.write
        || next_is_write
        || state.transition.stage != next.stage
        || state.transition.access != next.access
}

fn image_barrier_needed(
    state: TrackedImageState,
    next: ImageLayoutTransition,
    next_is_write: bool,
) -> bool {
    !state.initialized
        || state.write
        || next_is_write
        || !same_image_layout(state.transition.layout, next.layout)
        || state.transition.stage != next.stage
        || state.transition.access != next.access
}

fn same_image_layout(lhs: ImageLayout, rhs: ImageLayout) -> bool {
    match (lhs, rhs) {
        (ImageLayout::Unified, ImageLayout::Unified) => true,
        (ImageLayout::Unified, _) | (_, ImageLayout::Unified) => false,
        _ => vk::ImageLayout::from(lhs) == vk::ImageLayout::from(rhs),
    }
}

fn depth_attachment_transition() -> ImageLayoutTransition {
    ImageLayoutTransition::custom(
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    )
}

fn infer_image_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graphics_only() -> QueueAvailability {
        QueueAvailability {
            has_async: false,
            async_same_family: false,
            has_copy: false,
            copy_same_family: false,
        }
    }

    #[test]
    fn async_queue_falls_back_when_missing() {
        let queue = resolve_queue(
            PassKind::Compute,
            PassQueue::AsyncCompute,
            false,
            graphics_only(),
        );

        assert_eq!(queue, QueueKind::Graphics);
    }

    #[test]
    fn render_pass_cannot_resolve_to_async() {
        let queue = resolve_queue(
            PassKind::Render,
            PassQueue::AsyncCompute,
            false,
            QueueAvailability {
                has_async: true,
                async_same_family: true,
                has_copy: false,
                copy_same_family: false,
            },
        );

        assert_eq!(queue, QueueKind::Graphics);
    }

    #[test]
    fn transient_buffer_read_before_write_is_rejected() {
        let graph = GraphMetadata {
            buffers: vec![BufferMetadata {
                name: "tmp".into(),
                initial: BufferAccessTransition::NONE,
                initialized: false,
                kind: BufferKind::Transient,
            }],
            images: vec![],
            passes: vec![PassMetadata {
                name: "reader".into(),
                kind: PassKind::Compute,
                requested_queue: PassQueue::Graphics,
                buffer_uses: vec![BufferUse {
                    buffer: GraphBuffer(0),
                    access: BufferAccess::StorageComputeRead,
                }],
                image_uses: vec![],
            }],
        };

        let prepared = prepare_metadata(&graph, graphics_only()).unwrap();
        let err = compile_metadata(&prepared.metadata, graphics_only()).unwrap_err();

        match err {
            RenderGraphError::Validation(message) => {
                assert!(message.contains("reads transient or uninitialized buffer"));
            }
            _ => panic!("expected validation error"),
        }
    }

    #[test]
    fn write_then_read_image_creates_barrier() {
        let graph = GraphMetadata {
            buffers: vec![],
            images: vec![
                ImageMetadata {
                    name: "img".into(),
                    extent: vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                    aspect: vk::ImageAspectFlags::COLOR,
                    initial: ImageLayoutTransition::UNDEFINED,
                    initialized: false,
                    kind: ImageKind::Transient,
                },
                ImageMetadata {
                    name: "rt".into(),
                    extent: vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                    aspect: vk::ImageAspectFlags::COLOR,
                    initial: ImageLayoutTransition::UNDEFINED,
                    initialized: false,
                    kind: ImageKind::Transient,
                },
            ],
            passes: vec![
                PassMetadata {
                    name: "writer".into(),
                    kind: PassKind::Compute,
                    requested_queue: PassQueue::Graphics,
                    buffer_uses: vec![],
                    image_uses: vec![ImageUse::General {
                        image: GraphImage(0),
                        access: ImageAccess::StorageComputeWrite,
                    }],
                },
                PassMetadata {
                    name: "reader".into(),
                    kind: PassKind::Render,
                    requested_queue: PassQueue::Graphics,
                    buffer_uses: vec![],
                    image_uses: vec![
                        ImageUse::General {
                            image: GraphImage(0),
                            access: ImageAccess::SampledFragment,
                        },
                        ImageUse::ColorAttachment {
                            image: GraphImage(1),
                            desc: ColorAttachmentDesc::dont_care(),
                        },
                    ],
                },
            ],
        };

        let plan = compile_metadata(&graph, graphics_only()).unwrap();

        assert_eq!(plan.passes[1].image_barriers.len(), 2);
        assert!(same_image_layout(
            plan.passes[1].image_barriers[0].from.layout,
            ImageLayout::General,
        ));
    }

    #[test]
    fn swapchain_usage_forces_graphics_queue_and_present_barrier() {
        let graph = GraphMetadata {
            buffers: vec![],
            images: vec![ImageMetadata {
                name: "backbuffer".into(),
                extent: vk::Extent3D {
                    width: 32,
                    height: 32,
                    depth: 1,
                },
                aspect: vk::ImageAspectFlags::COLOR,
                initial: ImageLayoutTransition::UNDEFINED,
                initialized: false,
                kind: ImageKind::Swapchain,
            }],
            passes: vec![PassMetadata {
                name: "copy-to-backbuffer".into(),
                kind: PassKind::Copy,
                requested_queue: PassQueue::Copy,
                buffer_uses: vec![],
                image_uses: vec![ImageUse::General {
                    image: GraphImage(0),
                    access: ImageAccess::TransferDst,
                }],
            }],
        };

        let plan = compile_metadata(
            &graph,
            QueueAvailability {
                has_async: false,
                async_same_family: false,
                has_copy: true,
                copy_same_family: true,
            },
        )
        .unwrap();

        assert_eq!(plan.passes[0].queue, QueueKind::Graphics);
        assert!(plan.passes[0].waits_on_swapchain_acquire);
        assert!(plan.passes[0].signals_swapchain_present);
        assert!(same_image_layout(
            plan.passes[0].post_image_barriers[0].to.layout,
            ImageLayout::Present,
        ));
    }

    #[test]
    fn conflicting_duplicate_image_usage_is_rejected() {
        let graph = GraphMetadata {
            buffers: vec![],
            images: vec![ImageMetadata {
                name: "present".into(),
                extent: vk::Extent3D {
                    width: 64,
                    height: 64,
                    depth: 1,
                },
                aspect: vk::ImageAspectFlags::COLOR,
                initial: ImageLayoutTransition::UNDEFINED,
                initialized: true,
                kind: ImageKind::Imported,
            }],
            passes: vec![PassMetadata {
                name: "bad-pass".into(),
                kind: PassKind::Render,
                requested_queue: PassQueue::Graphics,
                buffer_uses: vec![],
                image_uses: vec![
                    ImageUse::General {
                        image: GraphImage(0),
                        access: ImageAccess::SampledFragment,
                    },
                    ImageUse::ColorAttachment {
                        image: GraphImage(0),
                        desc: ColorAttachmentDesc::dont_care(),
                    },
                ],
            }],
        };

        let err = prepare_metadata(&graph, graphics_only()).unwrap_err();
        match err {
            RenderGraphError::Validation(message) => {
                assert!(message.contains("conflicting usages"));
            }
            _ => panic!("expected validation error"),
        }
    }

    #[test]
    fn unused_transient_output_generates_warning() {
        let graph = GraphMetadata {
            buffers: vec![],
            images: vec![ImageMetadata {
                name: "scratch".into(),
                extent: vk::Extent3D {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
                aspect: vk::ImageAspectFlags::COLOR,
                initial: ImageLayoutTransition::UNDEFINED,
                initialized: false,
                kind: ImageKind::Transient,
            }],
            passes: vec![PassMetadata {
                name: "orphan".into(),
                kind: PassKind::Compute,
                requested_queue: PassQueue::Graphics,
                buffer_uses: vec![],
                image_uses: vec![ImageUse::General {
                    image: GraphImage(0),
                    access: ImageAccess::StorageComputeWrite,
                }],
            }],
        };

        let prepared = prepare_metadata(&graph, graphics_only()).unwrap();
        assert!(
            prepared
                .warnings
                .iter()
                .any(|warning| warning.contains("pass node 'orphan' is unused"))
        );
    }

    #[test]
    fn signature_ignores_clear_color() {
        let base_pass = |color: [f32; 4]| PassMetadata {
            name: "present".into(),
            kind: PassKind::Render,
            requested_queue: PassQueue::Graphics,
            buffer_uses: vec![],
            image_uses: vec![ImageUse::ColorAttachment {
                image: GraphImage(0),
                desc: ColorAttachmentDesc::clear(color),
            }],
        };

        let base_image = ImageMetadata {
            name: "backbuffer".into(),
            extent: vk::Extent3D {
                width: 32,
                height: 32,
                depth: 1,
            },
            aspect: vk::ImageAspectFlags::COLOR,
            initial: ImageLayoutTransition::UNDEFINED,
            initialized: false,
            kind: ImageKind::Swapchain,
        };

        let graph_a = GraphMetadata {
            buffers: vec![],
            images: vec![base_image.clone()],
            passes: vec![base_pass([1.0, 0.0, 0.0, 1.0])],
        };
        let graph_b = GraphMetadata {
            buffers: vec![],
            images: vec![base_image],
            passes: vec![base_pass([0.0, 0.0, 1.0, 1.0])],
        };

        let prepared_a = prepare_metadata(&graph_a, graphics_only()).unwrap();
        let prepared_b = prepare_metadata(&graph_b, graphics_only()).unwrap();

        assert_eq!(prepared_a.signature, prepared_b.signature);
    }

    #[test]
    fn cached_render_targets_refresh_clear_color_from_current_metadata() {
        let base_pass = |color: [f32; 4]| PassMetadata {
            name: "present".into(),
            kind: PassKind::Render,
            requested_queue: PassQueue::Graphics,
            buffer_uses: vec![],
            image_uses: vec![ImageUse::ColorAttachment {
                image: GraphImage(0),
                desc: ColorAttachmentDesc::clear(color),
            }],
        };

        let base_image = ImageMetadata {
            name: "backbuffer".into(),
            extent: vk::Extent3D {
                width: 32,
                height: 32,
                depth: 1,
            },
            aspect: vk::ImageAspectFlags::COLOR,
            initial: ImageLayoutTransition::UNDEFINED,
            initialized: false,
            kind: ImageKind::Swapchain,
        };

        let graph_a = GraphMetadata {
            buffers: vec![],
            images: vec![base_image.clone()],
            passes: vec![base_pass([1.0, 0.0, 0.0, 1.0])],
        };
        let graph_b = GraphMetadata {
            buffers: vec![],
            images: vec![base_image],
            passes: vec![base_pass([0.0, 0.0, 1.0, 1.0])],
        };

        let prepared_a = prepare_metadata(&graph_a, graphics_only()).unwrap();
        let prepared_b = prepare_metadata(&graph_b, graphics_only()).unwrap();
        assert_eq!(prepared_a.signature, prepared_b.signature);

        let mut cached_plan = compile_metadata(&prepared_a.metadata, graphics_only()).unwrap();
        for (compiled, pass) in cached_plan
            .passes
            .iter_mut()
            .zip(&prepared_b.metadata.passes)
        {
            if pass.kind == PassKind::Render {
                compiled.render = Some(build_render_targets(pass, &prepared_b.metadata).unwrap());
            }
        }

        let render = cached_plan.passes[0].render.as_ref().unwrap();
        assert_eq!(render.colors[0].desc.clear.float32, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn dot_dump_contains_queue_cluster_and_edges() {
        let graph = GraphMetadata {
            buffers: vec![],
            images: vec![
                ImageMetadata {
                    name: "source".into(),
                    extent: vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                    aspect: vk::ImageAspectFlags::COLOR,
                    initial: ImageLayoutTransition::UNDEFINED,
                    initialized: true,
                    kind: ImageKind::Imported,
                },
                ImageMetadata {
                    name: "target".into(),
                    extent: vk::Extent3D {
                        width: 16,
                        height: 16,
                        depth: 1,
                    },
                    aspect: vk::ImageAspectFlags::COLOR,
                    initial: ImageLayoutTransition::UNDEFINED,
                    initialized: false,
                    kind: ImageKind::Swapchain,
                },
            ],
            passes: vec![PassMetadata {
                name: "present".into(),
                kind: PassKind::Render,
                requested_queue: PassQueue::Graphics,
                buffer_uses: vec![],
                image_uses: vec![
                    ImageUse::General {
                        image: GraphImage(0),
                        access: ImageAccess::SampledFragment,
                    },
                    ImageUse::ColorAttachment {
                        image: GraphImage(1),
                        desc: ColorAttachmentDesc::dont_care(),
                    },
                ],
            }],
        };

        let prepared = prepare_metadata(&graph, graphics_only()).unwrap();
        let plan = compile_metadata(&prepared.metadata, graphics_only()).unwrap();
        let dot = format_graph_dot(&prepared.metadata, &plan);

        assert!(dot.contains("cluster_graphics"));
        assert!(dot.contains("i0 -> p0"));
        assert!(dot.contains("p0 -> i1"));
    }
}
