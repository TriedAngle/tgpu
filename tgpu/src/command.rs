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
    GPUError, Queue, Semaphore,
    raw::{QueueImpl, RawDevice},
};

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

impl CommandRecorder {
    pub fn finish(&mut self) -> CommandBuffer {
        let recorder = unsafe { self.inner.get().as_mut().unwrap() };
        let buffer = unsafe { recorder.finish() };
        CommandBuffer { inner: buffer }
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

    // pub unsafe fn bind_pipeline(&mut self, pipeline: &RenderPipelineImpl) {
    //     unsafe {
    //         self.device.handle.cmd_bind_pipeline(
    //             self.handle,
    //             vk::PipelineBindPoint::GRAPHICS,
    //             pipeline.handle,
    //         );
    //     }
    // }

    pub unsafe fn viewport(&mut self, viewport: vk::Viewport) {
        unsafe {
            self.device
                .handle
                .cmd_set_viewport(self.buffer.handle, 0, &[viewport]);
        }
    }

    pub unsafe fn scissor(&mut self, scissor: vk::Rect2D) {
        unsafe {
            self.device
                .handle
                .cmd_set_scissor(self.buffer.handle, 0, &[scissor]);
        }
    }

    pub unsafe fn draw(&mut self, vertex: ops::Range<u32>, instance: ops::Range<u32>) {
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
}

pub struct SubmitInfo<'a> {
    records: &'a [CommandBuffer],
    wait_binary: &'a [(Semaphore, vk::PipelineStageFlags)],
    wait_timeline: &'a [(Semaphore, u64, vk::PipelineStageFlags)],
    signal_binary: &'a [Semaphore],
    signal_timeline: &'a [(Semaphore, u64)],
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

        let recorder = CommandRecorder {
            inner: Rc::new(UnsafeCell::new(inner)),
        };
        recorder
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
        for (_thread, pool) in self.pools.get_mut() {
            let ready = pool.ready.borrow();
            let ready = ready.iter().map(|b| b.handle).collect::<Vec<_>>();

            let dropped = pool.dropped.borrow();
            let dropped = dropped.iter().map(|b| b.handle).collect::<Vec<_>>();

            unsafe { self.device.wait_idle() };

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
