#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryPreset {
    #[default]
    GpuOnly,
    Upload,
    Readback,
    Dynamic,
    TransientAttachment,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HostAccess {
    #[default]
    None,
    WriteSequential,
    ReadRandom,
    ReadWriteRandom,
}
