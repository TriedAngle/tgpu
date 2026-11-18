use std::{
    ffi::{c_char, c_void},
    slice,
};

use tgpu::{
    BufferDesc, BufferImageCopy, BufferUses, CopyBufferToImageInfo, Extent2D, Format, HostAccess,
    Image, ImageAspect, ImageDesc, ImageLayout, ImageLayoutTransition, ImageView, MemoryPreset,
    Sampler, SamplerCreateInfo, SubmitInfo, ViewImage, ViewImageDesc,
};

use crate::{
    error::{Error, check_size, guard_box, label},
    instance::{TgpuDevice, TgpuQueue},
    types::{
        TgpuExtent3D, TgpuFilter, TgpuHostAccess, TgpuImageLayout, TgpuMemoryPreset,
        TgpuSamplerAddressMode, TgpuSamplerMipmapMode, image_uses,
    },
};

/// Owned image + view + optional sampler in one handle.
pub struct TgpuViewImage {
    handle: ViewImage,
}

impl TgpuViewImage {
    pub fn image(&self) -> TgpuImage {
        TgpuImage::new(&self.handle.image)
    }

    pub fn view(&self) -> TgpuImageView {
        TgpuImageView::new(&self.handle.view)
    }

    pub fn sampler(&self) -> TgpuSampler {
        match &self.handle.sampler {
            Some(sampler) => TgpuSampler::new(sampler),
            None => TgpuSampler::null(),
        }
    }
}

/// A borrowed image handle.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuImage {
    pub handle: *const c_void,
}

impl TgpuImage {
    pub fn new(image: &Image) -> Self {
        Self {
            handle: image as *const Image as *const c_void,
        }
    }

    pub fn null() -> Self {
        Self {
            handle: std::ptr::null(),
        }
    }

    pub fn as_tgpu(&self) -> Option<&Image> {
        unsafe { (self.handle as *const Image).as_ref() }
    }
}

/// A borrowed image view handle.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuImageView {
    pub handle: *const c_void,
}

impl TgpuImageView {
    pub fn new(view: &ImageView) -> Self {
        Self {
            handle: view as *const ImageView as *const c_void,
        }
    }

    pub fn null() -> Self {
        Self {
            handle: std::ptr::null(),
        }
    }

    pub fn as_tgpu(&self) -> Option<&ImageView> {
        unsafe { (self.handle as *const ImageView).as_ref() }
    }
}

/// A borrowed sampler handle.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuSampler {
    pub handle: *const c_void,
}

impl TgpuSampler {
    pub fn new(sampler: &Sampler) -> Self {
        Self {
            handle: sampler as *const Sampler as *const c_void,
        }
    }

    pub fn null() -> Self {
        Self {
            handle: std::ptr::null(),
        }
    }

    pub fn as_tgpu(&self) -> Option<&Sampler> {
        unsafe { (self.handle as *const Sampler).as_ref() }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuImageDesc {
    pub struct_size: usize,
    pub label: *const c_char,
    /// A `VkFormat` value.
    pub format: i32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    /// A combination of `TGPU_IMAGE_USAGE_*`.
    pub usage: u32,
    pub memory: TgpuMemoryPreset,
    pub host_access: TgpuHostAccess,
    pub initial_layout: TgpuImageLayout,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuSamplerDesc {
    pub struct_size: usize,
    pub label: *const c_char,
    pub mag: TgpuFilter,
    pub min: TgpuFilter,
    pub mipmap: TgpuSamplerMipmapMode,
    pub address_u: TgpuSamplerAddressMode,
    pub address_v: TgpuSamplerAddressMode,
    pub address_w: TgpuSamplerAddressMode,
    /// `0` disables anisotropic filtering.
    pub anisotropy: f32,
    pub min_lod: f32,
    pub max_lod: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TgpuViewImageDesc {
    pub struct_size: usize,
    pub image: TgpuImageDesc,
    /// Optional; pass NULL for an image view without a sampler.
    pub sampler: *const TgpuSamplerDesc,
}

impl TgpuImageDesc {
    fn to_tgpu(&self) -> Result<ImageDesc<'static>, Error> {
        Ok(ImageDesc {
            label: unsafe { label(self.label) },
            format: Format::from_raw(self.format),
            extent: TgpuExtent3D {
                width: self.width,
                height: self.height,
                depth: self.depth.max(1),
            }
            .into(),
            mip_levels: self.mip_levels.max(1),
            array_layers: self.array_layers.max(1),
            usage: image_uses(self.usage),
            memory: self.memory.into(),
            host_access: self.host_access.into(),
            initial_layout: self.initial_layout.into(),
            ..Default::default()
        })
    }
}

fn sampler_desc(desc: &TgpuSamplerDesc) -> SamplerCreateInfo<'static> {
    SamplerCreateInfo {
        label: unsafe { label(desc.label) },
        mag: desc.mag.into(),
        min: desc.min.into(),
        mipmap: desc.mipmap.into(),
        address_u: desc.address_u.into(),
        address_v: desc.address_v.into(),
        address_w: desc.address_w.into(),
        anisotropy: (desc.anisotropy > 0.0).then_some(desc.anisotropy),
        compare: None,
        min_lod: desc.min_lod,
        max_lod: desc.max_lod,
    }
}

fn create_view_image(device: &tgpu::Device, desc: &TgpuViewImageDesc) -> Result<ViewImage, Error> {
    check_size::<TgpuViewImageDesc>(desc.struct_size, "TgpuViewImageDesc")?;
    check_size::<TgpuImageDesc>(desc.image.struct_size, "TgpuImageDesc")?;

    let sampler = if desc.sampler.is_null() {
        None
    } else {
        let sampler = unsafe { &*desc.sampler };
        check_size::<TgpuSamplerDesc>(sampler.struct_size, "TgpuSamplerDesc")?;
        Some(sampler_desc(sampler))
    };

    Ok(device.create_view_image(&ViewImageDesc {
        image: desc.image.to_tgpu()?,
        sampler,
        // TODO: expose view type/format/aspect/mips/layers.
        ..Default::default()
    })?)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_device_create_view_image(
    device: *mut TgpuDevice,
    desc: *const TgpuViewImageDesc,
) -> *mut TgpuViewImage {
    guard_box(|| {
        let device =
            unsafe { device.as_ref() }.ok_or_else(|| Error::validation("device is NULL"))?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| Error::validation("TgpuViewImageDesc is NULL"))?;
        Ok(Box::new(TgpuViewImage {
            handle: create_view_image(device.handle(), desc)?,
        }))
    })
}

/// Creates a 2D texture and uploads tightly packed RGBA8 pixels into it.
///
/// The image is left in `TGPU_IMAGE_LAYOUT_FRAGMENT` for sampling. Pass `data`
/// as NULL to create an uninitialised texture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_queue_create_texture_2d(
    queue: *const TgpuQueue,
    device: *mut TgpuDevice,
    desc: *const TgpuViewImageDesc,
    data: *const u8,
    data_len: usize,
) -> *mut TgpuViewImage {
    guard_box(|| {
        let queue = unsafe { queue.as_ref() }.ok_or_else(|| Error::validation("queue is NULL"))?;
        let device =
            unsafe { device.as_ref() }.ok_or_else(|| Error::validation("device is NULL"))?;
        let desc = unsafe { desc.as_ref() }
            .ok_or_else(|| Error::validation("TgpuViewImageDesc is NULL"))?;

        let image = create_view_image(device.handle(), desc)?;
        if data.is_null() || data_len == 0 {
            return Ok(Box::new(TgpuViewImage { handle: image }));
        }

        let upload = device.handle().create_buffer(&BufferDesc {
            label: None,
            size: data_len,
            usage: BufferUses::COPY_SRC,
            memory: MemoryPreset::Upload,
            host_access: HostAccess::WriteSequential,
            ..Default::default()
        })?;
        upload.write(unsafe { slice::from_raw_parts(data, data_len) }, 0);

        let extent = Extent2D::new(desc.image.width, desc.image.height);
        let mut recorder = queue.handle().record();
        recorder.image_transition(
            &image.image,
            ImageLayoutTransition::new(ImageLayout::TransferDst),
        );
        recorder.copy_buffer_to_image(&CopyBufferToImageInfo {
            src: &upload,
            dst: &image.image,
            dst_layout: ImageLayout::TransferDst,
            regions: &[BufferImageCopy::image_2d(ImageAspect::COLOR, extent)],
        });
        recorder.image_transition(&image.image, ImageLayoutTransition::FRAGMENT);

        let signal = queue.handle().submit(SubmitInfo {
            records: &[recorder.finish()],
            ..Default::default()
        });
        queue.handle().timeline.wait(signal, None);
        drop(upload);

        Ok(Box::new(TgpuViewImage { handle: image }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_view_image_destroy(image: *mut TgpuViewImage) {
    if !image.is_null() {
        drop(unsafe { Box::from_raw(image) });
    }
}

/// Borrowed image handle. Valid while the [`TgpuViewImage`] is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_view_image_image(image: *const TgpuViewImage) -> TgpuImage {
    match unsafe { image.as_ref() } {
        Some(image) => image.image(),
        None => TgpuImage::null(),
    }
}

/// Borrowed view handle. Valid while the [`TgpuViewImage`] is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_view_image_view(image: *const TgpuViewImage) -> TgpuImageView {
    match unsafe { image.as_ref() } {
        Some(image) => image.view(),
        None => TgpuImageView::null(),
    }
}

/// Borrowed sampler handle; `handle` is NULL when the image has no sampler.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tgpu_view_image_sampler(image: *const TgpuViewImage) -> TgpuSampler {
    match unsafe { image.as_ref() } {
        Some(image) => image.sampler(),
        None => TgpuSampler::null(),
    }
}
