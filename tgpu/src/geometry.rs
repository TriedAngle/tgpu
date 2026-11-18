use ash::vk;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn to_3d(self, depth: u32) -> Extent3D {
        Extent3D {
            width: self.width,
            height: self.height,
            depth,
        }
    }
}

impl From<[u32; 2]> for Extent2D {
    fn from(value: [u32; 2]) -> Self {
        Self {
            width: value[0],
            height: value[1],
        }
    }
}

impl From<(u32, u32)> for Extent2D {
    fn from(value: (u32, u32)) -> Self {
        Self {
            width: value.0,
            height: value.1,
        }
    }
}

impl From<Extent2D> for [u32; 2] {
    fn from(value: Extent2D) -> Self {
        [value.width, value.height]
    }
}

impl From<vk::Extent2D> for Extent2D {
    fn from(value: vk::Extent2D) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

impl From<Extent2D> for vk::Extent2D {
    fn from(value: Extent2D) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3D {
    pub const fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    pub const fn xy(self) -> Extent2D {
        Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

impl From<[u32; 3]> for Extent3D {
    fn from(value: [u32; 3]) -> Self {
        Self {
            width: value[0],
            height: value[1],
            depth: value[2],
        }
    }
}

impl From<(u32, u32, u32)> for Extent3D {
    fn from(value: (u32, u32, u32)) -> Self {
        Self {
            width: value.0,
            height: value.1,
            depth: value.2,
        }
    }
}

impl From<Extent3D> for [u32; 3] {
    fn from(value: Extent3D) -> Self {
        [value.width, value.height, value.depth]
    }
}

impl From<vk::Extent3D> for Extent3D {
    fn from(value: vk::Extent3D) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

impl From<Extent3D> for vk::Extent3D {
    fn from(value: Extent3D) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Offset {
    pub x: i32,
    pub y: i32,
}

impl Offset {
    pub const ZERO: Self = Self { x: 0, y: 0 };

    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

impl From<[i32; 2]> for Offset {
    fn from(value: [i32; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<(i32, i32)> for Offset {
    fn from(value: (i32, i32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<vk::Offset2D> for Offset {
    fn from(value: vk::Offset2D) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<Offset> for vk::Offset2D {
    fn from(value: Offset) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Offset3D {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl From<[i32; 3]> for Offset3D {
    fn from(value: [i32; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<(i32, i32, i32)> for Offset3D {
    fn from(value: (i32, i32, i32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl From<vk::Offset3D> for Offset3D {
    fn from(value: vk::Offset3D) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Offset3D> for vk::Offset3D {
    fn from(value: Offset3D) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Rect {
    pub offset: Offset,
    pub extent: Extent2D,
}

impl Rect {
    pub const fn new(offset: Offset, extent: Extent2D) -> Self {
        Self { offset, extent }
    }

    pub const fn from_extent(extent: Extent2D) -> Self {
        Self {
            offset: Offset::ZERO,
            extent,
        }
    }
}

impl From<vk::Rect2D> for Rect {
    fn from(value: vk::Rect2D) -> Self {
        Self {
            offset: value.offset.into(),
            extent: value.extent.into(),
        }
    }
}

impl From<Rect> for vk::Rect2D {
    fn from(value: Rect) -> Self {
        Self {
            offset: value.offset.into(),
            extent: value.extent.into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Viewport {
    pub const fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        }
    }

    pub fn from_extent(extent: Extent2D) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Builds a viewport covering `rect`. Use this instead of
    /// [`Viewport::from_extent`] when rendering to a sub-region, since the
    /// viewport origin is in framebuffer space.
    pub fn from_rect(rect: Rect) -> Self {
        Self {
            x: rect.offset.x as f32,
            y: rect.offset.y as f32,
            width: rect.extent.width as f32,
            height: rect.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    pub fn from_extent_y_up(extent: Extent2D) -> Self {
        Self {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

impl From<vk::Viewport> for Viewport {
    fn from(value: vk::Viewport) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
            min_depth: value.min_depth,
            max_depth: value.max_depth,
        }
    }
}

impl From<Viewport> for vk::Viewport {
    fn from(value: Viewport) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
            min_depth: value.min_depth,
            max_depth: value.max_depth,
        }
    }
}
