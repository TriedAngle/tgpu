#ifndef TGPU_H
#define TGPU_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define TGPU_BUFFER_USAGE_COPY_SRC (1 << 0)

#define TGPU_BUFFER_USAGE_COPY_DST (1 << 1)

#define TGPU_BUFFER_USAGE_INDEX (1 << 2)

#define TGPU_BUFFER_USAGE_VERTEX (1 << 3)

#define TGPU_BUFFER_USAGE_UNIFORM (1 << 4)

#define TGPU_BUFFER_USAGE_STORAGE (1 << 5)

#define TGPU_BUFFER_USAGE_DEVICE_ADDRESS (1 << 6)

#define TGPU_BUFFER_USAGE_INDIRECT (1 << 7)

/**
 * Returned by the `tgpu_bindless_add_*` functions on failure.
 */
#define TGPU_INVALID_HANDLE UINT32_MAX

#define TGPU_IMAGE_USAGE_COPY_SRC (1 << 0)

#define TGPU_IMAGE_USAGE_COPY_DST (1 << 1)

#define TGPU_IMAGE_USAGE_SAMPLED (1 << 2)

#define TGPU_IMAGE_USAGE_STORAGE (1 << 3)

#define TGPU_IMAGE_USAGE_COLOR_ATTACHMENT (1 << 4)

#define TGPU_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT (1 << 5)

#define TGPU_IMAGE_USAGE_INPUT_ATTACHMENT (1 << 6)

#define TGPU_IMAGE_USAGE_TRANSIENT_ATTACHMENT (1 << 7)

#define TGPU_QUEUE_GRAPHICS (1 << 0)

#define TGPU_QUEUE_COMPUTE (1 << 1)

#define TGPU_QUEUE_TRANSFER (1 << 2)

#define TGPU_QUEUE_SPARSE_BINDING (1 << 3)

#define TGPU_QUEUE_PROTECTED (1 << 4)

#define TGPU_QUEUE_GRAPHICS_COMPUTE (TGPU_QUEUE_GRAPHICS | TGPU_QUEUE_COMPUTE)

#define TGPU_DEVICE_FEATURE_FILL_MODE_NON_SOLID (1 << 0)

#define TGPU_DEVICE_FEATURE_MULTI_DRAW_INDIRECT (1 << 1)

#define TGPU_DEVICE_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE (1 << 2)

#define TGPU_DEVICE_FEATURE_DESCRIPTOR_INDEXING (1 << 3)

#define TGPU_DEVICE_FEATURE_BUFFER_DEVICE_ADDRESS (1 << 4)

/**
 * The features the rest of this library assumes are available.
 */
#define TGPU_DEVICE_FEATURES_MODERN ((TGPU_DEVICE_FEATURE_MULTI_DRAW_INDIRECT | TGPU_DEVICE_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE) | TGPU_DEVICE_FEATURE_DESCRIPTOR_INDEXING)

#define TGPU_ADAPTER_DEVICE_TYPE_OTHER 0

#define TGPU_ADAPTER_DEVICE_TYPE_INTEGRATED_GPU 1

#define TGPU_ADAPTER_DEVICE_TYPE_DISCRETE_GPU 2

#define TGPU_ADAPTER_DEVICE_TYPE_VIRTUAL_GPU 3

#define TGPU_ADAPTER_DEVICE_TYPE_CPU 4

/**
 * Result of every fallible call in the C API.
 *
 * Calls returning an opaque handle return `NULL` instead; the reason is
 * available through [`tgpu_last_error`].
 */
typedef enum {
    TGPU_STATUS_OK = 0,
    TGPU_STATUS_INVALID_ARGUMENT = 1,
    TGPU_STATUS_VALIDATION = 2,
    TGPU_STATUS_VULKAN = 3,
    TGPU_STATUS_SHADER = 4,
    /**
     * A Rust panic was caught at the ABI boundary. The library is still
     * usable, but the call that panicked had no effect.
     */
    TGPU_STATUS_PANIC = 5,
} TgpuStatus;

typedef enum {
    TGPU_IMAGE_LAYOUT_UNDEFINED = 0,
    TGPU_IMAGE_LAYOUT_GENERAL = 1,
    TGPU_IMAGE_LAYOUT_COMPUTE = 2,
    TGPU_IMAGE_LAYOUT_FRAGMENT = 3,
    TGPU_IMAGE_LAYOUT_COLOR = 4,
    TGPU_IMAGE_LAYOUT_DEPTH_STENCIL = 5,
    TGPU_IMAGE_LAYOUT_TRANSFER_DST = 6,
    TGPU_IMAGE_LAYOUT_PRESENT = 7,
} TgpuImageLayout;

typedef enum {
    TGPU_MEMORY_PRESET_GPU_ONLY = 0,
    TGPU_MEMORY_PRESET_UPLOAD = 1,
    TGPU_MEMORY_PRESET_READBACK = 2,
    TGPU_MEMORY_PRESET_DYNAMIC = 3,
    TGPU_MEMORY_PRESET_TRANSIENT_ATTACHMENT = 4,
} TgpuMemoryPreset;

typedef enum {
    TGPU_HOST_ACCESS_NONE = 0,
    TGPU_HOST_ACCESS_WRITE_SEQUENTIAL = 1,
    TGPU_HOST_ACCESS_READ_RANDOM = 2,
    TGPU_HOST_ACCESS_READ_WRITE_RANDOM = 3,
} TgpuHostAccess;

/**
 * The stage/access pair for one side of a buffer barrier.
 */
typedef enum {
    TGPU_BUFFER_ACCESS_NONE = 0,
    TGPU_BUFFER_ACCESS_TRANSFER_SRC = 1,
    TGPU_BUFFER_ACCESS_TRANSFER_DST = 2,
    TGPU_BUFFER_ACCESS_INDIRECT = 3,
    TGPU_BUFFER_ACCESS_COMPUTE_STORAGE_READ = 4,
    TGPU_BUFFER_ACCESS_COMPUTE_STORAGE_WRITE = 5,
    TGPU_BUFFER_ACCESS_COMPUTE_STORAGE_READ_WRITE = 6,
    TGPU_BUFFER_ACCESS_UNIFORM_READ = 7,
    TGPU_BUFFER_ACCESS_VERTEX_INPUT_READ = 8,
} TgpuBufferAccess;

typedef enum {
    TGPU_LOAD_OP_LOAD = 0,
    TGPU_LOAD_OP_CLEAR = 1,
    TGPU_LOAD_OP_DONT_CARE = 2,
} TgpuLoadOp;

typedef enum {
    TGPU_STORE_OP_STORE = 0,
    TGPU_STORE_OP_DONT_CARE = 1,
} TgpuStoreOp;

typedef enum {
    TGPU_FILTER_NEAREST = 0,
    TGPU_FILTER_LINEAR = 1,
} TgpuFilter;

typedef enum {
    TGPU_SAMPLER_MIPMAP_MODE_NEAREST = 0,
    TGPU_SAMPLER_MIPMAP_MODE_LINEAR = 1,
} TgpuSamplerMipmapMode;

typedef enum {
    TGPU_SAMPLER_ADDRESS_MODE_REPEAT = 0,
    TGPU_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT = 1,
    TGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2,
    TGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3,
    TGPU_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE = 4,
} TgpuSamplerAddressMode;

typedef enum {
    TGPU_COMPARE_OP_NEVER = 0,
    TGPU_COMPARE_OP_LESS = 1,
    TGPU_COMPARE_OP_EQUAL = 2,
    TGPU_COMPARE_OP_LESS_OR_EQUAL = 3,
    TGPU_COMPARE_OP_GREATER = 4,
    TGPU_COMPARE_OP_NOT_EQUAL = 5,
    TGPU_COMPARE_OP_GREATER_OR_EQUAL = 6,
    TGPU_COMPARE_OP_ALWAYS = 7,
} TgpuCompareOp;

typedef enum {
    TGPU_CULL_MODE_NONE = 0,
    TGPU_CULL_MODE_FRONT = 1,
    TGPU_CULL_MODE_BACK = 2,
    TGPU_CULL_MODE_FRONT_AND_BACK = 3,
} TgpuCullMode;

typedef enum {
    TGPU_FRONT_FACE_COUNTER_CLOCKWISE = 0,
    TGPU_FRONT_FACE_CLOCKWISE = 1,
} TgpuFrontFace;

typedef enum {
    TGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 0,
    TGPU_PRIMITIVE_TOPOLOGY_POINT_LIST = 1,
    TGPU_PRIMITIVE_TOPOLOGY_LINE_LIST = 2,
    TGPU_PRIMITIVE_TOPOLOGY_LINE_STRIP = 3,
    TGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4,
    TGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN = 5,
} TgpuPrimitiveTopology;

typedef enum {
    TGPU_BLEND_DISABLED = 0,
    TGPU_BLEND_ALPHA = 1,
} TgpuBlend;

typedef enum {
    TGPU_VERTEX_INPUT_RATE_VERTEX = 0,
    TGPU_VERTEX_INPUT_RATE_INSTANCE = 1,
} TgpuVertexInputRate;

typedef enum {
    TGPU_WINDOW_KIND_APP_KIT = 1,
    TGPU_WINDOW_KIND_XLIB = 2,
    TGPU_WINDOW_KIND_WIN32 = 3,
    TGPU_WINDOW_KIND_WAYLAND = 4,
} TgpuWindowKind;

typedef enum {
    TGPU_PRESENT_MODE_FIFO = 0,
    TGPU_PRESENT_MODE_FIFO_RELAXED = 1,
    TGPU_PRESENT_MODE_IMMEDIATE = 2,
    TGPU_PRESENT_MODE_MAILBOX = 3,
} TgpuPresentMode;

typedef struct TgpuBindlessHeap TgpuBindlessHeap;

typedef struct TgpuBuffer TgpuBuffer;

typedef struct TgpuCommandBuffer TgpuCommandBuffer;

typedef struct TgpuCommandRecorder TgpuCommandRecorder;

typedef struct TgpuComputePipeline TgpuComputePipeline;

/**
 * Owns the Vulkan device and every queue created with it.
 *
 * Queues are borrowed handles obtained from [`tgpu_device_queue`]; they live
 * as long as the device.
 */
typedef struct TgpuDevice TgpuDevice;

typedef struct TgpuInstance TgpuInstance;

typedef struct TgpuQueue TgpuQueue;

typedef struct TgpuRenderPipeline TgpuRenderPipeline;

typedef struct TgpuShader TgpuShader;

typedef struct TgpuSwapchain TgpuSwapchain;

/**
 * Owned image + view + optional sampler in one handle.
 */
typedef struct TgpuViewImage TgpuViewImage;

typedef struct {
    size_t struct_size;
    uint32_t max_read_buffers;
    uint32_t max_rw_buffers;
    uint32_t max_sampled_images;
    uint32_t max_storage_images;
    uint32_t max_samplers;
    uint32_t max_uniform_buffers;
} TgpuBindlessDesc;

/**
 * A borrowed image view handle.
 */
typedef struct {
    const void *handle;
} TgpuImageView;

/**
 * A borrowed sampler handle.
 */
typedef struct {
    const void *handle;
} TgpuSampler;

typedef struct {
    size_t struct_size;
    const char *label;
    uint64_t size;
    /**
     * A combination of `TGPU_BUFFER_USAGE_*`.
     */
    uint32_t usage;
    TgpuMemoryPreset memory;
    TgpuHostAccess host_access;
} TgpuBufferDesc;

/**
 * A borrowed image handle.
 */
typedef struct {
    const void *handle;
} TgpuImage;

typedef struct {
    uint32_t index;
    size_t flight_index;
    bool suboptimal;
    bool out_of_date;
} TgpuFrame;

typedef struct {
    TgpuImageView view;
    TgpuImageLayout layout;
    TgpuLoadOp load;
    TgpuStoreOp store;
    /**
     * Used when `load` is `TGPU_LOAD_OP_CLEAR`.
     */
    float clear_color[4];
} TgpuColorAttachment;

typedef struct {
    TgpuImageView view;
    TgpuImageLayout layout;
    TgpuLoadOp load;
    TgpuStoreOp store;
    float clear_depth;
    uint32_t clear_stencil;
} TgpuDepthAttachment;

typedef struct {
    size_t struct_size;
    int32_t x;
    int32_t y;
    uint32_t width;
    uint32_t height;
    const TgpuColorAttachment *colors;
    size_t color_count;
    const TgpuDepthAttachment *depth;
    const TgpuDepthAttachment *stencil;
} TgpuRenderDesc;

typedef struct {
    size_t struct_size;
    const char *label;
    /**
     * A `VkFormat` value.
     */
    int32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_levels;
    uint32_t array_layers;
    /**
     * A combination of `TGPU_IMAGE_USAGE_*`.
     */
    uint32_t usage;
    TgpuMemoryPreset memory;
    TgpuHostAccess host_access;
    TgpuImageLayout initial_layout;
} TgpuImageDesc;

typedef struct {
    size_t struct_size;
    const char *label;
    TgpuFilter mag;
    TgpuFilter min;
    TgpuSamplerMipmapMode mipmap;
    TgpuSamplerAddressMode address_u;
    TgpuSamplerAddressMode address_v;
    TgpuSamplerAddressMode address_w;
    /**
     * `0` disables anisotropic filtering.
     */
    float anisotropy;
    float min_lod;
    float max_lod;
} TgpuSamplerDesc;

typedef struct {
    size_t struct_size;
    TgpuImageDesc image;
    /**
     * Optional; pass NULL for an image view without a sampler.
     */
    const TgpuSamplerDesc *sampler;
} TgpuViewImageDesc;

typedef struct {
    size_t struct_size;
    const char *app_name;
    const char *engine_name;
} TgpuInstanceDesc;

typedef struct {
    size_t struct_size;
    /**
     * A combination of `TGPU_QUEUE_*`.
     */
    uint32_t flags;
    /**
     * Fail instead of accepting a family with extra capabilities.
     */
    bool strict;
    /**
     * Allow several requests to share one queue.
     */
    bool allow_fallback_share;
} TgpuQueueDesc;

typedef struct {
    size_t struct_size;
    /**
     * A combination of `TGPU_DEVICE_FEATURE_*`.
     */
    uint32_t features;
    size_t queue_count;
    /**
     * `queue_count` entries, or NULL for one graphics+compute queue.
     */
    const TgpuQueueDesc *queues;
} TgpuDeviceDesc;

typedef struct {
    /**
     * NUL-terminated UTF-8, truncated to 256 bytes.
     */
    char name[256];
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t device_type;
    /**
     * Raw `VkPhysicalDeviceProperties::apiVersion`.
     */
    uint32_t api_version;
    uint32_t driver_version;
} TgpuAdapterInfo;

typedef struct {
    size_t struct_size;
    const char *label;
    const TgpuShader *shader;
    const char *entry;
    /**
     * Optional bindless heap used as descriptor set 0.
     */
    const TgpuBindlessHeap *bindless;
    /**
     * `0` for no push constants.
     */
    uint32_t push_constant_size;
} TgpuComputePipelineDesc;

/**
 * A `VkFormat` value.
 */
typedef int32_t TgpuFormat;

typedef struct {
    uint32_t location;
    uint32_t binding;
    /**
     * A `VkFormat` value.
     */
    int32_t format;
    uint32_t offset;
} TgpuVertexAttribute;

typedef struct {
    uint32_t binding;
    uint32_t stride;
    TgpuVertexInputRate input_rate;
    const TgpuVertexAttribute *attributes;
    size_t attribute_count;
} TgpuVertexBufferLayout;

typedef struct {
    size_t struct_size;
    const char *label;
    const TgpuShader *vertex_shader;
    const char *vertex_entry;
    const TgpuShader *fragment_shader;
    const char *fragment_entry;
    const TgpuFormat *color_formats;
    size_t color_format_count;
    /**
     * `TGPU_FORMAT_UNDEFINED` for no depth attachment.
     */
    TgpuFormat depth_format;
    bool depth_test;
    bool depth_write;
    TgpuCompareOp depth_compare;
    /**
     * Optional bindless heap used as descriptor set 0.
     */
    const TgpuBindlessHeap *bindless;
    uint32_t push_constant_size;
    TgpuCullMode cull;
    TgpuFrontFace front_face;
    TgpuPrimitiveTopology topology;
    TgpuBlend blend;
    const TgpuVertexBufferLayout *vertex_buffers;
    size_t vertex_buffer_count;
} TgpuRenderPipelineDesc;

/**
 * A raw window and display, tagged by platform.
 *
 * `window` is an `NSView*` (AppKit), `Window` (Xlib), `HWND` (Win32) or
 * `wl_surface*` (Wayland). `display` is `Display*` on Xlib, `wl_display*` on
 * Wayland, and unused (NULL) on AppKit and Win32.
 */
typedef struct {
    TgpuWindowKind kind;
    void *display;
    void *window;
} TgpuWindow;

typedef struct {
    size_t struct_size;
    TgpuWindow window;
    uint32_t width;
    uint32_t height;
    /**
     * `0` uses the default (3).
     */
    uint32_t image_count;
    TgpuPresentMode present_mode;
} TgpuSwapchainDesc;

typedef struct {
    uint32_t width;
    uint32_t height;
} TgpuExtent2D;

#define TGPU_FORMAT_UNDEFINED 0

/**
 * `VK_FORMAT_R8G8B8A8_UNORM`
 */
#define TGPU_FORMAT_RGBA8_UNORM 37

/**
 * `VK_FORMAT_R32G32_SFLOAT`
 */
#define TGPU_FORMAT_RG32_FLOAT 103

/**
 * `VK_FORMAT_R32G32B32_SFLOAT`
 */
#define TGPU_FORMAT_RGB32_FLOAT 106

/**
 * `VK_FORMAT_D32_SFLOAT`
 */
#define TGPU_FORMAT_DEPTH32_FLOAT 126

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Version of the C ABI, bumped whenever an exported signature or
 * `#[repr(C)]` type changes incompatibly.
 */
uint32_t tgpu_api_version(void);

TgpuBindlessHeap *tgpu_device_create_bindless(TgpuDevice *device, const TgpuBindlessDesc *desc);

void tgpu_bindless_destroy(TgpuBindlessHeap *heap);

uint32_t tgpu_bindless_add_read_buffer(const TgpuBindlessHeap *heap_ptr,
                                       const TgpuBuffer *buffer_ptr);

uint32_t tgpu_bindless_add_rw_buffer(const TgpuBindlessHeap *heap_ptr,
                                     const TgpuBuffer *buffer_ptr);

uint32_t tgpu_bindless_add_uniform_buffer(const TgpuBindlessHeap *heap_ptr,
                                          const TgpuBuffer *buffer_ptr,
                                          uint64_t range);

TgpuStatus tgpu_bindless_update_uniform_buffer(const TgpuBindlessHeap *heap_ptr,
                                               uint32_t handle,
                                               const TgpuBuffer *buffer_ptr,
                                               uint64_t range);

uint32_t tgpu_bindless_add_storage_image(const TgpuBindlessHeap *heap_ptr,
                                         TgpuImageView image_view,
                                         TgpuImageLayout layout);

uint32_t tgpu_bindless_add_sampled_image(const TgpuBindlessHeap *heap_ptr,
                                         TgpuImageView image_view,
                                         TgpuImageLayout layout);

uint32_t tgpu_bindless_add_sampler(const TgpuBindlessHeap *heap_ptr, TgpuSampler sampler_value);

TgpuStatus tgpu_bindless_update_storage_image(const TgpuBindlessHeap *heap_ptr,
                                              uint32_t handle,
                                              TgpuImageView image_view,
                                              TgpuImageLayout layout);

TgpuStatus tgpu_bindless_update_sampled_image(const TgpuBindlessHeap *heap_ptr,
                                              uint32_t handle,
                                              TgpuImageView image_view,
                                              TgpuImageLayout layout);

TgpuStatus tgpu_bindless_update_sampler(const TgpuBindlessHeap *heap_ptr,
                                        uint32_t handle,
                                        TgpuSampler sampler_value);

/**
 * Creates a buffer and initialises it from `data` when non-NULL.
 */
TgpuBuffer *tgpu_device_create_buffer(TgpuDevice *device,
                                      const TgpuBufferDesc *desc,
                                      const uint8_t *data,
                                      size_t data_len);

void tgpu_buffer_destroy(TgpuBuffer *buffer);

uint64_t tgpu_buffer_size(const TgpuBuffer *buffer);

/**
 * Copies `len` bytes from `data` into the buffer at `offset`.
 */
TgpuStatus tgpu_buffer_write(TgpuBuffer *buffer, uint64_t offset, const uint8_t *data, size_t len);

/**
 * Copies `len` bytes out of the buffer at `offset` into `data`.
 */
TgpuStatus tgpu_buffer_read(TgpuBuffer *buffer, uint64_t offset, uint8_t *data, size_t len);

TgpuCommandRecorder *tgpu_queue_begin_commands(const TgpuQueue *queue);

/**
 * Finishes recording and consumes the recorder handle.
 */
TgpuCommandBuffer *tgpu_recorder_finish(TgpuCommandRecorder *recorder);

void tgpu_command_buffer_destroy(TgpuCommandBuffer *command_buffer);

TgpuStatus tgpu_recorder_transition_image(TgpuCommandRecorder *recorder,
                                          TgpuImage image,
                                          TgpuImageLayout layout);

TgpuStatus tgpu_recorder_transition_frame(TgpuCommandRecorder *recorder,
                                          const TgpuSwapchain *swapchain,
                                          TgpuFrame frame,
                                          TgpuImageLayout layout);

TgpuStatus tgpu_recorder_buffer_transition(TgpuCommandRecorder *recorder,
                                           const TgpuBuffer *buffer,
                                           TgpuBufferAccess from,
                                           TgpuBufferAccess to);

TgpuStatus tgpu_recorder_update_buffer(TgpuCommandRecorder *recorder,
                                       const TgpuBuffer *buffer,
                                       uint64_t offset,
                                       const uint8_t *data,
                                       size_t len);

TgpuStatus tgpu_recorder_bind_vertex_buffer(TgpuCommandRecorder *recorder,
                                            uint32_t slot,
                                            const TgpuBuffer *buffer,
                                            uint64_t offset);

TgpuStatus tgpu_recorder_bind_compute(TgpuCommandRecorder *recorder,
                                      const TgpuComputePipeline *pipeline,
                                      const TgpuBindlessHeap *heap,
                                      const uint8_t *push_constants,
                                      size_t push_constant_len);

TgpuStatus tgpu_recorder_dispatch(TgpuCommandRecorder *recorder,
                                  uint32_t x,
                                  uint32_t y,
                                  uint32_t z);

TgpuStatus tgpu_recorder_bind_render(TgpuCommandRecorder *recorder,
                                     const TgpuRenderPipeline *pipeline,
                                     const TgpuBindlessHeap *heap,
                                     const uint8_t *push_constants,
                                     size_t push_constant_len);

TgpuStatus tgpu_recorder_begin_render(TgpuCommandRecorder *recorder, const TgpuRenderDesc *desc);

TgpuStatus tgpu_recorder_end_render(TgpuCommandRecorder *recorder);

TgpuStatus tgpu_recorder_draw(TgpuCommandRecorder *recorder,
                              uint32_t first_vertex,
                              uint32_t vertex_count,
                              uint32_t first_instance,
                              uint32_t instance_count);

/**
 * Copies the last error message into `buffer` (always NUL terminated) and
 * returns its status. Returns [`TgpuStatus::Ok`] when nothing failed.
 */
TgpuStatus tgpu_last_error(char *buffer, size_t capacity);

TgpuViewImage *tgpu_device_create_view_image(TgpuDevice *device, const TgpuViewImageDesc *desc);

/**
 * Creates a 2D texture and uploads tightly packed RGBA8 pixels into it.
 *
 * The image is left in `TGPU_IMAGE_LAYOUT_FRAGMENT` for sampling. Pass `data`
 * as NULL to create an uninitialised texture.
 */
TgpuViewImage *tgpu_queue_create_texture_2d(const TgpuQueue *queue,
                                            TgpuDevice *device,
                                            const TgpuViewImageDesc *desc,
                                            const uint8_t *data,
                                            size_t data_len);

void tgpu_view_image_destroy(TgpuViewImage *image);

/**
 * Borrowed image handle. Valid while the [`TgpuViewImage`] is alive.
 */
TgpuImage tgpu_view_image_image(const TgpuViewImage *image);

/**
 * Borrowed view handle. Valid while the [`TgpuViewImage`] is alive.
 */
TgpuImageView tgpu_view_image_view(const TgpuViewImage *image);

/**
 * Borrowed sampler handle; `handle` is NULL when the image has no sampler.
 */
TgpuSampler tgpu_view_image_sampler(const TgpuViewImage *image);

/**
 * Creates a Vulkan instance.
 *
 * Pass `NULL` for `desc` to use the defaults. Returns `NULL` on failure; call
 * [`crate::tgpu_last_error`] for the reason.
 */
TgpuInstance *tgpu_instance_create(const TgpuInstanceDesc *desc);

void tgpu_instance_destroy(TgpuInstance *instance);

/**
 * Creates a device with the requested queues.
 *
 * `desc` may be NULL, in which case one graphics+compute queue is created.
 * Creates a device and writes the requested queue handles into `out_queues`.
 *
 * `out_queues` must have room for `desc.queue_count` entries (or one when
 * `desc` is NULL). Each queue is owned by the caller; destroy it with
 * [`tgpu_queue_destroy`].
 */
TgpuDevice *tgpu_device_create(TgpuInstance *instance,
                               const TgpuDeviceDesc *desc,
                               TgpuQueue **out_queues,
                               size_t out_capacity);

void tgpu_device_destroy(TgpuDevice *device);

void tgpu_device_wait_idle(TgpuDevice *device);

/**
 * Fills `out` with information about the selected adapter.
 */
TgpuStatus tgpu_device_adapter_info(TgpuDevice *device, TgpuAdapterInfo *out);

void tgpu_queue_destroy(TgpuQueue *queue);

/**
 * Copies the adapter name into `buffer`. Returns the number of bytes written
 * (excluding the NUL) or 0 when there is nothing to copy.
 */
size_t tgpu_device_name(TgpuDevice *device, char *buffer, size_t capacity);

TgpuShader *tgpu_device_create_shader(TgpuDevice *device,
                                      const char *label_ptr,
                                      const uint32_t *words,
                                      size_t word_count);

void tgpu_shader_destroy(TgpuShader *shader);

TgpuComputePipeline *tgpu_device_create_compute_pipeline(TgpuDevice *device,
                                                         const TgpuComputePipelineDesc *desc);

void tgpu_compute_pipeline_destroy(TgpuComputePipeline *pipeline);

TgpuRenderPipeline *tgpu_device_create_render_pipeline(TgpuDevice *device,
                                                       const TgpuRenderPipelineDesc *desc);

void tgpu_render_pipeline_destroy(TgpuRenderPipeline *pipeline);

TgpuSwapchain *tgpu_device_create_swapchain(TgpuDevice *device, const TgpuSwapchainDesc *desc);

void tgpu_swapchain_destroy(TgpuSwapchain *swapchain);

TgpuStatus tgpu_swapchain_extent(const TgpuSwapchain *swapchain, TgpuExtent2D *out);

/**
 * Returns the surface format as a `VkFormat` value.
 */
int32_t tgpu_swapchain_format(const TgpuSwapchain *swapchain);

size_t tgpu_swapchain_max_frames_in_flight(const TgpuSwapchain *swapchain);

/**
 * Waits for the frame slot, acquires an image, and recreates the swapchain
 * internally when the surface is out of date.
 */
TgpuStatus tgpu_swapchain_begin_frame(const TgpuQueue *queue,
                                      TgpuSwapchain *swapchain,
                                      TgpuFrame *out);

/**
 * Submits `records`, presents `frame`, and records the timeline value.
 */
TgpuStatus tgpu_swapchain_end_frame(const TgpuQueue *queue,
                                    TgpuSwapchain *swapchain,
                                    TgpuFrame frame,
                                    TgpuCommandBuffer *const *records,
                                    size_t record_count);

/**
 * Borrowed view of the acquired image, valid while the swapchain is alive.
 */
TgpuImageView tgpu_swapchain_view(const TgpuSwapchain *swapchain, TgpuFrame frame);

void tgpu_swapchain_set_present_mode(TgpuSwapchain *swapchain, TgpuPresentMode present_mode);

void tgpu_swapchain_set_extent(TgpuSwapchain *swapchain, TgpuExtent2D extent);

TgpuStatus tgpu_swapchain_recreate(TgpuSwapchain *swapchain);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* TGPU_H */
