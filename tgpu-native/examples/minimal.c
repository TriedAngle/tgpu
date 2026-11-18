// minimal_latest, ported to the tgpu C API.
//
// The Rust example draws the scene into a G-buffer and shows it through egui.
// This port keeps the G-buffer and adds a fullscreen pass that samples it onto
// the swapchain. The egui panel is replaced by keys:
//
//   1 / 2  select texture      v  toggle vsync      esc  quit

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glfw_window.h"
#include "tgpu.h"

#define START_WIDTH 800
#define START_HEIGHT 600
#define MAX_FLIGHT 4
#define TEXTURE_SIZE 256

typedef struct {
    float position[3];
    float color[3];
    float tex_coord[2];
} Vertex;

typedef struct {
    float resolution[2];
    float anim_value;
    uint32_t num_data;
    uint32_t tex_id;
    uint32_t pad0[3];
} SceneInfo;

typedef struct {
    uint32_t vertex_buffer;
    float rotation_angle;
    uint32_t num_vertex;
    uint32_t pad0;
    uint32_t scene_info;
    uint32_t points;
    uint32_t texture_image;
    uint32_t texture_sampler;
    float color[4];
} PushData;

typedef struct {
    TgpuViewImage *color;
    TgpuViewImage *depth;
    TgpuExtent2D extent;
} GBuffer;

typedef struct {
    TgpuBuffer *buffer;
    uint32_t handle;
} SceneSlot;

static const Vertex VERTICES[6] = {
    {{0.0f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {0.5f, 0.5f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}},
    {{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.5f}},
    {{0.1f, -0.4f, 0.75f}, {0.3f, 0.3f, 0.3f}, {0.5f, 1.0f}},
    {{-0.4f, 0.6f, 0.25f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
    {{0.6f, 0.6f, 0.75f}, {0.7f, 0.7f, 0.7f}, {0.0f, 0.0f}},
};

static const float POINTS[4][2] = {
    {0.05f, 0.0f},
    {-0.05f, 0.0f},
    {0.0f, -0.05f},
    {0.0f, 0.05f},
};

static int g_image_id = 0;
static bool g_vsync = false;
static bool g_rebuild_swapchain = false;
static float g_clear_color[4] = {0.2f, 0.2f, 0.3f, 1.0f};

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    if (action != GLFW_PRESS) {
        return;
    }
    switch (key) {
        case GLFW_KEY_1:
            g_image_id = 0;
            break;
        case GLFW_KEY_2:
            g_image_id = 1;
            break;
        case GLFW_KEY_V:
            g_vsync = !g_vsync;
            g_rebuild_swapchain = true;
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        default:
            break;
    }
}

static void check(TgpuStatus status, const char *what) {
    if (status == TGPU_STATUS_OK) {
        return;
    }
    char message[512];
    tgpu_last_error(message, sizeof message);
    fprintf(stderr, "%s failed (%d): %s\n", what, (int)status, message);
    exit(EXIT_FAILURE);
}

static void check_pointer(const void *pointer, const char *what) {
    if (pointer) {
        return;
    }
    char message[512];
    tgpu_last_error(message, sizeof message);
    fprintf(stderr, "%s returned NULL: %s\n", what, message);
    exit(EXIT_FAILURE);
}

static uint32_t check_handle(uint32_t handle, const char *what) {
    if (handle != TGPU_INVALID_HANDLE) {
        return handle;
    }
    char message[512];
    tgpu_last_error(message, sizeof message);
    fprintf(stderr, "%s failed: %s\n", what, message);
    exit(EXIT_FAILURE);
}

static uint32_t *load_spirv(const char *path, size_t *out_words) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size <= 0 || size % 4 != 0) {
        fprintf(stderr, "%s is not a SPIR-V file\n", path);
        exit(EXIT_FAILURE);
    }
    uint32_t *words = malloc((size_t)size);
    if (!words || fread(words, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(file);
    *out_words = (size_t)size / 4;
    return words;
}

static uint8_t *make_texture(uint32_t size, uint32_t variant) {
    uint8_t *pixels = malloc((size_t)size * size * 4);
    if (!pixels) {
        return NULL;
    }
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            size_t offset = ((size_t)y * size + x) * 4;
            float fx = (float)x / (float)(size - 1);
            float fy = (float)y / (float)(size - 1);
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            float r, g, b;
            if (variant == 0) {
                float pulse = checker ? 1.0f : 0.35f;
                r = 255.0f * fx;
                g = 180.0f * pulse;
                b = 255.0f * (1.0f - fy);
            } else {
                float dx = fx - 0.5f;
                float dy = fy - 0.5f;
                float ring = fabsf(sinf(sqrtf(dx * dx + dy * dy) * 18.0f));
                r = 255.0f * ring;
                g = 255.0f * fy;
                b = 80.0f + 120.0f * fx;
            }
            pixels[offset + 0] = (uint8_t)r;
            pixels[offset + 1] = (uint8_t)g;
            pixels[offset + 2] = (uint8_t)b;
            pixels[offset + 3] = 255;
        }
    }
    return pixels;
}

static TgpuSamplerDesc default_sampler_desc(void) {
    TgpuSamplerDesc desc = {
        .struct_size = sizeof desc,
        .mag = TGPU_FILTER_LINEAR,
        .min = TGPU_FILTER_LINEAR,
        .mipmap = TGPU_SAMPLER_MIPMAP_MODE_LINEAR,
        .address_u = TGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .address_v = TGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .address_w = TGPU_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropy = 0.0f,
        .min_lod = 0.0f,
        .max_lod = 0.0f,
    };
    return desc;
}

static GBuffer create_gbuffer(TgpuDevice *device, TgpuExtent2D extent) {
    TgpuSamplerDesc sampler = default_sampler_desc();

    TgpuViewImageDesc color_desc = {
        .struct_size = sizeof color_desc,
        .image =
            {
                .struct_size = sizeof(TgpuImageDesc),
                .label = "gbuffer color",
                .format = TGPU_FORMAT_RGBA8_UNORM,
                .width = extent.width,
                .height = extent.height,
                .depth = 1,
                .mip_levels = 1,
                .array_layers = 1,
                .usage = TGPU_IMAGE_USAGE_COLOR_ATTACHMENT | TGPU_IMAGE_USAGE_SAMPLED,
                .memory = TGPU_MEMORY_PRESET_GPU_ONLY,
                .host_access = TGPU_HOST_ACCESS_NONE,
                .initial_layout = TGPU_IMAGE_LAYOUT_UNDEFINED,
            },
        .sampler = &sampler,
    };

    TgpuViewImageDesc depth_desc = {
        .struct_size = sizeof depth_desc,
        .image =
            {
                .struct_size = sizeof(TgpuImageDesc),
                .label = "gbuffer depth",
                .format = TGPU_FORMAT_DEPTH32_FLOAT,
                .width = extent.width,
                .height = extent.height,
                .depth = 1,
                .mip_levels = 1,
                .array_layers = 1,
                .usage = TGPU_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT,
                .memory = TGPU_MEMORY_PRESET_GPU_ONLY,
                .host_access = TGPU_HOST_ACCESS_NONE,
                .initial_layout = TGPU_IMAGE_LAYOUT_UNDEFINED,
            },
        .sampler = NULL,
    };

    GBuffer gbuffer = {0};
    gbuffer.color = tgpu_device_create_view_image(device, &color_desc);
    check_pointer(gbuffer.color, "create gbuffer color");
    gbuffer.depth = tgpu_device_create_view_image(device, &depth_desc);
    check_pointer(gbuffer.depth, "create gbuffer depth");
    gbuffer.extent = extent;
    return gbuffer;
}

static TgpuRenderPipeline *create_graphics_pipeline(
    TgpuDevice *device,
    TgpuBindlessHeap *bindless,
    TgpuShader *shader,
    const char *fragment_entry,
    TgpuFormat color_format,
    TgpuFormat depth_format,
    const char *label) {
    TgpuVertexAttribute attributes[3] = {
        {.location = 0, .binding = 0, .format = TGPU_FORMAT_RGB32_FLOAT, .offset = 0},
        {.location = 1,
         .binding = 0,
         .format = TGPU_FORMAT_RGB32_FLOAT,
         .offset = offsetof(Vertex, color)},
        {.location = 2,
         .binding = 0,
         .format = TGPU_FORMAT_RG32_FLOAT,
         .offset = offsetof(Vertex, tex_coord)},
    };
    TgpuVertexBufferLayout vertex_layout = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .input_rate = TGPU_VERTEX_INPUT_RATE_VERTEX,
        .attributes = attributes,
        .attribute_count = 3,
    };

    TgpuRenderPipelineDesc desc = {
        .struct_size = sizeof desc,
        .label = label,
        .vertex_shader = shader,
        .vertex_entry = "vertexMain",
        .fragment_shader = shader,
        .fragment_entry = fragment_entry,
        .color_formats = &color_format,
        .color_format_count = 1,
        .depth_format = depth_format,
        .depth_test = true,
        .depth_write = true,
        .depth_compare = TGPU_COMPARE_OP_LESS_OR_EQUAL,
        .bindless = bindless,
        .push_constant_size = sizeof(PushData),
        .cull = TGPU_CULL_MODE_NONE,
        .front_face = TGPU_FRONT_FACE_COUNTER_CLOCKWISE,
        .topology = TGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .blend = TGPU_BLEND_ALPHA,
        .vertex_buffers = &vertex_layout,
        .vertex_buffer_count = 1,
    };
    TgpuRenderPipeline *pipeline = tgpu_device_create_render_pipeline(device, &desc);
    check_pointer(pipeline, "create graphics pipeline");
    return pipeline;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    GLFWwindow *window = example_window_create("tgpu minimal latest (C)", START_WIDTH, START_HEIGHT);
    if (!window) {
        fprintf(stderr, "failed to create window\n");
        return EXIT_FAILURE;
    }
    glfwSetKeyCallback(window, key_callback);

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    TgpuInstanceDesc instance_desc = {
        .struct_size = sizeof instance_desc,
        .app_name = "tgpu minimal latest (C)",
        .engine_name = "tgpu-native",
    };
    TgpuInstance *instance = tgpu_instance_create(&instance_desc);
    check_pointer(instance, "tgpu_instance_create");

    TgpuQueueDesc queue_desc = {
        .struct_size = sizeof queue_desc,
        .flags = TGPU_QUEUE_GRAPHICS_COMPUTE,
        .strict = false,
        .allow_fallback_share = true,
    };
    TgpuDeviceDesc device_desc = {
        .struct_size = sizeof device_desc,
        .features = TGPU_DEVICE_FEATURES_MODERN,
        .queue_count = 1,
        .queues = &queue_desc,
    };
    TgpuQueue *queues[1] = {0};
    TgpuDevice *device = tgpu_device_create(instance, &device_desc, queues, 1);
    check_pointer(device, "tgpu_device_create");
    check_pointer(queues[0], "tgpu_device_create (queue)");
    const TgpuQueue *queue = queues[0];

    TgpuAdapterInfo adapter = {0};
    check(tgpu_device_adapter_info(device, &adapter), "tgpu_device_adapter_info");
    printf("adapter: %s\n", adapter.name);
    printf("keys: 1/2 texture, v vsync, esc quit\n");

    TgpuWindow native_window = example_window_tgpu_window(window);
    check_pointer(native_window.window, "example_window_tgpu_window");

    TgpuSwapchainDesc swapchain_desc = {
        .struct_size = sizeof swapchain_desc,
        .window = native_window,
        .width = (uint32_t)width,
        .height = (uint32_t)height,
        .image_count = 3,
        .present_mode = g_vsync ? TGPU_PRESENT_MODE_FIFO : TGPU_PRESENT_MODE_IMMEDIATE,
    };
    TgpuSwapchain *swapchain = tgpu_device_create_swapchain(device, &swapchain_desc);
    check_pointer(swapchain, "tgpu_device_create_swapchain");

    TgpuExtent2D extent = {0};
    check(tgpu_swapchain_extent(swapchain, &extent), "tgpu_swapchain_extent");
    TgpuFormat swapchain_format = tgpu_swapchain_format(swapchain);

    GBuffer gbuffer = create_gbuffer(device, extent);

    TgpuBindlessDesc bindless_desc = {
        .struct_size = sizeof bindless_desc,
        .max_read_buffers = 1,
        .max_rw_buffers = 1,
        .max_sampled_images = 3,
        .max_samplers = 1,
        .max_uniform_buffers = MAX_FLIGHT,
    };
    TgpuBindlessHeap *bindless = tgpu_device_create_bindless(device, &bindless_desc);
    check_pointer(bindless, "tgpu_device_create_bindless");

    uint32_t gbuffer_handle = check_handle(
        tgpu_bindless_add_sampled_image(
            bindless, tgpu_view_image_view(gbuffer.color), TGPU_IMAGE_LAYOUT_FRAGMENT),
        "add gbuffer sampled image");

    TgpuBufferDesc vertex_buffer_desc = {
        .struct_size = sizeof vertex_buffer_desc,
        .label = "minimal latest vertices",
        .size = sizeof VERTICES,
        .usage = TGPU_BUFFER_USAGE_VERTEX | TGPU_BUFFER_USAGE_STORAGE,
        .memory = TGPU_MEMORY_PRESET_DYNAMIC,
        .host_access = TGPU_HOST_ACCESS_WRITE_SEQUENTIAL,
    };
    TgpuBuffer *vertex_buffer = tgpu_device_create_buffer(
        device, &vertex_buffer_desc, (const uint8_t *)VERTICES, sizeof VERTICES);
    check_pointer(vertex_buffer, "create vertex buffer");

    TgpuBufferDesc points_buffer_desc = {
        .struct_size = sizeof points_buffer_desc,
        .label = "minimal latest points",
        .size = sizeof POINTS,
        .usage = TGPU_BUFFER_USAGE_STORAGE | TGPU_BUFFER_USAGE_COPY_DST,
        .memory = TGPU_MEMORY_PRESET_DYNAMIC,
        .host_access = TGPU_HOST_ACCESS_WRITE_SEQUENTIAL,
    };
    TgpuBuffer *points_buffer =
        tgpu_device_create_buffer(device, &points_buffer_desc, (const uint8_t *)POINTS, sizeof POINTS);
    check_pointer(points_buffer, "create points buffer");

    uint32_t vertex_handle = check_handle(
        tgpu_bindless_add_rw_buffer(bindless, vertex_buffer), "add vertex buffer");
    uint32_t points_handle = check_handle(
        tgpu_bindless_add_read_buffer(bindless, points_buffer), "add points buffer");

    TgpuSamplerDesc texture_sampler = default_sampler_desc();
    texture_sampler.address_u = TGPU_SAMPLER_ADDRESS_MODE_REPEAT;
    texture_sampler.address_v = TGPU_SAMPLER_ADDRESS_MODE_REPEAT;
    texture_sampler.address_w = TGPU_SAMPLER_ADDRESS_MODE_REPEAT;

    TgpuViewImageDesc texture_desc = {
        .struct_size = sizeof texture_desc,
        .image =
            {
                .struct_size = sizeof(TgpuImageDesc),
                .label = "texture",
                .format = TGPU_FORMAT_RGBA8_UNORM,
                .width = TEXTURE_SIZE,
                .height = TEXTURE_SIZE,
                .depth = 1,
                .mip_levels = 1,
                .array_layers = 1,
                .usage = TGPU_IMAGE_USAGE_COPY_DST | TGPU_IMAGE_USAGE_SAMPLED,
                .memory = TGPU_MEMORY_PRESET_GPU_ONLY,
                .host_access = TGPU_HOST_ACCESS_NONE,
                .initial_layout = TGPU_IMAGE_LAYOUT_UNDEFINED,
            },
        .sampler = &texture_sampler,
    };

    TgpuViewImage *textures[2] = {0};
    uint32_t texture_handles[2] = {0};
    for (uint32_t i = 0; i < 2; ++i) {
        uint8_t *pixels = make_texture(TEXTURE_SIZE, i);
        check_pointer(pixels, "make_texture");
        textures[i] = tgpu_queue_create_texture_2d(
            queue, device, &texture_desc, pixels, TEXTURE_SIZE * TEXTURE_SIZE * 4);
        check_pointer(textures[i], "create texture");
        free(pixels);

        texture_handles[i] = check_handle(
            tgpu_bindless_add_sampled_image(
                bindless, tgpu_view_image_view(textures[i]), TGPU_IMAGE_LAYOUT_FRAGMENT),
            "add texture");
    }

    uint32_t sampler_handle = check_handle(
        tgpu_bindless_add_sampler(bindless, tgpu_view_image_sampler(textures[0])),
        "add sampler");

    size_t flight_count = tgpu_swapchain_max_frames_in_flight(swapchain);
    if (flight_count > MAX_FLIGHT) {
        flight_count = MAX_FLIGHT;
    }
    SceneSlot scenes[MAX_FLIGHT] = {0};
    for (size_t i = 0; i < flight_count; ++i) {
        TgpuBufferDesc scene_desc = {
            .struct_size = sizeof scene_desc,
            .label = "scene info",
            .size = sizeof(SceneInfo),
            .usage = TGPU_BUFFER_USAGE_UNIFORM | TGPU_BUFFER_USAGE_COPY_DST,
            .memory = TGPU_MEMORY_PRESET_GPU_ONLY,
            .host_access = TGPU_HOST_ACCESS_NONE,
        };
        scenes[i].buffer = tgpu_device_create_buffer(device, &scene_desc, NULL, 0);
        check_pointer(scenes[i].buffer, "create scene buffer");
        scenes[i].handle = check_handle(
            tgpu_bindless_add_uniform_buffer(bindless, scenes[i].buffer, sizeof(SceneInfo)),
            "add scene buffer");
    }

    size_t spirv_words = 0;
    uint32_t *spirv = load_spirv("minimal.spv", &spirv_words);
    TgpuShader *shader = tgpu_device_create_shader(device, "minimal latest", spirv, spirv_words);
    check_pointer(shader, "create shader");
    free(spirv);

    TgpuComputePipelineDesc compute_desc = {
        .struct_size = sizeof compute_desc,
        .label = "vertex rotation",
        .shader = shader,
        .entry = "computeMain",
        .bindless = bindless,
        .push_constant_size = sizeof(PushData),
    };
    TgpuComputePipeline *compute_pipeline =
        tgpu_device_create_compute_pipeline(device, &compute_desc);
    check_pointer(compute_pipeline, "create compute pipeline");

    TgpuRenderPipeline *color_pipeline = create_graphics_pipeline(
        device, bindless, shader, "fragmentMainColor", TGPU_FORMAT_RGBA8_UNORM,
        TGPU_FORMAT_DEPTH32_FLOAT, "scene color");
    TgpuRenderPipeline *textured_pipeline = create_graphics_pipeline(
        device, bindless, shader, "fragmentMainTextured", TGPU_FORMAT_RGBA8_UNORM,
        TGPU_FORMAT_DEPTH32_FLOAT, "scene textured");

    TgpuRenderPipelineDesc present_desc = {
        .struct_size = sizeof present_desc,
        .label = "present",
        .vertex_shader = shader,
        .vertex_entry = "presentVertex",
        .fragment_shader = shader,
        .fragment_entry = "presentFragment",
        .color_formats = &swapchain_format,
        .color_format_count = 1,
        .depth_format = TGPU_FORMAT_UNDEFINED,
        .bindless = bindless,
        .push_constant_size = sizeof(PushData),
        .cull = TGPU_CULL_MODE_NONE,
        .front_face = TGPU_FRONT_FACE_COUNTER_CLOCKWISE,
        .topology = TGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .blend = TGPU_BLEND_DISABLED,
    };
    TgpuRenderPipeline *present_pipeline =
        tgpu_device_create_render_pipeline(device, &present_desc);
    check_pointer(present_pipeline, "create present pipeline");

    double last_time = glfwGetTime();
    double start_time = last_time;
    float smoothed_dt = 1.0f / 60.0f;
    bool vertex_was_read = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int frame_width = 0, frame_height = 0;
        glfwGetFramebufferSize(window, &frame_width, &frame_height);
        if (frame_width <= 0 || frame_height <= 0) {
            continue;
        }

        if ((uint32_t)frame_width != extent.width || (uint32_t)frame_height != extent.height) {
            extent.width = (uint32_t)frame_width;
            extent.height = (uint32_t)frame_height;
            g_rebuild_swapchain = true;
        }

        if (g_rebuild_swapchain) {
            tgpu_swapchain_set_present_mode(
                swapchain, g_vsync ? TGPU_PRESENT_MODE_FIFO : TGPU_PRESENT_MODE_IMMEDIATE);
            tgpu_swapchain_set_extent(swapchain, extent);
            check(tgpu_swapchain_recreate(swapchain), "tgpu_swapchain_recreate");
            g_rebuild_swapchain = false;
        }

        double now = glfwGetTime();
        float measured = (float)(now - last_time);
        last_time = now;
        measured = fmaxf(measured, 1.0f / 240.0f);
        measured = fminf(measured, 1.0f / 15.0f);
        smoothed_dt = smoothed_dt * 0.9f + measured * 0.1f;

        TgpuFrame frame = {0};
        check(tgpu_swapchain_begin_frame(queue, swapchain, &frame), "begin_frame");
        if (frame.flight_index >= flight_count) {
            fprintf(stderr, "flight index %zu out of range\n", frame.flight_index);
            return EXIT_FAILURE;
        }

        TgpuExtent2D current = {0};
        check(tgpu_swapchain_extent(swapchain, &current), "swapchain_extent");
        if (current.width != gbuffer.extent.width ||
            current.height != gbuffer.extent.height) {
            tgpu_device_wait_idle(device);
            tgpu_view_image_destroy(gbuffer.color);
            tgpu_view_image_destroy(gbuffer.depth);
            gbuffer = create_gbuffer(device, current);
            check(tgpu_bindless_update_sampled_image(
                      bindless, gbuffer_handle, tgpu_view_image_view(gbuffer.color),
                      TGPU_IMAGE_LAYOUT_FRAGMENT),
                  "update gbuffer image");
        }

        float elapsed = (float)(now - start_time);
        SceneInfo scene = {
            .resolution = {(float)gbuffer.extent.width, (float)gbuffer.extent.height},
            .anim_value = 0.5f * (sinf(5.0f * elapsed) + 1.0f) + 0.5f,
            .num_data = 4,
            .tex_id = (uint32_t)g_image_id,
        };

        TgpuCommandRecorder *recorder = tgpu_queue_begin_commands(queue);
        check_pointer(recorder, "begin commands");

        TgpuBuffer *scene_buffer = scenes[frame.flight_index].buffer;
        uint32_t scene_handle = scenes[frame.flight_index].handle;

        check(tgpu_recorder_buffer_transition(
                  recorder, scene_buffer, TGPU_BUFFER_ACCESS_UNIFORM_READ,
                  TGPU_BUFFER_ACCESS_TRANSFER_DST),
              "scene barrier in");
        check(tgpu_recorder_update_buffer(recorder, scene_buffer, 0, (const uint8_t *)&scene, sizeof scene),
              "update scene buffer");
        check(tgpu_recorder_buffer_transition(
                  recorder, scene_buffer, TGPU_BUFFER_ACCESS_TRANSFER_DST,
                  TGPU_BUFFER_ACCESS_UNIFORM_READ),
              "scene barrier out");

        check(tgpu_recorder_buffer_transition(
                  recorder, vertex_buffer,
                  vertex_was_read ? TGPU_BUFFER_ACCESS_VERTEX_INPUT_READ : TGPU_BUFFER_ACCESS_NONE,
                  TGPU_BUFFER_ACCESS_COMPUTE_STORAGE_WRITE),
              "vertex barrier in");

        PushData compute_push = {
            .vertex_buffer = vertex_handle,
            .rotation_angle = 1.2f * smoothed_dt,
            .num_vertex = 3,
            .scene_info = scene_handle,
            .points = points_handle,
            .texture_image = texture_handles[g_image_id],
            .texture_sampler = sampler_handle,
        };
        check(tgpu_recorder_bind_compute(
                  recorder, compute_pipeline, bindless, (const uint8_t *)&compute_push,
                  sizeof compute_push),
              "bind compute");
        check(tgpu_recorder_dispatch(recorder, 1, 1, 1), "dispatch");
        check(tgpu_recorder_buffer_transition(
                  recorder, vertex_buffer, TGPU_BUFFER_ACCESS_COMPUTE_STORAGE_WRITE,
                  TGPU_BUFFER_ACCESS_VERTEX_INPUT_READ),
              "vertex barrier out");
        vertex_was_read = true;

        check(tgpu_recorder_transition_image(
                  recorder, tgpu_view_image_image(gbuffer.color), TGPU_IMAGE_LAYOUT_COLOR),
              "gbuffer color transition");
        check(tgpu_recorder_transition_image(
                  recorder, tgpu_view_image_image(gbuffer.depth), TGPU_IMAGE_LAYOUT_DEPTH_STENCIL),
              "gbuffer depth transition");

        TgpuColorAttachment gbuffer_color = {
            .view = tgpu_view_image_view(gbuffer.color),
            .layout = TGPU_IMAGE_LAYOUT_COLOR,
            .load = TGPU_LOAD_OP_CLEAR,
            .store = TGPU_STORE_OP_STORE,
            .clear_color = {g_clear_color[0], g_clear_color[1], g_clear_color[2], 1.0f},
        };
        TgpuDepthAttachment gbuffer_depth = {
            .view = tgpu_view_image_view(gbuffer.depth),
            .layout = TGPU_IMAGE_LAYOUT_DEPTH_STENCIL,
            .load = TGPU_LOAD_OP_CLEAR,
            .store = TGPU_STORE_OP_STORE,
            .clear_depth = 1.0f,
            .clear_stencil = 0,
        };
        TgpuRenderDesc scene_render = {
            .struct_size = sizeof scene_render,
            .width = gbuffer.extent.width,
            .height = gbuffer.extent.height,
            .colors = &gbuffer_color,
            .color_count = 1,
            .depth = &gbuffer_depth,
        };

        check(tgpu_recorder_begin_render(recorder, &scene_render), "begin scene render");
        check(tgpu_recorder_bind_vertex_buffer(recorder, 0, vertex_buffer, 0), "bind vertices");

        PushData color_push = {
            .vertex_buffer = vertex_handle,
            .scene_info = scene_handle,
            .points = points_handle,
            .texture_image = texture_handles[g_image_id],
            .texture_sampler = sampler_handle,
            .color = {1.0f, 0.0f, 0.0f, 1.0f},
        };
        check(tgpu_recorder_bind_render(
                  recorder, color_pipeline, bindless, (const uint8_t *)&color_push,
                  sizeof color_push),
              "bind color pipeline");
        check(tgpu_recorder_draw(recorder, 0, 3, 0, 1), "draw color");

        PushData textured_push = color_push;
        textured_push.color[0] = 0.0f;
        textured_push.color[1] = 1.0f;
        check(tgpu_recorder_bind_render(
                  recorder, textured_pipeline, bindless, (const uint8_t *)&textured_push,
                  sizeof textured_push),
              "bind textured pipeline");
        check(tgpu_recorder_draw(recorder, 3, 3, 0, 1), "draw textured");
        check(tgpu_recorder_end_render(recorder), "end scene render");

        check(tgpu_recorder_transition_image(
                  recorder, tgpu_view_image_image(gbuffer.color), TGPU_IMAGE_LAYOUT_FRAGMENT),
              "gbuffer to fragment");

        check(tgpu_recorder_transition_frame(recorder, swapchain, frame, TGPU_IMAGE_LAYOUT_COLOR),
              "frame to color");

        TgpuColorAttachment present_color = {
            .view = tgpu_swapchain_view(swapchain, frame),
            .layout = TGPU_IMAGE_LAYOUT_COLOR,
            .load = TGPU_LOAD_OP_CLEAR,
            .store = TGPU_STORE_OP_STORE,
            .clear_color = {0.0f, 0.0f, 0.0f, 1.0f},
        };
        TgpuRenderDesc present_render = {
            .struct_size = sizeof present_render,
            .width = extent.width,
            .height = extent.height,
            .colors = &present_color,
            .color_count = 1,
        };
        PushData present_push = {
            .texture_image = gbuffer_handle,
            .texture_sampler = sampler_handle,
        };
        check(tgpu_recorder_bind_render(
                  recorder, present_pipeline, bindless, (const uint8_t *)&present_push,
                  sizeof present_push),
              "bind present pipeline");
        check(tgpu_recorder_begin_render(recorder, &present_render), "begin present render");
        check(tgpu_recorder_draw(recorder, 0, 3, 0, 1), "draw present");
        check(tgpu_recorder_end_render(recorder), "end present render");

        check(tgpu_recorder_transition_frame(recorder, swapchain, frame, TGPU_IMAGE_LAYOUT_PRESENT),
              "frame to present");

        TgpuCommandBuffer *commands[1] = {tgpu_recorder_finish(recorder)};
        check_pointer(commands[0], "finish commands");
        check(tgpu_swapchain_end_frame(queue, swapchain, frame, commands, 1), "end_frame");
        tgpu_command_buffer_destroy(commands[0]);
    }

    tgpu_device_wait_idle(device);

    tgpu_render_pipeline_destroy(present_pipeline);
    tgpu_render_pipeline_destroy(textured_pipeline);
    tgpu_render_pipeline_destroy(color_pipeline);
    tgpu_compute_pipeline_destroy(compute_pipeline);
    tgpu_shader_destroy(shader);
    for (size_t i = 0; i < flight_count; ++i) {
        tgpu_buffer_destroy(scenes[i].buffer);
    }
    for (uint32_t i = 0; i < 2; ++i) {
        tgpu_view_image_destroy(textures[i]);
    }
    tgpu_bindless_destroy(bindless);
    tgpu_buffer_destroy(points_buffer);
    tgpu_buffer_destroy(vertex_buffer);
    tgpu_view_image_destroy(gbuffer.color);
    tgpu_view_image_destroy(gbuffer.depth);
    tgpu_swapchain_destroy(swapchain);
    tgpu_queue_destroy(queues[0]);
    tgpu_device_destroy(device);
    tgpu_instance_destroy(instance);
    example_window_destroy(window);

    return EXIT_SUCCESS;
}
