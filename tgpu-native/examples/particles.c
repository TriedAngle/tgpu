// Particles, ported to the tgpu C API.
//
// Shows a compute shader mutating a storage buffer and a storage image, then
// a fullscreen triangle sampling that image. The UI of the Rust example is
// replaced by mouse interaction only.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glfw_window.h"
#include "tgpu.h"

#define PARTICLE_COUNT 10000

typedef struct {
    float position[2];
    float velocity[2];
    float color[4];
} Particle;

typedef struct {
    uint32_t output_image;
    uint32_t particles;
    uint32_t present_texture;
    uint32_t present_sampler;
    uint32_t window[2];
    float mouse[2];
    float dt;
    float pad;
} PushConstants;

static float g_mouse[2] = {0.0f, 0.0f};

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

static void cursor_callback(GLFWwindow *window, double x, double y) {
    int window_width = 0, window_height = 0;
    int frame_width = 0, frame_height = 0;
    glfwGetWindowSize(window, &window_width, &window_height);
    glfwGetFramebufferSize(window, &frame_width, &frame_height);

    float scale_x = window_width > 0 ? (float)frame_width / (float)window_width : 1.0f;
    float scale_y = window_height > 0 ? (float)frame_height / (float)window_height : 1.0f;

    g_mouse[0] = (float)x * scale_x;
    g_mouse[1] = (float)frame_height - (float)y * scale_y;
}

static void fill_particles(Particle *particles, int width, int height) {
    srand(1);
    for (int i = 0; i < PARTICLE_COUNT; ++i) {
        particles[i].position[0] = (float)rand() / (float)RAND_MAX * (float)width;
        particles[i].position[1] = (float)rand() / (float)RAND_MAX * (float)height;
        particles[i].velocity[0] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 30.0f;
        particles[i].velocity[1] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 30.0f;
        particles[i].color[0] = 0.5f + (float)rand() / (float)RAND_MAX * 0.5f;
        particles[i].color[1] = 0.5f + (float)rand() / (float)RAND_MAX * 0.5f;
        particles[i].color[2] = 0.5f + (float)rand() / (float)RAND_MAX * 0.5f;
        particles[i].color[3] = 1.0f;
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    GLFWwindow *window = example_window_create("tgpu particles (C)", 800, 600);
    if (!window) {
        fprintf(stderr, "failed to create window\n");
        return EXIT_FAILURE;
    }
    glfwSetCursorPosCallback(window, cursor_callback);

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    TgpuInstanceDesc instance_desc = {
        .struct_size = sizeof instance_desc,
        .app_name = "tgpu particles (C)",
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
    printf("queues:  %d\n", 1);

    TgpuWindow native_window = example_window_tgpu_window(window);
    check_pointer(native_window.window, "example_window_tgpu_window");

    TgpuSwapchainDesc swapchain_desc = {
        .struct_size = sizeof swapchain_desc,
        .window = native_window,
        .width = (uint32_t)width,
        .height = (uint32_t)height,
        .image_count = 3,
        .present_mode = TGPU_PRESENT_MODE_FIFO,
    };
    TgpuSwapchain *swapchain = tgpu_device_create_swapchain(device, &swapchain_desc);
    check_pointer(swapchain, "tgpu_device_create_swapchain");

    int32_t color_format = tgpu_swapchain_format(swapchain);
    TgpuExtent2D extent = {(uint32_t)width, (uint32_t)height};
    check(tgpu_swapchain_extent(swapchain, &extent), "tgpu_swapchain_extent");

    Particle *particles = malloc(sizeof(Particle) * PARTICLE_COUNT);
    if (!particles) {
        return EXIT_FAILURE;
    }
    fill_particles(particles, (int)extent.width, (int)extent.height);

    TgpuBufferDesc buffer_desc = {
        .struct_size = sizeof buffer_desc,
        .label = "particle buffer",
        .size = sizeof(Particle) * PARTICLE_COUNT,
        .usage = TGPU_BUFFER_USAGE_STORAGE | TGPU_BUFFER_USAGE_COPY_DST,
        .memory = TGPU_MEMORY_PRESET_DYNAMIC,
        .host_access = TGPU_HOST_ACCESS_READ_WRITE_RANDOM,
    };
    TgpuBuffer *particle_buffer =
        tgpu_device_create_buffer(device, &buffer_desc, (const uint8_t *)particles, buffer_desc.size);
    check_pointer(particle_buffer, "tgpu_device_create_buffer");
    free(particles);

    TgpuSamplerDesc sampler_desc = {
        .struct_size = sizeof sampler_desc,
        .label = "present sampler",
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

    TgpuViewImageDesc present_desc = {
        .struct_size = sizeof present_desc,
        .image =
            {
                .struct_size = sizeof(TgpuImageDesc),
                .label = "present image",
                .format = color_format,
                .width = extent.width,
                .height = extent.height,
                .depth = 1,
                .mip_levels = 1,
                .array_layers = 1,
                .usage = TGPU_IMAGE_USAGE_COPY_SRC | TGPU_IMAGE_USAGE_COPY_DST |
                         TGPU_IMAGE_USAGE_STORAGE | TGPU_IMAGE_USAGE_COLOR_ATTACHMENT |
                         TGPU_IMAGE_USAGE_SAMPLED,
                .memory = TGPU_MEMORY_PRESET_GPU_ONLY,
                .host_access = TGPU_HOST_ACCESS_NONE,
                .initial_layout = TGPU_IMAGE_LAYOUT_UNDEFINED,
            },
        .sampler = &sampler_desc,
    };
    TgpuViewImage *present_image = tgpu_device_create_view_image(device, &present_desc);
    check_pointer(present_image, "tgpu_device_create_view_image");

    TgpuBindlessDesc bindless_desc = {
        .struct_size = sizeof bindless_desc,
        .max_rw_buffers = 1,
        .max_sampled_images = 1,
        .max_storage_images = 1,
        .max_samplers = 1,
    };
    TgpuBindlessHeap *bindless = tgpu_device_create_bindless(device, &bindless_desc);
    check_pointer(bindless, "tgpu_device_create_bindless");

    PushConstants pc = {0};
    pc.particles = check_handle(
        tgpu_bindless_add_rw_buffer(bindless, particle_buffer), "add rw buffer");
    pc.output_image = check_handle(
        tgpu_bindless_add_storage_image(
            bindless, tgpu_view_image_view(present_image), TGPU_IMAGE_LAYOUT_GENERAL),
        "add storage image");
    pc.present_texture = check_handle(
        tgpu_bindless_add_sampled_image(
            bindless, tgpu_view_image_view(present_image), TGPU_IMAGE_LAYOUT_FRAGMENT),
        "add sampled image");
    pc.present_sampler = check_handle(
        tgpu_bindless_add_sampler(bindless, tgpu_view_image_sampler(present_image)),
        "add sampler");
    pc.window[0] = extent.width;
    pc.window[1] = extent.height;

    size_t spirv_words = 0;
    uint32_t *spirv = load_spirv("particles.spv", &spirv_words);
    TgpuShader *shader = tgpu_device_create_shader(device, "particles", spirv, spirv_words);
    check_pointer(shader, "tgpu_device_create_shader");
    free(spirv);

    TgpuComputePipelineDesc compute_desc = {
        .struct_size = sizeof compute_desc,
        .label = "particle compute",
        .shader = shader,
        .entry = "computeMain",
        .bindless = bindless,
        .push_constant_size = sizeof(PushConstants),
    };
    TgpuComputePipeline *compute_pipeline =
        tgpu_device_create_compute_pipeline(device, &compute_desc);
    check_pointer(compute_pipeline, "tgpu_device_create_compute_pipeline");

    TgpuComputePipelineDesc clear_desc = {
        .struct_size = sizeof clear_desc,
        .label = "clear compute",
        .shader = shader,
        .entry = "computeClear",
        .bindless = bindless,
        .push_constant_size = sizeof(PushConstants),
    };
    TgpuComputePipeline *clear_pipeline =
        tgpu_device_create_compute_pipeline(device, &clear_desc);
    check_pointer(clear_pipeline, "tgpu_device_create_compute_pipeline");

    TgpuRenderPipelineDesc render_desc = {
        .struct_size = sizeof render_desc,
        .label = "present render",
        .vertex_shader = shader,
        .vertex_entry = "vertexMain",
        .fragment_shader = shader,
        .fragment_entry = "fragmentMain",
        .color_formats = &color_format,
        .color_format_count = 1,
        .depth_format = TGPU_FORMAT_UNDEFINED,
        .bindless = bindless,
        .push_constant_size = sizeof(PushConstants),
        .cull = TGPU_CULL_MODE_BACK,
        .front_face = TGPU_FRONT_FACE_COUNTER_CLOCKWISE,
        .topology = TGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .blend = TGPU_BLEND_DISABLED,
    };
    TgpuRenderPipeline *render_pipeline =
        tgpu_device_create_render_pipeline(device, &render_desc);
    check_pointer(render_pipeline, "tgpu_device_create_render_pipeline");

    double last_time = glfwGetTime();
    float smoothed_dt = 1.0f / 60.0f;
    unsigned long frame_count = 0;

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
            tgpu_swapchain_set_extent(swapchain, extent);
            check(tgpu_swapchain_recreate(swapchain), "tgpu_swapchain_recreate");
        }

        double now = glfwGetTime();
        float measured = (float)(now - last_time);
        last_time = now;
        measured = measured < 1.0f / 240.0f ? 1.0f / 240.0f : measured;
        measured = measured > 1.0f / 15.0f ? 1.0f / 15.0f : measured;
        smoothed_dt = smoothed_dt * 0.9f + measured * 0.1f;

        TgpuFrame frame = {0};
        check(tgpu_swapchain_begin_frame(queue, swapchain, &frame), "begin_frame");

        TgpuExtent2D current = {0};
        check(tgpu_swapchain_extent(swapchain, &current), "swapchain_extent");
        if (current.width != extent.width || current.height != extent.height) {
            extent = current;
            tgpu_device_wait_idle(device);
            tgpu_view_image_destroy(present_image);

            present_desc.image.width = extent.width;
            present_desc.image.height = extent.height;
            present_image = tgpu_device_create_view_image(device, &present_desc);
            check_pointer(present_image, "recreate present image");

            check(tgpu_bindless_update_storage_image(
                      bindless, pc.output_image, tgpu_view_image_view(present_image),
                      TGPU_IMAGE_LAYOUT_GENERAL),
                  "update storage image");
            check(tgpu_bindless_update_sampled_image(
                      bindless, pc.present_texture, tgpu_view_image_view(present_image),
                      TGPU_IMAGE_LAYOUT_FRAGMENT),
                  "update sampled image");
            check(tgpu_bindless_update_sampler(
                      bindless, pc.present_sampler,
                      tgpu_view_image_sampler(present_image)),
                  "update sampler");

            pc.window[0] = extent.width;
            pc.window[1] = extent.height;
        }

        pc.mouse[0] = g_mouse[0];
        pc.mouse[1] = g_mouse[1];
        pc.dt = smoothed_dt * 6.0f;

        TgpuCommandRecorder *recorder = tgpu_queue_begin_commands(queue);
        check_pointer(recorder, "tgpu_queue_begin_commands");

        check(tgpu_recorder_transition_image(
                  recorder, tgpu_view_image_image(present_image), TGPU_IMAGE_LAYOUT_COMPUTE),
              "transition present image");

        check(tgpu_recorder_bind_compute(recorder, compute_pipeline, bindless, (const uint8_t *)&pc, sizeof pc),
              "bind compute");
        check(tgpu_recorder_dispatch(recorder, (PARTICLE_COUNT + 255) / 256, 1, 1), "dispatch");

        check(tgpu_recorder_bind_compute(recorder, clear_pipeline, bindless, (const uint8_t *)&pc, sizeof pc),
              "bind clear");
        check(tgpu_recorder_dispatch(recorder, (extent.width + 15) / 16, (extent.height + 15) / 16, 1),
              "clear dispatch");

        check(tgpu_recorder_transition_image(
                  recorder, tgpu_view_image_image(present_image), TGPU_IMAGE_LAYOUT_FRAGMENT),
              "transition present image for sampling");

        check(tgpu_recorder_transition_frame(recorder, swapchain, frame, TGPU_IMAGE_LAYOUT_COLOR),
              "transition frame to color");

        float color = fabsf(sinf((float)frame_count / 120.0f));
        TgpuColorAttachment attachment = {
            .view = tgpu_swapchain_view(swapchain, frame),
            .layout = TGPU_IMAGE_LAYOUT_COLOR,
            .load = TGPU_LOAD_OP_CLEAR,
            .store = TGPU_STORE_OP_STORE,
            .clear_color = {0.0f, color, 1.0f - color, 1.0f},
        };
        TgpuRenderDesc render_info = {
            .struct_size = sizeof render_info,
            .width = extent.width,
            .height = extent.height,
            .colors = &attachment,
            .color_count = 1,
        };

        check(tgpu_recorder_bind_render(recorder, render_pipeline, bindless, (const uint8_t *)&pc, sizeof pc),
              "bind render");
        check(tgpu_recorder_begin_render(recorder, &render_info), "begin render");
        check(tgpu_recorder_draw(recorder, 0, 6, 0, 1), "draw");
        check(tgpu_recorder_end_render(recorder), "end render");

        check(tgpu_recorder_transition_frame(recorder, swapchain, frame, TGPU_IMAGE_LAYOUT_PRESENT),
              "transition frame to present");

        TgpuCommandBuffer *commands[1] = {tgpu_recorder_finish(recorder)};
        check_pointer(commands[0], "tgpu_recorder_finish");

        check(tgpu_swapchain_end_frame(queue, swapchain, frame, commands, 1), "end_frame");
        tgpu_command_buffer_destroy(commands[0]);

        ++frame_count;
    }

    tgpu_device_wait_idle(device);

    tgpu_render_pipeline_destroy(render_pipeline);
    tgpu_compute_pipeline_destroy(clear_pipeline);
    tgpu_compute_pipeline_destroy(compute_pipeline);
    tgpu_shader_destroy(shader);
    tgpu_view_image_destroy(present_image);
    tgpu_bindless_destroy(bindless);
    tgpu_buffer_destroy(particle_buffer);
    tgpu_swapchain_destroy(swapchain);
    tgpu_queue_destroy(queues[0]);
    tgpu_device_destroy(device);
    tgpu_instance_destroy(instance);
    example_window_destroy(window);

    return EXIT_SUCCESS;
}
