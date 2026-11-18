#include "glfw_window.h"

#if defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#endif

GLFWwindow *example_window_create(const char *title, int width, int height) {
    if (!glfwInit()) {
        return NULL;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    return glfwCreateWindow(width, height, title, NULL, NULL);
}

void example_window_destroy(GLFWwindow *window) {
    if (window) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

#ifdef __APPLE__
/* Implemented in glfw_window_macos.m */
void *example_window_ns_view(GLFWwindow *window);
#endif

TgpuWindow example_window_tgpu_window(GLFWwindow *window) {
    TgpuWindow result = {0};

#if defined(__APPLE__)
    result.kind = TGPU_WINDOW_KIND_APP_KIT;
    result.window = example_window_ns_view(window);
#elif defined(__linux__)
    result.kind = TGPU_WINDOW_KIND_XLIB;
    result.display = glfwGetX11Display();
    result.window = (void *)(uintptr_t)glfwGetX11Window(window);
#elif defined(_WIN32)
    result.kind = TGPU_WINDOW_KIND_WIN32;
    result.window = (void *)glfwGetWin32Window(window);
#else
    (void)window;
#endif

    return result;
}
