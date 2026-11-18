#ifndef TGPU_EXAMPLE_GLFW_WINDOW_H
#define TGPU_EXAMPLE_GLFW_WINDOW_H

#include <GLFW/glfw3.h>

#include "tgpu.h"

GLFWwindow *example_window_create(const char *title, int width, int height);
void example_window_destroy(GLFWwindow *window);

/// The tgpu description of this window's native handle.
TgpuWindow example_window_tgpu_window(GLFWwindow *window);

#endif
