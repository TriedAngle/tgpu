#include "glfw_window.h"

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#import <Cocoa/Cocoa.h>

void *example_window_ns_view(GLFWwindow *window) {
    if (!window) {
        return NULL;
    }
    NSWindow *ns_window = glfwGetCocoaWindow(window);
    if (!ns_window) {
        return NULL;
    }
    return (void *)[ns_window contentView];
}
