TGPU_NATIVE := ..
WORKSPACE   := ../..
TARGET      := $(WORKSPACE)/target/release

CC    ?= cc
UNAME := $(shell uname -s)

INCLUDES := -I$(TGPU_NATIVE)/include -Icommon $(shell pkg-config --cflags glfw3)
CFLAGS   := -std=c11 -O2 -Wall -Wextra -Werror $(INCLUDES)
LDLIBS   := $(shell pkg-config --libs glfw3) -L$(TARGET) -ltgpu_native

PLATFORM_OBJS :=
ifeq ($(UNAME),Darwin)
  PLATFORM_OBJS += common/glfw_window_macos.o
  LDLIBS += -framework Cocoa
endif

OBJS := particles.o minimal.o common/glfw_window.o $(PLATFORM_OBJS)

LDLIBS += -Wl,-rpath,$(abspath $(TARGET))

.PHONY: all lib clean

all: particles minimal

lib:
	cd $(WORKSPACE) && cargo build -p tgpu-native --release

particles.spv: $(WORKSPACE)/examples/particles/src/shader.slang
	slangc $< -target spirv -profile spirv_1_3 -fvk-use-entrypoint-name -emit-spirv-directly -o $@

minimal.spv: $(WORKSPACE)/examples/minimal_latest/src/shader.slang
	slangc $< -target spirv -profile spirv_1_3 -fvk-use-entrypoint-name -emit-spirv-directly -o $@

particles: particles.o common/glfw_window.o $(PLATFORM_OBJS) particles.spv lib
	$(CC) particles.o common/glfw_window.o $(PLATFORM_OBJS) $(LDLIBS) -o $@

minimal: minimal.o common/glfw_window.o $(PLATFORM_OBJS) minimal.spv lib
	$(CC) minimal.o common/glfw_window.o $(PLATFORM_OBJS) $(LDLIBS) -o $@

HEADERS := ../include/tgpu.h common/glfw_window.h

particles.o minimal.o common/glfw_window.o common/glfw_window_macos.o: $(HEADERS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.m
	$(CC) -O2 -Wall -Wextra -Werror $(INCLUDES) -c $< -o $@

clean:
	rm -f particles minimal particles.spv minimal.spv $(OBJS)
