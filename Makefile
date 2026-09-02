# Makefile for tiny-inference-engine on macOS (Apple Silicon Metal & MPS)
CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -D__APPLE__=1 -fobjc-arc -Iinclude
LDFLAGS ?= -framework Metal -framework Foundation -framework MetalPerformanceShaders

SRCS = src/tensor_metal.mm \
       src/kernels_metal.mm \
       src/safetensors.cpp \
       src/transformer.cpp \
       src/main.cpp

TARGET = tiny_inference

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
