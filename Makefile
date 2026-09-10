# Makefile for tiny-inference-engine on macOS (Apple Silicon Metal & MPS)
CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -D__APPLE__=1 -fobjc-arc -Iinclude
LDFLAGS ?= -framework Metal -framework Foundation -framework MetalPerformanceShaders

SRCS = src/tensor_metal.mm \
       src/kernels_metal.mm \
       src/safetensors.cpp \
       src/weight_loader.cpp \
       src/transformer.cpp \
       src/main.cpp

TARGET = tiny_inference
TEST_TARGETS = test_kernels test_model_loader

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

test_kernels: tests/test_kernels.cpp src/tensor_metal.mm src/kernels_metal.mm
	$(CXX) $(CXXFLAGS) tests/test_kernels.cpp src/tensor_metal.mm src/kernels_metal.mm $(LDFLAGS) -o $@

test_model_loader: tests/test_model_loader.cpp src/tensor_metal.mm src/kernels_metal.mm src/safetensors.cpp src/weight_loader.cpp src/transformer.cpp
	$(CXX) $(CXXFLAGS) tests/test_model_loader.cpp src/tensor_metal.mm src/kernels_metal.mm src/safetensors.cpp src/weight_loader.cpp src/transformer.cpp $(LDFLAGS) -o $@

test: $(TEST_TARGETS)
	./test_kernels
	./test_model_loader

clean:
	rm -f $(TARGET) $(TEST_TARGETS)

.PHONY: all test clean
