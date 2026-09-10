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
TEST_TARGET = test_kernels
TEST_SRCS = tests/test_kernels.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

$(TEST_TARGET): $(TEST_SRCS) src/tensor_metal.mm src/kernels_metal.mm
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) src/tensor_metal.mm src/kernels_metal.mm $(LDFLAGS) -o $(TEST_TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET)

.PHONY: all test clean
