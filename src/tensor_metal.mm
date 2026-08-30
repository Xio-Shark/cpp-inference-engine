#if defined(__APPLE__)
#include "tensor.h"
#import <Metal/Metal.h>
#include <stdexcept>
#include <cstring>

// Singleton accessor for Metal Device in Objective-C++
id<MTLDevice> get_default_metal_device() {
    static id<MTLDevice> device = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        device = MTLCreateSystemDefaultDevice();
    });
    if (!device) {
        throw std::runtime_error("No Metal-compatible GPU found on this system.");
    }
    return device;
}

GpuTensor::GpuTensor(std::vector<int> shape) : shape_(std::move(shape)) {
    size_t n = numel();
    if (n == 0) {
        data_ = nullptr;
        native_buffer_ = nullptr;
        owned_ = false;
        return;
    }

    size_t byte_len = n * sizeof(half);
    // Align to at least 16 bytes for SIMD/Metal buffer constraints
    if (byte_len < 16) byte_len = 16;

    @autoreleasepool {
        id<MTLDevice> device = get_default_metal_device();
        // Allocate buffer with MTLResourceStorageModeShared for zero-copy CPU/GPU unified memory
        id<MTLBuffer> buf = [device newBufferWithLength:byte_len
                                                options:MTLResourceStorageModeShared];
        if (!buf) {
            throw std::runtime_error("Failed to allocate Metal buffer of size " + std::to_string(byte_len));
        }

        // Bridge to void* and retain in native_buffer_
        native_buffer_ = (void*)CFBridgingRetain(buf);
        data_ = reinterpret_cast<half*>([buf contents]);
        owned_ = true;
        register_metal_buffer(data_, byte_len, (__bridge void*)buf);
    }
}

GpuTensor::~GpuTensor() {
    if (owned_ && native_buffer_) {
        unregister_metal_buffer(data_);
        @autoreleasepool {
            id<MTLBuffer> buf = CFBridgingRelease(native_buffer_);
            buf = nil;
        }
        native_buffer_ = nullptr;
        data_ = nullptr;
    }
}

GpuTensor::GpuTensor(GpuTensor&& o) noexcept
    : data_(o.data_), shape_(std::move(o.shape_)), owned_(o.owned_), native_buffer_(o.native_buffer_) {
    o.data_ = nullptr;
    o.native_buffer_ = nullptr;
    o.owned_ = false;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& o) noexcept {
    if (this != &o) {
        if (owned_ && native_buffer_) {
            unregister_metal_buffer(data_);
            @autoreleasepool {
                id<MTLBuffer> buf = CFBridgingRelease(native_buffer_);
                buf = nil;
            }
        }
        data_ = o.data_;
        shape_ = std::move(o.shape_);
        owned_ = o.owned_;
        native_buffer_ = o.native_buffer_;

        o.data_ = nullptr;
        o.native_buffer_ = nullptr;
        o.owned_ = false;
    }
    return *this;
}

size_t GpuTensor::numel() const {
    if (shape_.empty()) return 0;
    size_t n = 1;
    for (int d : shape_) n *= static_cast<size_t>(d);
    return n;
}

void GpuTensor::load_from_host(const half* src, size_t n) {
    if (data_ && src && n > 0) {
        std::memcpy(data_, src, n * sizeof(half));
    }
}

void GpuTensor::copy_to_host(half* dst, size_t n) const {
    if (data_ && dst && n > 0) {
        std::memcpy(dst, data_, n * sizeof(half));
    }
}

#endif
