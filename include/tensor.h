#pragma once
#include "device_utils.h"
#include <vector>
#include <cstddef>
#include <cstring>

/// Owning FP16 GPU tensor with RAII semantics.
/// On macOS (Metal): Backed by MTLBuffer allocated in Unified Memory (Shared Storage Mode).
/// On Linux (CUDA): Backed by cudaMalloc device pointer.
class GpuTensor {
public:
    GpuTensor() = default;

    /// Allocate GPU memory for given shape.
    explicit GpuTensor(std::vector<int> shape);

    ~GpuTensor();

    // Move only
    GpuTensor(GpuTensor&& o) noexcept;
    GpuTensor& operator=(GpuTensor&& o) noexcept;
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;

    /// Non-owning view into existing GPU memory
    static GpuTensor from_borrowed(half* ptr, std::vector<int> shape, void* native_buf = nullptr) {
        GpuTensor t;
        t.data_ = ptr;
        t.shape_ = std::move(shape);
        t.owned_ = false;
        t.native_buffer_ = native_buf;
        return t;
    }

    void load_from_host(const half* src, size_t n);
    void copy_to_host(half* dst, size_t n) const;

    half*       data()       { return data_; }
    const half* data() const { return data_; }
    const std::vector<int>& shape() const { return shape_; }
    int  dim(int i) const { return shape_[i]; }
    size_t numel() const;

    // Platform specific buffer handle (e.g., id<MTLBuffer> on macOS)
    void* native_buffer() const { return native_buffer_; }

private:
    half* data_ = nullptr;
    std::vector<int> shape_;
    bool owned_ = false;
    void* native_buffer_ = nullptr;
};
