#pragma once
#include "cuda_utils.cuh"
#include <vector>
#include <cstddef>

/// Owning FP16 GPU tensor with RAII semantics.
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

    void load_from_host(const half* src, size_t n);
    void copy_to_host(half* dst, size_t n) const;

    half*       data()       { return data_; }
    const half* data() const { return data_; }
    const std::vector<int>& shape() const { return shape_; }
    int  dim(int i) const { return shape_[i]; }
    size_t numel() const;

private:
    half* data_ = nullptr;
    std::vector<int> shape_;
    bool owned_ = false;
};
