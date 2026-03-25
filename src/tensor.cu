#include "tensor.cuh"

GpuTensor::GpuTensor(std::vector<int> shape) : shape_(std::move(shape)) {
    CUDA_CHECK(cudaMalloc(&data_, numel() * sizeof(half)));
    owned_ = true;
}

GpuTensor::~GpuTensor() {
    if (owned_ && data_) cudaFree(data_);
}

GpuTensor::GpuTensor(GpuTensor&& o) noexcept
    : data_(o.data_), shape_(std::move(o.shape_)), owned_(o.owned_) {
    o.data_ = nullptr;
    o.owned_ = false;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& o) noexcept {
    if (this != &o) {
        if (owned_ && data_) cudaFree(data_);
        data_ = o.data_;
        shape_ = std::move(o.shape_);
        owned_ = o.owned_;
        o.data_ = nullptr;
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
    CUDA_CHECK(cudaMemcpy(data_, src, n * sizeof(half),
                          cudaMemcpyHostToDevice));
}

void GpuTensor::copy_to_host(half* dst, size_t n) const {
    CUDA_CHECK(cudaMemcpy(dst, data_, n * sizeof(half),
                          cudaMemcpyDeviceToHost));
}
