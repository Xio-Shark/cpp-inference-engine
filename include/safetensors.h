#pragma once
#include "tensor.cuh"
#include <string>
#include <unordered_map>
#include <vector>

struct TensorMeta {
    std::string dtype;
    std::vector<int> shape;
    size_t offset;   // byte offset into data region
    size_t nbytes;   // byte length
};

/// Zero-copy safetensors file reader (mmap-based).
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    /// Load tensor to GPU by name.
    GpuTensor load_tensor(const std::string& name) const;
    bool has(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    void parse();
    std::string path_;
    int fd_ = -1;
    void* map_ = nullptr;
    size_t file_sz_ = 0;
    size_t data_start_ = 0;
    std::unordered_map<std::string, TensorMeta> meta_;
};
