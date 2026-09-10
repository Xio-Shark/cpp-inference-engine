#include "safetensors.h"
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path);

    struct stat st{};
    fstat(fd_, &st);
    file_sz_ = static_cast<size_t>(st.st_size);

    map_ = mmap(nullptr, file_sz_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed: " + path);
    }
    parse();
}

SafetensorsFile::~SafetensorsFile() {
    if (map_ && map_ != MAP_FAILED) munmap(map_, file_sz_);
    if (fd_ >= 0) close(fd_);
}

void SafetensorsFile::parse() {
    auto* base = static_cast<const uint8_t*>(map_);

    if (file_sz_ < 8) {
        throw std::runtime_error("safetensors file too small: " + path_);
    }

    // First 8 bytes: little-endian uint64 header length
    uint64_t hdr_len = 0;
    std::memcpy(&hdr_len, base, 8);
    if (hdr_len > file_sz_ - 8) {
        throw std::runtime_error("invalid safetensors header length: " + path_);
    }
    data_start_ = 8 + hdr_len;

    std::string json_str(reinterpret_cast<const char*>(base + 8), hdr_len);
    auto j = nlohmann::json::parse(json_str);

    for (auto& [key, val] : j.items()) {
        if (key == "__metadata__") continue;

        TensorMeta m;
        m.dtype = val["dtype"].get<std::string>();
        for (auto& s : val["shape"]) m.shape.push_back(s.get<int>());
        auto off = val["data_offsets"];
        const size_t begin = off[0].get<size_t>();
        const size_t end = off[1].get<size_t>();
        if (begin > end || end > file_sz_ - data_start_) {
            throw std::runtime_error("invalid tensor data range for " + key + " in " + path_);
        }
        m.offset = begin;
        m.nbytes = end - begin;

        meta_[key] = std::move(m);
    }
}

GpuTensor SafetensorsFile::load_tensor(const std::string& name) const {
    auto it = meta_.find(name);
    if (it == meta_.end())
        throw std::runtime_error("tensor not found: " + name);

    const auto& m = it->second;
    size_t element_bytes = 0;
    if (m.dtype == "F16" || m.dtype == "BF16") {
        element_bytes = sizeof(half);
    } else if (m.dtype == "F32") {
        element_bytes = sizeof(float);
    } else {
        throw std::runtime_error(
            "unsupported tensor dtype for " + name + ": " + m.dtype + " in " + path_);
    }

    GpuTensor t(m.shape);
    const size_t numel = t.numel();
    if (m.nbytes != numel * element_bytes) {
        throw std::runtime_error(
            "tensor byte size mismatch for " + name + " in " + path_);
    }

    const auto* src = static_cast<const uint8_t*>(map_) + data_start_ + m.offset;

    if (m.dtype == "F16") {
        t.load_from_host(reinterpret_cast<const half*>(src), numel);
    } else if (m.dtype == "BF16") {
        const auto* bf16_data = reinterpret_cast<const uint16_t*>(src);
        std::vector<half> converted(numel);
        for (size_t i = 0; i < numel; ++i) {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16_data[i]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &fp32_bits, sizeof(float));
            converted[i] = __float2half(value);
        }
        t.load_from_host(converted.data(), numel);
    } else {
        const auto* fp32_data = reinterpret_cast<const float*>(src);
        std::vector<half> converted(numel);
        for (size_t i = 0; i < numel; ++i) {
            converted[i] = __float2half(fp32_data[i]);
        }
        t.load_from_host(converted.data(), numel);
    }
    return t;
}

bool SafetensorsFile::has(const std::string& name) const {
    return meta_.count(name) > 0;
}

std::vector<std::string> SafetensorsFile::names() const {
    std::vector<std::string> out;
    out.reserve(meta_.size());
    for (const auto& [k, _] : meta_) out.push_back(k);
    return out;
}
