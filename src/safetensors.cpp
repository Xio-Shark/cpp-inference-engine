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

    // First 8 bytes: little-endian uint64 header length
    uint64_t hdr_len = 0;
    std::memcpy(&hdr_len, base, 8);
    data_start_ = 8 + hdr_len;

    std::string json_str(reinterpret_cast<const char*>(base + 8), hdr_len);
    auto j = nlohmann::json::parse(json_str);

    for (auto& [key, val] : j.items()) {
        if (key == "__metadata__") continue;

        TensorMeta m;
        m.dtype = val["dtype"].get<std::string>();
        for (auto& s : val["shape"]) m.shape.push_back(s.get<int>());
        auto off = val["data_offsets"];
        m.offset = off[0].get<size_t>();
        m.nbytes = off[1].get<size_t>() - m.offset;

        meta_[key] = std::move(m);
    }
}

GpuTensor SafetensorsFile::load_tensor(const std::string& name) const {
    auto it = meta_.find(name);
    if (it == meta_.end())
        throw std::runtime_error("tensor not found: " + name);

    const auto& m = it->second;
    GpuTensor t(m.shape);
    size_t numel = t.numel();

    auto* src = static_cast<const uint8_t*>(map_) + data_start_ + m.offset;

    if (m.dtype == "BF16") {
        // BF16 → FP32 → FP16 conversion on host
        auto* bf16_data = reinterpret_cast<const uint16_t*>(src);
        std::vector<uint16_t> fp16_buf(numel);
        for (size_t i = 0; i < numel; ++i) {
            // BF16 → FP32: left-shift 16 bits (BF16 = upper 16 bits of FP32)
            uint32_t fp32_bits = static_cast<uint32_t>(bf16_data[i]) << 16;
            float fval;
            std::memcpy(&fval, &fp32_bits, sizeof(float));
            // FP32 → FP16: extract sign, exponent, mantissa
            uint32_t f = fp32_bits;
            uint16_t sign = (f >> 16) & 0x8000;
            int exp = ((f >> 23) & 0xFF) - 127 + 15;
            uint16_t mant = (f >> 13) & 0x03FF;
            uint16_t h;
            if (exp <= 0)       h = sign;                // underflow → zero
            else if (exp >= 31) h = sign | 0x7C00;       // overflow → inf
            else                h = sign | (exp << 10) | mant;
            fp16_buf[i] = h;
        }
        CUDA_CHECK(cudaMemcpy(t.data(), fp16_buf.data(),
                              numel * sizeof(uint16_t), cudaMemcpyHostToDevice));
    } else {
        // F16 or F32: direct copy (F16 assumed)
        CUDA_CHECK(cudaMemcpy(t.data(), src, m.nbytes,
                              cudaMemcpyHostToDevice));
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
