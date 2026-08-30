#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>

#if defined(__APPLE__)
    // Apple Silicon / macOS Metal & MPS support
    #include <TargetConditionals.h>
    
    // Half-precision float support on Apple Clang
    using half = _Float16;
    
    inline half __float2half(float val) {
        return static_cast<half>(val);
    }
    
    inline float __half2float(half val) {
        return static_cast<float>(val);
    }

    #define METAL_CHECK(cond, msg) do {                                    \
        if (!(cond)) {                                                     \
            fprintf(stderr, "Metal error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, (msg));                            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

    // Metal buffer tracking functions
    void register_metal_buffer(const void* ptr, size_t bytes, void* mtl_buf);
    void unregister_metal_buffer(const void* ptr);

    #ifndef CUDA_CHECK
        #define CUDA_CHECK(call) do { (void)(call); } while (0)
    #endif
    #ifndef CUBLAS_CHECK
        #define CUBLAS_CHECK(call) do { (void)(call); } while (0)
    #endif

#elif defined(USE_CUDA) || defined(__CUDACC__)
    // NVIDIA CUDA / cuBLAS support
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cublas_v2.h>

    #define CUDA_CHECK(call) do {                                          \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

    #define CUBLAS_CHECK(call) do {                                        \
        cublasStatus_t s = (call);                                         \
        if (s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                    \
                    __FILE__, __LINE__, static_cast<int>(s));              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
#endif

// Forward declaration of backend-specific implementation
class DeviceContextImpl;

/// Unified device context:
/// On macOS: manages MTLDevice, MTLCommandQueue, and Metal Compute Pipelines.
/// On CUDA: manages cublasHandle_t and CUDA streams.
class DeviceContext {
public:
    DeviceContext();
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&&) noexcept;
    DeviceContext& operator=(DeviceContext&&) noexcept;

    void* get_native_handle() const;
    void synchronize();

private:
    std::unique_ptr<DeviceContextImpl> impl_;
};

// Compatibility alias for existing code referencing CublasHandle
using CublasHandle = DeviceContext;
