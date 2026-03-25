#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

#define CUBLAS_CHECK(call) do {                                    \
    cublasStatus_t s = (call);                                     \
    if (s != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",               \
                __FILE__, __LINE__, static_cast<int>(s));          \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

/// RAII wrapper for cublasHandle_t
class CublasHandle {
public:
    CublasHandle()  { CUBLAS_CHECK(cublasCreate(&h_)); }
    ~CublasHandle() { cublasDestroy(h_); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    cublasHandle_t get() const { return h_; }
private:
    cublasHandle_t h_;
};
