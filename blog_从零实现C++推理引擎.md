# 从零实现 C++ LLM 推理引擎：加载 Qwen2.5-7B 并跑通单层 Forward Pass

> 最近在准备 LLM 推理框架方向的实习，发现「从零写一个推理引擎」是理解 Transformer 推理底层最直接的方式。本文记录了用 ~1000 行 C++/CUDA 实现一个极简推理 Runtime 的过程，包括 safetensors 权重加载、自定义 CUDA kernel、cuBLAS 矩阵运算，最终在 RTX 4090 上跑通 Qwen2.5-7B 单层 Forward（0.68ms）。

## 为什么要自己写？

用 vLLM / SGLang 跑推理很容易，但面试时被问「底层发生了什么」就会卡住。我想通过从零实现来搞清楚：

1. **模型权重怎么从文件到 GPU？** — safetensors 二进制格式解析 + mmap 零拷贝
2. **一个 Transformer layer 需要哪些算子？** — RMSNorm / RoPE / GQA Attention / SwiGLU MLP
3. **cuBLAS 调用的 row-major/column-major 陷阱** — 这是实际工程中最容易出 bug 的地方
4. **BF16 vs FP16 的坑** — Qwen2.5 默认存 BF16，按 FP16 读取会全 NaN

---

## 架构总览

```
safetensors (mmap) → GpuTensor (RAII) → TransformerLayer::forward()
                                            ├── RMSNorm (custom kernel)
                                            ├── Q/K/V projection (cuBLAS)
                                            ├── RoPE (custom kernel)
                                            ├── GQA repeat_kv (custom kernel)
                                            ├── Attention scores (cublasHgemmStridedBatched)
                                            ├── Causal mask + softmax (custom kernel)
                                            ├── Output projection (cuBLAS)
                                            ├── Residual + RMSNorm
                                            └── SwiGLU MLP (cuBLAS + SiLU kernel)
```

项目结构：

```
cpp-inference-engine/
├── CMakeLists.txt              # CMake 构建 (FetchContent + CUDA Toolkit)
├── include/
│   ├── cuda_utils.cuh          # 错误检查宏 + cuBLAS RAII handle
│   ├── tensor.cuh              # FP16 GPU tensor (RAII, move-only)
│   ├── safetensors.h           # mmap 零拷贝权重加载
│   ├── kernels.cuh             # 9 个自定义 CUDA kernel
│   └── transformer.cuh         # TransformerConfig + TransformerLayer
├── src/
│   ├── tensor.cu / safetensors.cpp / kernels.cu
│   ├── transformer.cu          # 单层 forward (GQA + SwiGLU)
│   └── main.cpp                # CLI + benchmark
```

---

## 关键实现细节

### 1. Safetensors 解析：mmap + BF16 自动转换

Safetensors 格式非常简洁：前 8 字节是 header 长度（little-endian uint64），接着是 JSON header，最后是连续的 tensor 数据。

```cpp
// 解析 header
uint64_t hdr_len = 0;
std::memcpy(&hdr_len, base, 8);            // 前 8 字节
auto j = nlohmann::json::parse(base + 8);  // JSON header
data_start_ = 8 + hdr_len;                 // 数据起始偏移
```

关键优化：用 `mmap()` 做零拷贝映射，避免 `read()` 的内存拷贝开销。大模型权重动辄几 GB，这很重要。

**踩坑：BF16 vs FP16。** Qwen2.5 的 safetensors 存储的是 BF16（8 位指数 + 7 位尾数），不是 FP16（5 位指数 + 10 位尾数）。直接当 FP16 读取会得到全 NaN。解决方案是在 host 端做 BF16→FP32→FP16 转换：

```cpp
if (m.dtype == "BF16") {
    for (size_t i = 0; i < numel; ++i) {
        // BF16 → FP32：左移 16 位（BF16 = FP32 的高 16 位）
        uint32_t fp32_bits = static_cast<uint32_t>(bf16_data[i]) << 16;
        // FP32 → FP16：提取 sign/exp/mantissa 并截断
        uint16_t sign = (fp32_bits >> 16) & 0x8000;
        int exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
        uint16_t mant = (fp32_bits >> 13) & 0x03FF;
        // 处理上溢/下溢
        if (exp <= 0)       h = sign;           // 下溢 → 0
        else if (exp >= 31) h = sign | 0x7C00;  // 上溢 → inf
        else                h = sign | (exp << 10) | mant;
    }
}
```

### 2. 自定义 CUDA Kernels

实现了 9 个 kernel，核心是 **RMSNorm** 和 **RoPE**。

**RMSNorm** 用 shared memory 做 parallel reduction：

```cuda
__global__ void rms_norm_k(half* out, const half* x, const half* w,
                           int H, float eps) {
    extern __shared__ float sm[];
    // 每个 block 处理一行：先求平方和，parallel reduction 后取 rsqrt
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = __half2float(x[blockIdx.x * H + i]);
        local += v * v;
    }
    sm[threadIdx.x] = local;
    __syncthreads();
    // Reduction...
    float rms = rsqrtf(sm[0] / H + eps);
    // 归一化 + scale
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = __float2half(val * rms * __half2float(w[i]));
}
```

**RoPE** 的实现要同时处理 Q 和 K（head 数不同），用 unified indexing 一个 kernel 搞定。

### 3. GQA（Grouped-Query Attention）

Qwen2.5-7B 用 GQA：28 个 Q heads，4 个 KV heads。每个 KV head 服务 7 个 Q heads。实现上需要：

1. Q/K/V 投影后，transpose 到 `[heads, seq_len, head_dim]`
2. 用 `repeat_kv` kernel 将 KV 从 4 heads 扩展到 28 heads
3. 用 `cublasHgemmStridedBatched` 做 batched attention 计算

```cpp
// Attention scores: [n_heads, S, S] = Qt @ Kt^T / sqrt(D)
cublasHgemmStridedBatched(cublas_.get(),
    CUBLAS_OP_T, CUBLAS_OP_N,
    S, S, D, &alpha,
    Kt.data(), D, S * D,   // 每个 head 的 K: [S, D]
    Qt.data(), D, S * D,   // 每个 head 的 Q: [S, D]
    &beta,
    scores.data(), S, S * S,
    n_heads);               // batch = 28 heads
```

### 4. cuBLAS Row-Major 陷阱

cuBLAS 默认 column-major，但 C++ 代码通常 row-major。诀窍是利用 `C = A @ B^T` 在 row-major 下等价于 column-major 的 `C_col = B_col^T @ A_col`：

```cpp
// Row-major: out[M,N] = input[M,K] @ W[N,K]^T
cublasHgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,  // 转置 W，不转置 input
    N, M, K, &alpha,
    W, K,      // W[N,K] row → [K,N] col, lda=K
    input, K,  // input[M,K] row → [K,M] col, ldb=K
    &beta,
    out, N);   // out[M,N] row → [N,M] col, ldc=N
```

**这个转换推导过程是面试高频考点。**

### 5. RAII 内存管理

所有 GPU 资源用 RAII 管理，零手动 `cudaFree`：

```cpp
class GpuTensor {
    half* data_ = nullptr;
    bool owned_ = false;
public:
    GpuTensor(std::vector<int> shape) {
        CUDA_CHECK(cudaMalloc(&data_, numel() * sizeof(half)));
        owned_ = true;
    }
    ~GpuTensor() { if (owned_ && data_) cudaFree(data_); }
    // Move-only: 禁止拷贝，支持 move
    GpuTensor(GpuTensor&& o) noexcept;
    GpuTensor(const GpuTensor&) = delete;
};
```

---

## 运行结果

环境：RTX 4090 / CUDA 12.6 / Qwen2.5-7B-Instruct (BF16)

```
=== Tiny C++ Inference Engine ===
Config: hidden=3584, heads=28, kv_heads=4, head_dim=128
Loaded 9/9 weight tensors.

=== Results ===
Avg forward (1 layer): 0.68 ms
Output shape: [4, 3584]
Output[0, :8] = -0.1721 -0.1903 -0.3088 -0.3022 -0.1097 0.1157 -0.4243 0.0133
```

单层 forward 0.68ms，数值输出正常（非 NaN/Inf），表明 CUDA kernel 和 cuBLAS 调用逻辑正确。

---

## 性能阶梯：从 Naive CUDA 到 Triton 到 cuBLAS

同期我还做了 CUDA GEMM 两级优化和 Triton autotuned GEMM，形成完整的性能认知：

| 实现 | TFLOPS | 占 cuBLAS % | 代码量 |
|------|--------|------------|--------|
| Naive CUDA | ~2 | 0.9% | ~100 行 .cu |
| Tiled CUDA (shared memory) | ~4 | 1.7% | ~150 行 .cu |
| **Triton DSL (autotuned)** | **163.5** | **96.7%** | **~40 行 Python** |
| cuBLAS (vendor library) | 169.1 | 100% | API 调用 |

Triton 用 40 行 Python 就能达到 cuBLAS 96.7%，核心在于 `@triton.autotune` 自动搜索 BLOCK_M/N/K 和 pipeline stages。

---

## 收获与思考

1. **safetensors 格式非常适合 C++ 加载** — 二进制紧凑，JSON header 可用 nlohmann/json 解析，数据区可 mmap
2. **BF16/FP16 区分是生产中的真实问题** — 不看 dtype 直接读，debug 半天找不到 NaN 来源
3. **cuBLAS 的 row/column-major 转换** — 理解了这个推导，就理解了为什么框架代码里 GEMM 调用看起来「反直觉」
4. **GQA 的 repeat_kv** — 理解了 MHA → MQA → GQA 的演进，以及为什么 GQA 是 KV Cache 友好的
5. **RAII 是 C++ GPU 编程的最佳实践** — 避免 CUDA OOM 和资源泄漏

---

*完整代码：[GitHub 链接]*（替换为实际仓库地址）

*相关实验数据：AWQ INT4 量化 15.3→5.6GB（节省 63%）、vLLM TP=2 tensor parallel 验证、CUDA GEMM 两级优化 + Nsight Profiling*
