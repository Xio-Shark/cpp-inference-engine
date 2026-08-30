# tiny-inference-engine

极简 C++ LLM 推理引擎 — 从零实现单层 Transformer Forward Pass，现已全面支持 **macOS (Apple Silicon Metal / MPS)** 与 **Linux (NVIDIA CUDA / cuBLAS)** 双平台。

## 目标

展示对 LLM 推理底层实现的深度理解：
- **C++ 工程能力**：现代 CMake 跨平台构建、RAII 资源管理、move-only 语义
- **多后端加速体系**：
  - **macOS (Apple Silicon)**：Apple 原生 Metal Compute Shaders (MSL) + MetalPerformanceShaders (MPS) FP16 矩阵计算，基于 Unified Memory（统一内存架构）实现 CPU/GPU 零拷贝
  - **Linux (NVIDIA)**：CUDA Custom Kernels + cuBLAS `cublasHgemm` / `cublasHgemmStridedBatched`
- **模型格式解析**：safetensors 二进制格式（mmap 零拷贝加载，支持 BF16/FP16 自动转换）
- **Transformer 架构**：Pre-norm + GQA (Grouped-Query Attention) + SwiGLU MLP + RoPE 位置编码

## 架构

```
safetensors (mmap) → GpuTensor (RAII / Unified Memory) → TransformerLayer::forward()
                                                            ├── RMSNorm (Metal / CUDA 自定义内核)
                                                            ├── Q/K/V projection (MPS / cuBLAS GEMM)
                                                            ├── RoPE (Metal / CUDA 自定义内核)
                                                            ├── GQA repeat_kv (Metal / CUDA 自定义内核)
                                                            ├── Attention scores (Batched MPS / cuBLAS)
                                                            ├── Causal mask + softmax (Metal / CUDA)
                                                            ├── Attention output (Batched MPS / cuBLAS)
                                                            ├── Output projection (MPS / cuBLAS GEMM)
                                                            ├── Residual + RMSNorm
                                                            ├── SwiGLU MLP (MPS / cuBLAS + SiLU 内核)
                                                            └── Residual
```

## 构建

### 1. macOS (Apple Silicon: M1/M2/M3/M4)

macOS 平台自动启用 Metal 与 MetalPerformanceShaders (MPS) 原生编译：

```bash
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### 2. Linux (NVIDIA GPU: Ampere, Ada, Hopper 等)

Linux 平台自动检测并启用 CUDA 与 cuBLAS 编译：

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;89"
make -j$(nproc)
```

## 运行

```bash
# 加载 Qwen2.5-7B-Instruct FP16 模型的 layer 0，seq_len=4
./tiny_inference /path/to/Qwen2.5-7B-Instruct 0 4
```

### 输出示例

```
=== Tiny C++ Inference Engine ===
Backend: Apple Metal & MPS (Apple Silicon)
Model: Qwen2.5-7B-Instruct (FP16)
Config: hidden=3584, heads=28, kv_heads=4, head_dim=128
Layer: 0, Seq length: 4

Loading layer 0 weights from 2 shard(s)...
  ✓ model.layers.0.self_attn.q_proj.weight
  ✓ model.layers.0.self_attn.k_proj.weight
  ...
Loaded 9/9 weight tensors.

=== Results ===
Avg forward (1 layer): 2.18 ms
Output shape: [4, 3584]
Output[0, :8] = -0.0312 0.0156 ...

Done. Compare output with PyTorch to verify correctness.
```

## 项目结构

```
cpp-inference-engine/
├── CMakeLists.txt              CMake 跨平台自动检测与构建
├── include/
│   ├── device_utils.h          跨平台设备上下文 (MTLDevice/cublasHandle) 与半精度 half 抽象
│   ├── tensor.h                跨平台 FP16 GPU Tensor (RAII，move-only，统一内存)
│   ├── safetensors.h           safetensors 解析器 (mmap + JSON)
│   ├── kernels.h               跨平台计算算子接口声明
│   ├── transformer.h           TransformerConfig + TransformerLayer 核心抽象
│   ├── cuda_utils.cuh          [兼容层] CUDA 头文件重定向
│   ├── tensor.cuh              [兼容层] Tensor 头文件重定向
│   ├── kernels.cuh             [兼容层] 内核声明重定向
│   └── transformer.cuh         [兼容层] Transformer 头文件重定向
├── src/
│   ├── kernels.metal           Apple Metal 计算着色器源码 (RMSNorm, RoPE, SiLU, Softmax, Mask 等)
│   ├── kernels_metal.mm        macOS Metal 管线分发与 MPS FP16 GEMM 算子实现
│   ├── tensor_metal.mm         macOS Metal MTLBuffer 统一内存与生命周期实现
│   ├── kernels.cu              Linux CUDA 自定义内核与 cuBLAS GEMM 实现
│   ├── tensor.cu               Linux CUDA cudaMalloc 内存管理实现
│   ├── transformer.cpp         跨平台单层 Transformer Forward Pass (GQA + SwiGLU)
│   ├── safetensors.cpp         跨平台 safetensors 权重零拷贝解析
│   └── main.cpp                CLI 入口 + 跨平台计时 Benchmark
└── README.md
```

## 技术要点

| 组件 | macOS (Apple Silicon) | Linux (NVIDIA) |
|------|----------------------|----------------|
| **内存管理** | `MTLResourceStorageModeShared` 统一内存零拷贝 | RAII `cudaMalloc` / `cudaFree` |
| **权重加载** | `mmap()` 零拷贝直接注入 Metal 统一内存 | `mmap()` + `cudaMemcpy` H2D |
| **矩阵运算** | `MPSMatrixMultiplication` (FP16) | `cublasHgemm` + Strided Batched |
| **注意力机制** | GQA (`repeat_kv_kernel` MSL) + causal mask | GQA (`repeat_k` CUDA) + causal mask |
| **激活函数** | Metal Compute Shader `silu_kernel` ⊙ up | CUDA `silu_k` ⊙ up |
| **位置编码** | Metal Compute Shader `rope_kernel` | CUDA `rope_k` |
| **层归一化** | Metal threadgroup shared memory 规约 RMSNorm | CUDA shared-memory 并行规约 RMSNorm |

## 性能验证

将输出与 PyTorch 参考实现对比：
```python
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                              torch_dtype=torch.float16)
# 对比 model.model.layers[0] 的输出
```
