# tiny-inference-engine

极简 C++ LLM 推理引擎 — 从零实现单层 Transformer Forward Pass，现已全面支持 **macOS (Apple Silicon Metal / MPS)** 与 **Linux (NVIDIA CUDA / cuBLAS)** 双平台。

## 目标

展示对 LLM 推理底层实现的深度理解：
- **C++ 工程能力**：现代 CMake 跨平台构建、RAII 资源管理、move-only 语义
- **多后端加速体系**：
  - **macOS (Apple Silicon)**：Apple 原生 Metal Compute Shaders (MSL) + MetalPerformanceShaders (MPS) FP16 矩阵计算，基于 Unified Memory（统一内存架构）实现 CPU/GPU 零拷贝
  - **Linux (NVIDIA)**：CUDA Custom Kernels + cuBLAS `cublasHgemm` / `cublasHgemmStridedBatched`
- **模型格式解析**：safetensors 二进制格式（mmap 零拷贝加载，支持 FP16/BF16 自动转换，FP32 降精度加载）
- **权重加载健壮性**：支持单文件 `model.safetensors`、`model.safetensors.index.json` 分片索引，以及 legacy `model-*.safetensors` 分片；加载时校验 9 个权重是否齐全、shape 是否匹配
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
│   ├── weight_loader.h         权重文件解析、单层权重加载与 shape 校验
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
│   ├── weight_loader.cpp       单文件/index/分片解析与 9 权重校验
│   └── main.cpp                CLI 入口 + 跨平台计时 Benchmark
└── README.md
```

## 技术要点与深度优化

| 组件 / 算子 | 原版基础实现 | 深度性能优化方案 (当前状态) |
|-------------|--------------|------------------------------|
| **显存生命周期** | 每轮推理动态 `newBuffer`/`cudaMalloc` 17 次 | **预热后稳态 0 次动态分配**：预分配连续 Scratchpad 与输出缓冲，各阶段复用偏移，彻底消除驱动锁与分配开销 |
| **多头注意力 (GEMM)** | CPU 循环分发 28 次单头 GEMM + 84 个 `MPSMatrix` | **原生 Batched GEMM**：使用 `[MPSMatrixDescriptor matrixDescriptorWithRows:... columns:... matrices:... rowBytes:... matrixBytes:... dataType:...]` 与 `matmul.batchSize = batch`，单次分发 |
| **GQA KV 处理** | 转置 `transpose_012_to_102` + 广播 `repeat_kv`（2 次显存往返） | **Fused Transpose & Repeat**：直接由 `[S, n_kv, D]` 计算出 `[n_heads, S, D]`，消除中间张量与显存搬运 |
| **注意力归一化** | 写入 mask (-1e4) + 读回计算 Softmax（2 次全局显存往返） | **Fused Causal Softmax**：单个 GPU 线程组原地结合因果条件与两遍规约，消除掩码大矩阵显存分配与冗余访存 |
| **MLP 激活函数** | `silu_inplace` (读/写) + `ewise_mul` (读2/写1)（共 5 次读写） | **Fused SwiGLU**：单个 GPU 内核完成 `silu(gate) * up` 计算，访存降至 2 读 1 写（访存带宽节约 40%） |
| **内存管理** | `MTLResourceStorageModeShared` 统一内存零拷贝 | RAII `cudaMalloc` / `cudaFree` (CUDA) |
| **权重加载** | `mmap()` 零拷贝直接注入 Metal 统一内存 | `mmap()` + `cudaMemcpy` H2D (CUDA) |
| **位置编码** | Metal Compute Shader `rope_kernel` | CUDA `rope_k` |

## 优化效果分析

1. **CPU/GPU 调度瓶颈破除**：通过 MPS 原生 Batched GEMM，避免了每一层 28 个注意力头的 CPU 循环分发与 Objective-C 对象开辟，使 GPU 队列持续饱满。
2. **预热后稳态 0 显存分配 (Zero-Allocation)**：`TransformerLayer::forward` 在预热阶段预分配满足最大序列长度的共享工作区与输出缓冲，循环推理过程中稳态显存分配次数为 **0 次**，避免多线程锁竞争与显存抖动。
3. **全局访存流量断崖式下降**：融合 SwiGLU、因果 Softmax 与 Transpose-Repeat KV 后，每层推理减少了 4 个中间全量张量的落地存储，计算强度显著提高。

## 正确性测试

无需模型权重，可直接在 macOS Metal 环境运行 kernel 级 CPU 参考对比：

```bash
# 方式一：CMake
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure

# 方式二：macOS Makefile
make test
```

`tests/test_kernels.cpp` 会独立实现 CPU 参考，并校验 RMSNorm、RoPE、GQA KV transpose/repeat、SwiGLU、causal softmax、transpose 以及 MPS/cuBLAS GEMM 的数值结果。

`tests/test_model_loader.cpp` 不下载真实模型，而是生成 tiny synthetic safetensors 文件，覆盖：单文件、`model.safetensors.index.json` 分片、legacy 分片、缺权重报错、shape 不匹配报错，以及 config/layer 边界校验。

> 当前仓库仍是 Qwen2.5 单层 forward demo：尚未覆盖完整模型、KV cache 与端到端生成。与 PyTorch 逐层全模型对齐是下一阶段目标。

## 性能验证

后续将输出与 PyTorch 参考实现对比，并给出可复现的逐层数值误差与 benchmark 脚本：

```python
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                              torch_dtype=torch.float16)
# 对比 model.model.layers[0] 的输出
```
