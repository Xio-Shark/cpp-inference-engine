# tiny-inference-engine

极简 C++ LLM 推理引擎 — 从零实现单层 Transformer Forward Pass。

## 目标

展示对 LLM 推理底层实现的理解：
- **C++ 工程能力**：CMake 构建、RAII 内存管理、move 语义
- **CUDA 内核开发**：RMSNorm、RoPE、SiLU、Softmax、layout transform
- **cuBLAS 集成**：`cublasHgemm` + `cublasHgemmStridedBatched` FP16 矩阵运算
- **模型格式解析**：safetensors 二进制格式（mmap 零拷贝加载）
- **Transformer 架构**：Pre-norm + GQA (Grouped-Query Attention) + SwiGLU MLP

## 架构

```
safetensors (mmap) → GpuTensor (RAII) → TransformerLayer::forward()
                                            ├── RMSNorm (custom kernel)
                                            ├── Q/K/V projection (cuBLAS)
                                            ├── RoPE (custom kernel)
                                            ├── GQA repeat_kv (custom kernel)
                                            ├── Attention scores (batched cuBLAS)
                                            ├── Causal mask + softmax (custom kernel)
                                            ├── Attention output (batched cuBLAS)
                                            ├── Output projection (cuBLAS)
                                            ├── Residual + RMSNorm
                                            ├── SwiGLU MLP (cuBLAS + custom kernels)
                                            └── Residual
```

## 构建

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
Model: Qwen2.5-7B-Instruct (FP16)
Config: hidden=3584, heads=28, kv_heads=4, head_dim=128
Layer: 0, Seq length: 4

Loading layer 0 weights from 2 shard(s)...
  ✓ model.layers.0.self_attn.q_proj.weight
  ✓ model.layers.0.self_attn.k_proj.weight
  ...
Loaded 9/9 weight tensors.

=== Results ===
Avg forward (1 layer): 2.35 ms
Output shape: [4, 3584]
Output[0, :8] = -0.0312 0.0156 ...
```

## 项目结构

```
cpp-inference-engine/
├── CMakeLists.txt              CMake 构建（FetchContent 获取 nlohmann/json）
├── include/
│   ├── cuda_utils.cuh          CUDA/cuBLAS 错误检查 + RAII handle
│   ├── tensor.cuh              FP16 GPU Tensor（RAII，move-only）
│   ├── safetensors.h           safetensors 解析器（mmap + JSON）
│   ├── kernels.cuh             自定义 CUDA 内核声明
│   └── transformer.cuh         TransformerConfig + TransformerLayer
├── src/
│   ├── tensor.cu               Tensor 实现
│   ├── safetensors.cpp         safetensors 解析实现
│   ├── kernels.cu              9 个 CUDA 内核实现
│   ├── transformer.cu          单层 forward（GQA + SwiGLU）
│   └── main.cpp                CLI 入口 + benchmark
└── README.md
```

## 技术要点

| 组件 | 技术 |
|------|------|
| 内存管理 | RAII + move 语义，零手动 `cudaFree` |
| 权重加载 | `mmap()` 零拷贝 + `cudaMemcpy` H2D |
| 矩阵运算 | `cublasHgemm` (FP16) + Strided Batched |
| 注意力 | GQA (`repeat_kv` kernel) + causal mask |
| 激活函数 | SwiGLU = SiLU(gate) ⊙ up |
| 位置编码 | RoPE (rotary position embedding) |
| 归一化 | RMSNorm (shared-memory parallel reduction) |

## 性能验证

将输出与 PyTorch 参考实现对比：
```python
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                              torch_dtype=torch.float16)
# 对比 model.model.layers[0] 的输出
```
