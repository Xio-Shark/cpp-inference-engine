#pragma once
#include "tensor.h"
#include "device_utils.h"
#include <string>

/// Model hyper-parameters (matches Qwen2.5 config.json).
struct TransformerConfig {
    int hidden_size        = 3584;
    int intermediate_size  = 18944;
    int num_heads          = 28;
    int num_kv_heads       = 4;
    int head_dim           = 128;
    int vocab_size         = 152064;
    int num_hidden_layers  = 28;
    float rms_norm_eps     = 1e-6f;
    float rope_theta       = 1000000.0f;

    static TransformerConfig from_json(const std::string& path);
};

/// Weights for a single transformer layer.
struct LayerWeights {
    GpuTensor q_proj, k_proj, v_proj, o_proj;  // attention
    GpuTensor gate_proj, up_proj, down_proj;    // MLP
    GpuTensor input_norm, post_norm;            // RMSNorm
};

/// Single transformer decoder layer (pre-norm + GQA + SwiGLU MLP).
class TransformerLayer {
public:
    TransformerLayer(const TransformerConfig& cfg, DeviceContext& ctx);

    /// input: [seq_len, hidden] -> output: [seq_len, hidden].
    /// The returned reference points at an internal, reusable output buffer.
    /// It stays valid until the next call to forward() with a different seq_len.
    GpuTensor& forward(const GpuTensor& input, int seq_len, int pos_offset = 0);

    LayerWeights& weights() { return w_; }
    const TransformerConfig& config() const { return cfg_; }

private:
    void linear(half* out, const half* in, const half* weight,
                int M, int N, int K);
    void attention_out(half* out, const half* x_normed, int seq_len, int offset);
    void mlp_out(half* out, const half* x_normed, int seq_len);
    void ensure_workspace(int seq_len);

    TransformerConfig cfg_;
    DeviceContext& ctx_;
    LayerWeights w_;
    GpuTensor workspace_;
    GpuTensor output_;
    int workspace_seq_len_ = 0;
};
