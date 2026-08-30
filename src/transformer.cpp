// transformer.cpp — Single decoder layer forward pass (Qwen2.5 architecture)
// Pre-norm + GQA self-attention + SwiGLU MLP
#include "transformer.h"
#include "kernels.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cmath>

// ---- Config loader ----
TransformerConfig TransformerConfig::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    auto j = nlohmann::json::parse(f);

    TransformerConfig c;
    c.hidden_size       = j.value("hidden_size", c.hidden_size);
    c.intermediate_size = j.value("intermediate_size", c.intermediate_size);
    c.num_heads         = j.value("num_attention_heads", c.num_heads);
    c.num_kv_heads      = j.value("num_key_value_heads", c.num_kv_heads);
    c.head_dim          = j.value("head_dim", c.head_dim);
    c.vocab_size        = j.value("vocab_size", c.vocab_size);
    c.rms_norm_eps      = j.value("rms_norm_eps", c.rms_norm_eps);
    c.rope_theta        = j.value("rope_theta", c.rope_theta);
    return c;
}

TransformerLayer::TransformerLayer(const TransformerConfig& cfg,
                                   DeviceContext& ctx)
    : cfg_(cfg), ctx_(ctx) {}

// ---- Linear: out[M,N] = in[M,K] @ W[N,K]^T ----
void TransformerLayer::linear(half* out, const half* in,
                               const half* W, int M, int N, int K) {
    gemm_linear(ctx_, out, in, W, M, N, K);
}

// ---- GQA Self-Attention ----
GpuTensor TransformerLayer::attention(const GpuTensor& x,
                                       int S, int off) {
    int H = cfg_.hidden_size;
    int nh = cfg_.num_heads;
    int nkv = cfg_.num_kv_heads;
    int D = cfg_.head_dim;
    int qd = nh * D, kvd = nkv * D;
    int rep = nh / nkv;

    // 1. Q/K/V projections
    GpuTensor Q({S, qd}), K({S, kvd}), V({S, kvd});
    linear(Q.data(), x.data(), w_.q_proj.data(), S, qd, H);
    linear(K.data(), x.data(), w_.k_proj.data(), S, kvd, H);
    linear(V.data(), x.data(), w_.v_proj.data(), S, kvd, H);

    // 2. RoPE on Q [S, nh, D] and K [S, nkv, D]
    apply_rope(Q.data(), K.data(), S, nh, nkv, D, off, cfg_.rope_theta);

    // 3. Transpose to [heads, S, D]
    GpuTensor Qt({nh, S, D}), Kt0({nkv, S, D}), Vt0({nkv, S, D});
    transpose_012_to_102(Qt.data(), Q.data(), S, nh, D);
    transpose_012_to_102(Kt0.data(), K.data(), S, nkv, D);
    transpose_012_to_102(Vt0.data(), V.data(), S, nkv, D);

    // 4. GQA: repeat K/V → [nh, S, D]
    GpuTensor Kt({nh, S, D}), Vt({nh, S, D});
    repeat_kv(Kt.data(), Kt0.data(), nkv, rep, S, D);
    repeat_kv(Vt.data(), Vt0.data(), nkv, rep, S, D);

    // 5. Attention scores: [nh, S, S] = Qt @ Kt^T / sqrt(D)
    GpuTensor scores({nh, S, S});
    float alpha = 1.0f / std::sqrt(static_cast<float>(D));
    // For each head h: scores[h] = Qt[h] [S, D] @ Kt[h]^T [D, S] -> [S, S]
    gemm_batched(ctx_,
                 scores.data(), Qt.data(), Kt.data(),
                 nh, S, S, D,
                 false, true, alpha);

    // 6. Causal mask + softmax
    causal_mask(scores.data(), nh, S);
    softmax_rows(scores.data(), nh * S, S);

    // 7. Attention output: [nh, S, D] = scores @ Vt
    // scores[h] [S, S] @ Vt[h] [S, D] -> [S, D]
    GpuTensor attn({nh, S, D});
    gemm_batched(ctx_,
                 attn.data(), scores.data(), Vt.data(),
                 nh, S, D, S,
                 false, false, 1.0f);

    // 8. Transpose back: [nh, S, D] → [S, nh, D] = [S, qd]
    GpuTensor attn_cat({S, qd});
    transpose_012_to_102(attn_cat.data(), attn.data(), nh, S, D);

    // 9. Output projection
    GpuTensor out({S, H});
    linear(out.data(), attn_cat.data(), w_.o_proj.data(), S, H, qd);
    return out;
}

// ---- SwiGLU MLP ----
GpuTensor TransformerLayer::mlp(const GpuTensor& x, int S) {
    int H = cfg_.hidden_size;
    int I = cfg_.intermediate_size;

    GpuTensor gate({S, I}), up({S, I});
    linear(gate.data(), x.data(), w_.gate_proj.data(), S, I, H);
    linear(up.data(), x.data(), w_.up_proj.data(), S, I, H);

    silu_inplace(gate.data(), S * I);
    ewise_mul(gate.data(), gate.data(), up.data(), S * I);

    GpuTensor out({S, H});
    linear(out.data(), gate.data(), w_.down_proj.data(), S, H, I);
    return out;
}

// ---- Full layer forward ----
GpuTensor TransformerLayer::forward(const GpuTensor& input,
                                     int S, int off) {
    int H = cfg_.hidden_size;
    int n = S * H;

    // Pre-norm → attention → residual
    GpuTensor normed({S, H});
    rms_norm(normed.data(), input.data(), w_.input_norm.data(),
             S, H, cfg_.rms_norm_eps);

    GpuTensor attn_out = attention(normed, S, off);

    GpuTensor res1({S, H});
    ewise_add(res1.data(), input.data(), attn_out.data(), n);

    // Post-norm → MLP → residual
    GpuTensor normed2({S, H});
    rms_norm(normed2.data(), res1.data(), w_.post_norm.data(),
             S, H, cfg_.rms_norm_eps);

    GpuTensor mlp_out = mlp(normed2, S);

    GpuTensor output({S, H});
    ewise_add(output.data(), res1.data(), mlp_out.data(), n);
    return output;
}
