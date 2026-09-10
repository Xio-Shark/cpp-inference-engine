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

// ---- Workspace allocation ----
void TransformerLayer::ensure_workspace(int S) {
    int H = cfg_.hidden_size;

    if (workspace_seq_len_ >= S && workspace_.data() != nullptr) {
        if (output_.numel() != static_cast<size_t>(S) * H) {
            output_ = GpuTensor({S, H});
        }
        return;
    }

    int nh = cfg_.num_heads;
    int nkv = cfg_.num_kv_heads;
    int D = cfg_.head_dim;
    int I = cfg_.intermediate_size;
    int qd = nh * D;
    int kvd = nkv * D;

    // Buffer layout in workspace:
    // [0, S*H):               res1 (persists across Attention and MLP)
    // [S*H, 2*S*H):           normed (pre-norm and post-norm buffer)
    // [2*S*H, 2*S*H + scratch): Scratchpad (reused between Attention and MLP phases)
    size_t persistent_needed = static_cast<size_t>(S) * H * 2;

    // Attention scratchpad sizes:
    // Q (S*qd), K (S*kvd), V (S*kvd)
    // Qt (S*qd), Kt (S*qd), Vt (S*qd)
    // scores (nh*S*S)
    // attn (S*qd), attn_cat (S*qd)
    // attn_out (S*H)
    size_t attn_needed = static_cast<size_t>(S) * (
        qd + kvd + kvd +     // Q, K, V
        qd + qd + qd +       // Qt, Kt, Vt
        nh * S +             // scores
        qd + qd +            // attn, attn_cat
        H                    // attn_out
    );

    // MLP scratchpad sizes:
    // gate (S*I), up (S*I), mlp_out (S*H)
    size_t mlp_needed = static_cast<size_t>(S) * (
        I + I + H
    );

    size_t max_scratch = std::max(attn_needed, mlp_needed);
    size_t total_elements = persistent_needed + max_scratch;

    // Align to 256 elements
    total_elements = ((total_elements + 255) / 256) * 256;

    workspace_ = GpuTensor({static_cast<int>(total_elements)});
    if (output_.numel() != static_cast<size_t>(S) * H) {
        output_ = GpuTensor({S, H});
    }
    workspace_seq_len_ = S;
}

// ---- GQA Self-Attention ----
void TransformerLayer::attention_out(half* out, const half* x_normed,
                                     int S, int off) {
    int H = cfg_.hidden_size;
    int nh = cfg_.num_heads;
    int nkv = cfg_.num_kv_heads;
    int D = cfg_.head_dim;
    int qd = nh * D, kvd = nkv * D;
    int rep = nh / nkv;

    half* ws_base = workspace_.data();
    half* ws = ws_base + 2 * static_cast<size_t>(S) * H;
    size_t cur = 0;

    half* ptr_Q        = ws + cur; cur += static_cast<size_t>(S) * qd;
    half* ptr_K        = ws + cur; cur += static_cast<size_t>(S) * kvd;
    half* ptr_V        = ws + cur; cur += static_cast<size_t>(S) * kvd;
    half* ptr_Qt       = ws + cur; cur += static_cast<size_t>(S) * qd;
    half* ptr_Kt       = ws + cur; cur += static_cast<size_t>(S) * qd;
    half* ptr_Vt       = ws + cur; cur += static_cast<size_t>(S) * qd;
    half* ptr_scores   = ws + cur; cur += static_cast<size_t>(nh) * S * S;
    half* ptr_attn     = ws + cur; cur += static_cast<size_t>(S) * qd;
    half* ptr_attn_cat = ws + cur; cur += static_cast<size_t>(S) * qd;

    // 1. Q/K/V projections
    linear(ptr_Q, x_normed, w_.q_proj.data(), S, qd, H);
    linear(ptr_K, x_normed, w_.k_proj.data(), S, kvd, H);
    linear(ptr_V, x_normed, w_.v_proj.data(), S, kvd, H);

    // 2. RoPE on Q [S, nh, D] and K [S, nkv, D]
    apply_rope(ptr_Q, ptr_K, S, nh, nkv, D, off, cfg_.rope_theta);

    // 3. Transpose Q: [S, nh, D] -> [nh, S, D]
    transpose_012_to_102(ptr_Qt, ptr_Q, S, nh, D);

    // 4. Fused Transpose & Repeat KV: [S, nkv, D] -> [nh, S, D]
    // Eliminates separate transpose_012_to_102 and repeat_kv intermediate buffers!
    transpose_and_repeat_kv(ptr_Kt, ptr_K, nkv, rep, S, D);
    transpose_and_repeat_kv(ptr_Vt, ptr_V, nkv, rep, S, D);

    // 5. Attention scores: [nh, S, S] = Qt @ Kt^T / sqrt(D)
    float alpha = 1.0f / std::sqrt(static_cast<float>(D));
    gemm_batched(ctx_,
                 ptr_scores, ptr_Qt, ptr_Kt,
                 nh, S, S, D,
                 false, true, alpha);

    // 6. Fused causal mask + softmax in a single pass (eliminates writing intermediate masked scores)
    fused_causal_softmax(ptr_scores, nh, S);

    // 7. Attention output: [nh, S, D] = scores @ Vt
    gemm_batched(ctx_,
                 ptr_attn, ptr_scores, ptr_Vt,
                 nh, S, D, S,
                 false, false, 1.0f);

    // 8. Transpose back: [nh, S, D] → [S, nh, D] = [S, qd]
    transpose_012_to_102(ptr_attn_cat, ptr_attn, nh, S, D);

    // 9. Output projection: [S, H]
    linear(out, ptr_attn_cat, w_.o_proj.data(), S, H, qd);
}

// ---- SwiGLU MLP ----
void TransformerLayer::mlp_out(half* out, const half* x_normed, int S) {
    int H = cfg_.hidden_size;
    int I = cfg_.intermediate_size;

    half* ws_base = workspace_.data();
    half* ws = ws_base + 2 * static_cast<size_t>(S) * H;
    half* ptr_gate = ws;
    half* ptr_up   = ws + static_cast<size_t>(S) * I;

    linear(ptr_gate, x_normed, w_.gate_proj.data(), S, I, H);
    linear(ptr_up, x_normed, w_.up_proj.data(), S, I, H);

    // Fused SwiGLU activation: gate = silu(gate) * up in a single memory pass
    swiglu(ptr_gate, ptr_gate, ptr_up, S * I);

    linear(out, ptr_gate, w_.down_proj.data(), S, H, I);
}

// ---- Full layer forward ----
GpuTensor& TransformerLayer::forward(const GpuTensor& input,
                                      int S, int off) {
    int H = cfg_.hidden_size;
    int n = S * H;

    ensure_workspace(S);

    half* ws_base = workspace_.data();
    half* ptr_res1    = ws_base;
    half* ptr_normed  = ws_base + n;
    half* ptr_scratch = ws_base + 2 * n;

    // 1. Pre-norm: normed = rms_norm(input)
    rms_norm(ptr_normed, input.data(), w_.input_norm.data(),
             S, H, cfg_.rms_norm_eps);

    // 2. Attention: output goes into dedicated attn_out slot in scratchpad
    half* ptr_attn_out = ptr_scratch;
    attention_out(ptr_attn_out, ptr_normed, S, off);

    // 3. Residual 1: res1 = input + attn_out
    ewise_add(ptr_res1, input.data(), ptr_attn_out, n);

    // 4. Post-norm: normed = rms_norm(res1)
    rms_norm(ptr_normed, ptr_res1, w_.post_norm.data(),
             S, H, cfg_.rms_norm_eps);

    // 5. MLP: gate & up in scratchpad, output goes into non-overlapping mlp_out slot
    half* ptr_mlp_out = ptr_scratch + static_cast<size_t>(S) * (2 * cfg_.intermediate_size);
    mlp_out(ptr_mlp_out, ptr_normed, S);

    // 6. Residual 2: output_ = res1 + mlp_out (reuse the preallocated buffer)
    ewise_add(output_.data(), ptr_res1, ptr_mlp_out, n);
    return output_;
}

