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
    if (workspace_seq_len_ >= S && workspace_.data() != nullptr) {
        return;
    }

    int H = cfg_.hidden_size;
    int nh = cfg_.num_heads;
    int nkv = cfg_.num_kv_heads;
    int D = cfg_.head_dim;
    int I = cfg_.intermediate_size;
    int qd = nh * D;
    int kvd = nkv * D;

    // Attention buffer sizes (in half elements):
    // Q: [S, qd]
    // K: [S, kvd]
    // V: [S, kvd]
    // Qt: [nh, S, D] == S * qd
    // Kt: [nh, S, D] == S * qd
    // Vt: [nh, S, D] == S * qd
    // scores: [nh, S, S]
    // attn: [nh, S, D] == S * qd
    // attn_cat: [S, qd] == S * qd
    // attn_out: [S, H]
    //
    // MLP buffer sizes:
    // gate: [S, I]
    // up: [S, I]
    // mlp_out: [S, H]
    //
    // Forward layer-level:
    // normed: [S, H]
    // res1: [S, H]
    //
    // Peak memory needed simultaneously during Attention:
    // Q (S*qd), K (S*kvd), V (S*kvd),
    // Qt (S*qd), Kt (S*qd), Vt (S*qd),
    // scores (nh*S*S), attn (S*qd), attn_cat (S*qd), attn_out (S*H),
    // normed (S*H), res1 (S*H)
    //
    // Peak memory needed during MLP:
    // gate (S*I), up (S*I), mlp_out (S*H), res1 (S*H)

    size_t attn_needed = static_cast<size_t>(S) * (
        qd + kvd + kvd +     // Q, K, V
        qd + qd + qd +       // Qt, Kt, Vt
        nh * S +             // scores
        qd + qd +            // attn, attn_cat
        H + H + H            // attn_out, normed, res1
    );

    size_t mlp_needed = static_cast<size_t>(S) * (
        I + I + H + H        // gate, up, mlp_out, res1
    );

    size_t total_elements = std::max(attn_needed, mlp_needed);
    // Align to 256 elements
    total_elements = ((total_elements + 255) / 256) * 256;

    workspace_ = GpuTensor({static_cast<int>(total_elements)});
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

    half* ws = workspace_.data();
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

    half* ws = workspace_.data();
    half* ptr_gate = ws;
    half* ptr_up   = ws + static_cast<size_t>(S) * I;

    linear(ptr_gate, x_normed, w_.gate_proj.data(), S, I, H);
    linear(ptr_up, x_normed, w_.up_proj.data(), S, I, H);

    // Fused SwiGLU activation: gate = silu(gate) * up in a single memory pass
    swiglu(ptr_gate, ptr_gate, ptr_up, S * I);

    linear(out, ptr_gate, w_.down_proj.data(), S, H, I);
}

// ---- Full layer forward ----
GpuTensor TransformerLayer::forward(const GpuTensor& input,
                                     int S, int off) {
    int H = cfg_.hidden_size;
    int n = S * H;

    ensure_workspace(S);

    // Workspace offsets for persistent buffers during this forward step
    // Attention buffers end before the persistent tail of workspace
    int nh = cfg_.num_heads;
    int D = cfg_.head_dim;
    int qd = nh * D;
    int kvd = cfg_.num_kv_heads * D;

    size_t attn_scratch_size = static_cast<size_t>(S) * (
        qd + kvd + kvd +     // Q, K, V
        qd + qd + qd +       // Qt, Kt, Vt
        nh * S +             // scores
        qd + qd              // attn, attn_cat
    );

    half* ws_base = workspace_.data();
    half* ptr_normed   = ws_base + attn_scratch_size;
    half* ptr_attn_out = ptr_normed + static_cast<size_t>(n);
    half* ptr_res1     = ptr_attn_out + static_cast<size_t>(n);

    // 1. Pre-norm
    rms_norm(ptr_normed, input.data(), w_.input_norm.data(),
             S, H, cfg_.rms_norm_eps);

    // 2. Attention
    attention_out(ptr_attn_out, ptr_normed, S, off);

    // 3. Residual 1: res1 = input + attn_out
    ewise_add(ptr_res1, input.data(), ptr_attn_out, n);

    // 4. Post-norm: normed2 -> ptr_normed (reuse ptr_normed buffer)
    rms_norm(ptr_normed, ptr_res1, w_.post_norm.data(),
             S, H, cfg_.rms_norm_eps);

    // 5. MLP: mlp_out -> ptr_attn_out (reuse ptr_attn_out buffer)
    mlp_out(ptr_attn_out, ptr_normed, S);

    // 6. Residual 2: output = res1 + mlp_out
    GpuTensor output({S, H});
    ewise_add(output.data(), ptr_res1, ptr_attn_out, n);
    return output;
}
