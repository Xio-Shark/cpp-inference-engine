#pragma once
#include "device_utils.h"

// ---- Normalization ----
void rms_norm(half* out, const half* x, const half* w,
              int rows, int hidden, float eps);

// ---- Positional Encoding ----
void apply_rope(half* q, half* k, int seq_len,
                int n_q_heads, int n_kv_heads,
                int head_dim, int offset, float theta);

// ---- Activations ----
void silu_inplace(half* data, int n);
void swiglu(half* out, const half* gate, const half* up, int n);

// ---- Element-wise ----
void ewise_mul(half* out, const half* a, const half* b, int n);
void ewise_add(half* out, const half* a, const half* b, int n);

// ---- Attention helpers ----
void softmax_rows(half* data, int rows, int cols);
void causal_mask(half* scores, int n_heads, int seq_len);
void fused_causal_softmax(half* scores, int n_heads, int seq_len);

// ---- Layout ----
/// Transpose [A, B, D] → [B, A, D]
void transpose_012_to_102(half* out, const half* in, int A, int B, int D);
/// Repeat KV heads: [n_kv, S, D] → [n_heads, S, D]
void repeat_kv(half* out, const half* in,
               int n_kv, int repeats, int S, int D);
/// Fused Transpose & Repeat KV: [S, n_kv, D] → [n_heads, S, D]
void transpose_and_repeat_kv(half* out, const half* in,
                             int n_kv, int repeats, int S, int D);

// ---- Linear / GEMM operations ----
/// Linear projection: out[M, N] = in[M, K] @ W[N, K]^T
void gemm_linear(DeviceContext& ctx, half* out, const half* in, const half* weight,
                 int M, int N, int K);

/// Batched GEMM: C[batch, M, N] = A[batch, M, K] @ B[batch, K, N] (or transposed)
void gemm_batched(DeviceContext& ctx,
                  half* C, const half* A, const half* B,
                  int batch, int M, int N, int K,
                  bool trans_a, bool trans_b, float alpha);
