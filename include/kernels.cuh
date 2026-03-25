#pragma once
#include <cuda_fp16.h>

// ---- Normalization ----
void rms_norm(half* out, const half* x, const half* w,
              int rows, int hidden, float eps);

// ---- Positional Encoding ----
void apply_rope(half* q, half* k, int seq_len,
                int n_q_heads, int n_kv_heads,
                int head_dim, int offset, float theta);

// ---- Activations ----
void silu_inplace(half* data, int n);

// ---- Element-wise ----
void ewise_mul(half* out, const half* a, const half* b, int n);
void ewise_add(half* out, const half* a, const half* b, int n);

// ---- Attention helpers ----
void softmax_rows(half* data, int rows, int cols);
void causal_mask(half* scores, int n_heads, int seq_len);

// ---- Layout ----
/// Transpose [A, B, D] → [B, A, D]
void transpose_012_to_102(half* out, const half* in, int A, int B, int D);
/// Repeat KV heads: [n_kv, S, D] → [n_heads, S, D]
void repeat_kv(half* out, const half* in,
               int n_kv, int repeats, int S, int D);
