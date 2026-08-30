#include <metal_stdlib>
using namespace metal;

// ===================== RMSNorm =====================
// One threadgroup per row. Threadgroup memory reduction.
kernel void rms_norm_kernel(
    device half* out                  [[buffer(0)]],
    device const half* x              [[buffer(1)]],
    device const half* w              [[buffer(2)]],
    constant int& H                   [[buffer(3)]],
    constant float& eps               [[buffer(4)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint threads_per_group            [[threads_per_threadgroup]]
) {
    device const half* xi = x + row * H;
    device half* yi = out + row * H;

    threadgroup float sm[256];

    float local = 0.0f;
    for (int i = tid; i < H; i += threads_per_group) {
        float v = float(xi[i]);
        local += v * v;
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] += sm[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(sm[0] / float(H) + eps);
    for (int i = tid; i < H; i += threads_per_group) {
        yi[i] = half(float(xi[i]) * rms * float(w[i]));
    }
}

// ===================== RoPE =====================
struct RoPEParams {
    int S;
    int nq;
    int nkv;
    int D;
    int off;
    float theta;
};

kernel void rope_kernel(
    device half* q                    [[buffer(0)]],
    device half* k                    [[buffer(1)]],
    constant RoPEParams& p            [[buffer(2)]],
    uint idx                          [[thread_position_in_grid]]
) {
    int half_d = p.D / 2;
    int total_q = p.S * p.nq * half_d;
    int total_k = p.S * p.nkv * half_d;
    if (idx >= uint(total_q + total_k)) return;

    bool is_q = (int(idx) < total_q);
    int li = is_q ? int(idx) : (int(idx) - total_q);
    int nh = is_q ? p.nq : p.nkv;
    device half* d = is_q ? q : k;

    int pos_in_half = li % half_d;
    int h = (li / half_d) % nh;
    int s = li / (nh * half_d);
    int pos = s + p.off;

    float freq = 1.0f / pow(p.theta, 2.0f * float(pos_in_half) / float(p.D));
    float ang = float(pos) * freq;
    float c = cos(ang);
    float sn = sin(ang);

    int base = s * nh * p.D + h * p.D;
    float x0 = float(d[base + pos_in_half]);
    float x1 = float(d[base + pos_in_half + half_d]);
    d[base + pos_in_half]          = half(x0 * c - x1 * sn);
    d[base + pos_in_half + half_d] = half(x0 * sn + x1 * c);
}

// ===================== SiLU =====================
kernel void silu_kernel(
    device half* data                 [[buffer(0)]],
    constant int& n                   [[buffer(1)]],
    uint idx                          [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        float v = float(data[idx]);
        data[idx] = half(v / (1.0f + exp(-v)));
    }
}

// ===================== Element-wise =====================
kernel void ewise_mul_kernel(
    device half* out                  [[buffer(0)]],
    device const half* a              [[buffer(1)]],
    device const half* b              [[buffer(2)]],
    constant int& n                   [[buffer(3)]],
    uint idx                          [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        out[idx] = half(float(a[idx]) * float(b[idx]));
    }
}

kernel void ewise_add_kernel(
    device half* out                  [[buffer(0)]],
    device const half* a              [[buffer(1)]],
    device const half* b              [[buffer(2)]],
    constant int& n                   [[buffer(3)]],
    uint idx                          [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        out[idx] = half(float(a[idx]) + float(b[idx]));
    }
}

// ===================== Softmax =====================
kernel void softmax_rows_kernel(
    device half* data                 [[buffer(0)]],
    constant int& C                   [[buffer(1)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint threads_per_group            [[threads_per_threadgroup]]
) {
    device half* r = data + row * C;
    threadgroup float sm[256];

    float mx = -1e30f;
    for (int i = tid; i < C; i += threads_per_group) {
        mx = max(mx, float(r[i]));
    }
    sm[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] = max(sm[tid], sm[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mv = sm[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float es = 0.0f;
    for (int i = tid; i < C; i += threads_per_group) {
        es += exp(float(r[i]) - mv);
    }
    sm[tid] = es;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] += sm[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sv = sm[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tid; i < C; i += threads_per_group) {
        r[i] = half(exp(float(r[i]) - mv) / sv);
    }
}

// ===================== Causal Mask =====================
struct MaskParams {
    int H;
    int S;
};

kernel void causal_mask_kernel(
    device half* scores               [[buffer(0)]],
    constant MaskParams& p            [[buffer(1)]],
    uint3 thread_id                   [[thread_position_in_grid]]
) {
    int c = thread_id.x;
    int r = thread_id.y;
    int h = thread_id.z;

    if (c < p.S && r < p.S && h < p.H) {
        if (c > r) {
            scores[h * p.S * p.S + r * p.S + c] = half(-1e4f);
        }
    }
}

// ===================== Layout Transforms =====================
struct TransposeParams {
    int A;
    int B;
    int D;
};

kernel void transpose_012_to_102_kernel(
    device half* out                  [[buffer(0)]],
    device const half* in             [[buffer(1)]],
    constant TransposeParams& p       [[buffer(2)]],
    uint idx                          [[thread_position_in_grid]]
) {
    int total = p.A * p.B * p.D;
    if (idx >= uint(total)) return;

    int d = idx % p.D;
    int b = (idx / p.D) % p.B;
    int a = idx / (p.B * p.D);

    out[b * p.A * p.D + a * p.D + d] = in[a * p.B * p.D + b * p.D + d];
}

struct RepeatKVParams {
    int nkv;
    int rep;
    int S;
    int D;
};

kernel void repeat_kv_kernel(
    device half* out                  [[buffer(0)]],
    device const half* in             [[buffer(1)]],
    constant RepeatKVParams& p        [[buffer(2)]],
    uint idx                          [[thread_position_in_grid]]
) {
    int total = p.nkv * p.rep * p.S * p.D;
    if (idx >= uint(total)) return;

    int d = idx % p.D;
    int s = (idx / p.D) % p.S;
    int h = idx / (p.S * p.D);

    out[idx] = in[(h / p.rep) * p.S * p.D + s * p.D + d];
}

// ===================== SwiGLU Fused Kernel =====================
kernel void swiglu_kernel(
    device half* out                  [[buffer(0)]],
    device const half* gate           [[buffer(1)]],
    device const half* up             [[buffer(2)]],
    constant int& n                   [[buffer(3)]],
    uint idx                          [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        float g = float(gate[idx]);
        float u = float(up[idx]);
        float silu_g = g / (1.0f + exp(-g));
        out[idx] = half(silu_g * u);
    }
}

// ===================== Fused Causal Softmax =====================
// Each threadgroup processes one row (one query position of one head)
// Row length is S (columns). Causal condition: c > r is masked to -inf.
kernel void fused_causal_softmax_kernel(
    device half* scores               [[buffer(0)]],
    constant int& S                   [[buffer(1)]],
    uint row_idx                      [[threadgroup_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint threads_per_group            [[threads_per_threadgroup]]
) {
    int r = row_idx % S; // position in sequence
    device half* row_ptr = scores + row_idx * S;
    threadgroup float sm[256];

    // 1. Find max for valid positions c <= r
    float mx = -1e30f;
    for (int c = tid; c <= r; c += threads_per_group) {
        mx = max(mx, float(row_ptr[c]));
    }
    sm[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] = max(sm[tid], sm[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mv = sm[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute sum of exp(val - mv) for c <= r
    float es = 0.0f;
    for (int c = tid; c <= r; c += threads_per_group) {
        es += exp(float(row_ptr[c]) - mv);
    }
    sm[tid] = es;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sm[tid] += sm[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = (sm[0] > 0.0f) ? (1.0f / sm[0]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Write normalized probabilities for c <= r, and 0 for c > r
    for (int c = tid; c < S; c += threads_per_group) {
        if (c <= r) {
            row_ptr[c] = half(exp(float(row_ptr[c]) - mv) * inv_sum);
        } else {
            row_ptr[c] = half(0.0f);
        }
    }
}

// ===================== Fused Transpose & Repeat KV =====================
// Input:  [S, n_kv, D]
// Output: [n_heads, S, D] where n_heads = n_kv * repeats
kernel void transpose_and_repeat_kv_kernel(
    device half* out                  [[buffer(0)]],
    device const half* in             [[buffer(1)]],
    constant RepeatKVParams& p        [[buffer(2)]],
    uint idx                          [[thread_position_in_grid]]
) {
    int total = p.nkv * p.rep * p.S * p.D;
    if (idx >= uint(total)) return;

    int d = idx % p.D;
    int s = (idx / p.D) % p.S;
    int h = idx / (p.S * p.D);
    int kv_h = h / p.rep;

    // in is [S, nkv, D]
    out[idx] = in[s * (p.nkv * p.D) + kv_h * p.D + d];
}

