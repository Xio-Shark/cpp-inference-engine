#if defined(__APPLE__)
#include "kernels.h"
#include "tensor.h"
#include "device_utils.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

// Forward declaration of Metal device singleton
id<MTLDevice> get_default_metal_device();

// ===================== Buffer Registry =====================
struct MetalBufferSpan {
    uintptr_t start;
    size_t length;
    id<MTLBuffer> buffer;
};

static std::map<uintptr_t, MetalBufferSpan> g_metal_buffers;
static std::mutex g_metal_buffers_mutex;

void register_metal_buffer(const void* ptr, size_t bytes, void* mtl_buf) {
    if (!ptr || !mtl_buf) return;
    std::lock_guard<std::mutex> lock(g_metal_buffers_mutex);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    g_metal_buffers[addr] = {addr, bytes, (__bridge id<MTLBuffer>)mtl_buf};
}

void unregister_metal_buffer(const void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_metal_buffers_mutex);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    g_metal_buffers.erase(addr);
}

id<MTLBuffer> get_metal_buffer_for_pointer(const void* ptr, size_t* offset_out) {
    if (!ptr) return nil;
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(g_metal_buffers_mutex);

    auto it = g_metal_buffers.upper_bound(addr);
    if (it != g_metal_buffers.begin()) {
        --it;
        if (addr >= it->second.start && addr < it->second.start + it->second.length) {
            if (offset_out) {
                *offset_out = addr - it->second.start;
            }
            return it->second.buffer;
        }
    }
    return nil;
}

// Embedded Metal Shading Source to ensure zero external file dependency
static const char* kMetalShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

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
kernel void fused_causal_softmax_kernel(
    device half* scores               [[buffer(0)]],
    constant int& S                   [[buffer(1)]],
    uint row_idx                      [[threadgroup_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint threads_per_group            [[threads_per_threadgroup]]
) {
    int r = row_idx % S;
    device half* row_ptr = scores + row_idx * S;
    threadgroup float sm[256];

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

    for (int c = tid; c < S; c += threads_per_group) {
        if (c <= r) {
            row_ptr[c] = half(exp(float(row_ptr[c]) - mv) * inv_sum);
        } else {
            row_ptr[c] = half(0.0f);
        }
    }
}

// ===================== Fused Transpose & Repeat KV =====================
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

    out[idx] = in[s * (p.nkv * p.D) + kv_h * p.D + d];
}
)METAL";

// ===================== DeviceContext Implementation =====================
class DeviceContextImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> active_cmd_buf = nil;
    id<MTLComputeCommandEncoder> active_encoder = nil;

    id<MTLComputePipelineState> pso_rms_norm;
    id<MTLComputePipelineState> pso_rope;
    id<MTLComputePipelineState> pso_silu;
    id<MTLComputePipelineState> pso_mul;
    id<MTLComputePipelineState> pso_add;
    id<MTLComputePipelineState> pso_softmax;
    id<MTLComputePipelineState> pso_mask;
    id<MTLComputePipelineState> pso_transpose;
    id<MTLComputePipelineState> pso_repeat;
    id<MTLComputePipelineState> pso_swiglu;
    id<MTLComputePipelineState> pso_fused_causal_softmax;
    id<MTLComputePipelineState> pso_transpose_and_repeat_kv;

    DeviceContextImpl() {
        @autoreleasepool {
            device = get_default_metal_device();
            queue = [device newCommandQueue];
            compile_pipelines();
        }
    }

    ~DeviceContextImpl() {
        synchronize();
    }

    void compile_pipelines() {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:kMetalShaderSource];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
        if (!lib) {
            std::string err_str = error ? [[error localizedDescription] UTF8String] : "Unknown error";
            throw std::runtime_error("Failed to compile Metal library: " + err_str);
        }

        auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) throw std::runtime_error(std::string("Missing Metal kernel function: ") + name);
            NSError* pso_err = nil;
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&pso_err];
            if (!pso) {
                std::string err_str = pso_err ? [[pso_err localizedDescription] UTF8String] : "Unknown error";
                throw std::runtime_error("Failed to create PSO for " + std::string(name) + ": " + err_str);
            }
            return pso;
        };

        pso_rms_norm                   = make_pso("rms_norm_kernel");
        pso_rope                       = make_pso("rope_kernel");
        pso_silu                       = make_pso("silu_kernel");
        pso_mul                        = make_pso("ewise_mul_kernel");
        pso_add                        = make_pso("ewise_add_kernel");
        pso_softmax                    = make_pso("softmax_rows_kernel");
        pso_mask                       = make_pso("causal_mask_kernel");
        pso_transpose                  = make_pso("transpose_012_to_102_kernel");
        pso_repeat                     = make_pso("repeat_kv_kernel");
        pso_swiglu                     = make_pso("swiglu_kernel");
        pso_fused_causal_softmax       = make_pso("fused_causal_softmax_kernel");
        pso_transpose_and_repeat_kv   = make_pso("transpose_and_repeat_kv_kernel");
    }

    id<MTLCommandBuffer> get_or_create_command_buffer() {
        if (!active_cmd_buf) {
            active_cmd_buf = [queue commandBuffer];
        }
        return active_cmd_buf;
    }

    id<MTLComputeCommandEncoder> get_compute_encoder() {
        if (!active_encoder) {
            id<MTLCommandBuffer> cb = get_or_create_command_buffer();
            active_encoder = [cb computeCommandEncoder];
        }
        return active_encoder;
    }

    void end_compute_encoder() {
        if (active_encoder) {
            [active_encoder endEncoding];
            active_encoder = nil;
        }
    }

    void synchronize() {
        end_compute_encoder();
        if (active_cmd_buf) {
            [active_cmd_buf commit];
            [active_cmd_buf waitUntilCompleted];
            active_cmd_buf = nil;
        }
    }
};

DeviceContext::DeviceContext() : impl_(std::make_unique<DeviceContextImpl>()) {}
DeviceContext::~DeviceContext() = default;
DeviceContext::DeviceContext(DeviceContext&&) noexcept = default;
DeviceContext& DeviceContext::operator=(DeviceContext&&) noexcept = default;

void* DeviceContext::get_native_handle() const {
    return (__bridge void*)impl_->device;
}

void DeviceContext::synchronize() {
    if (impl_) impl_->synchronize();
}

DeviceContext& get_default_device_context() {
    static DeviceContext ctx;
    return ctx;
}

// ===================== Helper for Buffer Binding =====================
static void bind_buffer_to_encoder(id<MTLComputeCommandEncoder> enc,
                                  const void* ptr,
                                  NSUInteger index) {
    size_t offset = 0;
    id<MTLBuffer> buf = get_metal_buffer_for_pointer(ptr, &offset);
    if (buf) {
        [enc setBuffer:buf offset:offset atIndex:index];
    } else {
        fprintf(stderr, "Error: pointer %p is not registered in Metal buffer registry\n", ptr);
    }
}

// ===================== Custom Kernel Dispatchers =====================

void rms_norm(half* out, const half* x, const half* w,
              int rows, int hidden, float eps) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_rms_norm];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, x, 1);
    bind_buffer_to_encoder(enc, w, 2);
    [enc setBytes:&hidden length:sizeof(int) atIndex:3];
    [enc setBytes:&eps length:sizeof(float) atIndex:4];

    int t = std::min(256, hidden);
    MTLSize tg_size = MTLSizeMake(t, 1, 1);
    MTLSize grid_size = MTLSizeMake(rows, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void apply_rope(half* q, half* k, int seq_len,
                int n_q_heads, int n_kv_heads,
                int head_dim, int offset, float theta) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_rope];

    bind_buffer_to_encoder(enc, q, 0);
    bind_buffer_to_encoder(enc, k, 1);

    struct {
        int S, nq, nkv, D, off;
        float theta;
    } params = {seq_len, n_q_heads, n_kv_heads, head_dim, offset, theta};
    [enc setBytes:&params length:sizeof(params) atIndex:2];

    int n = seq_len * (n_q_heads + n_kv_heads) * (head_dim / 2);
    MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void silu_inplace(half* data, int n) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_silu];

    bind_buffer_to_encoder(enc, data, 0);
    [enc setBytes:&n length:sizeof(int) atIndex:1];

    MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void ewise_mul(half* out, const half* a, const half* b, int n) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_mul];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, a, 1);
    bind_buffer_to_encoder(enc, b, 2);
    [enc setBytes:&n length:sizeof(int) atIndex:3];

    MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void ewise_add(half* out, const half* a, const half* b, int n) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_add];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, a, 1);
    bind_buffer_to_encoder(enc, b, 2);
    [enc setBytes:&n length:sizeof(int) atIndex:3];

    MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void softmax_rows(half* data, int rows, int cols) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_softmax];

    bind_buffer_to_encoder(enc, data, 0);
    [enc setBytes:&cols length:sizeof(int) atIndex:1];

    int t = std::min(256, cols);
    MTLSize grid_size = MTLSizeMake(rows, 1, 1);
    MTLSize tg_size = MTLSizeMake(t, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void causal_mask(half* scores, int n_heads, int seq_len) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_mask];

    bind_buffer_to_encoder(enc, scores, 0);
    struct { int H, S; } params = {n_heads, seq_len};
    [enc setBytes:&params length:sizeof(params) atIndex:1];

    MTLSize grid_size = MTLSizeMake((seq_len + 15) / 16, (seq_len + 15) / 16, n_heads);
    MTLSize tg_size = MTLSizeMake(16, 16, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void transpose_012_to_102(half* out, const half* in, int A, int B, int D) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_transpose];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, in, 1);
    struct { int A, B, D; } params = {A, B, D};
    [enc setBytes:&params length:sizeof(params) atIndex:2];

    int total = A * B * D;
    MTLSize grid_size = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void repeat_kv(half* out, const half* in, int n_kv, int repeats, int S, int D) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_repeat];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, in, 1);
    struct { int nkv, rep, S, D; } params = {n_kv, repeats, S, D};
    [enc setBytes:&params length:sizeof(params) atIndex:2];

    int total = n_kv * repeats * S * D;
    MTLSize grid_size = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void swiglu(half* out, const half* gate, const half* up, int n) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_swiglu];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, gate, 1);
    bind_buffer_to_encoder(enc, up, 2);
    [enc setBytes:&n length:sizeof(int) atIndex:3];

    MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void fused_causal_softmax(half* scores, int n_heads, int seq_len) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_fused_causal_softmax];

    bind_buffer_to_encoder(enc, scores, 0);
    [enc setBytes:&seq_len length:sizeof(int) atIndex:1];

    int total_rows = n_heads * seq_len;
    int t = std::min(256, seq_len);
    if (t < 32) t = 32;
    MTLSize grid_size = MTLSizeMake(total_rows, 1, 1);
    MTLSize tg_size = MTLSizeMake(t, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

void transpose_and_repeat_kv(half* out, const half* in, int n_kv, int repeats, int S, int D) {
    DeviceContext& ctx = get_default_device_context();
    id<MTLComputeCommandEncoder> enc = ctx.impl_->get_compute_encoder();
    [enc setComputePipelineState:ctx.impl_->pso_transpose_and_repeat_kv];

    bind_buffer_to_encoder(enc, out, 0);
    bind_buffer_to_encoder(enc, in, 1);
    struct { int nkv, rep, S, D; } params = {n_kv, repeats, S, D};
    [enc setBytes:&params length:sizeof(params) atIndex:2];

    int total = n_kv * repeats * S * D;
    MTLSize grid_size = MTLSizeMake((total + 255) / 256, 1, 1);
    MTLSize tg_size = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];
}

// ===================== MPS Matrix Multiplication =====================

void gemm_linear(DeviceContext& ctx, half* out, const half* in, const half* weight,
                 int M, int N, int K) {
    ctx.impl_->end_compute_encoder();
    id<MTLCommandBuffer> cb = ctx.impl_->get_or_create_command_buffer();

    size_t off_out = 0, off_in = 0, off_w = 0;
    id<MTLBuffer> buf_out = get_metal_buffer_for_pointer(out, &off_out);
    id<MTLBuffer> buf_in  = get_metal_buffer_for_pointer(in, &off_in);
    id<MTLBuffer> buf_w   = get_metal_buffer_for_pointer(weight, &off_w);

    if (!buf_out || !buf_in || !buf_w) {
        throw std::runtime_error("Invalid buffers provided to gemm_linear");
    }

    // in: [M, K] row-major -> rowBytes = K * sizeof(half)
    // weight: [N, K] row-major -> rowBytes = K * sizeof(half)
    // out: [M, N] row-major -> rowBytes = N * sizeof(half)
    // Computation: out[M, N] = in[M, K] @ weight[N, K]^T
    MPSMatrixDescriptor* desc_in = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                         columns:K
                                                                        rowBytes:K * sizeof(half)
                                                                        dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* desc_w = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                        columns:K
                                                                       rowBytes:K * sizeof(half)
                                                                       dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* desc_out = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                          columns:N
                                                                         rowBytes:N * sizeof(half)
                                                                         dataType:MPSDataTypeFloat16];

    MPSMatrix* mat_in  = [[MPSMatrix alloc] initWithBuffer:buf_in offset:off_in descriptor:desc_in];
    MPSMatrix* mat_w   = [[MPSMatrix alloc] initWithBuffer:buf_w offset:off_w descriptor:desc_w];
    MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:buf_out offset:off_out descriptor:desc_out];

    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx.impl_->device
        transposeLeft:NO
        transposeRight:YES
        resultRows:M
        resultColumns:N
        interiorColumns:K
        alpha:1.0
        beta:0.0];

    [matmul encodeToCommandBuffer:cb leftMatrix:mat_in rightMatrix:mat_w resultMatrix:mat_out];
}

void gemm_batched(DeviceContext& ctx,
                  half* C, const half* A, const half* B,
                  int batch, int M, int N, int K,
                  bool trans_a, bool trans_b, float alpha) {
    ctx.impl_->end_compute_encoder();
    id<MTLCommandBuffer> cb = ctx.impl_->get_or_create_command_buffer();

    size_t base_c = 0, base_a = 0, base_b = 0;
    id<MTLBuffer> buf_c = get_metal_buffer_for_pointer(C, &base_c);
    id<MTLBuffer> buf_a = get_metal_buffer_for_pointer(A, &base_a);
    id<MTLBuffer> buf_b = get_metal_buffer_for_pointer(B, &base_b);

    if (!buf_c || !buf_a || !buf_b) {
        throw std::runtime_error("Invalid buffers provided to gemm_batched");
    }

    int rowsA = trans_a ? K : M;
    int colsA = trans_a ? M : K;
    int rowsB = trans_b ? N : K;
    int colsB = trans_b ? K : N;

    size_t stride_a = rowsA * colsA * sizeof(half);
    size_t stride_b = rowsB * colsB * sizeof(half);
    size_t stride_c = M * N * sizeof(half);

    // Native batched descriptors: single MPSMatrix descriptor for the entire batch
    MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsA
                                                                        columns:colsA
                                                                       matrices:batch
                                                                       rowBytes:colsA * sizeof(half)
                                                                    matrixBytes:stride_a
                                                                       dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor matrixDescriptorWithRows:rowsB
                                                                        columns:colsB
                                                                       matrices:batch
                                                                       rowBytes:colsB * sizeof(half)
                                                                    matrixBytes:stride_b
                                                                       dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:N
                                                                       matrices:batch
                                                                       rowBytes:N * sizeof(half)
                                                                    matrixBytes:stride_c
                                                                       dataType:MPSDataTypeFloat16];

    MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a offset:base_a descriptor:desc_a];
    MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b offset:base_b descriptor:desc_b];
    MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c offset:base_c descriptor:desc_c];

    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx.impl_->device
        transposeLeft:trans_a ? YES : NO
        transposeRight:trans_b ? YES : NO
        resultRows:M
        resultColumns:N
        interiorColumns:K
        alpha:alpha
        beta:0.0];
    matmul.batchStart = 0;
    matmul.batchSize = batch;

    [matmul encodeToCommandBuffer:cb leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
}

#endif
