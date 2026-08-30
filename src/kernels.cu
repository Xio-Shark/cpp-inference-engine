#if defined(USE_CUDA) || defined(__CUDACC__)
// kernels.cu — Custom CUDA kernels for transformer inference
// RMSNorm, RoPE, SiLU, element-wise ops, softmax, layout transforms
#include "kernels.h"
#include "device_utils.h"
#include <cfloat>

static constexpr int BLK = 256;

// ===================== RMSNorm =====================
// One block per row. Shared-memory parallel reduction.
__global__ void rms_norm_k(half* out, const half* x, const half* w,
                           int H, float eps) {
    int row = blockIdx.x;
    const half* xi = x + row * H;
    half* yi = out + row * H;
    extern __shared__ float sm[];

    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = __half2float(xi[i]);
        local += v * v;
    }
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sm[0] / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yi[i] = __float2half(__half2float(xi[i]) * rms * __half2float(w[i]));
}

void rms_norm(half* out, const half* x, const half* w,
              int rows, int H, float eps) {
    int t = min(BLK, H);
    rms_norm_k<<<rows, t, t * sizeof(float)>>>(out, x, w, H, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== RoPE =====================
__global__ void rope_k(half* q, half* k, int S,
                       int nq, int nkv, int D, int off, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = D / 2;
    int total_q = S * nq * half_d;
    int total_k = S * nkv * half_d;
    if (idx >= total_q + total_k) return;

    bool is_q = idx < total_q;
    int li = is_q ? idx : idx - total_q;
    int nh = is_q ? nq : nkv;
    half* d = is_q ? q : k;

    int p = li % half_d;
    int h = (li / half_d) % nh;
    int s = li / (nh * half_d);
    int pos = s + off;

    float freq = 1.0f / powf(theta, 2.0f * p / D);
    float ang = pos * freq;
    float c = cosf(ang), sn = sinf(ang);

    int base = s * nh * D + h * D;
    float x0 = __half2float(d[base + p]);
    float x1 = __half2float(d[base + p + half_d]);
    d[base + p]          = __float2half(x0 * c - x1 * sn);
    d[base + p + half_d] = __float2half(x0 * sn + x1 * c);
}

void apply_rope(half* q, half* k, int S,
                int nq, int nkv, int D, int off, float theta) {
    int n = S * (nq + nkv) * (D / 2);
    rope_k<<<(n + BLK - 1) / BLK, BLK>>>(q, k, S, nq, nkv, D, off, theta);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== SiLU =====================
__global__ void silu_k(half* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(d[i]);
        d[i] = __float2half(v / (1.f + expf(-v)));
    }
}
void silu_inplace(half* d, int n) {
    silu_k<<<(n + BLK - 1) / BLK, BLK>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Element-wise =====================
__global__ void mul_k(half* o, const half* a, const half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}
void ewise_mul(half* o, const half* a, const half* b, int n) {
    mul_k<<<(n + BLK - 1) / BLK, BLK>>>(o, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void add_k(half* o, const half* a, const half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}
void ewise_add(half* o, const half* a, const half* b, int n) {
    add_k<<<(n + BLK - 1) / BLK, BLK>>>(o, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Softmax =====================
__global__ void softmax_k(half* d, int C) {
    int row = blockIdx.x;
    half* r = d + row * C;
    extern __shared__ float sm[];

    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        mx = fmaxf(mx, __half2float(r[i]));
    sm[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    float mv = sm[0]; __syncthreads();

    float es = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        es += expf(__half2float(r[i]) - mv);
    sm[threadIdx.x] = es;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    float sv = sm[0];
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        r[i] = __float2half(expf(__half2float(r[i]) - mv) / sv);
}
void softmax_rows(half* d, int R, int C) {
    int t = min(BLK, C);
    softmax_k<<<R, t, t * sizeof(float)>>>(d, C);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Causal mask =====================
__global__ void mask_k(half* s, int S) {
    int h = blockIdx.z, r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < S && c > r)
        s[h * S * S + r * S + c] = __float2half(-1e4f);
}
void causal_mask(half* s, int H, int S) {
    dim3 grid((S + BLK - 1) / BLK, S, H);
    mask_k<<<grid, BLK>>>(s, S);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== Layout transforms =====================
__global__ void transpose_k(half* o, const half* in, int A, int B, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= A * B * D) return;
    int d = idx % D, b = (idx / D) % B, a = idx / (B * D);
    o[b * A * D + a * D + d] = in[a * B * D + b * D + d];
}
void transpose_012_to_102(half* o, const half* in, int A, int B, int D) {
    int n = A * B * D;
    transpose_k<<<(n + BLK - 1) / BLK, BLK>>>(o, in, A, B, D);
    CUDA_CHECK(cudaGetLastError());
__global__ void repeat_k(half* o, const half* in, int nkv, int rep, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nkv * rep * S * D;
    if (idx >= total) return;
    int d = idx % D, s = (idx / D) % S;
    int h = idx / (S * D);
    o[idx] = in[(h / rep) * S * D + s * D + d];
}
void repeat_kv(half* o, const half* in, int nkv, int rep, int S, int D) {
    int n = nkv * rep * S * D;
    repeat_k<<<(n + BLK - 1) / BLK, BLK>>>(o, in, nkv, rep, S, D);
    CUDA_CHECK(cudaGetLastError());
}

// ===================== cuBLAS GEMM Implementations =====================
#if defined(USE_CUDA) || defined(__CUDACC__)
void gemm_linear(DeviceContext& ctx, half* out, const half* in, const half* weight,
                 int M, int N, int K) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.get_native_handle());
    half alpha = __float2half(1.0f);
    half beta  = __float2half(0.0f);
    CUBLAS_CHECK(cublasHgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &alpha, weight, K, in, K, &beta, out, N));
}

void gemm_batched(DeviceContext& ctx,
                  half* C, const half* A, const half* B,
                  int batch, int M, int N, int K,
                  bool trans_a, bool trans_b, float alpha_val) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.get_native_handle());
    half alpha = __float2half(alpha_val);
    half beta  = __float2half(0.0f);
    cublasOperation_t opA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;

    CUBLAS_CHECK(cublasHgemmStridedBatched(handle,
        opB, opA,
        N, M, K,
        &alpha,
        B, ldb, strideB,
        A, lda, strideA,
        &beta,
        C, N, strideC,
        batch));
}
#endif // USE_CUDA
#endif // file level USE_CUDA
