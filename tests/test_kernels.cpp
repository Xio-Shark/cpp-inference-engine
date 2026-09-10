// CPU-reference sanity tests for the GPU kernels.
//
// These tests do not need model weights. They generate small deterministic
// tensors, run each kernel through the real backend (Metal on macOS), and
// compare the result against an independent CPU implementation.
#include "kernels.h"
#include "device_utils.h"
#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

half to_half(float value) {
    return __float2half(value);
}

float to_float(half value) {
    return __half2float(value);
}

std::vector<half> make_values(size_t count, float scale = 1.0f) {
    std::vector<half> values(count);
    uint32_t state = 0x12345678u;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f;
        values[i] = to_half((unit * 2.0f - 1.0f) * scale);
    }
    return values;
}

bool close_enough(float got, float want, float atol, float rtol) {
    if (std::isnan(got) || std::isnan(want)) {
        return false;
    }
    return std::fabs(got - want) <= atol + rtol * std::max(std::fabs(got), std::fabs(want));
}

void expect_close(const std::string& name,
                  const std::vector<half>& got,
                  const std::vector<half>& want,
                  float atol = 5e-3f,
                  float rtol = 5e-3f) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "FAIL %s: size mismatch (%zu vs %zu)\n", name.c_str(), got.size(), want.size());
        ++g_failures;
        return;
    }

    float max_abs = 0.0f;
    size_t bad_index = 0;
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const float a = to_float(got[i]);
        const float b = to_float(want[i]);
        max_abs = std::max(max_abs, std::fabs(a - b));
        if (close_enough(a, b, atol, rtol) == false) {
            ok = false;
            bad_index = i;
            break;
        }
    }
    if (ok == false) {
        std::fprintf(stderr,
                     "FAIL %s: index=%zu got=%f want=%f max_abs=%f\n",
                     name.c_str(),
                     bad_index,
                     static_cast<double>(to_float(got[bad_index])),
                     static_cast<double>(to_float(want[bad_index])),
                     static_cast<double>(max_abs));
        ++g_failures;
        return;
    }
    std::printf("PASS %s (max_abs=%.3e)\n", name.c_str(), static_cast<double>(max_abs));
}

GpuTensor on_device(const std::vector<int>& shape, const std::vector<half>& host) {
    GpuTensor tensor(shape);
    if (host.empty() == false) {
        tensor.load_from_host(host.data(), std::min(host.size(), tensor.numel()));
    }
    return tensor;
}

void test_rms_norm(DeviceContext& ctx) {
    constexpr int kRows = 4;
    constexpr int kHidden = 7;
    constexpr float kEps = 1e-6f;

    const auto x = make_values(kRows * kHidden, 2.0f);
    const auto w = make_values(kHidden, 1.0f);
    GpuTensor dx = on_device({kRows, kHidden}, x);
    GpuTensor dw = on_device({kHidden}, w);
    GpuTensor dout({kRows, kHidden});

    rms_norm(dout.data(), dx.data(), dw.data(), kRows, kHidden, kEps);
    ctx.synchronize();

    std::vector<half> got(kRows * kHidden);
    dout.copy_to_host(got.data(), got.size());

    std::vector<half> want(kRows * kHidden);
    for (int row = 0; row < kRows; ++row) {
        float sum_sq = 0.0f;
        for (int col = 0; col < kHidden; ++col) {
            const float value = to_float(x[row * kHidden + col]);
            sum_sq += value * value;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(kHidden) + kEps);
        for (int col = 0; col < kHidden; ++col) {
            const float value = to_float(x[row * kHidden + col]);
            want[row * kHidden + col] = to_half(value * inv_rms * to_float(w[col]));
        }
    }
    expect_close("rms_norm", got, want);
}

void test_rope(DeviceContext& ctx) {
    constexpr int kSeq = 3;
    constexpr int kQHeads = 2;
    constexpr int kKvHeads = 1;
    constexpr int kHeadDim = 4;
    constexpr int kOffset = 1;
    constexpr float kTheta = 10000.0f;

    const auto q = make_values(kSeq * kQHeads * kHeadDim, 2.0f);
    const auto k = make_values(kSeq * kKvHeads * kHeadDim, 2.0f);
    GpuTensor dq = on_device({kSeq, kQHeads, kHeadDim}, q);
    GpuTensor dk = on_device({kSeq, kKvHeads, kHeadDim}, k);

    apply_rope(dq.data(), dk.data(), kSeq, kQHeads, kKvHeads, kHeadDim, kOffset, kTheta);
    ctx.synchronize();

    std::vector<half> got_q(kSeq * kQHeads * kHeadDim);
    std::vector<half> got_k(kSeq * kKvHeads * kHeadDim);
    dq.copy_to_host(got_q.data(), got_q.size());
    dk.copy_to_host(got_k.data(), got_k.size());

    auto reference = [&](const std::vector<half>& src, int heads, std::vector<half>& dst) {
        const int half_dim = kHeadDim / 2;
        dst.assign(src.size(), to_half(0.0f));
        for (int s = 0; s < kSeq; ++s) {
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < half_dim; ++d) {
                    const int base = (s * heads + h) * kHeadDim;
                    const float freq = 1.0f / std::pow(
                        kTheta,
                        2.0f * static_cast<float>(d) / static_cast<float>(kHeadDim));
                    const float angle = static_cast<float>(s + kOffset) * freq;
                    const float c = std::cos(angle);
                    const float sn = std::sin(angle);
                    const float x0 = to_float(src[base + d]);
                    const float x1 = to_float(src[base + d + half_dim]);
                    dst[base + d] = to_half(x0 * c - x1 * sn);
                    dst[base + d + half_dim] = to_half(x0 * sn + x1 * c);
                }
            }
        }
    };

    std::vector<half> want_q;
    std::vector<half> want_k;
    reference(q, kQHeads, want_q);
    reference(k, kKvHeads, want_k);
    expect_close("apply_rope.q", got_q, want_q, 1e-2f, 1e-2f);
    expect_close("apply_rope.k", got_k, want_k, 1e-2f, 1e-2f);
}

void test_transpose_and_repeat_kv(DeviceContext& ctx) {
    constexpr int kSeq = 2;
    constexpr int kKvHeads = 2;
    constexpr int kRepeats = 3;
    constexpr int kHeadDim = 4;
    constexpr int kHeads = kKvHeads * kRepeats;

    const auto input = make_values(kSeq * kKvHeads * kHeadDim, 2.0f);
    GpuTensor din = on_device({kSeq, kKvHeads, kHeadDim}, input);
    GpuTensor dout({kHeads, kSeq, kHeadDim});

    transpose_and_repeat_kv(dout.data(), din.data(), kKvHeads, kRepeats, kSeq, kHeadDim);
    ctx.synchronize();

    std::vector<half> got(kHeads * kSeq * kHeadDim);
    dout.copy_to_host(got.data(), got.size());

    std::vector<half> want(kHeads * kSeq * kHeadDim);
    for (int h = 0; h < kHeads; ++h) {
        for (int s = 0; s < kSeq; ++s) {
            for (int d = 0; d < kHeadDim; ++d) {
                want[h * kSeq * kHeadDim + s * kHeadDim + d] =
                    input[s * kKvHeads * kHeadDim + (h / kRepeats) * kHeadDim + d];
            }
        }
    }
    expect_close("transpose_and_repeat_kv", got, want);
}

void test_swiglu_and_ewise(DeviceContext& ctx) {
    constexpr int kCount = 32;
    const auto gate = make_values(kCount, 3.0f);
    const auto up = make_values(kCount, 2.0f);
    GpuTensor dgate = on_device({kCount}, gate);
    GpuTensor dup = on_device({kCount}, up);
    GpuTensor dout({kCount});

    swiglu(dout.data(), dgate.data(), dup.data(), kCount);
    ctx.synchronize();

    std::vector<half> got(kCount);
    dout.copy_to_host(got.data(), got.size());
    std::vector<half> want(kCount);
    for (int i = 0; i < kCount; ++i) {
        const float g = to_float(gate[i]);
        const float u = to_float(up[i]);
        want[i] = to_half((g / (1.0f + std::exp(-g))) * u);
    }
    expect_close("swiglu", got, want, 2e-2f, 1e-2f);

    GpuTensor dsum({kCount});
    ewise_add(dsum.data(), dgate.data(), dup.data(), kCount);
    ctx.synchronize();
    std::vector<half> got_sum(kCount);
    dsum.copy_to_host(got_sum.data(), got_sum.size());
    std::vector<half> want_sum(kCount);
    for (int i = 0; i < kCount; ++i) {
        want_sum[i] = to_half(to_float(gate[i]) + to_float(up[i]));
    }
    expect_close("ewise_add", got_sum, want_sum);
}

void test_fused_causal_softmax(DeviceContext& ctx) {
    constexpr int kHeads = 2;
    constexpr int kSeq = 4;
    const auto scores = make_values(kHeads * kSeq * kSeq, 4.0f);
    GpuTensor dscores = on_device({kHeads, kSeq, kSeq}, scores);

    fused_causal_softmax(dscores.data(), kHeads, kSeq);
    ctx.synchronize();

    std::vector<half> got(kHeads * kSeq * kSeq);
    dscores.copy_to_host(got.data(), got.size());

    std::vector<half> want(kHeads * kSeq * kSeq, to_half(0.0f));
    for (int h = 0; h < kHeads; ++h) {
        for (int r = 0; r < kSeq; ++r) {
            const int row = (h * kSeq + r) * kSeq;
            float max_value = -1e30f;
            for (int c = 0; c <= r; ++c) {
                max_value = std::max(max_value, to_float(scores[row + c]));
            }
            float sum = 0.0f;
            for (int c = 0; c <= r; ++c) {
                sum += std::exp(to_float(scores[row + c]) - max_value);
            }
            for (int c = 0; c <= r; ++c) {
                want[row + c] = to_half(
                    std::exp(to_float(scores[row + c]) - max_value) / sum);
            }
        }
    }
    expect_close("fused_causal_softmax", got, want, 5e-3f, 5e-3f);
}

void test_transpose_012(DeviceContext& ctx) {
    constexpr int kA = 2;
    constexpr int kB = 3;
    constexpr int kD = 4;
    const auto input = make_values(kA * kB * kD, 2.0f);
    GpuTensor din = on_device({kA, kB, kD}, input);
    GpuTensor dout({kB, kA, kD});

    transpose_012_to_102(dout.data(), din.data(), kA, kB, kD);
    ctx.synchronize();

    std::vector<half> got(kB * kA * kD);
    dout.copy_to_host(got.data(), got.size());

    std::vector<half> want(kB * kA * kD);
    for (int a = 0; a < kA; ++a) {
        for (int b = 0; b < kB; ++b) {
            for (int d = 0; d < kD; ++d) {
                want[b * kA * kD + a * kD + d] = input[a * kB * kD + b * kD + d];
            }
        }
    }
    expect_close("transpose_012_to_102", got, want);
}

void test_gemm_linear(DeviceContext& ctx) {
    constexpr int kM = 2;
    constexpr int kN = 3;
    constexpr int kK = 4;
    const auto in = make_values(kM * kK, 1.5f);
    const auto weight = make_values(kN * kK, 1.5f);
    GpuTensor din = on_device({kM, kK}, in);
    GpuTensor dw = on_device({kN, kK}, weight);
    GpuTensor dout({kM, kN});

    gemm_linear(ctx, dout.data(), din.data(), dw.data(), kM, kN, kK);
    ctx.synchronize();

    std::vector<half> got(kM * kN);
    dout.copy_to_host(got.data(), got.size());

    std::vector<half> want(kM * kN);
    for (int m = 0; m < kM; ++m) {
        for (int n = 0; n < kN; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < kK; ++k) {
                acc += to_float(in[m * kK + k]) * to_float(weight[n * kK + k]);
            }
            want[m * kN + n] = to_half(acc);
        }
    }
    expect_close("gemm_linear", got, want, 2e-2f, 1e-2f);
}

void test_gemm_batched(DeviceContext& ctx) {
    constexpr int kBatch = 2;
    constexpr int kM = 3;
    constexpr int kN = 4;
    constexpr int kK = 5;
    const auto a = make_values(kBatch * kM * kK, 1.0f);
    const auto b = make_values(kBatch * kK * kN, 1.0f);
    GpuTensor da = on_device({kBatch, kM, kK}, a);
    GpuTensor db = on_device({kBatch, kK, kN}, b);
    GpuTensor dc({kBatch, kM, kN});

    gemm_batched(ctx, dc.data(), da.data(), db.data(), kBatch, kM, kN, kK, false, false, 1.0f);
    ctx.synchronize();

    std::vector<half> got(kBatch * kM * kN);
    dc.copy_to_host(got.data(), got.size());

    std::vector<half> want(kBatch * kM * kN);
    for (int batch = 0; batch < kBatch; ++batch) {
        for (int m = 0; m < kM; ++m) {
            for (int n = 0; n < kN; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < kK; ++k) {
                    const size_t a_index = (static_cast<size_t>(batch) * kM + m) * kK + k;
                    const size_t b_index = (static_cast<size_t>(batch) * kK + k) * kN + n;
                    acc += to_float(a[a_index]) * to_float(b[b_index]);
                }
                const size_t c_index = (static_cast<size_t>(batch) * kM + m) * kN + n;
                want[c_index] = to_half(acc);
            }
        }
    }
    expect_close("gemm_batched", got, want, 2e-2f, 1e-2f);
}

}  // namespace

int main() {
    try {
        DeviceContext ctx;

        std::printf("=== tiny_inference kernel correctness tests ===\n");
        test_rms_norm(ctx);
        test_rope(ctx);
        test_transpose_and_repeat_kv(ctx);
        test_swiglu_and_ewise(ctx);
        test_fused_causal_softmax(ctx);
        test_transpose_012(ctx);
        test_gemm_linear(ctx);
        test_gemm_batched(ctx);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FATAL: %s\n", error.what());
        return 2;
    }

    if (g_failures == 0) {
        std::printf("\nAll kernel correctness tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d kernel test(s) FAILED\n", g_failures);
    return 1;
}
