// Synthetic full-layer integration tests.
//
// These tests build tiny random/constant weights directly on the device. They
// do not touch a real checkpoint: the goal is to catch workspace layout,
// residual, and normalization wiring bugs that kernel-only tests cannot see.
#include "device_utils.h"
#include "tensor.h"
#include "transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) {
        std::printf("PASS %s\n", label.c_str());
    } else {
        std::fprintf(stderr, "FAIL %s\n", label.c_str());
        ++g_failures;
    }
}

std::vector<half> constant_values(size_t count, float value) {
    std::vector<half> values(count);
    for (auto& item : values) item = __float2half(value);
    return values;
}

std::vector<half> ramp_values(size_t count, float scale) {
    std::vector<half> values(count);
    for (size_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>(i % 11) - 5;
        values[i] = __float2half(static_cast<float>(centered) * scale);
    }
    return values;
}

void assign(GpuTensor& destination, const std::vector<int>& shape, const std::vector<half>& values) {
    GpuTensor tensor(shape);
    tensor.load_from_host(values.data(), values.size());
    destination = std::move(tensor);
}

void test_zero_projections_preserve_residual(DeviceContext& ctx) {
    TransformerConfig cfg;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 8;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 2;
    cfg.num_hidden_layers = 2;

    TransformerLayer layer(cfg, ctx);
    LayerWeights& weights = layer.weights();

    assign(weights.q_proj, {4, 4}, constant_values(16, 0.0f));
    assign(weights.k_proj, {2, 4}, constant_values(8, 0.0f));
    assign(weights.v_proj, {2, 4}, constant_values(8, 0.0f));
    assign(weights.o_proj, {4, 4}, constant_values(16, 0.0f));
    assign(weights.gate_proj, {8, 4}, constant_values(32, 0.0f));
    assign(weights.up_proj, {8, 4}, constant_values(32, 0.0f));
    assign(weights.down_proj, {4, 8}, constant_values(32, 0.0f));
    assign(weights.input_norm, {4}, constant_values(4, 1.0f));
    assign(weights.post_norm, {4}, constant_values(4, 1.0f));

    constexpr int kSeq = 3;
    constexpr int kHidden = 4;
    const auto input_values = ramp_values(kSeq * kHidden, 0.125f);
    GpuTensor input({kSeq, kHidden});
    input.load_from_host(input_values.data(), input_values.size());

    GpuTensor& output = layer.forward(input, kSeq);
    ctx.synchronize();

    std::vector<half> host_output(static_cast<size_t>(kSeq) * kHidden);
    output.copy_to_host(host_output.data(), host_output.size());

    bool same = true;
    float max_diff = 0.0f;
    for (size_t i = 0; i < host_output.size(); ++i) {
        const float got = __half2float(host_output[i]);
        const float want = __half2float(input_values[i]);
        max_diff = std::max(max_diff, std::fabs(got - want));
        if (std::fabs(got - want) > 1e-3f) same = false;
    }
    check(same, "zero projections preserve residual path (max diff " + std::to_string(max_diff) + ")");
}

}  // namespace

int main() {
    try {
        DeviceContext ctx;
        test_zero_projections_preserve_residual(ctx);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FATAL: %s\n", error.what());
        return 2;
    }

    if (g_failures == 0) {
        std::printf("\nAll layer forward tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d layer forward test(s) FAILED\n", g_failures);
    return 1;
}
