// main.cpp — Minimal C++ LLM inference engine demo
// Loads one transformer layer from Qwen2.5-7B safetensors and runs forward pass.
#include "transformer.h"
#include "weight_loader.h"
#include "device_utils.h"
#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <exception>
#include <stdexcept>

struct Args {
    std::string model_dir;
    int layer_idx = 0;
    int seq_len   = 4;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [layer_idx] [seq_len]\n", argv[0]);
        fprintf(stderr, "  model_dir: Qwen2.5-style model directory with config.json and safetensors\n");
        exit(1);
    }
    a.model_dir = argv[1];
    if (argc > 2) a.layer_idx = std::atoi(argv[2]);
    if (argc > 3) a.seq_len   = std::atoi(argv[3]);
    return a;
}

namespace {

int run(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    auto cfg = TransformerConfig::from_json(args.model_dir + "/config.json");
    validate_config_for_layer(cfg, args.layer_idx);

    printf("=== Tiny C++ Inference Engine ===\n");
#if defined(__APPLE__)
    printf("Backend: Apple Metal & MPS (Apple Silicon)\n");
#else
    printf("Backend: NVIDIA CUDA & cuBLAS\n");
#endif
    printf("Model dir: %s\n", args.model_dir.c_str());
    printf("Config: hidden=%d, layers=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
           cfg.hidden_size, cfg.num_hidden_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("Layer: %d, Seq length: %d\n\n", args.layer_idx, args.seq_len);

    DeviceContext ctx;
    TransformerLayer layer(cfg, ctx);

    const std::vector<std::string> weight_files =
        load_layer_weights(layer, args.model_dir, args.layer_idx);
    printf("Loaded 9/9 layer tensors from %zu file(s):\n", weight_files.size());
    for (const auto& file : weight_files) {
        printf("  - %s\n", file.c_str());
    }
    printf("\n");

    // Create deterministic input [seq_len, hidden_size].
    const int S = args.seq_len;
    const int H = cfg.hidden_size;
    if (S <= 0 || H <= 0) {
        throw std::runtime_error("seq_len and hidden_size must be positive");
    }
    std::vector<half> h_input(static_cast<size_t>(S) * H);
    srand(42);
    for (auto& value : h_input) {
        value = __float2half((rand() % 200 - 100) / 1000.0f);
    }

    GpuTensor input({S, H});
    input.load_from_host(h_input.data(), h_input.size());

    printf("Running warmup...\n");
    GpuTensor& out = layer.forward(input, S);
    ctx.synchronize();

    constexpr int kIters = 10;
    ctx.synchronize();
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
        layer.forward(input, S);
    }
    ctx.synchronize();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\n=== Results ===\n");
    printf("Avg forward (1 layer): %.2f ms\n", ms / kIters);
    printf("Output shape: [%d, %d]\n", S, H);

    std::vector<half> h_out(static_cast<size_t>(S) * H);
    out.copy_to_host(h_out.data(), h_out.size());
    printf("Output[0, :8] = ");
    for (int i = 0; i < 8 && i < H; ++i) {
        printf("%.4f ", __half2float(h_out[i]));
    }
    printf("...\n");
    printf("\nDone. Kernel-level correctness tests live in tests/test_kernels.cpp.\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
