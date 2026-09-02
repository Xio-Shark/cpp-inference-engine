// main.cpp — Minimal C++ LLM inference engine demo
// Loads one transformer layer from Qwen2.5-7B safetensors and runs forward pass.
#include "transformer.h"
#include "safetensors.h"
#include "device_utils.h"
#include <cstdio>
#include <string>
#include <chrono>
#include <vector>
#include <cstdlib>

struct Args {
    std::string model_dir;
    int layer_idx = 0;
    int seq_len   = 4;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [layer_idx] [seq_len]\n", argv[0]);
        fprintf(stderr, "  model_dir: path to Qwen2.5-7B-Instruct (FP16 safetensors)\n");
        exit(1);
    }
    a.model_dir = argv[1];
    if (argc > 2) a.layer_idx = std::atoi(argv[2]);
    if (argc > 3) a.seq_len   = std::atoi(argv[3]);
    return a;
}

/// Load layer weights from sharded safetensors files.
static void load_layer(TransformerLayer& layer, const Args& args,
                        const TransformerConfig& /*cfg*/) {
    int idx = args.layer_idx;
    auto prefix = "model.layers." + std::to_string(idx) + ".";

    // Auto-detect shard count (try common totals: 1-8)
    std::vector<std::string> shards;
    for (int total = 1; total <= 8; ++total) {
        for (int i = 1; i <= total; ++i) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "model-%05d-of-%05d.safetensors", i, total);
            std::string path = args.model_dir + "/" + buf;
            FILE* f = fopen(path.c_str(), "r");
            if (f) { fclose(f); shards.push_back(path); }
        }
        if (!shards.empty()) break;  // found valid shard set
    }
    if (shards.empty()) {
        fprintf(stderr, "No safetensors files found in %s\n",
                args.model_dir.c_str());
        exit(1);
    }

    printf("Loading layer %d weights from %zu shard(s)...\n",
           idx, shards.size());

    struct WMap { const char* suffix; GpuTensor* dst; };
    auto& w = layer.weights();
    WMap mapping[] = {
        {"self_attn.q_proj.weight", &w.q_proj},
        {"self_attn.k_proj.weight", &w.k_proj},
        {"self_attn.v_proj.weight", &w.v_proj},
        {"self_attn.o_proj.weight", &w.o_proj},
        {"mlp.gate_proj.weight",    &w.gate_proj},
        {"mlp.up_proj.weight",      &w.up_proj},
        {"mlp.down_proj.weight",    &w.down_proj},
        {"input_layernorm.weight",  &w.input_norm},
        {"post_attention_layernorm.weight", &w.post_norm},
    };

    int loaded = 0;
    for (const auto& shard : shards) {
        SafetensorsFile sf(shard);
        for (auto& m : mapping) {
            std::string name = prefix + m.suffix;
            if (sf.has(name)) {
                *m.dst = sf.load_tensor(name);
                printf("  ✓ %s\n", name.c_str());
                ++loaded;
            }
        }
    }
    printf("Loaded %d/9 weight tensors.\n\n", loaded);
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    // Load config
    auto cfg = TransformerConfig::from_json(args.model_dir + "/config.json");
    printf("=== Tiny C++ Inference Engine ===\n");
#if defined(__APPLE__)
    printf("Backend: Apple Metal & MPS (Apple Silicon)\n");
#else
    printf("Backend: NVIDIA CUDA & cuBLAS\n");
#endif
    printf("Model: Qwen2.5-7B-Instruct (FP16)\n");
    printf("Config: hidden=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
           cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("Layer: %d, Seq length: %d\n\n", args.layer_idx, args.seq_len);

    // Setup device context
    DeviceContext ctx;
    TransformerLayer layer(cfg, ctx);
    load_layer(layer, args, cfg);

    // Create random input [seq_len, hidden_size]
    int S = args.seq_len, H = cfg.hidden_size;
    std::vector<half> h_input(S * H);
    srand(42);
    for (auto& v : h_input) v = __float2half((rand() % 200 - 100) / 1000.0f);

    GpuTensor input({S, H});
    input.load_from_host(h_input.data(), h_input.size());

    // Warmup
    printf("Running warmup...\n");
    auto out = layer.forward(input, S);
    ctx.synchronize();

    // Benchmark
    constexpr int ITERS = 10;
    ctx.synchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        out = layer.forward(input, S);
    }
    ctx.synchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\n=== Results ===\n");
    printf("Avg forward (1 layer): %.2f ms\n", ms / ITERS);
    printf("Output shape: [%d, %d]\n", S, H);

    // Print first few output values for verification
    std::vector<half> h_out(S * H);
    out.copy_to_host(h_out.data(), h_out.size());
    printf("Output[0, :8] = ");
    for (int i = 0; i < 8 && i < H; ++i)
        printf("%.4f ", __half2float(h_out[i]));
    printf("...\n");

    printf("\nDone. Compare output with PyTorch to verify correctness.\n");
    return 0;
}
