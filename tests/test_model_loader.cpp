// Synthetic safetensors tests for the model weight loader.
//
// These tests create tiny fake models on disk with the same file layout and
// tensor schema as a Qwen2.5 checkpoint. No real model weights are downloaded
// or executed; only file resolution, dtype loading, and shape validation are
// exercised.
#include "transformer.h"
#include "weight_loader.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;
int g_temp_counter = 0;

void check(bool condition, const std::string& label) {
    if (condition) {
        std::printf("PASS %s\n", label.c_str());
    } else {
        std::fprintf(stderr, "FAIL %s\n", label.c_str());
        ++g_failures;
    }
}

template <typename Fn>
void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
        check(false, label + " (expected an exception)");
    } catch (const std::exception&) {
        check(true, label);
    }
}

struct TempDir {
    fs::path path;

    explicit TempDir(const std::string& label) {
        path = fs::temp_directory_path()
            / ("tiny_inference_loader_" + label + "_" + std::to_string(++g_temp_counter));
        fs::create_directories(path);
    }

    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }

    std::string str() const { return path.string(); }
};

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write " + path.string());
    output << text;
}

struct SyntheticTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<half> data;
};

size_t numel(const std::vector<int>& shape) {
    size_t count = 1;
    for (int dim : shape) count *= static_cast<size_t>(dim);
    return count;
}

SyntheticTensor make_tensor(const std::string& name,
                            const std::vector<int>& shape,
                            int seed) {
    SyntheticTensor tensor;
    tensor.name = name;
    tensor.shape = shape;
    tensor.data.resize(numel(shape));
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        const int value = static_cast<int>((i + static_cast<size_t>(seed)) % 17) - 8;
        tensor.data[i] = __float2half(static_cast<float>(value) * 0.01f);
    }
    return tensor;
}

void write_safetensors(const fs::path& path, const std::vector<SyntheticTensor>& tensors) {
    nlohmann::json header = nlohmann::json::object();
    size_t offset = 0;
    for (const auto& tensor : tensors) {
        const size_t bytes = tensor.data.size() * sizeof(half);
        header[tensor.name] = {
            {"dtype", "F16"},
            {"shape", tensor.shape},
            {"data_offsets", {offset, offset + bytes}},
        };
        offset += bytes;
    }

    const std::string header_text = header.dump();
    const uint64_t header_length = static_cast<uint64_t>(header_text.size());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot write " + path.string());
    output.write(reinterpret_cast<const char*>(&header_length), sizeof(header_length));
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const auto& tensor : tensors) {
        output.write(
            reinterpret_cast<const char*>(tensor.data.data()),
            static_cast<std::streamsize>(tensor.data.size() * sizeof(half)));
    }
}

TransformerConfig tiny_config() {
    TransformerConfig cfg;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 8;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 2;
    cfg.vocab_size = 32;
    cfg.num_hidden_layers = 2;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;
    return cfg;
}

void write_config(const fs::path& path, const TransformerConfig& cfg) {
    nlohmann::json json = {
        {"hidden_size", cfg.hidden_size},
        {"intermediate_size", cfg.intermediate_size},
        {"num_attention_heads", cfg.num_heads},
        {"num_key_value_heads", cfg.num_kv_heads},
        {"head_dim", cfg.head_dim},
        {"vocab_size", cfg.vocab_size},
        {"num_hidden_layers", cfg.num_hidden_layers},
        {"rms_norm_eps", cfg.rms_norm_eps},
        {"rope_theta", cfg.rope_theta},
    };
    write_text(path, json.dump(2));
}

std::vector<SyntheticTensor> make_layer_tensors(const TransformerConfig& cfg, int layer_idx) {
    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const int hidden = cfg.hidden_size;
    const int intermediate = cfg.intermediate_size;
    const int q_dim = cfg.num_heads * cfg.head_dim;
    const int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    return {
        make_tensor(prefix + "self_attn.q_proj.weight", {q_dim, hidden}, 1),
        make_tensor(prefix + "self_attn.k_proj.weight", {kv_dim, hidden}, 2),
        make_tensor(prefix + "self_attn.v_proj.weight", {kv_dim, hidden}, 3),
        make_tensor(prefix + "self_attn.o_proj.weight", {hidden, q_dim}, 4),
        make_tensor(prefix + "mlp.gate_proj.weight", {intermediate, hidden}, 5),
        make_tensor(prefix + "mlp.up_proj.weight", {intermediate, hidden}, 6),
        make_tensor(prefix + "mlp.down_proj.weight", {hidden, intermediate}, 7),
        make_tensor(prefix + "input_layernorm.weight", {hidden}, 8),
        make_tensor(prefix + "post_attention_layernorm.weight", {hidden}, 9),
    };
}

void test_resolve_single_file() {
    TempDir dir("resolve_single");
    write_text(dir.path / "model.safetensors", "placeholder");
    const auto files = resolve_weight_files(dir.str());
    check(files.size() == 1, "resolve single file count");
    check(fs::path(files[0]).filename() == "model.safetensors", "resolve single file name");
}

void test_resolve_index_files() {
    TempDir dir("resolve_index");
    const std::string index = R"({
        "weight_map": {
            "model.layers.0.a.weight": "model-00002-of-00002.safetensors",
            "model.layers.0.b.weight": "model-00001-of-00002.safetensors"
        }
    })";
    write_text(dir.path / "model.safetensors.index.json", index);
    write_text(dir.path / "model-00001-of-00002.safetensors", "placeholder");
    write_text(dir.path / "model-00002-of-00002.safetensors", "placeholder");

    const auto files = resolve_weight_files(dir.str());
    check(files.size() == 2, "resolve index file count");
    check(fs::path(files[0]).filename() == "model-00001-of-00002.safetensors", "resolve index sorted first");
    check(fs::path(files[1]).filename() == "model-00002-of-00002.safetensors", "resolve index sorted second");
}

void test_resolve_legacy_shards() {
    TempDir dir("resolve_legacy");
    write_text(dir.path / "model-00002-of-00002.safetensors", "placeholder");
    write_text(dir.path / "model-00001-of-00002.safetensors", "placeholder");

    const auto files = resolve_weight_files(dir.str());
    check(files.size() == 2, "resolve legacy shard count");
    check(fs::path(files[0]).filename() == "model-00001-of-00002.safetensors", "resolve legacy first");
    check(fs::path(files[1]).filename() == "model-00002-of-00002.safetensors", "resolve legacy second");
}

void test_resolve_missing() {
    TempDir dir("resolve_missing");
    expect_throws([&] { resolve_weight_files(dir.str()); }, "resolve missing directory throws");
}

void test_resolve_ambiguous_files() {
    TempDir dir("resolve_ambiguous");
    write_text(dir.path / "a.safetensors", "placeholder");
    write_text(dir.path / "b.safetensors", "placeholder");
    expect_throws([&] { resolve_weight_files(dir.str()); }, "resolve ambiguous files throws");
}

void check_layer_shapes(TransformerLayer& layer,
                        const TransformerConfig& cfg,
                        const std::string& label) {
    const int hidden = cfg.hidden_size;
    const int intermediate = cfg.intermediate_size;
    const int q_dim = cfg.num_heads * cfg.head_dim;
    const int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    const auto& weights = layer.weights();
    check(weights.q_proj.shape() == std::vector<int>({q_dim, hidden}), label + " q_proj shape");
    check(weights.k_proj.shape() == std::vector<int>({kv_dim, hidden}), label + " k_proj shape");
    check(weights.v_proj.shape() == std::vector<int>({kv_dim, hidden}), label + " v_proj shape");
    check(weights.o_proj.shape() == std::vector<int>({hidden, q_dim}), label + " o_proj shape");
    check(weights.gate_proj.shape() == std::vector<int>({intermediate, hidden}), label + " gate_proj shape");
    check(weights.up_proj.shape() == std::vector<int>({intermediate, hidden}), label + " up_proj shape");
    check(weights.down_proj.shape() == std::vector<int>({hidden, intermediate}), label + " down_proj shape");
    check(weights.input_norm.shape() == std::vector<int>({hidden}), label + " input_norm shape");
    check(weights.post_norm.shape() == std::vector<int>({hidden}), label + " post_norm shape");
}

void test_load_single_file(DeviceContext& ctx) {
    TempDir dir("load_single");
    const TransformerConfig cfg = tiny_config();
    write_config(dir.path / "config.json", cfg);
    write_safetensors(dir.path / "model.safetensors", make_layer_tensors(cfg, 0));

    TransformerLayer layer(cfg, ctx);
    const auto files = load_layer_weights(layer, dir.str(), 0);
    check(files.size() == 1, "load single file count");
    check_layer_shapes(layer, cfg, "load single file");
}

void test_load_index_shards(DeviceContext& ctx) {
    TempDir dir("load_index");
    const TransformerConfig cfg = tiny_config();
    write_config(dir.path / "config.json", cfg);

    auto tensors = make_layer_tensors(cfg, 1);
    std::vector<SyntheticTensor> first(tensors.begin(), tensors.begin() + 5);
    std::vector<SyntheticTensor> second(tensors.begin() + 5, tensors.end());
    const std::string shard_a = "model-00001-of-00002.safetensors";
    const std::string shard_b = "model-00002-of-00002.safetensors";
    write_safetensors(dir.path / shard_a, first);
    write_safetensors(dir.path / shard_b, second);

    nlohmann::json weight_map = nlohmann::json::object();
    for (const auto& tensor : first) weight_map[tensor.name] = shard_a;
    for (const auto& tensor : second) weight_map[tensor.name] = shard_b;
    write_text(
        dir.path / "model.safetensors.index.json",
        nlohmann::json({{"weight_map", weight_map}}).dump(2));

    TransformerLayer layer(cfg, ctx);
    const auto files = load_layer_weights(layer, dir.str(), 1);
    check(files.size() == 2, "load index file count");
    check_layer_shapes(layer, cfg, "load index shards");
}

void test_missing_tensor_fails(DeviceContext& ctx) {
    TempDir dir("load_missing_tensor");
    const TransformerConfig cfg = tiny_config();
    auto tensors = make_layer_tensors(cfg, 0);
    tensors.pop_back();
    write_safetensors(dir.path / "model.safetensors", tensors);

    TransformerLayer layer(cfg, ctx);
    expect_throws([&] { load_layer_weights(layer, dir.str(), 0); }, "missing tensor throws");
}

void test_wrong_shape_fails(DeviceContext& ctx) {
    TempDir dir("load_wrong_shape");
    const TransformerConfig cfg = tiny_config();
    auto tensors = make_layer_tensors(cfg, 0);
    const std::string q_name = "model.layers.0.self_attn.q_proj.weight";
    tensors[0] = make_tensor(
        q_name,
        {cfg.num_heads * cfg.head_dim + 1, cfg.hidden_size},
        1);
    write_safetensors(dir.path / "model.safetensors", tensors);

    TransformerLayer layer(cfg, ctx);
    expect_throws([&] { load_layer_weights(layer, dir.str(), 0); }, "wrong shape throws");
}

void test_validate_config() {
    TransformerConfig cfg = tiny_config();
    check(true, "valid config accepted");
    validate_config_for_layer(cfg, 0);
    check(true, "valid layer index accepted");

    TransformerConfig bad_ratio = tiny_config();
    bad_ratio.num_heads = 3;
    expect_throws([&] { validate_config_for_layer(bad_ratio, 0); }, "indivisible head ratio throws");

    TransformerConfig bad_hidden = tiny_config();
    bad_hidden.head_dim = 3;
    expect_throws([&] { validate_config_for_layer(bad_hidden, 0); }, "head_dim mismatch throws");

    expect_throws([&] { validate_config_for_layer(cfg, 2); }, "layer index out of range throws");
    expect_throws([&] { validate_config_for_layer(cfg, -1); }, "negative layer index throws");
}

}  // namespace

int main() {
    try {
        DeviceContext ctx;

        test_resolve_single_file();
        test_resolve_index_files();
        test_resolve_legacy_shards();
        test_resolve_missing();
        test_resolve_ambiguous_files();

        test_load_single_file(ctx);
        test_load_index_shards(ctx);
        test_missing_tensor_fails(ctx);
        test_wrong_shape_fails(ctx);
        test_validate_config();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FATAL: %s\n", error.what());
        return 2;
    }

    if (g_failures == 0) {
        std::printf("\nAll model loader tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d model loader test(s) FAILED\n", g_failures);
    return 1;
}
