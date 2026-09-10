#include "weight_loader.h"

#include "safetensors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

std::string shape_to_string(const std::vector<int>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

void validate_shape(const GpuTensor& tensor,
                    const std::vector<int>& expected,
                    const std::string& name) {
    if (tensor.shape() != expected) {
        throw std::runtime_error(
            "shape mismatch for " + name + ": expected " + shape_to_string(expected)
            + ", got " + shape_to_string(tensor.shape()));
    }
}

std::vector<std::string> resolve_from_index(const fs::path& model_dir) {
    const fs::path index_path = model_dir / "model.safetensors.index.json";
    std::ifstream input(index_path);
    if (!input) throw std::runtime_error("cannot open " + index_path.string());

    nlohmann::json doc;
    input >> doc;
    if (!doc.contains("weight_map") || !doc["weight_map"].is_object()) {
        throw std::runtime_error(
            "invalid index file " + index_path.string() + ": missing object \"weight_map\"");
    }

    std::set<std::string> shard_names;
    for (const auto& entry : doc["weight_map"].items()) {
        if (!entry.value().is_string()) {
            throw std::runtime_error(
                "invalid index file " + index_path.string() + ": non-string shard for "
                + entry.key());
        }
        shard_names.insert(entry.value().get<std::string>());
    }
    if (shard_names.empty()) {
        throw std::runtime_error("index file references no shards: " + index_path.string());
    }

    std::vector<std::string> files;
    files.reserve(shard_names.size());
    for (const auto& name : shard_names) {
        const fs::path shard = model_dir / name;
        if (!fs::is_regular_file(shard)) {
            throw std::runtime_error("index references missing shard: " + shard.string());
        }
        files.push_back(shard.string());
    }
    return files;
}

std::vector<std::string> resolve_fallback(const fs::path& model_dir) {
    std::vector<fs::path> all;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".safetensors") {
            all.push_back(entry.path());
        }
    }
    if (all.empty()) {
        throw std::runtime_error("no .safetensors file found in " + model_dir.string());
    }
    std::sort(all.begin(), all.end());

    if (all.size() == 1) {
        return {all.front().string()};
    }

    std::vector<std::string> legacy;
    for (const auto& path : all) {
        if (path.filename().string().find("-of-") != std::string::npos) {
            legacy.push_back(path.string());
        }
    }
    if (!legacy.empty()) {
        return legacy;
    }

    std::string listing;
    for (const auto& path : all) {
        if (!listing.empty()) listing += ", ";
        listing += path.filename().string();
    }
    throw std::runtime_error(
        "multiple .safetensors files found but no model.safetensors or "
        "model.safetensors.index.json to disambiguate: " + listing);
}

}  // namespace

std::vector<std::string> resolve_weight_files(const std::string& model_dir) {
    const fs::path dir(model_dir);
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("model directory does not exist: " + model_dir);
    }

    const fs::path index = dir / "model.safetensors.index.json";
    if (fs::is_regular_file(index)) {
        return resolve_from_index(dir);
    }

    const fs::path single = dir / "model.safetensors";
    if (fs::is_regular_file(single)) {
        return {single.string()};
    }

    return resolve_fallback(dir);
}

void validate_config_for_layer(const TransformerConfig& config, int layer_idx) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0
        || config.num_heads <= 0 || config.num_kv_heads <= 0
        || config.head_dim <= 0 || config.num_hidden_layers <= 0) {
        throw std::runtime_error("invalid transformer config: dimensions must be positive");
    }
    if (config.num_heads % config.num_kv_heads != 0) {
        throw std::runtime_error(
            "invalid transformer config: num_attention_heads must be divisible by "
            "num_key_value_heads");
    }
    if (config.num_heads * config.head_dim != config.hidden_size) {
        throw std::runtime_error(
            "invalid transformer config: num_attention_heads * head_dim != hidden_size");
    }
    if (layer_idx < 0 || layer_idx >= config.num_hidden_layers) {
        throw std::runtime_error(
            "layer index out of range: " + std::to_string(layer_idx)
            + " (model has " + std::to_string(config.num_hidden_layers) + " layers)");
    }
}

namespace {

struct TensorSpec {
    std::string suffix;
    GpuTensor* target;
    std::vector<int> shape;
};

std::vector<TensorSpec> layer_tensor_specs(const TransformerConfig& cfg, LayerWeights& weights) {
    const int hidden = cfg.hidden_size;
    const int intermediate = cfg.intermediate_size;
    const int q_dim = cfg.num_heads * cfg.head_dim;
    const int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    return {
        {"self_attn.q_proj.weight", &weights.q_proj, {q_dim, hidden}},
        {"self_attn.k_proj.weight", &weights.k_proj, {kv_dim, hidden}},
        {"self_attn.v_proj.weight", &weights.v_proj, {kv_dim, hidden}},
        {"self_attn.o_proj.weight", &weights.o_proj, {hidden, q_dim}},
        {"mlp.gate_proj.weight", &weights.gate_proj, {intermediate, hidden}},
        {"mlp.up_proj.weight", &weights.up_proj, {intermediate, hidden}},
        {"mlp.down_proj.weight", &weights.down_proj, {hidden, intermediate}},
        {"input_layernorm.weight", &weights.input_norm, {hidden}},
        {"post_attention_layernorm.weight", &weights.post_norm, {hidden}},
    };
}

}  // namespace

void load_layer_weights(TransformerLayer& layer,
                        const std::vector<std::string>& weight_files,
                        int layer_idx) {
    const TransformerConfig& config = layer.config();
    validate_config_for_layer(config, layer_idx);
    if (weight_files.empty()) {
        throw std::runtime_error("no weight files provided");
    }

    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    std::vector<TensorSpec> specs = layer_tensor_specs(config, layer.weights());
    std::vector<bool> found(specs.size(), false);

    for (const auto& file : weight_files) {
        SafetensorsFile source(file);
        for (size_t i = 0; i < specs.size(); ++i) {
            const std::string name = prefix + specs[i].suffix;
            if (!source.has(name)) continue;
            if (found[i]) {
                throw std::runtime_error("duplicate layer tensor: " + name);
            }
            GpuTensor tensor = source.load_tensor(name);
            validate_shape(tensor, specs[i].shape, name);
            *specs[i].target = std::move(tensor);
            found[i] = true;
        }
    }

    std::string missing;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (found[i]) continue;
        if (!missing.empty()) missing += ", ";
        missing += prefix + specs[i].suffix;
    }
    if (!missing.empty()) {
        throw std::runtime_error("missing layer weights: " + missing);
    }
}

std::vector<std::string> load_layer_weights(TransformerLayer& layer,
                                            const std::string& model_dir,
                                            int layer_idx) {
    std::vector<std::string> files = resolve_weight_files(model_dir);
    load_layer_weights(layer, files, layer_idx);
    return files;
}
