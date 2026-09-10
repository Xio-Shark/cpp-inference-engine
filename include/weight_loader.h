#pragma once

#include "transformer.h"

#include <string>
#include <vector>

/// Resolve the safetensors files that make up a model directory.
///
/// Priority:
///   1. model.safetensors.index.json (weight_map shards)
///   2. model.safetensors
///   3. legacy model-XXXXX-of-XXXXX.safetensors shards
///   4. a single *.safetensors file with another name
///
/// Throws std::runtime_error when the directory is missing, contains no
/// safetensors file, or an index references a missing shard.
std::vector<std::string> resolve_weight_files(const std::string& model_dir);

/// Validate dimensions and the requested layer index before touching the GPU.
void validate_config_for_layer(const TransformerConfig& config, int layer_idx);

/// Load one layer's nine weights from an already resolved file list.
/// Throws on missing tensors, duplicate tensor names, or shape mismatch.
void load_layer_weights(TransformerLayer& layer,
                        const std::vector<std::string>& weight_files,
                        int layer_idx);

/// Convenience wrapper: resolve first, then load.
std::vector<std::string> load_layer_weights(TransformerLayer& layer,
                                            const std::string& model_dir,
                                            int layer_idx);
