#include "carat/gemma4_config.h"

#include "carat/json.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace carat {
namespace {

std::uint32_t u32(const Json &value, const char *field) {
  const std::uint64_t parsed = value.as_u64();
  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string(field) + " exceeds uint32");
  }

  return static_cast<std::uint32_t>(parsed);
}

int token_id(const Json &value, const char *field) {
  const std::uint32_t parsed = u32(value, field);
  if (parsed > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(field) + " exceeds int");
  }

  return static_cast<int>(parsed);
}

double number_at(const Json &object, std::string_view field) {
  return object.at(field).as_double();
}

std::uint32_t exact_rotary_dimension(std::uint32_t head_dimension, const Json &rope) {
  const Json *factor = rope.find("partial_rotary_factor");
  const double fraction = factor == nullptr ? 1.0 : factor->as_double();
  const double dimension = static_cast<double>(head_dimension) * fraction;
  if (dimension < 2.0 || dimension > head_dimension || std::floor(dimension) != dimension ||
      static_cast<std::uint64_t>(dimension) % 2U != 0U) {
    throw std::runtime_error("invalid rotary dimension");
  }
  return static_cast<std::uint32_t>(dimension);
}

} // namespace

std::uint64_t Gemma4LayerConfig::query_width() const {
  return static_cast<std::uint64_t>(query_heads) * head_dimension;
}

std::uint64_t Gemma4LayerConfig::kv_width() const {
  return static_cast<std::uint64_t>(kv_heads) * head_dimension;
}

Gemma4Config Gemma4Config::load(const std::string &config_path) {
  const Json root = Json::parse(read_text_file(config_path));
  if (root.at("model_type").as_string() != "gemma4") {
    throw std::runtime_error("expected a Gemma 4 model");
  }

  const Json &text = root.at("text_config");
  if (text.at("model_type").as_string() != "gemma4_text") {
    throw std::runtime_error("expected Gemma 4 text configuration");
  }

  if (!text.at("tie_word_embeddings").as_bool()) {
    throw std::runtime_error("untied Gemma 4 embeddings are not supported");
  }

  if (text.at("attention_bias").as_bool()) {
    throw std::runtime_error("attention bias is not supported");
  }

  if (u32(text.at("hidden_size_per_layer_input"), "hidden_size_per_layer_input") != 0U) {
    throw std::runtime_error("per-layer input embeddings are not supported");
  }

  if (u32(text.at("num_kv_shared_layers"), "num_kv_shared_layers") != 0U) {
    throw std::runtime_error("shared assistant KV layers are not valid for the target engine");
  }

  Gemma4Config result{
      .hidden_size = u32(text.at("hidden_size"), "hidden_size"),
      .intermediate_size = u32(text.at("intermediate_size"), "intermediate_size"),
      .vocabulary_size = u32(text.at("vocab_size"), "vocab_size"),
      .sliding_window = u32(text.at("sliding_window"), "sliding_window"),
      .maximum_positions = u32(text.at("max_position_embeddings"), "max_position_embeddings"),
      .rms_norm_epsilon = number_at(text, "rms_norm_eps"),
      .final_logit_softcap = number_at(text, "final_logit_softcapping"),
      .eos_token_ids = {},
      .layers = {},
  };
  const Json &eos = root.at("eos_token_id");
  for (const Json &token : eos.as_array()) {
    result.eos_token_ids.push_back(token_id(token, "eos_token_id"));
  }
  if (result.eos_token_ids.empty()) {
    throw std::runtime_error("eos_token_id must not be empty");
  }

  const std::uint32_t query_heads = u32(text.at("num_attention_heads"), "num_attention_heads");
  const std::uint32_t sliding_kv_heads = u32(text.at("num_key_value_heads"), "num_key_value_heads");
  const std::uint32_t sliding_head_dimension = u32(text.at("head_dim"), "head_dim");
  const std::uint32_t global_kv_heads =
      u32(text.at("num_global_key_value_heads"), "num_global_key_value_heads");
  const std::uint32_t global_head_dimension = u32(text.at("global_head_dim"), "global_head_dim");
  const bool key_equals_value = text.at("attention_k_eq_v").as_bool();
  const Json &ropes = text.at("rope_parameters");
  const Json &sliding_rope = ropes.at("sliding_attention");
  const Json &global_rope = ropes.at("full_attention");
  const auto &layer_types = text.at("layer_types").as_array();
  if (layer_types.size() != u32(text.at("num_hidden_layers"), "num_hidden_layers")) {
    throw std::runtime_error("layer_types length does not match num_hidden_layers");
  }

  result.layers.reserve(layer_types.size());
  for (const Json &layer_type : layer_types) {
    const bool sliding = layer_type.as_string() == "sliding_attention";
    if (!sliding && layer_type.as_string() != "full_attention") {
      throw std::runtime_error("unsupported Gemma 4 attention type: " + layer_type.as_string());
    }

    const Json &rope = sliding ? sliding_rope : global_rope;
    const std::uint32_t head_dimension = sliding ? sliding_head_dimension : global_head_dimension;
    result.layers.push_back({
        .attention = sliding ? AttentionKind::sliding : AttentionKind::global,
        .query_heads = query_heads,
        .kv_heads = sliding ? sliding_kv_heads : global_kv_heads,
        .head_dimension = head_dimension,
        .rotary_dimension = exact_rotary_dimension(head_dimension, rope),
        .rope_theta = number_at(rope, "rope_theta"),
        .key_equals_value = !sliding && key_equals_value,
    });
  }
  return result;
}

std::uint64_t Gemma4Config::text_parameter_count() const {
  std::uint64_t parameters =
      static_cast<std::uint64_t>(vocabulary_size) * hidden_size + hidden_size;
  for (const auto &layer : layers) {
    parameters += 4ULL * hidden_size + 1ULL;
    parameters += 2ULL * static_cast<std::uint64_t>(intermediate_size) * hidden_size;
    parameters += static_cast<std::uint64_t>(hidden_size) * intermediate_size;
    parameters += layer.query_width() * hidden_size;
    parameters += layer.kv_width() * hidden_size;
    if (!layer.key_equals_value)
      parameters += layer.kv_width() * hidden_size;
    parameters += static_cast<std::uint64_t>(hidden_size) * layer.query_width();
    parameters += 2ULL * layer.head_dimension;
  }
  return parameters;
}

std::uint64_t Gemma4Config::full_kv_bytes_per_token() const {
  std::uint64_t bytes = 0;
  for (const auto &layer : layers) {
    bytes += 2ULL * layer.kv_width() * 2ULL;
  }
  return bytes;
}

std::uint64_t Gemma4Config::resident_sliding_kv_bytes_per_sequence() const {
  std::uint64_t bytes = 0;
  for (const auto &layer : layers) {
    if (layer.attention == AttentionKind::sliding) {
      bytes += 2ULL * layer.kv_width() * 2ULL * sliding_window;
    }
  }
  return bytes;
}

void require_native_gemma4_target(const Gemma4Config &config) {
  for (const Gemma4LayerConfig &layer : config.layers) {
    const bool sliding = layer.attention == AttentionKind::sliding;
    const std::string kind = sliding ? "sliding" : "global";
    const auto require = [&](bool supported, std::string_view field, std::string_view expected,
                             auto actual) {
      if (supported)
        return;

      std::ostringstream message;
      message << "native Gemma 4 kernels require " << kind << ' ' << field << ' ' << expected
              << ", found " << actual;
      throw std::runtime_error(message.str());
    };

    const std::uint32_t head_dimension = sliding ? 256 : 512;
    const double rope_theta = sliding ? 10000.0 : 1000000.0;
    const std::uint32_t rotary_dimension =
        sliding ? layer.head_dimension : layer.head_dimension / 4;

    require(layer.head_dimension == head_dimension, "head_dimension",
            std::to_string(head_dimension), layer.head_dimension);
    require(layer.rope_theta == rope_theta, "rope_theta",
            std::to_string(static_cast<int>(rope_theta)), layer.rope_theta);
    require(layer.rotary_dimension == rotary_dimension, "rotary_dimension",
            std::to_string(rotary_dimension), layer.rotary_dimension);
    require(layer.key_equals_value == !sliding, "key_equals_value", sliding ? "false" : "true",
            layer.key_equals_value ? "true" : "false");
  }
}

} // namespace carat
