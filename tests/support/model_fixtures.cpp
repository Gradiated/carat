#include "model_fixtures.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace carat::testing {
namespace {

std::uint64_t dtype_bytes(std::string_view dtype) {
  if (dtype == "BF16")
    return 2;
  if (dtype == "F32")
    return 4;
  throw std::invalid_argument("fixture dtype not supported: " + std::string(dtype));
}

std::string layer_tensor(std::size_t layer, std::string_view suffix) {
  return "model.language_model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

void add_tiny_layer(std::vector<TensorSpec> &tensors, std::size_t layer, std::uint64_t query_width,
                    std::uint64_t kv_width, std::uint64_t head_dimension, bool value_projection) {
  constexpr std::uint64_t hidden = 8;
  constexpr std::uint64_t intermediate = 16;

  for (const std::string_view norm :
       {"input_layernorm.weight", "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"}) {
    tensors.push_back({layer_tensor(layer, norm), "BF16", {hidden}});
  }
  tensors.push_back({layer_tensor(layer, "layer_scalar"), "BF16", {1}});
  tensors.push_back({layer_tensor(layer, "mlp.gate_proj.weight"), "BF16", {intermediate, hidden}});
  tensors.push_back({layer_tensor(layer, "mlp.up_proj.weight"), "BF16", {intermediate, hidden}});
  tensors.push_back({layer_tensor(layer, "mlp.down_proj.weight"), "BF16", {hidden, intermediate}});
  tensors.push_back(
      {layer_tensor(layer, "self_attn.q_proj.weight"), "BF16", {query_width, hidden}});
  tensors.push_back({layer_tensor(layer, "self_attn.k_proj.weight"), "BF16", {kv_width, hidden}});
  if (value_projection) {
    tensors.push_back({layer_tensor(layer, "self_attn.v_proj.weight"), "BF16", {kv_width, hidden}});
  }
  tensors.push_back(
      {layer_tensor(layer, "self_attn.o_proj.weight"), "BF16", {hidden, query_width}});
  tensors.push_back({layer_tensor(layer, "self_attn.q_norm.weight"), "BF16", {head_dimension}});
  tensors.push_back({layer_tensor(layer, "self_attn.k_norm.weight"), "BF16", {head_dimension}});
}

} // namespace

TemporaryDirectory::TemporaryDirectory() {
  static std::atomic<unsigned> sequence{0};
  const auto clock = std::chrono::steady_clock::now().time_since_epoch().count();
  path_ = std::filesystem::temp_directory_path() /
          ("carat-test-" + std::to_string(getpid()) + "-" + std::to_string(clock) + "-" +
           std::to_string(sequence.fetch_add(1)));
  std::filesystem::create_directories(path_);
}

TemporaryDirectory::~TemporaryDirectory() {
  std::error_code ignored;
  std::filesystem::remove_all(path_, ignored);
}

const std::filesystem::path &TemporaryDirectory::path() const {
  return path_;
}

std::uint64_t tensor_bytes(const TensorSpec &tensor) {
  std::uint64_t bytes = dtype_bytes(tensor.dtype);
  for (const std::uint64_t dimension : tensor.shape)
    bytes *= dimension;
  return bytes;
}

void write_text(const std::filesystem::path &path, std::string_view text) {
  std::ofstream output(path, std::ios::binary);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!output)
    throw std::runtime_error("fixture write failed: " + path.string());
}

void write_safetensors_raw(const std::filesystem::path &path, std::uint64_t declared_header_size,
                           std::string_view header, std::uint64_t file_size) {
  {
    std::ofstream output(path, std::ios::binary);
    for (unsigned index = 0; index < 8; ++index) {
      output.put(static_cast<char>((declared_header_size >> (index * 8U)) & 0xffU));
    }
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!output)
      throw std::runtime_error("fixture write failed: " + path.string());
  }

  std::filesystem::resize_file(path, file_size);
}

std::string safetensors_header(std::span<const TensorSpec> tensors) {
  std::string header = R"({"__metadata__":{"format":"pt"})";
  std::uint64_t offset = 0;

  for (const TensorSpec &tensor : tensors) {
    std::string shape;
    for (const std::uint64_t dimension : tensor.shape) {
      if (!shape.empty())
        shape += ',';
      shape += std::to_string(dimension);
    }
    const std::uint64_t end = offset + tensor_bytes(tensor);
    header += ",\"" + tensor.name + "\":{\"dtype\":\"" + tensor.dtype + "\",\"shape\":[" + shape +
              "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(end) + "]}";
    offset = end;
  }

  header += '}';
  while (header.size() % 8 != 0)
    header.push_back(' ');
  return header;
}

void write_safetensors(const std::filesystem::path &path, std::span<const TensorSpec> tensors) {
  const std::string header = safetensors_header(tensors);

  std::uint64_t data_bytes = 0;
  for (const TensorSpec &tensor : tensors)
    data_bytes += tensor_bytes(tensor);

  write_safetensors_raw(path, header.size(), header, 8 + header.size() + data_bytes);
}

void write_sharded_safetensors(const std::filesystem::path &directory,
                               std::span<const std::vector<TensorSpec>> shards) {
  std::string weight_map;

  for (std::size_t shard = 0; shard < shards.size(); ++shard) {
    const std::string shard_name = "model-" + std::to_string(shard + 1) + ".safetensors";
    write_safetensors(directory / shard_name, shards[shard]);
    for (const TensorSpec &tensor : shards[shard]) {
      if (!weight_map.empty())
        weight_map += ',';
      weight_map += "\"" + tensor.name + "\":\"" + shard_name + "\"";
    }
  }

  write_text(directory / "model.safetensors.index.json",
             R"({"metadata":{},"weight_map":{)" + weight_map + "}}");
}

std::string replace_once(std::string text, std::string_view from, std::string_view to) {
  const std::size_t position = text.find(from);
  if (position == std::string::npos || text.find(from, position + 1) != std::string::npos) {
    throw std::invalid_argument("fixture text must contain exactly one: " + std::string(from));
  }

  text.replace(position, from.size(), to);
  return text;
}

std::string tiny_gemma4_config() {
  return R"({
    "model_type":"gemma4",
    "eos_token_id":[1,106],
    "text_config":{
      "model_type":"gemma4_text","tie_word_embeddings":true,"attention_bias":false,
      "hidden_size_per_layer_input":0,"num_kv_shared_layers":0,
      "hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"vocab_size":32,
      "sliding_window":4,"max_position_embeddings":128,"rms_norm_eps":1e-6,
      "final_logit_softcapping":30,"num_attention_heads":2,"num_key_value_heads":1,
      "head_dim":256,"num_global_key_value_heads":1,"global_head_dim":512,
      "attention_k_eq_v":true,"layer_types":["sliding_attention","full_attention"],
      "rope_parameters":{
        "sliding_attention":{"rope_theta":10000,"rope_type":"default"},
        "full_attention":{"rope_theta":1000000,"rope_type":"proportional","partial_rotary_factor":0.25}
      }
    }
  })";
}

std::vector<TensorSpec> tiny_gemma4_tensors() {
  std::vector<TensorSpec> tensors{
      {"model.language_model.embed_tokens.weight", "BF16", {32, 8}},
      {"model.language_model.norm.weight", "BF16", {8}},
  };
  add_tiny_layer(tensors, 0, 512, 256, 256, true);
  add_tiny_layer(tensors, 1, 1024, 512, 512, false);
  return tensors;
}

void write_tiny_gemma4_model(const std::filesystem::path &directory) {
  write_text(directory / "config.json", tiny_gemma4_config());

  const std::vector<TensorSpec> tensors = tiny_gemma4_tensors();
  const auto middle = tensors.begin() + static_cast<std::ptrdiff_t>(tensors.size() / 2);
  const std::vector<std::vector<TensorSpec>> shards{{tensors.begin(), middle},
                                                    {middle, tensors.end()}};
  write_sharded_safetensors(directory, shards);
}

std::vector<TensorSpec> gemma4_assistant_tensors() {
  constexpr std::uint64_t hidden = 1024;
  constexpr std::uint64_t backbone_hidden = 5376;
  constexpr std::uint64_t intermediate = 8192;
  constexpr std::uint64_t vocabulary = 262144;

  std::vector<TensorSpec> tensors{
      {"model.embed_tokens.weight", "BF16", {vocabulary, hidden}},
      {"pre_projection.weight", "BF16", {hidden, 2 * backbone_hidden}},
      {"post_projection.weight", "BF16", {backbone_hidden, hidden}},
      {"model.norm.weight", "BF16", {hidden}},
  };
  for (std::size_t layer = 0; layer < 4; ++layer) {
    const std::uint64_t head_dimension = layer == 3 ? 512 : 256;
    const std::uint64_t query_width = 32 * head_dimension;
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    tensors.push_back({prefix + "input_layernorm.weight", "BF16", {hidden}});
    tensors.push_back({prefix + "self_attn.q_norm.weight", "BF16", {head_dimension}});
    tensors.push_back({prefix + "self_attn.q_proj.weight", "BF16", {query_width, hidden}});
    tensors.push_back({prefix + "self_attn.o_proj.weight", "BF16", {hidden, query_width}});
    tensors.push_back({prefix + "post_attention_layernorm.weight", "BF16", {hidden}});
    tensors.push_back({prefix + "pre_feedforward_layernorm.weight", "BF16", {hidden}});
    tensors.push_back({prefix + "mlp.gate_proj.weight", "BF16", {intermediate, hidden}});
    tensors.push_back({prefix + "mlp.up_proj.weight", "BF16", {intermediate, hidden}});
    tensors.push_back({prefix + "mlp.down_proj.weight", "BF16", {hidden, intermediate}});
    tensors.push_back({prefix + "post_feedforward_layernorm.weight", "BF16", {hidden}});
    tensors.push_back({prefix + "layer_scalar", "BF16", {1}});
  }
  return tensors;
}

} // namespace carat::testing
