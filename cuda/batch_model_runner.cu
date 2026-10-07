#include "carat/batch_model_runner.h"

#include "carat/assistant_runner.h"
#include "carat/attention.h"
#include "carat/cuda_ops.h"
#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/layer_runner.h"
#include "carat/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace carat {
namespace {

constexpr std::uint64_t alignment = 256;
constexpr int speculative_depth = 8;
std::uint64_t align_up(std::uint64_t value) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

bool fp8_global_k_enabled() {
  const char *value = std::getenv("CARAT_FP8_GLOBAL_K");
  return value != nullptr && std::string_view(value) == "1";
}

bool fp8_global_kv_enabled() {
  const char *value = std::getenv("CARAT_FP8_GLOBAL_KV");
  return value != nullptr && std::string_view(value) == "1";
}

bool int8_global_kv_oracle_enabled() {
  const char *value = std::getenv("CARAT_INT8_GLOBAL_KV_ORACLE");
  return value != nullptr && std::string_view(value) == "1";
}

bool int8_global_k_enabled() {
  const char *value = std::getenv("CARAT_INT8_GLOBAL_K");
  return value != nullptr && std::string_view(value) == "1";
}

bool int8_global_kv_enabled() {
  const char *value = std::getenv("CARAT_INT8_GLOBAL_KV");
  return value != nullptr && std::string_view(value) == "1";
}

enum class Int8OracleMode { keys, values, both };

Int8OracleMode int8_global_kv_oracle_mode() {
  const char *value = std::getenv("CARAT_INT8_GLOBAL_KV_ORACLE_MODE");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "both") {
    return Int8OracleMode::both;
  }
  if (std::string_view(value) == "keys")
    return Int8OracleMode::keys;
  if (std::string_view(value) == "values")
    return Int8OracleMode::values;
  throw std::runtime_error("CARAT_INT8_GLOBAL_KV_ORACLE_MODE must be keys, values, or both");
}

int int8_global_kv_oracle_block_width() {
  const char *value = std::getenv("CARAT_INT8_GLOBAL_KV_ORACLE_BLOCK");
  if (value == nullptr || *value == '\0')
    return 512;
  const int parsed = std::stoi(value);
  if (parsed != 64 && parsed != 128 && parsed != 256 && parsed != 512) {
    throw std::runtime_error("CARAT_INT8_GLOBAL_KV_ORACLE_BLOCK must be 64, 128, 256, or 512");
  }
  return parsed;
}

bool bf16_lm_head_enabled() {
  const char *value = std::getenv("CARAT_FP8_BF16_LM_HEAD");
  return value != nullptr && std::string_view(value) == "1";
}

bool int4_lm_head_enabled() {
  const char *value = std::getenv("CARAT_INT4_LM_HEAD");
  return value != nullptr && std::string_view(value) == "1";
}

bool int4_bf16_lm_head_enabled() {
  const char *value = std::getenv("CARAT_INT4_BF16_LM_HEAD");
  return value != nullptr && std::string_view(value) == "1";
}

bool int4_bf16_lm_head_validation_enabled() {
  const char *value = std::getenv("CARAT_INT4_BF16_LM_HEAD_VALIDATE");
  return value != nullptr && std::string_view(value) == "1";
}

bool int4_lm_head_validation_enabled() {
  const char *value = std::getenv("CARAT_INT4_LM_HEAD_VALIDATE");
  return value != nullptr && std::string_view(value) == "1";
}

int int4_lm_head_segment_width() {
  const char *value = std::getenv("CARAT_INT4_LM_HEAD_SEGMENT");
  if (value == nullptr || *value == '\0')
    return 262144;
  try {
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size() || parsed < 256 || parsed % 256 != 0) {
      throw std::runtime_error("CARAT_INT4_LM_HEAD_SEGMENT must be a positive multiple of 256");
    }
    return parsed;
  } catch (const std::invalid_argument &) {
    throw std::runtime_error("CARAT_INT4_LM_HEAD_SEGMENT must be a positive multiple of 256");
  } catch (const std::out_of_range &) {
    throw std::runtime_error("CARAT_INT4_LM_HEAD_SEGMENT is out of range");
  }
}

bool int4_lm_head_oracle_enabled() {
  const char *value = std::getenv("CARAT_INT4_LM_HEAD_ORACLE");
  return value != nullptr && std::string_view(value) == "1";
}

int fp8_suffix_max_tokens() {
  const char *value = std::getenv("CARAT_FP8_SUFFIX_MAX_TOKENS");
  if (value == nullptr || *value == '\0')
    return 0;
  try {
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size() || parsed < 0) {
      throw std::runtime_error("CARAT_FP8_SUFFIX_MAX_TOKENS must be a non-negative integer");
    }
    return parsed;
  } catch (const std::invalid_argument &) {
    throw std::runtime_error("CARAT_FP8_SUFFIX_MAX_TOKENS must be a non-negative integer");
  } catch (const std::out_of_range &) {
    throw std::runtime_error("CARAT_FP8_SUFFIX_MAX_TOKENS is out of range");
  }
}

std::vector<bool> global_layer_selection(const Gemma4Config &config, bool enabled,
                                         const char *environment_name, const char *description) {
  std::vector<bool> selected(config.layers.size(), false);
  if (!enabled)
    return selected;
  const char *requested = std::getenv(environment_name);
  if (requested == nullptr || *requested == '\0') {
    for (std::size_t layer = 0; layer < config.layers.size(); ++layer) {
      selected[layer] = config.layers[layer].attention == AttentionKind::global;
    }
    return selected;
  }
  const std::string list(requested);
  std::size_t begin = 0;
  while (begin < list.size()) {
    const std::size_t end = list.find(',', begin);
    const std::string item =
        list.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    const int layer = std::stoi(item);
    if (layer < 0 || layer >= static_cast<int>(config.layers.size()) ||
        config.layers[static_cast<std::size_t>(layer)].attention != AttentionKind::global) {
      throw std::runtime_error(std::string(description) +
                               " layer selection contains an invalid layer");
    }
    selected[static_cast<std::size_t>(layer)] = true;
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return selected;
}

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class Allocation {
public:
  explicit Allocation(std::uint64_t bytes) {
    check(cudaMalloc(&pointer_, bytes), "allocate batch workspace");
  }
  Allocation(const Allocation &) = delete;
  Allocation &operator=(const Allocation &) = delete;

  ~Allocation() {
    cudaFree(pointer_);
  }
  void *get() {
    return pointer_;
  }

private:
  void *pointer_{nullptr};
};

struct CacheLocation {
  std::uint64_t offset;
  int capacity;
};

} // namespace

struct Gemma4BatchModelRunner::Implementation {
  struct Int4LmHeadSegment {
    int vocabulary_offset;
    std::unique_ptr<Int4Fp8Linear> linear;
  };

  struct Int4Bf16LmHeadSegment {
    int vocabulary_offset;
    std::unique_ptr<Int4Bf16Linear> linear;
  };

  struct DecodeGraph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};

    ~DecodeGraph() {
      if (executable != nullptr)
        cudaGraphExecDestroy(executable);
      if (graph != nullptr)
        cudaGraphDestroy(graph);
    }
  };

  struct SuffixLayerState {
    std::vector<int> slots;
    std::vector<const std::vector<int> *> prompts;
    std::vector<int> counts;
    std::vector<int> positions;
    std::vector<int> last_rows;
    std::vector<std::size_t> prompt_sizes;
    int packed_tokens{0};
    int next_layer{0};
    bool fp8_projections{false};
  };

  const Gemma4Config &config;
  const DeviceWeightArena &weights;
  const Fp8WeightArena *fp8_weights;
  const DeviceWeightArena *assistant_weights;
  const Fp8WeightArena *assistant_fp8_weights;
  int batch_size;
  int maximum_context;
  int current_position{0};
  int hidden;
  int vocabulary;
  int shared_sliding_layer{-1};
  int shared_global_layer{-1};
  std::vector<int> slot_positions;
  std::vector<CacheLocation> caches;
  std::uint64_t cache_payload_bytes;
  int fp8_suffix_token_limit;
  bool use_fp8_lm_head;
  bool use_int4_lm_head;
  bool use_int4_bf16_lm_head;
  bool int4_lm_head_validation_logged{false};
  bool int4_bf16_lm_head_validation_logged{false};
  bool use_fp8_global_kv;
  bool use_fp8_global_k;
  std::vector<bool> fp8_global_k_layers;
  bool use_int8_global_kv;
  bool use_int8_global_k;
  std::vector<bool> int8_global_k_layers;
  bool use_int8_global_kv_oracle;
  Int8OracleMode int8_oracle_mode;
  int int8_oracle_block_width;
  std::vector<bool> int8_global_kv_oracle_layers;
  std::vector<std::uint64_t> fp8_key_offsets;
  std::uint64_t fp8_key_payload_bytes;
  std::vector<std::uint64_t> int8_key_offsets;
  std::uint64_t int8_key_payload_bytes;
  std::vector<std::uint64_t> int8_key_scale_offsets;
  std::uint64_t int8_key_scale_payload_bytes;
  std::vector<std::uint64_t> int8_value_scale_offsets;
  std::uint64_t int8_value_scale_payload_bytes;
  std::vector<std::uint64_t> speculative_backup_offsets;
  std::uint64_t speculative_backup_payload_bytes;
  Gemma4LayerRunner layers;
  Bf16Linear lm_head;
  Fp8Linear fp8_lm_head;
  std::vector<Int4LmHeadSegment> int4_lm_head_segments;
  std::vector<Int4Bf16LmHeadSegment> int4_bf16_lm_head_segments;
  Allocation cache_arena;
  Allocation fp8_key_arena;
  Allocation fp8_value_arena;
  Allocation int8_key_arena;
  Allocation int8_key_scale_arena;
  Allocation int8_value_arena;
  Allocation int8_value_scale_arena;
  Allocation hidden_a;
  Allocation hidden_b;
  Allocation final_hidden;
  Allocation logits;
  Allocation device_input_tokens;
  Allocation device_output_tokens;
  Allocation device_positions;
  Allocation device_slots;
  Allocation argmax_workspace;
  Allocation fp8_final_hidden;
  Allocation fp8_final_scales;
  Allocation target_hidden_slots;
  Allocation speculative_backup_arena;
  Allocation device_cache_pointers;
  // Suffix-only hidden states survive decode interleaving and do not share decode workspaces.
  Allocation suffix_hidden_a;
  Allocation suffix_hidden_b;
  cudaStream_t decode_stream{};
  std::unique_ptr<Gemma4AssistantRunner> assistant;
  std::map<std::tuple<int, int, int>, std::unique_ptr<DecodeGraph>> decode_graphs;
  std::optional<SuffixLayerState> suffix_layer_state;

  static std::uint64_t cache_payload(const Gemma4Config &config, int batch, int maximum_context,
                                     std::vector<CacheLocation> *locations) {
    std::uint64_t bytes = 0;
    for (const auto &layer : config.layers) {
      bytes = align_up(bytes);
      const int capacity = layer.attention == AttentionKind::sliding
                               ? std::min(maximum_context, static_cast<int>(config.sliding_window))
                               : maximum_context;
      locations->push_back({bytes, capacity});
      bytes += 2ULL * batch * capacity * layer.kv_width();
    }
    return align_up(bytes);
  }

  static std::uint64_t speculative_backup_payload(const Gemma4Config &config, int batch,
                                                  std::vector<std::uint64_t> *offsets) {
    std::uint64_t bytes = 0;
    for (const auto &layer : config.layers) {
      if (layer.attention != AttentionKind::sliding) {
        offsets->push_back(std::numeric_limits<std::uint64_t>::max());
        continue;
      }
      bytes = align_up(bytes);
      offsets->push_back(bytes);
      bytes += 2ULL * batch * speculative_depth * layer.kv_width();
    }
    return align_up(bytes);
  }

  static std::uint64_t fp8_key_payload(const Gemma4Config &config, int batch, int maximum_context,
                                       const std::vector<bool> &selected,
                                       std::vector<std::uint64_t> *offsets) {
    std::uint64_t bytes = 0;
    for (const auto &layer : config.layers) {
      const std::size_t layer_index = offsets->size();
      if (!selected.at(layer_index)) {
        offsets->push_back(std::numeric_limits<std::uint64_t>::max());
        continue;
      }
      bytes = align_up(bytes);
      offsets->push_back(bytes);
      bytes += static_cast<std::uint64_t>(batch) * maximum_context * layer.kv_width();
    }
    return align_up(bytes);
  }

  static std::uint64_t int8_key_scale_payload(const Gemma4Config &config, int batch,
                                              int maximum_context,
                                              const std::vector<bool> &selected,
                                              std::vector<std::uint64_t> *offsets) {
    std::uint64_t bytes = 0;
    for (const auto &layer : config.layers) {
      const std::size_t layer_index = offsets->size();
      if (!selected.at(layer_index)) {
        offsets->push_back(std::numeric_limits<std::uint64_t>::max());
        continue;
      }
      bytes = align_up(bytes);
      offsets->push_back(bytes);
      bytes += static_cast<std::uint64_t>(batch) * maximum_context * layer.kv_heads * 4ULL *
               sizeof(float);
    }
    return align_up(bytes);
  }

  static std::uint64_t int8_value_scale_payload(const Gemma4Config &config, int batch,
                                                int maximum_context,
                                                const std::vector<bool> &selected,
                                                std::vector<std::uint64_t> *offsets) {
    std::uint64_t bytes = 0;
    for (const auto &layer : config.layers) {
      const std::size_t layer_index = offsets->size();
      if (!selected.at(layer_index)) {
        offsets->push_back(std::numeric_limits<std::uint64_t>::max());
        continue;
      }
      bytes = align_up(bytes);
      offsets->push_back(bytes);
      bytes += static_cast<std::uint64_t>(batch) * maximum_context * layer.kv_heads * 4ULL *
               sizeof(float);
    }
    return align_up(bytes);
  }

  Implementation(const Gemma4Config &model_config, const DeviceWeightArena &device_weights,
                 int requested_batch, int max_context, const Fp8WeightArena *decode_weights,
                 const DeviceWeightArena *draft_weights, const Fp8WeightArena *draft_fp8_weights)
      : config(model_config), weights(device_weights), fp8_weights(decode_weights),
        assistant_weights(draft_weights), assistant_fp8_weights(draft_fp8_weights),
        batch_size(requested_batch), maximum_context(max_context),
        hidden(static_cast<int>(model_config.hidden_size)),
        vocabulary(static_cast<int>(model_config.vocabulary_size)),
        slot_positions(static_cast<std::size_t>(requested_batch), 0), caches(),
        cache_payload_bytes(cache_payload(model_config, requested_batch, max_context, &caches)),
        fp8_suffix_token_limit(fp8_suffix_max_tokens()),
        use_fp8_lm_head(decode_weights != nullptr && !bf16_lm_head_enabled()),
        use_int4_lm_head(int4_lm_head_enabled()),
        use_int4_bf16_lm_head(int4_bf16_lm_head_enabled()),
        use_fp8_global_kv(fp8_global_kv_enabled()),
        use_fp8_global_k(fp8_global_k_enabled() || use_fp8_global_kv),
        fp8_global_k_layers(global_layer_selection(model_config, use_fp8_global_k,
                                                   "CARAT_FP8_GLOBAL_K_LAYERS", "FP8 global K")),
        use_int8_global_kv(int8_global_kv_enabled()),
        use_int8_global_k(int8_global_k_enabled() || use_int8_global_kv),
        int8_global_k_layers(global_layer_selection(model_config, use_int8_global_k,
                                                    "CARAT_INT8_GLOBAL_K_LAYERS", "INT8 global K")),
        use_int8_global_kv_oracle(int8_global_kv_oracle_enabled()),
        int8_oracle_mode(int8_global_kv_oracle_mode()),
        int8_oracle_block_width(int8_global_kv_oracle_block_width()),
        int8_global_kv_oracle_layers(global_layer_selection(model_config, use_int8_global_kv_oracle,
                                                            "CARAT_INT8_GLOBAL_KV_ORACLE_LAYERS",
                                                            "INT8 global KV oracle")),
        fp8_key_offsets(),
        fp8_key_payload_bytes(use_fp8_global_k
                                  ? fp8_key_payload(model_config, requested_batch, max_context,
                                                    fp8_global_k_layers, &fp8_key_offsets)
                                  : 0),
        int8_key_offsets(),
        int8_key_payload_bytes(use_int8_global_k
                                   ? fp8_key_payload(model_config, requested_batch, max_context,
                                                     int8_global_k_layers, &int8_key_offsets)
                                   : 0),
        int8_key_scale_offsets(),
        int8_key_scale_payload_bytes(use_int8_global_k
                                         ? int8_key_scale_payload(model_config, requested_batch,
                                                                  max_context, int8_global_k_layers,
                                                                  &int8_key_scale_offsets)
                                         : 0),
        int8_value_scale_offsets(),
        int8_value_scale_payload_bytes(
            use_int8_global_kv
                ? int8_value_scale_payload(model_config, requested_batch, max_context,
                                           int8_global_k_layers, &int8_value_scale_offsets)
                : 0),
        speculative_backup_offsets(),
        speculative_backup_payload_bytes(
            speculative_backup_payload(model_config, requested_batch, &speculative_backup_offsets)),
        layers(model_config, device_weights, max_context, max_context, requested_batch,
               decode_weights),
        lm_head(),
        fp8_lm_head(decode_weights == nullptr ? Fp8Scaling::tensor : decode_weights->scaling()),
        int4_lm_head_segments(), int4_bf16_lm_head_segments(),
        cache_arena(2ULL * cache_payload_bytes),
        fp8_key_arena(std::max<std::uint64_t>(1, fp8_key_payload_bytes)),
        fp8_value_arena(std::max<std::uint64_t>(1, use_fp8_global_kv ? fp8_key_payload_bytes : 0)),
        int8_key_arena(std::max<std::uint64_t>(1, int8_key_payload_bytes)),
        int8_key_scale_arena(std::max<std::uint64_t>(1, int8_key_scale_payload_bytes)),
        int8_value_arena(
            std::max<std::uint64_t>(1, use_int8_global_kv ? int8_key_payload_bytes : 0)),
        int8_value_scale_arena(std::max<std::uint64_t>(1, int8_value_scale_payload_bytes)),
        hidden_a(2ULL * std::max(requested_batch, max_context) * hidden),
        hidden_b(2ULL * std::max(requested_batch, max_context) * hidden),
        final_hidden(2ULL * speculative_depth * requested_batch * hidden),
        logits(2ULL * speculative_depth * requested_batch * vocabulary),
        device_input_tokens(sizeof(int) * std::max(requested_batch, max_context)),
        device_output_tokens(sizeof(int) * speculative_depth * requested_batch),
        device_positions(sizeof(int) * requested_batch),
        device_slots(sizeof(int) * requested_batch),
        argmax_workspace(argmax_bf16_workspace_bytes(vocabulary) * speculative_depth *
                         requested_batch),
        fp8_final_hidden(static_cast<std::uint64_t>(speculative_depth) * requested_batch * hidden),
        fp8_final_scales(sizeof(float) * static_cast<std::uint64_t>(speculative_depth) *
                         requested_batch * ((hidden + 127) / 128)),
        target_hidden_slots(2ULL * requested_batch * hidden),
        speculative_backup_arena(2ULL * speculative_backup_payload_bytes),
        device_cache_pointers(2ULL * sizeof(void *) * requested_batch * model_config.layers.size()),
        suffix_hidden_a(2ULL * max_context * hidden), suffix_hidden_b(2ULL * max_context * hidden) {
    if (requested_batch <= 0 || max_context <= 0)
      throw std::runtime_error("invalid batch model shape");
    if (fp8_suffix_token_limit > max_context) {
      throw std::runtime_error("CARAT_FP8_SUFFIX_MAX_TOKENS exceeds prefill workspace");
    }
    if (use_int4_lm_head && (decode_weights == nullptr || !use_fp8_lm_head ||
                             decode_weights->scaling() != Fp8Scaling::tensor)) {
      throw std::runtime_error("CARAT_INT4_LM_HEAD requires tensor-scaled FP8 decode weights");
    }
    if (use_int4_lm_head && int4_lm_head_oracle_enabled()) {
      throw std::runtime_error("physical and oracle INT4 LM-head modes cannot be combined");
    }
    if (use_int4_lm_head && use_int4_bf16_lm_head) {
      throw std::runtime_error(
          "FP8-input and BF16-input physical INT4 LM heads cannot be combined");
    }
    if (use_int4_bf16_lm_head && int4_lm_head_oracle_enabled()) {
      throw std::runtime_error(
          "physical BF16 INT4 and oracle INT4 LM-head modes cannot be combined");
    }
    if (use_int4_bf16_lm_head &&
        (decode_weights == nullptr || decode_weights->scaling() != Fp8Scaling::tensor)) {
      throw std::runtime_error("CARAT_INT4_BF16_LM_HEAD requires tensor-scaled FP8 decode weights");
    }
    if (use_int4_lm_head) {
      const int segment_width = int4_lm_head_segment_width();
      if (segment_width > vocabulary || vocabulary % segment_width != 0) {
        throw std::runtime_error("CARAT_INT4_LM_HEAD_SEGMENT must divide the model vocabulary");
      }
      const auto *embedding =
          static_cast<const unsigned char *>(decode_weights->at("token_embedding"));
      int4_lm_head_segments.reserve(vocabulary / segment_width);
      for (int offset = 0; offset < vocabulary; offset += segment_width) {
        int4_lm_head_segments.push_back(
            {offset,
             std::make_unique<Int4Fp8Linear>(embedding + static_cast<std::size_t>(offset) * hidden,
                                             hidden, segment_width, 256)});
      }
    }
    if (use_int4_bf16_lm_head) {
      const int segment_width = int4_lm_head_segment_width();
      if (segment_width > vocabulary || vocabulary % segment_width != 0) {
        throw std::runtime_error("CARAT_INT4_LM_HEAD_SEGMENT must divide the model vocabulary");
      }
      const auto *embedding =
          static_cast<const unsigned char *>(decode_weights->at("token_embedding"));
      int4_bf16_lm_head_segments.reserve(vocabulary / segment_width);
      for (int offset = 0; offset < vocabulary; offset += segment_width) {
        int4_bf16_lm_head_segments.push_back(
            {offset,
             std::make_unique<Int4Bf16Linear>(embedding + static_cast<std::size_t>(offset) * hidden,
                                              hidden, segment_width, 256, true)});
      }
    }
    if (use_int8_global_kv_oracle && use_fp8_global_k) {
      throw std::runtime_error(
          "INT8 global KV oracle cannot be combined with the FP8 global cache experiment");
    }
    if (use_int8_global_k && (use_fp8_global_k || use_int8_global_kv_oracle)) {
      throw std::runtime_error(
          "materialized INT8 global K cannot be combined with FP8 K or the INT8 oracle");
    }
    check(cudaStreamCreateWithFlags(&decode_stream, cudaStreamNonBlocking), "create decode stream");
    for (std::size_t layer = 0; layer < config.layers.size(); ++layer) {
      if (config.layers[layer].attention == AttentionKind::sliding) {
        shared_sliding_layer = static_cast<int>(layer);
      } else {
        shared_global_layer = static_cast<int>(layer);
      }
    }
    if (draft_weights != nullptr) {
      if (shared_sliding_layer < 0 || shared_global_layer < 0) {
        throw std::runtime_error("target does not expose both assistant KV types");
      }
      assistant = std::make_unique<Gemma4AssistantRunner>(*draft_weights, draft_fp8_weights,
                                                          requested_batch, max_context);
    }
    // Global batched suffix attention pads shorter requests to one GEMM width. Zeroing once makes
    // every unread cache tail finite; masked probabilities then contribute exactly zero in PV.
    check(cudaMemset(cache_arena.get(), 0, static_cast<std::size_t>(2ULL * cache_payload_bytes)),
          "initialize KV cache arena");
    if (use_fp8_global_k) {
      check(cudaMemset(fp8_key_arena.get(), 0, static_cast<std::size_t>(fp8_key_payload_bytes)),
            "initialize FP8 global K cache arena");
    }
    if (use_fp8_global_kv) {
      check(cudaMemset(fp8_value_arena.get(), 0, static_cast<std::size_t>(fp8_key_payload_bytes)),
            "initialize FP8 global V cache arena");
    }
    if (use_int8_global_k) {
      check(cudaMemset(int8_key_arena.get(), 0, static_cast<std::size_t>(int8_key_payload_bytes)),
            "initialize INT8 global K cache arena");
      check(cudaMemset(int8_key_scale_arena.get(), 0,
                       static_cast<std::size_t>(int8_key_scale_payload_bytes)),
            "initialize INT8 global K scale arena");
    }
    if (use_int8_global_kv) {
      check(cudaMemset(int8_value_arena.get(), 0, static_cast<std::size_t>(int8_key_payload_bytes)),
            "initialize INT8 global V cache arena");
      check(cudaMemset(int8_value_scale_arena.get(), 0,
                       static_cast<std::size_t>(int8_value_scale_payload_bytes)),
            "initialize INT8 global V scale arena");
    }
  }

  ~Implementation() {
    decode_graphs.clear();
    if (decode_stream != nullptr)
      cudaStreamDestroy(decode_stream);
  }

  void *key_cache(std::size_t layer) {
    return static_cast<unsigned char *>(cache_arena.get()) + caches.at(layer).offset;
  }
  void *value_cache(std::size_t layer) {
    return static_cast<unsigned char *>(cache_arena.get()) + cache_payload_bytes +
           caches.at(layer).offset;
  }
  void *slot_key_cache(std::size_t layer, int slot) {
    return static_cast<unsigned char *>(key_cache(layer)) +
           2ULL * slot * caches.at(layer).capacity * config.layers.at(layer).kv_width();
  }
  void *slot_value_cache(std::size_t layer, int slot) {
    return static_cast<unsigned char *>(value_cache(layer)) +
           2ULL * slot * caches.at(layer).capacity * config.layers.at(layer).kv_width();
  }
  void *fp8_key_cache(std::size_t layer) {
    if (!use_fp8_global_k || !fp8_global_k_layers.at(layer)) {
      return nullptr;
    }
    return static_cast<unsigned char *>(fp8_key_arena.get()) + fp8_key_offsets.at(layer);
  }
  void *slot_fp8_key_cache(std::size_t layer, int slot) {
    auto *base = static_cast<unsigned char *>(fp8_key_cache(layer));
    if (base == nullptr)
      return nullptr;
    return base + static_cast<std::uint64_t>(slot) * caches.at(layer).capacity *
                      config.layers.at(layer).kv_width();
  }
  void *fp8_value_cache(std::size_t layer) {
    if (!use_fp8_global_kv || !fp8_global_k_layers.at(layer))
      return nullptr;
    return static_cast<unsigned char *>(fp8_value_arena.get()) + fp8_key_offsets.at(layer);
  }
  void *slot_fp8_value_cache(std::size_t layer, int slot) {
    auto *base = static_cast<unsigned char *>(fp8_value_cache(layer));
    if (base == nullptr)
      return nullptr;
    return base + static_cast<std::uint64_t>(slot) * caches.at(layer).capacity *
                      config.layers.at(layer).kv_width();
  }
  void *int8_key_cache(std::size_t layer) {
    if (!use_int8_global_k || !int8_global_k_layers.at(layer))
      return nullptr;
    return static_cast<unsigned char *>(int8_key_arena.get()) + int8_key_offsets.at(layer);
  }
  void *slot_int8_key_cache(std::size_t layer, int slot) {
    auto *base = static_cast<unsigned char *>(int8_key_cache(layer));
    if (base == nullptr)
      return nullptr;
    return base + static_cast<std::uint64_t>(slot) * caches.at(layer).capacity *
                      config.layers.at(layer).kv_width();
  }
  float *int8_key_scales(std::size_t layer) {
    if (!use_int8_global_k || !int8_global_k_layers.at(layer))
      return nullptr;
    auto *base =
        static_cast<unsigned char *>(int8_key_scale_arena.get()) + int8_key_scale_offsets.at(layer);
    return reinterpret_cast<float *>(base);
  }
  float *slot_int8_key_scales(std::size_t layer, int slot) {
    auto *base = int8_key_scales(layer);
    if (base == nullptr)
      return nullptr;
    return base + static_cast<std::uint64_t>(slot) * config.layers.at(layer).kv_heads *
                      caches.at(layer).capacity * 4ULL;
  }
  void *int8_value_cache(std::size_t layer) {
    if (!use_int8_global_kv || !int8_global_k_layers.at(layer))
      return nullptr;
    return static_cast<unsigned char *>(int8_value_arena.get()) + int8_key_offsets.at(layer);
  }
  void *slot_int8_value_cache(std::size_t layer, int slot) {
    auto *base = static_cast<unsigned char *>(int8_value_cache(layer));
    if (base == nullptr)
      return nullptr;
    return base + static_cast<std::uint64_t>(slot) * caches.at(layer).capacity *
                      config.layers.at(layer).kv_width();
  }
  float *int8_value_scales(std::size_t layer) {
    if (!use_int8_global_kv || !int8_global_k_layers.at(layer))
      return nullptr;
    auto *base = static_cast<unsigned char *>(int8_value_scale_arena.get()) +
                 int8_value_scale_offsets.at(layer);
    return reinterpret_cast<float *>(base);
  }
  float *slot_int8_value_scales(std::size_t layer, int slot) {
    auto *base = int8_value_scales(layer);
    if (base == nullptr)
      return nullptr;
    return base + static_cast<std::uint64_t>(slot) * config.layers.at(layer).kv_heads *
                      caches.at(layer).capacity * 4ULL;
  }
  void refresh_int8_kv_span(std::size_t layer, int slot, int position, int tokens,
                            cudaStream_t stream) {
    void *destination = slot_int8_key_cache(layer, slot);
    if (destination == nullptr)
      return;
    const auto &layer_config = config.layers.at(layer);
    quantize_global_k_cache_span_int8_block128(
        slot_key_cache(layer, slot), destination, slot_int8_key_scales(layer, slot),
        static_cast<int>(layer_config.kv_heads), tokens,
        static_cast<int>(layer_config.head_dimension), caches.at(layer).capacity, position, stream);
    void *value_destination = slot_int8_value_cache(layer, slot);
    if (value_destination != nullptr) {
      quantize_grouped_global_value_span_wgmma_int8(
          slot_value_cache(layer, slot), value_destination, slot_int8_value_scales(layer, slot),
          static_cast<int>(layer_config.kv_heads), tokens, caches.at(layer).capacity,
          static_cast<int>(layer_config.head_dimension), position, stream);
    }
  }
  void refresh_fp8_kv_span(std::size_t layer, int slot, int position, int tokens,
                           cudaStream_t stream) {
    void *destination = slot_fp8_key_cache(layer, slot);
    if (destination == nullptr)
      return;
    const auto &layer_config = config.layers.at(layer);
    quantize_global_k_cache_span_fp8(
        slot_key_cache(layer, slot), destination, static_cast<int>(layer_config.kv_heads), tokens,
        static_cast<int>(layer_config.head_dimension), caches.at(layer).capacity, position, stream);
    void *value_destination = slot_fp8_value_cache(layer, slot);
    if (value_destination != nullptr) {
      quantize_grouped_global_value_span_wgmma_fp8(
          slot_value_cache(layer, slot), value_destination, static_cast<int>(layer_config.kv_heads),
          tokens, caches.at(layer).capacity, static_cast<int>(layer_config.head_dimension),
          position, stream);
    }
  }
  void refresh_global_cache_approximations(std::size_t layer, int slot, int position, int tokens,
                                           cudaStream_t stream) {
    if (int8_global_kv_oracle_layers.at(layer)) {
      const auto &layer_config = config.layers.at(layer);
      roundtrip_global_kv_int8_per_token_bf16(
          slot_key_cache(layer, slot), slot_value_cache(layer, slot),
          static_cast<int>(layer_config.kv_heads), tokens,
          static_cast<int>(layer_config.head_dimension), caches.at(layer).capacity, position,
          stream, int8_oracle_mode != Int8OracleMode::values,
          int8_oracle_mode != Int8OracleMode::keys, int8_oracle_block_width);
    }
    refresh_int8_kv_span(layer, slot, position, tokens, stream);
    refresh_fp8_kv_span(layer, slot, position, tokens, stream);
  }
  void *speculative_key_backup(std::size_t layer) {
    return static_cast<unsigned char *>(speculative_backup_arena.get()) +
           speculative_backup_offsets.at(layer);
  }
  void *speculative_value_backup(std::size_t layer) {
    return static_cast<unsigned char *>(speculative_backup_arena.get()) +
           speculative_backup_payload_bytes + speculative_backup_offsets.at(layer);
  }

  void run_decode_lm_head(const void *input, void *output, int rows, cudaStream_t stream) {
    if (use_int4_bf16_lm_head && !int4_bf16_lm_head_segments.empty() &&
        int4_bf16_lm_head_segments.front().linear->supports_rows(rows)) {
      const auto run_int4_bf16 = [&] {
        for (const auto &segment : int4_bf16_lm_head_segments) {
          auto *segment_output =
              static_cast<unsigned char *>(output) + 2ULL * segment.vocabulary_offset;
          segment.linear->run(input, segment_output, rows, stream, vocabulary);
        }
      };
      run_int4_bf16();
      if (int4_bf16_lm_head_validation_enabled() && !int4_bf16_lm_head_validation_logged) {
        std::vector<__nv_bfloat16> physical(static_cast<std::size_t>(vocabulary));
        std::vector<__nv_bfloat16> reference(static_cast<std::size_t>(vocabulary));
        std::vector<__nv_bfloat16> oracle(static_cast<std::size_t>(vocabulary));
        check(cudaStreamSynchronize(stream), "synchronize BF16 INT4 LM-head validation");
        check(cudaMemcpy(physical.data(), output, physical.size() * sizeof(physical[0]),
                         cudaMemcpyDeviceToHost),
              "copy BF16 INT4 LM-head validation logits");
        lm_head.run(input, weights.at("token_embedding"), output, rows, hidden, vocabulary, stream);
        check(cudaStreamSynchronize(stream), "synchronize BF16 LM-head validation reference");
        check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(reference[0]),
                         cudaMemcpyDeviceToHost),
              "copy BF16 LM-head validation reference logits");
        auto *validation_scales = static_cast<float *>(fp8_final_scales.get());
        quantize_bf16_to_fp8_e4m3(input, fp8_final_hidden.get(), validation_scales,
                                  static_cast<std::size_t>(rows) * hidden, stream);
        Allocation oracle_weight(static_cast<std::uint64_t>(vocabulary) * hidden);
        check(cudaMemcpyAsync(oracle_weight.get(), fp8_weights->at("token_embedding"),
                              static_cast<std::size_t>(vocabulary) * hidden,
                              cudaMemcpyDeviceToDevice, stream),
              "copy BF16 INT4 validation oracle weight");
        roundtrip_fp8_weight_via_int4_blocks(oracle_weight.get(), vocabulary, hidden, 256, stream);
        fp8_lm_head.run(fp8_final_hidden.get(), validation_scales, oracle_weight.get(),
                        fp8_weights->scale("token_embedding"), output, rows, hidden, vocabulary,
                        stream);
        check(cudaStreamSynchronize(stream), "synchronize BF16 INT4 validation oracle");
        check(cudaMemcpy(oracle.data(), output, oracle.size() * sizeof(oracle[0]),
                         cudaMemcpyDeviceToHost),
              "copy BF16 INT4 validation oracle logits");
        int physical_argmax = 0;
        int reference_argmax = 0;
        int oracle_argmax = 0;
        double dot = 0.0;
        double physical_squared = 0.0;
        double reference_squared = 0.0;
        double physical_oracle_dot = 0.0;
        double oracle_squared = 0.0;
        for (int column = 0; column < vocabulary; ++column) {
          const double physical_value = __bfloat162float(physical[column]);
          const double reference_value = __bfloat162float(reference[column]);
          const double oracle_value = __bfloat162float(oracle[column]);
          if (physical_value > __bfloat162float(physical[physical_argmax])) {
            physical_argmax = column;
          }
          if (reference_value > __bfloat162float(reference[reference_argmax])) {
            reference_argmax = column;
          }
          if (oracle_value > __bfloat162float(oracle[oracle_argmax])) {
            oracle_argmax = column;
          }
          dot += physical_value * reference_value;
          physical_squared += physical_value * physical_value;
          reference_squared += reference_value * reference_value;
          physical_oracle_dot += physical_value * oracle_value;
          oracle_squared += oracle_value * oracle_value;
        }
        std::cerr << "BF16 INT4 LM-head validation rows=" << rows
                  << " segment_width=" << int4_lm_head_segment_width()
                  << " physical_argmax=" << physical_argmax
                  << " reference_argmax=" << reference_argmax
                  << " cosine=" << dot / std::sqrt(physical_squared * reference_squared)
                  << " norm_ratio=" << std::sqrt(physical_squared / reference_squared)
                  << " oracle_argmax=" << oracle_argmax << " physical_oracle_cosine="
                  << physical_oracle_dot / std::sqrt(physical_squared * oracle_squared)
                  << " physical_oracle_norm_ratio=" << std::sqrt(physical_squared / oracle_squared)
                  << '\n';
        run_int4_bf16();
        int4_bf16_lm_head_validation_logged = true;
      }
      return;
    }
    if (!use_fp8_lm_head) {
      lm_head.run(input, weights.at("token_embedding"), output, rows, hidden, vocabulary, stream);
      return;
    }
    auto *scales = static_cast<float *>(fp8_final_scales.get());
    if (fp8_weights->scaling() == Fp8Scaling::tensor) {
      quantize_bf16_to_fp8_e4m3(input, fp8_final_hidden.get(), scales,
                                static_cast<std::size_t>(rows) * hidden, stream);
    } else if (fp8_weights->scaling() == Fp8Scaling::channel) {
      quantize_bf16_rows_to_fp8_e4m3(input, fp8_final_hidden.get(), scales, rows, hidden, stream);
    } else {
      quantize_bf16_activation_to_fp8_block_128(input, fp8_final_hidden.get(), scales, rows, hidden,
                                                stream);
    }
    if (use_int4_lm_head && !int4_lm_head_segments.empty() &&
        int4_lm_head_segments.front().linear->supports_rows(rows)) {
      const auto run_int4 = [&] {
        for (const auto &segment : int4_lm_head_segments) {
          auto *segment_output =
              static_cast<unsigned char *>(output) + 2ULL * segment.vocabulary_offset;
          segment.linear->run(fp8_final_hidden.get(), segment_output, rows, stream, vocabulary);
        }
      };
      run_int4();
      if (int4_lm_head_validation_enabled() && !int4_lm_head_validation_logged) {
        std::vector<__nv_bfloat16> physical(static_cast<std::size_t>(vocabulary));
        std::vector<__nv_bfloat16> reference(static_cast<std::size_t>(vocabulary));
        std::vector<__nv_bfloat16> oracle(static_cast<std::size_t>(vocabulary));
        check(cudaStreamSynchronize(stream), "synchronize physical INT4 LM-head validation");
        check(cudaMemcpy(physical.data(), output, physical.size() * sizeof(physical[0]),
                         cudaMemcpyDeviceToHost),
              "copy physical INT4 LM-head validation logits");
        fp8_lm_head.run(fp8_final_hidden.get(), scales, fp8_weights->at("token_embedding"),
                        fp8_weights->scale("token_embedding"), output, rows, hidden, vocabulary,
                        stream);
        check(cudaStreamSynchronize(stream), "synchronize FP8 LM-head validation reference");
        check(cudaMemcpy(reference.data(), output, reference.size() * sizeof(reference[0]),
                         cudaMemcpyDeviceToHost),
              "copy FP8 LM-head validation logits");
        Allocation oracle_weight(static_cast<std::uint64_t>(vocabulary) * hidden);
        check(cudaMemcpyAsync(oracle_weight.get(), fp8_weights->at("token_embedding"),
                              static_cast<std::size_t>(vocabulary) * hidden,
                              cudaMemcpyDeviceToDevice, stream),
              "copy INT4 LM-head validation oracle weight");
        roundtrip_fp8_weight_via_int4_blocks(oracle_weight.get(), vocabulary, hidden, 256, stream);
        fp8_lm_head.run(fp8_final_hidden.get(), scales, oracle_weight.get(),
                        fp8_weights->scale("token_embedding"), output, rows, hidden, vocabulary,
                        stream);
        check(cudaStreamSynchronize(stream), "synchronize INT4 LM-head validation oracle");
        check(cudaMemcpy(oracle.data(), output, oracle.size() * sizeof(oracle[0]),
                         cudaMemcpyDeviceToHost),
              "copy INT4 LM-head validation oracle logits");
        const auto compare = [&](const std::vector<__nv_bfloat16> &left,
                                 const std::vector<__nv_bfloat16> &right) {
          int left_argmax = 0;
          int right_argmax = 0;
          double dot = 0.0;
          double left_squared = 0.0;
          double right_squared = 0.0;
          for (int column = 0; column < vocabulary; ++column) {
            const double left_value = __bfloat162float(left[column]);
            const double right_value = __bfloat162float(right[column]);
            if (left_value > __bfloat162float(left[left_argmax]))
              left_argmax = column;
            if (right_value > __bfloat162float(right[right_argmax]))
              right_argmax = column;
            dot += left_value * right_value;
            left_squared += left_value * left_value;
            right_squared += right_value * right_value;
          }
          return std::tuple{left_argmax, right_argmax,
                            dot / std::sqrt(left_squared * right_squared),
                            std::sqrt(left_squared / right_squared)};
        };
        const auto [physical_argmax, reference_argmax, cosine, norm_ratio] =
            compare(physical, reference);
        const auto [physical_oracle_argmax, oracle_argmax, physical_oracle_cosine,
                    physical_oracle_norm_ratio] = compare(physical, oracle);
        const auto [oracle_reference_argmax, oracle_reference_target, oracle_reference_cosine,
                    oracle_reference_norm_ratio] = compare(oracle, reference);
        std::cerr << "INT4 LM-head validation rows=" << rows
                  << " segment_width=" << int4_lm_head_segment_width()
                  << " physical_argmax=" << physical_argmax
                  << " reference_argmax=" << reference_argmax << " cosine=" << cosine
                  << " norm_ratio=" << norm_ratio
                  << " physical_oracle_argmax=" << physical_oracle_argmax
                  << " oracle_argmax=" << oracle_argmax
                  << " physical_oracle_cosine=" << physical_oracle_cosine
                  << " physical_oracle_norm_ratio=" << physical_oracle_norm_ratio
                  << " oracle_reference_argmax=" << oracle_reference_argmax
                  << " oracle_reference_target=" << oracle_reference_target
                  << " oracle_reference_cosine=" << oracle_reference_cosine
                  << " oracle_reference_norm_ratio=" << oracle_reference_norm_ratio << '\n';
        run_int4();
        int4_lm_head_validation_logged = true;
      }
      return;
    }
    fp8_lm_head.run(fp8_final_hidden.get(), scales, fp8_weights->at("token_embedding"),
                    fp8_weights->scale("token_embedding"), output, rows, hidden, vocabulary,
                    stream);
  }

  [[nodiscard]] bool use_fp8_suffix(const int packed_tokens) const noexcept {
    return fp8_weights != nullptr && fp8_suffix_token_limit > 0 &&
           packed_tokens <= fp8_suffix_token_limit;
  }

  void enqueue_ragged_decode(int active, int maximum_context_length, int contiguous_slot_start) {
    embedding_batch_bf16(weights.at("token_embedding"),
                         static_cast<const int *>(device_input_tokens.get()), hidden_a.get(),
                         active, vocabulary, hidden, std::sqrt(static_cast<float>(hidden)),
                         decode_stream);
    void *input = hidden_a.get();
    void *output = hidden_b.get();
    for (std::size_t layer = 0; layer < config.layers.size(); ++layer) {
      const auto &cache = caches[layer];
      const int context = std::min(maximum_context_length, cache.capacity);
      layers.run_decode_ragged(
          static_cast<int>(layer), input, output, static_cast<const int *>(device_positions.get()),
          static_cast<const int *>(device_slots.get()), active, batch_size, context,
          key_cache(layer), value_cache(layer), fp8_key_cache(layer), fp8_value_cache(layer),
          int8_key_cache(layer), int8_key_scales(layer), int8_value_cache(layer),
          int8_value_scales(layer),
          int8_global_kv_oracle_layers.at(layer) && int8_oracle_mode != Int8OracleMode::values,
          int8_global_kv_oracle_layers.at(layer) && int8_oracle_mode != Int8OracleMode::keys,
          int8_oracle_block_width, cache.capacity, contiguous_slot_start, decode_stream);
      std::swap(input, output);
    }
    rms_norm_bf16(input, weights.at("final_norm"), final_hidden.get(), active, hidden,
                  static_cast<float>(config.rms_norm_epsilon), decode_stream);
    scatter_rows_bf16(final_hidden.get(), static_cast<const int *>(device_slots.get()),
                      target_hidden_slots.get(), active, hidden, decode_stream);
    run_decode_lm_head(final_hidden.get(), logits.get(), active, decode_stream);
    argmax_bf16_rows(logits.get(), active, vocabulary,
                     static_cast<int *>(device_output_tokens.get()), argmax_workspace.get(),
                     decode_stream);
  }

  DecodeGraph &capture_decode_graph(int active, int maximum_context_length,
                                    int contiguous_slot_start) {
    auto captured = std::make_unique<DecodeGraph>();
    check(cudaStreamBeginCapture(decode_stream, cudaStreamCaptureModeThreadLocal),
          "begin ragged decode graph capture");
    enqueue_ragged_decode(active, maximum_context_length, contiguous_slot_start);
    check(cudaStreamEndCapture(decode_stream, &captured->graph), "end ragged decode graph capture");
    check(cudaGraphInstantiate(&captured->executable, captured->graph, 0),
          "instantiate ragged decode graph");
    DecodeGraph &result = *captured;
    decode_graphs.emplace(std::make_tuple(active, maximum_context_length, contiguous_slot_start),
                          std::move(captured));
    return result;
  }
};

Gemma4BatchModelRunner::Gemma4BatchModelRunner(const Gemma4Config &config,
                                               const DeviceWeightArena &weights, int batch,
                                               int maximum_context,
                                               const Fp8WeightArena *fp8_weights,
                                               const DeviceWeightArena *assistant_weights,
                                               const Fp8WeightArena *assistant_fp8_weights)
    : implementation_(std::make_unique<Implementation>(config, weights, batch, maximum_context,
                                                       fp8_weights, assistant_weights,
                                                       assistant_fp8_weights)) {}
Gemma4BatchModelRunner::~Gemma4BatchModelRunner() = default;

std::vector<int> Gemma4BatchModelRunner::append(const std::vector<int> &token_ids) {
  auto &state = *implementation_;
  if (token_ids.size() != static_cast<std::size_t>(state.batch_size)) {
    throw std::runtime_error("token batch has the wrong size");
  }
  if (state.current_position >= state.maximum_context)
    throw std::runtime_error("batch context is full");
  check(cudaMemcpy(state.device_input_tokens.get(), token_ids.data(),
                   token_ids.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy batch input tokens");
  embedding_batch_bf16(state.weights.at("token_embedding"),
                       static_cast<const int *>(state.device_input_tokens.get()),
                       state.hidden_a.get(), state.batch_size, state.vocabulary, state.hidden,
                       std::sqrt(static_cast<float>(state.hidden)), nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    const auto &cache = state.caches[layer];
    state.layers.run_decode_batch(static_cast<int>(layer), input, output, state.batch_size,
                                  state.current_position, state.key_cache(layer),
                                  state.value_cache(layer), cache.capacity, nullptr);
    std::swap(input, output);
  }
  rms_norm_bf16(input, state.weights.at("final_norm"), state.final_hidden.get(), state.batch_size,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  state.run_decode_lm_head(state.final_hidden.get(), state.logits.get(), state.batch_size, nullptr);
  argmax_bf16_rows(state.logits.get(), state.batch_size, state.vocabulary,
                   static_cast<int *>(state.device_output_tokens.get()),
                   state.argmax_workspace.get(), nullptr);
  std::vector<int> result(state.batch_size);
  check(cudaMemcpy(result.data(), state.device_output_tokens.get(), result.size() * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "copy batch output tokens");
  ++state.current_position;
  std::fill(state.slot_positions.begin(), state.slot_positions.end(), state.current_position);
  return result;
}

int Gemma4BatchModelRunner::prefill_slot(int slot, const std::vector<int> &token_ids,
                                         int chunk_tokens) {
  reset_slot(slot);
  return prefill_slot_from_prefix(slot, token_ids, 0, chunk_tokens);
}

int Gemma4BatchModelRunner::prefill_slot_from_prefix(int slot, const std::vector<int> &token_ids,
                                                     int cached_prefix_tokens, int chunk_tokens) {
  auto &state = *implementation_;
  if (slot < 0 || slot >= state.batch_size || token_ids.empty() ||
      token_ids.size() > static_cast<std::size_t>(state.maximum_context) || chunk_tokens <= 0 ||
      chunk_tokens > state.maximum_context || cached_prefix_tokens < 0 ||
      cached_prefix_tokens >= static_cast<int>(token_ids.size()) ||
      state.slot_positions.at(static_cast<std::size_t>(slot)) != cached_prefix_tokens) {
    throw std::runtime_error("invalid slot prefill shape");
  }
  std::optional<int> prediction;
  while (!prediction)
    prediction = prefill_slot_chunk(slot, token_ids, chunk_tokens);
  return *prediction;
}

std::optional<int> Gemma4BatchModelRunner::prefill_slot_chunk(int slot,
                                                              const std::vector<int> &token_ids,
                                                              int chunk_tokens) {
  auto &state = *implementation_;
  if (slot < 0 || slot >= state.batch_size || token_ids.empty() ||
      token_ids.size() > static_cast<std::size_t>(state.maximum_context) || chunk_tokens <= 0 ||
      chunk_tokens > state.maximum_context) {
    throw std::runtime_error("invalid slot prefill chunk shape");
  }
  const int position = state.slot_positions.at(static_cast<std::size_t>(slot));
  if (position < 0 || position >= static_cast<int>(token_ids.size())) {
    throw std::runtime_error("slot prefill chunk does not extend resident context");
  }
  const int tokens = std::min(chunk_tokens, static_cast<int>(token_ids.size()) - position);
  for (int index = position; index < position + tokens; ++index) {
    const int token_id = token_ids[static_cast<std::size_t>(index)];
    if (token_id < 0 || token_id >= state.vocabulary) {
      throw std::runtime_error("slot prompt contains an invalid token id");
    }
  }
  check(cudaMemcpy(state.device_input_tokens.get(), token_ids.data() + position,
                   static_cast<std::size_t>(tokens) * sizeof(int), cudaMemcpyHostToDevice),
        "copy slot prompt chunk tokens");
  embedding_batch_bf16(state.weights.at("token_embedding"),
                       static_cast<const int *>(state.device_input_tokens.get()),
                       state.hidden_a.get(), tokens, state.vocabulary, state.hidden,
                       std::sqrt(static_cast<float>(state.hidden)), nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    const auto &cache = state.caches[layer];
    state.layers.run_prefill_chunk(static_cast<int>(layer), input, output, tokens, position,
                                   state.slot_key_cache(layer, slot),
                                   state.slot_value_cache(layer, slot), cache.capacity, nullptr);
    state.refresh_global_cache_approximations(layer, slot, position, tokens, nullptr);
    std::swap(input, output);
  }
  state.slot_positions.at(static_cast<std::size_t>(slot)) = position + tokens;
  if (position + tokens != static_cast<int>(token_ids.size()))
    return std::nullopt;

  const auto *last_hidden =
      static_cast<const unsigned char *>(input) + 2ULL * (tokens - 1) * state.hidden;
  rms_norm_bf16(last_hidden, state.weights.at("final_norm"), state.final_hidden.get(), 1,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  check(cudaMemcpy(state.device_slots.get(), &slot, sizeof(slot), cudaMemcpyHostToDevice),
        "copy prefill hidden slot");
  scatter_rows_bf16(state.final_hidden.get(), static_cast<const int *>(state.device_slots.get()),
                    state.target_hidden_slots.get(), 1, state.hidden, nullptr);
  state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                    state.logits.get(), 1, state.hidden, state.vocabulary, nullptr);
  argmax_bf16(state.logits.get(), state.vocabulary,
              static_cast<int *>(state.device_output_tokens.get()), state.argmax_workspace.get(),
              nullptr);
  int prediction = 0;
  check(cudaMemcpy(&prediction, state.device_output_tokens.get(), sizeof(prediction),
                   cudaMemcpyDeviceToHost),
        "copy slot prefill token");
  return prediction;
}

std::vector<int> Gemma4BatchModelRunner::prefill_slots_suffix(
    const std::vector<int> &slots, const std::vector<const std::vector<int> *> &token_ids) {
  auto &state = *implementation_;
  const int requests = static_cast<int>(slots.size());
  if (requests <= 0 || requests > state.batch_size || token_ids.size() != slots.size()) {
    throw std::runtime_error("invalid batched suffix prefill shape");
  }
  std::vector<bool> seen(static_cast<std::size_t>(state.batch_size), false);
  std::vector<int> counts(static_cast<std::size_t>(requests));
  std::vector<int> positions(static_cast<std::size_t>(requests));
  std::vector<int> last_rows(static_cast<std::size_t>(requests));
  std::vector<int> packed_tokens;
  for (int request = 0; request < requests; ++request) {
    const int slot = slots[static_cast<std::size_t>(request)];
    const auto *prompt = token_ids[static_cast<std::size_t>(request)];
    if (slot < 0 || slot >= state.batch_size || seen[static_cast<std::size_t>(slot)] ||
        prompt == nullptr || prompt->empty() ||
        prompt->size() > static_cast<std::size_t>(state.maximum_context)) {
      throw std::runtime_error("invalid batched suffix prefill request");
    }
    seen[static_cast<std::size_t>(slot)] = true;
    const int position = state.slot_positions[static_cast<std::size_t>(slot)];
    const int count = static_cast<int>(prompt->size()) - position;
    if (position < 0 || count <= 0) {
      throw std::runtime_error("batched suffix does not extend resident context");
    }
    positions[static_cast<std::size_t>(request)] = position;
    counts[static_cast<std::size_t>(request)] = count;
    for (int index = position; index < static_cast<int>(prompt->size()); ++index) {
      const int token = (*prompt)[static_cast<std::size_t>(index)];
      if (token < 0 || token >= state.vocabulary) {
        throw std::runtime_error("batched suffix contains an invalid token id");
      }
      packed_tokens.push_back(token);
    }
    last_rows[static_cast<std::size_t>(request)] = static_cast<int>(packed_tokens.size()) - 1;
  }
  if (packed_tokens.size() > static_cast<std::size_t>(state.maximum_context)) {
    throw std::runtime_error("batched suffix exceeds shared projection workspace");
  }
  const bool fp8_projections = state.use_fp8_suffix(static_cast<int>(packed_tokens.size()));

  check(cudaMemcpy(state.device_input_tokens.get(), packed_tokens.data(),
                   packed_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy batched suffix tokens");
  embedding_batch_bf16(state.weights.at("token_embedding"),
                       static_cast<const int *>(state.device_input_tokens.get()),
                       state.hidden_a.get(), static_cast<int>(packed_tokens.size()),
                       state.vocabulary, state.hidden, std::sqrt(static_cast<float>(state.hidden)),
                       nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    std::vector<void *> key_caches;
    std::vector<void *> value_caches;
    std::vector<int> capacities(static_cast<std::size_t>(requests), state.caches[layer].capacity);
    key_caches.reserve(static_cast<std::size_t>(requests));
    value_caches.reserve(static_cast<std::size_t>(requests));
    for (const int slot : slots) {
      key_caches.push_back(state.slot_key_cache(layer, slot));
      value_caches.push_back(state.slot_value_cache(layer, slot));
    }
    state.layers.run_prefill_chunks(static_cast<int>(layer), input, output, counts, positions,
                                    key_caches, value_caches, capacities, nullptr, fp8_projections);
    for (int request = 0; request < requests; ++request) {
      state.refresh_global_cache_approximations(layer, slots[static_cast<std::size_t>(request)],
                                                positions[static_cast<std::size_t>(request)],
                                                counts[static_cast<std::size_t>(request)], nullptr);
    }
    std::swap(input, output);
  }
  check(cudaMemcpy(state.device_positions.get(), last_rows.data(), last_rows.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy batched suffix final rows");
  gather_rows_bf16(input, static_cast<const int *>(state.device_positions.get()), output, requests,
                   state.hidden, nullptr);
  rms_norm_bf16(output, state.weights.at("final_norm"), state.final_hidden.get(), requests,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  check(cudaMemcpy(state.device_slots.get(), slots.data(), slots.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy suffix hidden slots");
  scatter_rows_bf16(state.final_hidden.get(), static_cast<const int *>(state.device_slots.get()),
                    state.target_hidden_slots.get(), requests, state.hidden, nullptr);
  if (fp8_projections) {
    state.run_decode_lm_head(state.final_hidden.get(), state.logits.get(), requests, nullptr);
  } else {
    state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                      state.logits.get(), requests, state.hidden, state.vocabulary, nullptr);
  }
  argmax_bf16_rows(state.logits.get(), requests, state.vocabulary,
                   static_cast<int *>(state.device_output_tokens.get()),
                   state.argmax_workspace.get(), nullptr);
  std::vector<int> predictions(static_cast<std::size_t>(requests));
  check(cudaMemcpy(predictions.data(), state.device_output_tokens.get(),
                   predictions.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy batched suffix predictions");
  for (int request = 0; request < requests; ++request) {
    state.slot_positions[static_cast<std::size_t>(slots[static_cast<std::size_t>(request)])] =
        static_cast<int>(token_ids[static_cast<std::size_t>(request)]->size());
  }
  return predictions;
}

std::vector<std::optional<int>>
Gemma4BatchModelRunner::prefill_slots_chunks(const std::vector<int> &slots,
                                             const std::vector<const std::vector<int> *> &token_ids,
                                             const int chunk_tokens) {
  auto &state = *implementation_;
  const int requests = static_cast<int>(slots.size());
  if (requests <= 0 || requests > state.batch_size || token_ids.size() != slots.size() ||
      chunk_tokens <= 0) {
    throw std::runtime_error("invalid batched chunk prefill shape");
  }
  std::vector<bool> seen(static_cast<std::size_t>(state.batch_size), false);
  std::vector<int> counts(static_cast<std::size_t>(requests));
  std::vector<int> positions(static_cast<std::size_t>(requests));
  std::vector<int> last_rows(static_cast<std::size_t>(requests));
  std::vector<int> packed_tokens;
  for (int request = 0; request < requests; ++request) {
    const int slot = slots[static_cast<std::size_t>(request)];
    const auto *prompt = token_ids[static_cast<std::size_t>(request)];
    if (slot < 0 || slot >= state.batch_size || seen[static_cast<std::size_t>(slot)] ||
        prompt == nullptr || prompt->empty() ||
        prompt->size() > static_cast<std::size_t>(state.maximum_context)) {
      throw std::runtime_error("invalid batched chunk prefill request");
    }
    seen[static_cast<std::size_t>(slot)] = true;
    const int position = state.slot_positions[static_cast<std::size_t>(slot)];
    const int remaining = static_cast<int>(prompt->size()) - position;
    if (position < 0 || remaining <= 0) {
      throw std::runtime_error("batched chunk does not extend resident context");
    }
    const int count = std::min(remaining, chunk_tokens);
    positions[static_cast<std::size_t>(request)] = position;
    counts[static_cast<std::size_t>(request)] = count;
    for (int index = position; index < position + count; ++index) {
      const int token = (*prompt)[static_cast<std::size_t>(index)];
      if (token < 0 || token >= state.vocabulary) {
        throw std::runtime_error("batched chunk contains an invalid token id");
      }
      packed_tokens.push_back(token);
    }
    last_rows[static_cast<std::size_t>(request)] = static_cast<int>(packed_tokens.size()) - 1;
  }
  if (packed_tokens.size() > static_cast<std::size_t>(state.maximum_context)) {
    throw std::runtime_error("batched chunks exceed shared projection workspace");
  }
  // Cold prefill uses BF16 even when decode uses FP8.
  const bool fp8_projections = false;
  check(cudaMemcpy(state.device_input_tokens.get(), packed_tokens.data(),
                   packed_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy batched chunk tokens");
  embedding_batch_bf16(state.weights.at("token_embedding"),
                       static_cast<const int *>(state.device_input_tokens.get()),
                       state.hidden_a.get(), static_cast<int>(packed_tokens.size()),
                       state.vocabulary, state.hidden, std::sqrt(static_cast<float>(state.hidden)),
                       nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    std::vector<void *> key_caches;
    std::vector<void *> value_caches;
    std::vector<int> capacities(static_cast<std::size_t>(requests), state.caches[layer].capacity);
    key_caches.reserve(static_cast<std::size_t>(requests));
    value_caches.reserve(static_cast<std::size_t>(requests));
    for (const int slot : slots) {
      key_caches.push_back(state.slot_key_cache(layer, slot));
      value_caches.push_back(state.slot_value_cache(layer, slot));
    }
    state.layers.run_prefill_chunks(static_cast<int>(layer), input, output, counts, positions,
                                    key_caches, value_caches, capacities, nullptr, fp8_projections);
    for (int request = 0; request < requests; ++request) {
      state.refresh_global_cache_approximations(layer, slots[static_cast<std::size_t>(request)],
                                                positions[static_cast<std::size_t>(request)],
                                                counts[static_cast<std::size_t>(request)], nullptr);
    }
    std::swap(input, output);
  }

  std::vector<int> completed_rows;
  std::vector<int> completed_slots;
  std::vector<int> completed_requests;
  for (int request = 0; request < requests; ++request) {
    const int resident =
        positions[static_cast<std::size_t>(request)] + counts[static_cast<std::size_t>(request)];
    state.slot_positions[static_cast<std::size_t>(slots[static_cast<std::size_t>(request)])] =
        resident;
    if (resident == static_cast<int>(token_ids[static_cast<std::size_t>(request)]->size())) {
      completed_rows.push_back(last_rows[static_cast<std::size_t>(request)]);
      completed_slots.push_back(slots[static_cast<std::size_t>(request)]);
      completed_requests.push_back(request);
    }
  }
  std::vector<std::optional<int>> predictions(static_cast<std::size_t>(requests));
  if (completed_rows.empty())
    return predictions;

  const int completed = static_cast<int>(completed_rows.size());
  check(cudaMemcpy(state.device_positions.get(), completed_rows.data(),
                   completed_rows.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy batched chunk final rows");
  gather_rows_bf16(input, static_cast<const int *>(state.device_positions.get()), output, completed,
                   state.hidden, nullptr);
  rms_norm_bf16(output, state.weights.at("final_norm"), state.final_hidden.get(), completed,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  check(cudaMemcpy(state.device_slots.get(), completed_slots.data(),
                   completed_slots.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy batched chunk completed slots");
  scatter_rows_bf16(state.final_hidden.get(), static_cast<const int *>(state.device_slots.get()),
                    state.target_hidden_slots.get(), completed, state.hidden, nullptr);
  if (fp8_projections) {
    state.run_decode_lm_head(state.final_hidden.get(), state.logits.get(), completed, nullptr);
  } else {
    state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                      state.logits.get(), completed, state.hidden, state.vocabulary, nullptr);
  }
  argmax_bf16_rows(state.logits.get(), completed, state.vocabulary,
                   static_cast<int *>(state.device_output_tokens.get()),
                   state.argmax_workspace.get(), nullptr);
  std::vector<int> completed_predictions(static_cast<std::size_t>(completed));
  check(cudaMemcpy(completed_predictions.data(), state.device_output_tokens.get(),
                   completed_predictions.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy batched chunk predictions");
  for (int row = 0; row < completed; ++row) {
    predictions[static_cast<std::size_t>(completed_requests[static_cast<std::size_t>(row)])] =
        completed_predictions[static_cast<std::size_t>(row)];
  }
  return predictions;
}

LayerSlicedSuffixPrefillResult Gemma4BatchModelRunner::prefill_slots_suffix_layer_slice(
    const std::vector<int> &slots, const std::vector<const std::vector<int> *> &token_ids,
    int maximum_layers) {
  auto &state = *implementation_;
  const int requests = static_cast<int>(slots.size());
  if (requests <= 0 || requests > state.batch_size || token_ids.size() != slots.size() ||
      maximum_layers <= 0) {
    throw std::runtime_error("invalid layer-sliced suffix prefill shape");
  }

  if (!state.suffix_layer_state) {
    Implementation::SuffixLayerState suffix;
    suffix.slots = slots;
    suffix.prompts = token_ids;
    suffix.counts.resize(static_cast<std::size_t>(requests));
    suffix.positions.resize(static_cast<std::size_t>(requests));
    suffix.last_rows.resize(static_cast<std::size_t>(requests));
    suffix.prompt_sizes.resize(static_cast<std::size_t>(requests));
    std::vector<bool> seen(static_cast<std::size_t>(state.batch_size), false);
    std::vector<int> packed_tokens;
    for (int request = 0; request < requests; ++request) {
      const int slot = slots[static_cast<std::size_t>(request)];
      const auto *prompt = token_ids[static_cast<std::size_t>(request)];
      if (slot < 0 || slot >= state.batch_size || seen[static_cast<std::size_t>(slot)] ||
          prompt == nullptr || prompt->empty() ||
          prompt->size() > static_cast<std::size_t>(state.maximum_context)) {
        throw std::runtime_error("invalid layer-sliced suffix request");
      }
      seen[static_cast<std::size_t>(slot)] = true;
      const int position = state.slot_positions[static_cast<std::size_t>(slot)];
      const int count = static_cast<int>(prompt->size()) - position;
      if (position < 0 || count <= 0) {
        throw std::runtime_error("layer-sliced suffix does not extend resident context");
      }
      suffix.positions[static_cast<std::size_t>(request)] = position;
      suffix.counts[static_cast<std::size_t>(request)] = count;
      suffix.prompt_sizes[static_cast<std::size_t>(request)] = prompt->size();
      for (int index = position; index < static_cast<int>(prompt->size()); ++index) {
        const int token = (*prompt)[static_cast<std::size_t>(index)];
        if (token < 0 || token >= state.vocabulary) {
          throw std::runtime_error("layer-sliced suffix contains an invalid token id");
        }
        packed_tokens.push_back(token);
      }
      suffix.last_rows[static_cast<std::size_t>(request)] =
          static_cast<int>(packed_tokens.size()) - 1;
    }
    if (packed_tokens.size() > static_cast<std::size_t>(state.maximum_context)) {
      throw std::runtime_error("layer-sliced suffix exceeds shared projection workspace");
    }
    suffix.packed_tokens = static_cast<int>(packed_tokens.size());
    suffix.fp8_projections = state.use_fp8_suffix(suffix.packed_tokens);
    check(cudaMemcpy(state.device_input_tokens.get(), packed_tokens.data(),
                     packed_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy layer-sliced suffix tokens");
    embedding_batch_bf16(state.weights.at("token_embedding"),
                         static_cast<const int *>(state.device_input_tokens.get()),
                         state.suffix_hidden_a.get(), suffix.packed_tokens, state.vocabulary,
                         state.hidden, std::sqrt(static_cast<float>(state.hidden)), nullptr);
    state.suffix_layer_state = std::move(suffix);
  } else {
    const auto &suffix = *state.suffix_layer_state;
    if (suffix.slots != slots || suffix.prompts != token_ids) {
      throw std::runtime_error("layer-sliced suffix cohort changed while in flight");
    }
    for (int request = 0; request < requests; ++request) {
      if (token_ids[static_cast<std::size_t>(request)] == nullptr ||
          token_ids[static_cast<std::size_t>(request)]->size() !=
              suffix.prompt_sizes[static_cast<std::size_t>(request)]) {
        throw std::runtime_error("layer-sliced suffix prompt changed while in flight");
      }
    }
  }

  auto &suffix = *state.suffix_layer_state;
  const int layer_count = static_cast<int>(state.config.layers.size());
  const int end_layer = std::min(layer_count, suffix.next_layer + maximum_layers);
  void *input =
      suffix.next_layer % 2 == 0 ? state.suffix_hidden_a.get() : state.suffix_hidden_b.get();
  void *output =
      suffix.next_layer % 2 == 0 ? state.suffix_hidden_b.get() : state.suffix_hidden_a.get();
  for (int layer = suffix.next_layer; layer < end_layer; ++layer) {
    std::vector<void *> key_caches;
    std::vector<void *> value_caches;
    std::vector<int> capacities(static_cast<std::size_t>(requests),
                                state.caches[static_cast<std::size_t>(layer)].capacity);
    key_caches.reserve(static_cast<std::size_t>(requests));
    value_caches.reserve(static_cast<std::size_t>(requests));
    for (const int slot : slots) {
      key_caches.push_back(state.slot_key_cache(static_cast<std::size_t>(layer), slot));
      value_caches.push_back(state.slot_value_cache(static_cast<std::size_t>(layer), slot));
    }
    state.layers.run_prefill_chunks(layer, input, output, suffix.counts, suffix.positions,
                                    key_caches, value_caches, capacities, nullptr,
                                    suffix.fp8_projections);
    for (int request = 0; request < requests; ++request) {
      state.refresh_global_cache_approximations(
          static_cast<std::size_t>(layer), slots[static_cast<std::size_t>(request)],
          suffix.positions[static_cast<std::size_t>(request)],
          suffix.counts[static_cast<std::size_t>(request)], nullptr);
    }
    std::swap(input, output);
  }
  suffix.next_layer = end_layer;
  if (end_layer != layer_count) {
    check(cudaDeviceSynchronize(), "synchronize layer-sliced suffix prefill");
    return {};
  }

  check(cudaMemcpy(state.device_positions.get(), suffix.last_rows.data(),
                   suffix.last_rows.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy layer-sliced suffix final rows");
  gather_rows_bf16(input, static_cast<const int *>(state.device_positions.get()), output, requests,
                   state.hidden, nullptr);
  rms_norm_bf16(output, state.weights.at("final_norm"), state.final_hidden.get(), requests,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  check(cudaMemcpy(state.device_slots.get(), slots.data(), slots.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy layer-sliced suffix slots");
  scatter_rows_bf16(state.final_hidden.get(), static_cast<const int *>(state.device_slots.get()),
                    state.target_hidden_slots.get(), requests, state.hidden, nullptr);
  if (suffix.fp8_projections) {
    state.run_decode_lm_head(state.final_hidden.get(), state.logits.get(), requests, nullptr);
  } else {
    state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                      state.logits.get(), requests, state.hidden, state.vocabulary, nullptr);
  }
  argmax_bf16_rows(state.logits.get(), requests, state.vocabulary,
                   static_cast<int *>(state.device_output_tokens.get()),
                   state.argmax_workspace.get(), nullptr);
  std::vector<int> predictions(static_cast<std::size_t>(requests));
  check(cudaMemcpy(predictions.data(), state.device_output_tokens.get(),
                   predictions.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy layer-sliced suffix predictions");
  for (int request = 0; request < requests; ++request) {
    state.slot_positions[static_cast<std::size_t>(slots[static_cast<std::size_t>(request)])] =
        static_cast<int>(suffix.prompt_sizes[static_cast<std::size_t>(request)]);
  }
  state.suffix_layer_state.reset();
  return LayerSlicedSuffixPrefillResult{true, std::move(predictions)};
}

void Gemma4BatchModelRunner::abort_suffix_layer_slices() {
  implementation_->suffix_layer_state.reset();
}

bool Gemma4BatchModelRunner::has_suffix_layer_slices() const {
  return implementation_->suffix_layer_state.has_value();
}

std::uint64_t Gemma4BatchModelRunner::clone_slot(int source_slot, int destination_slot) {
  auto &state = *implementation_;
  if (source_slot < 0 || source_slot >= state.batch_size || destination_slot < 0 ||
      destination_slot >= state.batch_size || source_slot == destination_slot) {
    throw std::runtime_error("invalid cache slot clone");
  }
  const int position = state.slot_positions.at(static_cast<std::size_t>(source_slot));
  if (position <= 0 || position >= state.maximum_context) {
    throw std::runtime_error("source cache slot is empty or full");
  }
  std::uint64_t copied_bytes = 0;
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    const auto &cache = state.caches[layer];
    const int resident_tokens = std::min(position, cache.capacity);
    const std::size_t bytes =
        2ULL * static_cast<std::size_t>(resident_tokens) * state.config.layers[layer].kv_width();
    clone_kv_cache_prefix_bf16(
        state.slot_key_cache(layer, source_slot), state.slot_value_cache(layer, source_slot),
        state.slot_key_cache(layer, destination_slot),
        state.slot_value_cache(layer, destination_slot),
        static_cast<int>(state.config.layers[layer].kv_heads), resident_tokens,
        static_cast<int>(state.config.layers[layer].head_dimension), cache.capacity, nullptr);
    state.refresh_int8_kv_span(layer, destination_slot, 0, resident_tokens, nullptr);
    state.refresh_fp8_kv_span(layer, destination_slot, 0, resident_tokens, nullptr);
    copied_bytes += 2ULL * bytes;
  }
  check(cudaMemcpy(static_cast<unsigned char *>(state.target_hidden_slots.get()) +
                       2ULL * destination_slot * state.hidden,
                   static_cast<unsigned char *>(state.target_hidden_slots.get()) +
                       2ULL * source_slot * state.hidden,
                   2ULL * state.hidden, cudaMemcpyDeviceToDevice),
        "clone slot target hidden state");
  check(cudaDeviceSynchronize(), "synchronize slot cache clone");
  state.slot_positions.at(static_cast<std::size_t>(destination_slot)) = position;
  return copied_bytes;
}

std::vector<int> Gemma4BatchModelRunner::append_ragged(const std::vector<int> &token_ids,
                                                       const std::vector<int> &slots) {
  auto &state = *implementation_;
  const int active = static_cast<int>(token_ids.size());
  if (active <= 0 || active > state.batch_size || slots.size() != token_ids.size()) {
    throw std::runtime_error("invalid ragged token batch");
  }
  std::vector<bool> seen(static_cast<std::size_t>(state.batch_size), false);
  std::vector<int> positions(static_cast<std::size_t>(active));
  int maximum_position = 0;
  for (int row = 0; row < active; ++row) {
    const int slot = slots[static_cast<std::size_t>(row)];
    const int token = token_ids[static_cast<std::size_t>(row)];
    if (slot < 0 || slot >= state.batch_size || seen[static_cast<std::size_t>(slot)] || token < 0 ||
        token >= state.vocabulary) {
      throw std::runtime_error("invalid or repeated ragged slot");
    }
    seen[static_cast<std::size_t>(slot)] = true;
    const int position = state.slot_positions[static_cast<std::size_t>(slot)];
    if (position <= 0 || position >= state.maximum_context) {
      throw std::runtime_error("ragged slot is empty or full");
    }
    positions[static_cast<std::size_t>(row)] = position;
    maximum_position = std::max(maximum_position, position);
  }
  check(cudaMemcpyAsync(state.device_input_tokens.get(), token_ids.data(),
                        static_cast<std::size_t>(active) * sizeof(int), cudaMemcpyHostToDevice,
                        state.decode_stream),
        "copy ragged input tokens");
  check(cudaMemcpyAsync(state.device_positions.get(), positions.data(),
                        static_cast<std::size_t>(active) * sizeof(int), cudaMemcpyHostToDevice,
                        state.decode_stream),
        "copy ragged positions");
  check(cudaMemcpyAsync(state.device_slots.get(), slots.data(),
                        static_cast<std::size_t>(active) * sizeof(int), cudaMemcpyHostToDevice,
                        state.decode_stream),
        "copy ragged slots");
  // A 256-token context bucket allows the captured launch graph to survive ordinary decode
  // steps. Per-row positions still mask every invalid score, so padding changes work, not math.
  constexpr int context_bucket = 256;
  const int maximum_context_length =
      std::min(state.maximum_context,
               ((maximum_position + 1 + context_bucket - 1) / context_bucket) * context_bucket);
  int contiguous_slot_start = slots.front();
  for (int row = 0; row < active; ++row) {
    if (slots[static_cast<std::size_t>(row)] != contiguous_slot_start + row) {
      contiguous_slot_start = -1;
      break;
    }
  }
  const auto graph_key = std::make_tuple(active, maximum_context_length, contiguous_slot_start);
  const auto existing = state.decode_graphs.find(graph_key);
  const bool needs_capture = existing == state.decode_graphs.end();
  if (needs_capture) {
    state.enqueue_ragged_decode(active, maximum_context_length, contiguous_slot_start);
  } else {
    check(cudaGraphLaunch(existing->second->executable, state.decode_stream),
          "launch ragged decode graph");
  }
  std::vector<int> result(static_cast<std::size_t>(active));
  check(cudaMemcpyAsync(result.data(), state.device_output_tokens.get(),
                        result.size() * sizeof(int), cudaMemcpyDeviceToHost, state.decode_stream),
        "copy ragged output tokens");
  check(cudaStreamSynchronize(state.decode_stream), "synchronize ragged decode");
  if (needs_capture) {
    static_cast<void>(
        state.capture_decode_graph(active, maximum_context_length, contiguous_slot_start));
  }
  for (const int slot : slots)
    ++state.slot_positions[static_cast<std::size_t>(slot)];
  return result;
}

GreedyVerificationResult Gemma4BatchModelRunner::verify_greedy_proposals(
    const std::vector<std::vector<int>> &proposals, const std::vector<int> &target_first_tokens,
    const std::vector<int> &slots, const std::vector<int> &maximum_accepted_tokens) {
  auto &state = *implementation_;
  const int requests = static_cast<int>(proposals.size());
  const int depth = requests == 0 ? 0 : static_cast<int>(proposals.front().size());
  if (state.fp8_weights == nullptr || requests <= 0 || requests > state.batch_size || depth <= 0 ||
      depth > speculative_depth || target_first_tokens.size() != proposals.size() ||
      slots.size() != proposals.size()) {
    throw std::runtime_error("invalid greedy verification batch");
  }
  if (!maximum_accepted_tokens.empty() && maximum_accepted_tokens.size() != proposals.size()) {
    throw std::runtime_error("speculative acceptance caps do not match request count");
  }
  std::vector<bool> seen(static_cast<std::size_t>(state.batch_size), false);
  std::vector<int> positions(static_cast<std::size_t>(requests));
  std::vector<int> counts(static_cast<std::size_t>(requests), depth);
  std::vector<int> capacities(static_cast<std::size_t>(requests));
  std::vector<int> packed_tokens;
  packed_tokens.reserve(static_cast<std::size_t>(requests * depth));
  for (int request = 0; request < requests; ++request) {
    const int slot = slots[static_cast<std::size_t>(request)];
    if (slot < 0 || slot >= state.batch_size || seen[static_cast<std::size_t>(slot)] ||
        static_cast<int>(proposals[static_cast<std::size_t>(request)].size()) != depth) {
      throw std::runtime_error("invalid or repeated speculative slot");
    }
    seen[static_cast<std::size_t>(slot)] = true;
    const int position = state.slot_positions[static_cast<std::size_t>(slot)];
    if (position <= 0 || position + depth > state.maximum_context) {
      throw std::runtime_error("speculative slot is empty or lacks context capacity");
    }
    positions[static_cast<std::size_t>(request)] = position;
    for (const int token : proposals[static_cast<std::size_t>(request)]) {
      if (token < 0 || token >= state.vocabulary) {
        throw std::runtime_error("speculative proposal contains an invalid token");
      }
      packed_tokens.push_back(token);
    }
  }

  check(cudaMemcpy(state.device_input_tokens.get(), packed_tokens.data(),
                   packed_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy speculative proposal tokens");
  check(cudaMemcpy(state.device_positions.get(), positions.data(), positions.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy speculative positions");
  embedding_batch_bf16(state.weights.at("token_embedding"),
                       static_cast<const int *>(state.device_input_tokens.get()),
                       state.hidden_a.get(), static_cast<int>(packed_tokens.size()),
                       state.vocabulary, state.hidden, std::sqrt(static_cast<float>(state.hidden)),
                       nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  std::vector<void *> key_caches(static_cast<std::size_t>(requests));
  std::vector<void *> value_caches(static_cast<std::size_t>(requests));
  const std::size_t pointers_per_layer = 2ULL * static_cast<std::size_t>(requests);
  std::vector<void *> layer_cache_pointers(state.config.layers.size() * pointers_per_layer);
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    auto *layer_pointers = layer_cache_pointers.data() + layer * pointers_per_layer;
    for (int request = 0; request < requests; ++request) {
      const int slot = slots[static_cast<std::size_t>(request)];
      layer_pointers[request] = state.slot_key_cache(layer, slot);
      layer_pointers[requests + request] = state.slot_value_cache(layer, slot);
    }
  }
  check(cudaMemcpy(state.device_cache_pointers.get(), layer_cache_pointers.data(),
                   layer_cache_pointers.size() * sizeof(void *), cudaMemcpyHostToDevice),
        "copy speculative cache pointer table");
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    std::fill(capacities.begin(), capacities.end(), state.caches[layer].capacity);
    auto *layer_pointers = layer_cache_pointers.data() + layer * pointers_per_layer;
    for (int request = 0; request < requests; ++request) {
      key_caches[static_cast<std::size_t>(request)] = layer_pointers[request];
      value_caches[static_cast<std::size_t>(request)] = layer_pointers[requests + request];
    }
    if (state.config.layers[layer].attention == AttentionKind::sliding) {
      auto **pointer_base =
          static_cast<void **>(state.device_cache_pointers.get()) + layer * pointers_per_layer;
      backup_kv_ring_spans_bf16(reinterpret_cast<const void *const *>(pointer_base),
                                reinterpret_cast<const void *const *>(pointer_base + requests),
                                state.speculative_key_backup(layer),
                                state.speculative_value_backup(layer),
                                static_cast<const int *>(state.device_positions.get()), requests,
                                static_cast<int>(state.config.layers[layer].kv_heads), depth,
                                static_cast<int>(state.config.layers[layer].head_dimension),
                                state.caches[layer].capacity, nullptr);
    }
    state.layers.run_prefill_chunks(static_cast<int>(layer), input, output, counts, positions,
                                    key_caches, value_caches, capacities, nullptr, true,
                                    state.config.layers[layer].attention == AttentionKind::sliding
                                        ? state.speculative_key_backup(layer)
                                        : nullptr,
                                    state.config.layers[layer].attention == AttentionKind::sliding
                                        ? state.speculative_value_backup(layer)
                                        : nullptr,
                                    static_cast<const int *>(state.device_positions.get()),
                                    static_cast<void **>(state.device_cache_pointers.get()) +
                                        layer * pointers_per_layer);
    for (int request = 0; request < requests; ++request) {
      state.refresh_global_cache_approximations(layer, slots[static_cast<std::size_t>(request)],
                                                positions[static_cast<std::size_t>(request)],
                                                counts[static_cast<std::size_t>(request)], nullptr);
    }
    std::swap(input, output);
  }

  const int rows = requests * depth;
  rms_norm_bf16(input, state.weights.at("final_norm"), state.final_hidden.get(), rows, state.hidden,
                static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  state.run_decode_lm_head(state.final_hidden.get(), state.logits.get(), rows, nullptr);
  argmax_bf16_rows(state.logits.get(), rows, state.vocabulary,
                   static_cast<int *>(state.device_output_tokens.get()),
                   state.argmax_workspace.get(), nullptr);
  std::vector<int> target_predictions(static_cast<std::size_t>(rows));
  check(cudaMemcpy(target_predictions.data(), state.device_output_tokens.get(),
                   target_predictions.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy speculative target predictions");

  GreedyVerificationResult result;
  result.accepted_tokens.resize(static_cast<std::size_t>(requests));
  result.next_tokens.resize(static_cast<std::size_t>(requests));
  for (int request = 0; request < requests; ++request) {
    int accepted = 0;
    const int row_start = request * depth;
    const int acceptance_cap =
        maximum_accepted_tokens.empty()
            ? depth
            : std::clamp(maximum_accepted_tokens[static_cast<std::size_t>(request)], 0, depth);
    while (accepted < acceptance_cap) {
      const int expected =
          accepted == 0 ? target_first_tokens[static_cast<std::size_t>(request)]
                        : target_predictions[static_cast<std::size_t>(row_start + accepted - 1)];
      if (proposals[static_cast<std::size_t>(request)][static_cast<std::size_t>(accepted)] !=
          expected) {
        break;
      }
      ++accepted;
    }
    result.accepted_tokens[static_cast<std::size_t>(request)] = accepted;
    result.next_tokens[static_cast<std::size_t>(request)] =
        accepted == 0 ? target_first_tokens[static_cast<std::size_t>(request)]
                      : target_predictions[static_cast<std::size_t>(row_start + accepted - 1)];
  }

  std::vector<int> hidden_rows;
  std::vector<int> hidden_slots;
  for (int request = 0; request < requests; ++request) {
    const int accepted = result.accepted_tokens[static_cast<std::size_t>(request)];
    if (accepted == 0)
      continue;
    hidden_rows.push_back(request * depth + accepted - 1);
    hidden_slots.push_back(slots[static_cast<std::size_t>(request)]);
  }
  if (!hidden_rows.empty()) {
    check(cudaMemcpy(state.device_positions.get(), hidden_rows.data(),
                     hidden_rows.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy accepted hidden rows");
    check(cudaMemcpy(state.device_slots.get(), hidden_slots.data(),
                     hidden_slots.size() * sizeof(int), cudaMemcpyHostToDevice),
          "copy accepted hidden slots");
    gather_rows_bf16(state.final_hidden.get(),
                     static_cast<const int *>(state.device_positions.get()), state.hidden_a.get(),
                     static_cast<int>(hidden_rows.size()), state.hidden, nullptr);
    scatter_rows_bf16(state.hidden_a.get(), static_cast<const int *>(state.device_slots.get()),
                      state.target_hidden_slots.get(), static_cast<int>(hidden_rows.size()),
                      state.hidden, nullptr);
  }

  check(cudaMemcpy(state.device_positions.get(), positions.data(), positions.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "restore speculative positions for rollback");
  check(cudaMemcpy(state.device_slots.get(), result.accepted_tokens.data(),
                   result.accepted_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
        "copy speculative accepted lengths");
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    if (state.config.layers[layer].attention != AttentionKind::sliding)
      continue;
    auto **pointer_base =
        static_cast<void **>(state.device_cache_pointers.get()) + layer * pointers_per_layer;
    restore_kv_ring_suffixes_bf16(
        state.speculative_key_backup(layer), state.speculative_value_backup(layer), pointer_base,
        pointer_base + requests, static_cast<const int *>(state.device_positions.get()),
        static_cast<const int *>(state.device_slots.get()), requests,
        static_cast<int>(state.config.layers[layer].kv_heads), depth,
        static_cast<int>(state.config.layers[layer].head_dimension), state.caches[layer].capacity,
        nullptr);
  }
  check(cudaDeviceSynchronize(), "synchronize speculative verification");
  for (int request = 0; request < requests; ++request) {
    state.slot_positions[static_cast<std::size_t>(slots[static_cast<std::size_t>(request)])] +=
        result.accepted_tokens[static_cast<std::size_t>(request)];
  }
  return result;
}

std::vector<std::vector<int>>
Gemma4BatchModelRunner::draft_four(const std::vector<int> &last_tokens,
                                   const std::vector<int> &slots,
                                   const std::vector<int> &forced_first_tokens, int depth) {
  auto &state = *implementation_;
  if (!state.assistant || last_tokens.size() != slots.size() || last_tokens.empty()) {
    throw std::runtime_error("assistant is not configured or draft batch is invalid");
  }
  std::vector<int> context_positions;
  context_positions.reserve(slots.size());
  for (const int slot : slots) {
    if (slot < 0 || slot >= state.batch_size ||
        state.slot_positions[static_cast<std::size_t>(slot)] <= 0) {
      throw std::runtime_error("assistant slot is empty or invalid");
    }
    context_positions.push_back(state.slot_positions[static_cast<std::size_t>(slot)] - 1);
  }
  const std::size_t sliding = static_cast<std::size_t>(state.shared_sliding_layer);
  const std::size_t global = static_cast<std::size_t>(state.shared_global_layer);
  return state.assistant->draft_four(
      state.weights.at("token_embedding"), state.target_hidden_slots.get(), last_tokens,
      context_positions, slots, forced_first_tokens, state.batch_size, depth,
      state.key_cache(sliding), state.value_cache(sliding), state.caches[sliding].capacity,
      state.key_cache(global), state.value_cache(global), state.caches[global].capacity);
}

void Gemma4BatchModelRunner::reset_slot(int slot) {
  auto &state = *implementation_;
  if (slot < 0 || slot >= state.batch_size)
    throw std::runtime_error("invalid cache slot");
  state.slot_positions[static_cast<std::size_t>(slot)] = 0;
}

void Gemma4BatchModelRunner::seed_empty_cache_for_benchmark(int position) {
  auto &state = *implementation_;
  if (position < 0 || position >= state.maximum_context) {
    throw std::runtime_error("invalid benchmark cache position");
  }
  check(cudaMemset(state.cache_arena.get(), 0,
                   static_cast<std::size_t>(2ULL * state.cache_payload_bytes)),
        "zero benchmark KV cache");
  if (state.use_fp8_global_k) {
    check(cudaMemset(state.fp8_key_arena.get(), 0,
                     static_cast<std::size_t>(state.fp8_key_payload_bytes)),
          "zero benchmark FP8 K cache");
  }
  if (state.use_fp8_global_kv) {
    check(cudaMemset(state.fp8_value_arena.get(), 0,
                     static_cast<std::size_t>(state.fp8_key_payload_bytes)),
          "zero benchmark FP8 V cache");
  }
  if (state.use_int8_global_k) {
    check(cudaMemset(state.int8_key_arena.get(), 0,
                     static_cast<std::size_t>(state.int8_key_payload_bytes)),
          "zero benchmark INT8 K cache");
    check(cudaMemset(state.int8_key_scale_arena.get(), 0,
                     static_cast<std::size_t>(state.int8_key_scale_payload_bytes)),
          "zero benchmark INT8 K scale cache");
  }
  if (state.use_int8_global_kv) {
    check(cudaMemset(state.int8_value_arena.get(), 0,
                     static_cast<std::size_t>(state.int8_key_payload_bytes)),
          "zero benchmark INT8 V cache");
    check(cudaMemset(state.int8_value_scale_arena.get(), 0,
                     static_cast<std::size_t>(state.int8_value_scale_payload_bytes)),
          "zero benchmark INT8 V scale cache");
  }
  state.current_position = position;
  std::fill(state.slot_positions.begin(), state.slot_positions.end(), position);
}

int Gemma4BatchModelRunner::batch() const {
  return implementation_->batch_size;
}
bool Gemma4BatchModelRunner::has_assistant() const {
  return implementation_->assistant != nullptr;
}
int Gemma4BatchModelRunner::position() const {
  return implementation_->current_position;
}
int Gemma4BatchModelRunner::slot_position(int slot) const {
  if (slot < 0 || slot >= implementation_->batch_size)
    throw std::runtime_error("invalid cache slot");
  return implementation_->slot_positions[static_cast<std::size_t>(slot)];
}

} // namespace carat
