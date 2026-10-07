#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace carat {

enum class AttentionKind { sliding, global };

struct Gemma4LayerConfig {
  AttentionKind attention;
  std::uint32_t query_heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dimension;
  std::uint32_t rotary_dimension;
  double rope_theta;
  bool key_equals_value;

  [[nodiscard]] std::uint64_t query_width() const;
  [[nodiscard]] std::uint64_t kv_width() const;
};

struct Gemma4Config {
  std::uint32_t hidden_size;
  std::uint32_t intermediate_size;
  std::uint32_t vocabulary_size;
  std::uint32_t sliding_window;
  std::uint32_t maximum_positions;
  double rms_norm_epsilon;
  double final_logit_softcap;
  std::vector<int> eos_token_ids;
  std::vector<Gemma4LayerConfig> layers;

  static Gemma4Config load(const std::string &config_path);
  [[nodiscard]] std::uint64_t text_parameter_count() const;
  [[nodiscard]] std::uint64_t full_kv_bytes_per_token() const;
  [[nodiscard]] std::uint64_t resident_sliding_kv_bytes_per_sequence() const;
};

// The CUDA kernels hard-code these values rather than reading them from the config. The final
// logit softcap is not checked: greedy argmax is invariant under the monotonic tanh softcap.
void require_native_gemma4_target(const Gemma4Config &config);

} // namespace carat
