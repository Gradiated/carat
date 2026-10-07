#pragma once

#include "runtime/slot_policy.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace carat {

using EnvironmentLookup = std::function<std::optional<std::string>(std::string_view name)>;

struct SchedulerOptions {
  int prefill_quantum_tokens{1024};
  int idle_prefill_quantum_tokens{1024};
  int short_prefill_batch_window_microseconds{10000};
  // Zero runs a suffix cohort through every layer at once. A positive value yields to decode only
  // at layer boundaries, so no prefill layer runs twice.
  int interleaved_suffix_layer_quantum{10};
  int interleaved_suffix_layer_minimum_decode{14};
  int speculative_depth{4};
  int speculative_minimum_batch{8};
  AdmissionOptions admission{.max_bypasses = 4, .idle_cache_reserve_slots = 0};
};

struct Int4LmHeadOracle {
  int block_width{128};
};

struct Fp8DecodeSettings {
  float weight_scale_multiplier{1.0F};
  std::optional<Int4LmHeadOracle> int4_lm_head_oracle;
};

enum class AssistantFp8Mode { off, lm_head, all };

struct AssistantSettings {
  std::string model_directory;
  AssistantFp8Mode fp8_mode{AssistantFp8Mode::off};
};

struct RuntimeSettings {
  std::string bind_host{"0.0.0.0"};
  std::uint16_t bind_port{30000};
  std::optional<std::string> api_key;
  std::optional<Fp8DecodeSettings> fp8_decode;
  std::optional<AssistantSettings> assistant;
  SchedulerOptions scheduler;
  bool ignore_model_eos{false};
};

// An empty variable counts as unset, except CARAT_ASSISTANT_FP8_MODE, which the CUDA library
// also reads. Every invalid value throws with the variable name.
[[nodiscard]] RuntimeSettings load_runtime_settings(const EnvironmentLookup &environment,
                                                    int model_layers);

} // namespace carat
