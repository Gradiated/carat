#pragma once

#include "runtime/capacity.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace carat {

struct RuntimeSettings;

using Counter = std::atomic<std::uint64_t>;

struct RuntimeMetrics {
  Counter requests{0};
  Counter invalid_requests{0};
  Counter failures{0};
  Counter cancelled{0};
  Counter input_tokens{0};
  Counter output_tokens{0};
  Counter inference_microseconds{0};
  Counter queued{0};
  Counter active{0};
  Counter decoding{0};
  Counter prefix_cache_hits{0};
  Counter prefix_tokens_reused{0};
  Counter cache_affinity_admissions{0};
  Counter cache_affinity_bypassed_requests{0};
  Counter prefill_tokens_computed{0};
  Counter kv_cache_clones{0};
  Counter kv_cache_clone_bytes{0};
  Counter suffix_prefill_batches{0};
  Counter interleaved_suffix_prefill_batches{0};
  Counter suffix_prefill_requests{0};
  Counter suffix_prefill_tokens{0};
  Counter suffix_prefill_microseconds{0};
  Counter maximum_suffix_prefill_microseconds{0};
  Counter suffix_prefill_layer_slices{0};
  Counter speculative_cycles{0};
  Counter speculative_proposed_tokens{0};
  Counter speculative_accepted_tokens{0};
  Counter speculative_first_token_rejections{0};
  Counter speculative_draft_microseconds{0};
  Counter speculative_verify_microseconds{0};
  Counter contiguous_decode_steps{0};
  Counter fragmented_decode_steps{0};
  std::array<Counter, maximum_slots + 1> decode_batch_steps{};
  std::array<Counter, maximum_slots + 1> suffix_prefill_batch_calls{};
};

void update_maximum(Counter &target, std::uint64_t value);

[[nodiscard]] std::string render_metrics(const RuntimeMetrics &metrics,
                                         const RuntimeSettings &settings);

} // namespace carat
