#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace carat {

struct GreedyVerificationResult {
  // Number of proposed tokens accepted for each request.  The returned next_tokens are the
  // target replacement at the first rejection, or the bonus token when every proposal matched.
  std::vector<int> accepted_tokens;
  std::vector<int> next_tokens;
};

struct LayerSlicedSuffixPrefillResult {
  bool complete{false};
  std::vector<int> predictions;
};

class Gemma4BatchModelRunner {
public:
  Gemma4BatchModelRunner(const struct Gemma4Config &config, const class DeviceWeightArena &weights,
                         int batch, int maximum_context,
                         const class Fp8WeightArena *fp8_weights = nullptr,
                         const class DeviceWeightArena *assistant_weights = nullptr,
                         const class Fp8WeightArena *assistant_fp8_weights = nullptr);
  ~Gemma4BatchModelRunner();
  Gemma4BatchModelRunner(const Gemma4BatchModelRunner &) = delete;
  Gemma4BatchModelRunner &operator=(const Gemma4BatchModelRunner &) = delete;

  std::vector<int> append(const std::vector<int> &token_ids);
  int prefill_slot(int slot, const std::vector<int> &token_ids, int chunk_tokens = 1024);
  // The first cached_prefix_tokens of token_ids must already be resident in the slot.
  int prefill_slot_from_prefix(int slot, const std::vector<int> &token_ids,
                               int cached_prefix_tokens, int chunk_tokens = 1024);
  // Returns the greedy next token only when the complete prompt is resident.
  std::optional<int> prefill_slot_chunk(int slot, const std::vector<int> &token_ids,
                                        int chunk_tokens);
  // An entry holds a prediction only when that request's complete prompt became resident.
  std::vector<std::optional<int>>
  prefill_slots_chunks(const std::vector<int> &slots,
                       const std::vector<const std::vector<int> *> &token_ids, int chunk_tokens);
  std::vector<int> prefill_slots_suffix(const std::vector<int> &slots,
                                        const std::vector<const std::vector<int> *> &token_ids);
  // Hidden states survive intervening decode waves, so a preempted cohort never repeats a layer.
  LayerSlicedSuffixPrefillResult
  prefill_slots_suffix_layer_slice(const std::vector<int> &slots,
                                   const std::vector<const std::vector<int> *> &token_ids,
                                   int maximum_layers);
  void abort_suffix_layer_slices();
  [[nodiscard]] bool has_suffix_layer_slices() const;
  std::uint64_t clone_slot(int source_slot, int destination_slot);
  std::vector<int> append_ragged(const std::vector<int> &token_ids, const std::vector<int> &slots);
  // Commits cache positions only through each accepted prefix. Rejected sliding-ring rows are
  // restored exactly, and stale global-cache tail rows stay unreachable.
  GreedyVerificationResult verify_greedy_proposals(
      const std::vector<std::vector<int>> &proposals, const std::vector<int> &target_first_tokens,
      const std::vector<int> &slots, const std::vector<int> &maximum_accepted_tokens = {});
  [[nodiscard]] bool has_assistant() const;
  std::vector<std::vector<int>> draft_four(const std::vector<int> &last_tokens,
                                           const std::vector<int> &slots,
                                           const std::vector<int> &forced_first_tokens = {},
                                           int depth = 4);
  void reset_slot(int slot);
  // Zero-fills every allocated KV page and advances logical time. Benchmarks and the runtime
  // warmup use it to build decode plans before any request arrives.
  void seed_empty_cache_for_benchmark(int position);
  [[nodiscard]] int batch() const;
  [[nodiscard]] int position() const;
  [[nodiscard]] int slot_position(int slot) const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
