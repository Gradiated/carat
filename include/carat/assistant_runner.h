#pragma once

#include <memory>
#include <vector>

namespace carat {

// Four-layer Gemma 4 MTP drafter. It consumes the target's final hidden state and the last
// sliding/global KV-sharing layers directly, avoiding a second KV cache and all framework
// tensor materialization.
class Gemma4AssistantRunner {
public:
  Gemma4AssistantRunner(const class DeviceWeightArena &assistant_weights,
                        const class Fp8WeightArena *fp8_weights, int maximum_batch,
                        int maximum_context);
  ~Gemma4AssistantRunner();
  Gemma4AssistantRunner(const Gemma4AssistantRunner &) = delete;
  Gemma4AssistantRunner &operator=(const Gemma4AssistantRunner &) = delete;

  std::vector<std::vector<int>>
  draft_four(const void *target_embedding_weight, const void *target_hidden_by_slot,
             const std::vector<int> &last_tokens, const std::vector<int> &context_positions,
             const std::vector<int> &slots, const std::vector<int> &forced_first_tokens,
             int maximum_slots, int depth, const void *sliding_keys, const void *sliding_values,
             int sliding_capacity, const void *global_keys, const void *global_values,
             int global_capacity);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
