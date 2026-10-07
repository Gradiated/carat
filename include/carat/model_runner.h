#pragma once

#include <memory>
#include <vector>

namespace carat {

class Gemma4ModelRunner {
public:
  Gemma4ModelRunner(const struct Gemma4Config &config, const class DeviceWeightArena &weights,
                    int maximum_context);
  ~Gemma4ModelRunner();
  Gemma4ModelRunner(const Gemma4ModelRunner &) = delete;
  Gemma4ModelRunner &operator=(const Gemma4ModelRunner &) = delete;

  int append(int token_id);
  int prefill(const std::vector<int> &token_ids);
  int prefill_chunked(const std::vector<int> &token_ids, int chunk_tokens);
  // Reuses plans and allocations for a new sequence. Prefill overwrites every cache entry it reads.
  void reset_sequence();
  [[nodiscard]] int position() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
