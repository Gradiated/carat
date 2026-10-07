#pragma once

#include <span>
#include <unordered_set>
#include <vector>

namespace carat {

// Caps verification at min(depth, remaining) and after the first proposed stop token. Index 0
// is the target's forced token, so only later proposals can end the cap early.
[[nodiscard]] int acceptance_cap(int depth, int remaining, std::span<const int> proposals,
                                 const std::unordered_set<int> &stop_ids);

struct SpeculativeCommit {
  std::vector<int> resident_tokens;
  std::vector<int> emitted_tokens;
  int next_token{0};
  bool stopped{false};
};

// The forced token was emitted in an earlier step and becomes resident now. After an accepted stop
// token, next_token is the prediction that follows the stop: it is kept for prefix reuse but not
// emitted.
[[nodiscard]] SpeculativeCommit commit_speculation(int forced_token,
                                                   std::span<const int> accepted_drafts,
                                                   int next_token,
                                                   const std::unordered_set<int> &stop_ids);

} // namespace carat
