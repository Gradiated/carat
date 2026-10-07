#include "runtime/speculation.h"

#include <algorithm>

namespace carat {

int acceptance_cap(int depth, int remaining, std::span<const int> proposals,
                   const std::unordered_set<int> &stop_ids) {
  const int cap = std::min(depth, remaining);
  const int inspected = std::min(cap, static_cast<int>(proposals.size()));

  for (int index = 1; index < inspected; ++index) {
    if (stop_ids.contains(proposals[static_cast<std::size_t>(index)]))
      return index + 1;
  }

  return cap;
}

SpeculativeCommit commit_speculation(int forced_token, std::span<const int> accepted_drafts,
                                     int next_token, const std::unordered_set<int> &stop_ids) {
  SpeculativeCommit commit{
      .resident_tokens = {forced_token}, .emitted_tokens = {}, .next_token = next_token};

  for (const int token : accepted_drafts) {
    commit.resident_tokens.push_back(token);
    commit.emitted_tokens.push_back(token);
    if (stop_ids.contains(token)) {
      commit.stopped = true;
      return commit;
    }
  }

  commit.emitted_tokens.push_back(next_token);
  return commit;
}

} // namespace carat
