#include "runtime/speculation.h"

#include "support/test_support.h"

#include <unordered_set>
#include <vector>

namespace {

using carat::acceptance_cap;
using carat::commit_speculation;

const std::vector<int> proposals{10, 11, 12, 13};

ENGINE_TEST(acceptance_cap_is_bounded_by_depth_and_remaining_tokens) {
  CHECK(acceptance_cap(4, 10, proposals, {}) == 4);
  CHECK(acceptance_cap(4, 2, proposals, {}) == 2);
  CHECK(acceptance_cap(4, 0, proposals, {}) == 0);
}

ENGINE_TEST(acceptance_cap_ends_after_the_first_proposed_stop_token) {
  CHECK(acceptance_cap(4, 10, proposals, {12}) == 3);
  CHECK(acceptance_cap(4, 10, proposals, {13, 11}) == 2);
}

ENGINE_TEST(acceptance_cap_ignores_stops_at_index_zero_and_beyond_the_cap) {
  CHECK(acceptance_cap(4, 10, proposals, {10}) == 4);
  CHECK(acceptance_cap(4, 2, proposals, {12}) == 2);
}

ENGINE_TEST(speculative_commit_emits_every_accepted_draft_and_the_verified_next_token) {
  const std::vector<int> drafts{11, 12};

  const auto commit = commit_speculation(10, drafts, 20, {99});

  CHECK(commit.resident_tokens == std::vector<int>({10, 11, 12}));
  CHECK(commit.emitted_tokens == std::vector<int>({11, 12, 20}));
  CHECK(commit.next_token == 20);
  CHECK(!commit.stopped);
}

ENGINE_TEST(speculative_commit_without_accepted_drafts_emits_only_the_next_token) {
  const auto commit = commit_speculation(10, {}, 20, {});

  CHECK(commit.resident_tokens == std::vector<int>({10}));
  CHECK(commit.emitted_tokens == std::vector<int>({20}));
  CHECK(commit.next_token == 20);
  CHECK(!commit.stopped);
}

ENGINE_TEST(speculative_commit_after_a_stop_keeps_the_verified_next_token_unemitted) {
  const std::vector<int> drafts{11, 12};

  const auto commit = commit_speculation(10, drafts, 20, {12});

  CHECK(commit.resident_tokens == std::vector<int>({10, 11, 12}));
  CHECK(commit.emitted_tokens == std::vector<int>({11, 12}));
  CHECK(commit.next_token == 20);
  CHECK(commit.stopped);
}

} // namespace
