#include "runtime/slot_policy.h"

#include "support/test_support.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <variant>
#include <vector>

namespace {

using carat::Admission;
using carat::AdmissionOptions;
using carat::charge_bypass;
using carat::NoFreeSlot;
using carat::PendingView;
using carat::Placement;
using carat::PlacementDecision;
using carat::ReserveBlocked;
using carat::select_admission;
using carat::select_placement;
using carat::SlotView;

struct SlotState {
  std::vector<int> cached_tokens;
  bool occupied{false};
  std::uint64_t age{0};
  bool prefilling{false};
};

struct PendingState {
  std::vector<int> input_ids;
  int priority{0};
  std::uint32_t bypasses{0};
};

std::vector<SlotView> slot_views(const std::vector<SlotState> &slots) {
  std::vector<SlotView> views;
  for (const SlotState &slot : slots) {
    views.push_back({slot.cached_tokens, slot.occupied, slot.age, slot.prefilling});
  }
  return views;
}

template <typename Container> std::vector<PendingView> pending_views(const Container &pending) {
  std::vector<PendingView> views;
  for (const PendingState &job : pending)
    views.push_back({job.input_ids, job.priority, job.bypasses});
  return views;
}

template <typename Container>
void admit(Container &pending, const Admission &admission, std::uint32_t max_bypasses) {
  for (const std::size_t index : admission.bypassed) {
    pending[index].bypasses = charge_bypass(pending[index].bypasses, max_bypasses);
  }

  pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(admission.pending_index));
}

PlacementDecision place(const std::vector<int> &input_ids, const std::vector<SlotState> &slots,
                        std::size_t idle_cache_reserve_slots = 0) {
  return select_placement(input_ids, slot_views(slots), idle_cache_reserve_slots);
}

Placement placed(const PlacementDecision &decision) {
  const auto *placement = std::get_if<Placement>(&decision);
  REQUIRE(placement != nullptr);
  return *placement;
}

ENGINE_TEST(placement_never_counts_an_empty_cache_as_a_hit) {
  const auto placement = placed(place({1, 2}, {{{}, false, 0}, {{9}, false, 0}}));

  CHECK(placement.destination == 0);
  CHECK(!placement.source);
  CHECK(placement.prefix_tokens == 0);
}

ENGINE_TEST(placement_reuses_the_longest_cached_prefix) {
  const auto placement =
      placed(place({1, 2, 3}, {{{1}, false, 0}, {{1, 2}, false, 0}, {{1, 2, 3, 4}, false, 0}}));

  CHECK(placement.destination == 1);
  CHECK(placement.source == 1);
  CHECK(placement.prefix_tokens == 2);
}

ENGINE_TEST(placement_prefers_an_idle_source_on_an_equal_prefix) {
  const auto idle_second = placed(place({1, 2, 3}, {{{1, 2}, true, 0}, {{1, 2}, false, 0}}));
  CHECK(idle_second.destination == 1);
  CHECK(idle_second.source == 1);

  const auto idle_first = placed(place({1, 2, 3}, {{{1, 2}, false, 0}, {{1, 2}, true, 0}}));
  CHECK(idle_first.destination == 0);
  CHECK(idle_first.source == 0);
}

ENGINE_TEST(placement_clones_an_occupied_source_into_a_free_slot) {
  const auto placement =
      placed(place({1, 2, 3}, {{{7}, false, 1}, {{1, 2}, true, 0}, {{}, false, 0}}));

  CHECK(placement.destination == 2);
  CHECK(placement.source == 1);
  CHECK(placement.prefix_tokens == 2);
}

ENGINE_TEST(placement_evicts_the_least_recently_used_idle_slot) {
  const auto placement =
      placed(place({1}, {{{5}, true, 0}, {{6}, false, 9}, {{7}, false, 3}, {{8}, false, 3}}));

  CHECK(placement.destination == 2);
  CHECK(!placement.source);
}

ENGINE_TEST(placement_never_reuses_a_prefilling_slot_that_covers_the_whole_input) {
  const std::vector<SlotState> slots{
      {{1, 2}, false, 0}, {{1, 2, 3}, true, 0, true}, {{}, false, 0}};

  const auto placement = placed(place({1, 2, 3}, slots));

  CHECK(placement.destination == 0);
  CHECK(placement.source == 0);
  CHECK(placement.prefix_tokens == 2);
}

ENGINE_TEST(placement_reuses_a_partial_prefix_of_a_prefilling_slot) {
  const std::vector<SlotState> slots{{{1, 2}, true, 0, true}, {{}, false, 0}};

  const auto placement = placed(place({1, 2, 3}, slots));

  CHECK(placement.destination == 1);
  CHECK(placement.source == 0);
  CHECK(placement.prefix_tokens == 2);
}

ENGINE_TEST(placement_reuses_a_decoding_slot_that_covers_the_whole_input) {
  const auto placement = placed(place({1, 2, 3}, {{{1, 2, 3}, true, 0, false}, {{}, false, 0}}));

  CHECK(placement.destination == 1);
  CHECK(placement.source == 0);
  CHECK(placement.prefix_tokens == 3);
}

ENGINE_TEST(placement_reports_no_free_slot_when_every_slot_is_occupied) {
  CHECK(std::holds_alternative<NoFreeSlot>(place({1}, {{{}, true, 0}, {{1}, true, 0}}, 1)));
}

ENGINE_TEST(placement_reserve_blocks_eviction_only_while_a_slot_is_active) {
  const std::vector<SlotState> one_idle_cache{{{5, 5, 5}, true, 0}, {{6, 6}, false, 1}};
  CHECK(std::holds_alternative<ReserveBlocked>(place({9}, one_idle_cache, 1)));
  CHECK(placed(place({9}, one_idle_cache, 0)).destination == 1);

  const std::vector<SlotState> nothing_active{{{6, 6}, false, 1}};
  CHECK(placed(place({9}, nothing_active, 1)).destination == 0);

  const std::vector<SlotState> two_idle_caches{
      {{5}, true, 0}, {{6, 6}, false, 1}, {{7, 7}, false, 2}};
  CHECK(std::holds_alternative<ReserveBlocked>(place({9}, two_idle_caches, 2)));
  CHECK(placed(place({9}, two_idle_caches, 1)).destination == 1);
}

ENGINE_TEST(placement_reserve_allows_eviction_of_a_shorter_cache_than_it_reuses) {
  const std::vector<SlotState> slots{{{5, 5}, true, 0}, {{6, 6}, false, 1}};

  const auto placement = placed(place({5, 5, 1}, slots, 1));

  CHECK(placement.destination == 1);
  CHECK(placement.source == 0);
  CHECK(placement.prefix_tokens == 2);
  CHECK(std::holds_alternative<ReserveBlocked>(
      place({5, 1}, {{{5}, true, 0}, {{6, 6}, false, 1}}, 1)));
}

ENGINE_TEST(placement_properties_hold_for_random_slot_states) {
  std::mt19937 random(20260923);
  std::uniform_int_distribution<int> token(1, 3);
  std::uniform_int_distribution<int> length(0, 4);
  std::uniform_int_distribution<int> slot_count(1, 5);
  std::bernoulli_distribution occupied(0.5);
  std::bernoulli_distribution prefilling(0.3);
  std::uniform_int_distribution<std::uint64_t> age(0, 4);
  std::uniform_int_distribution<std::size_t> reserve(0, 3);

  const auto tokens = [&] {
    std::vector<int> result(static_cast<std::size_t>(length(random)));
    for (int &value : result)
      value = token(random);
    return result;
  };

  for (int iteration = 0; iteration < 2000; ++iteration) {
    std::vector<SlotState> slots(static_cast<std::size_t>(slot_count(random)));
    for (SlotState &slot : slots) {
      slot = {tokens(), occupied(random), age(random)};
      slot.prefilling = slot.occupied && prefilling(random);
    }
    const std::vector<int> input = tokens();
    const std::size_t idle_cache_reserve_slots = reserve(random);

    const auto reuses = [&](const SlotState &slot) {
      const bool whole_input_without_prediction =
          slot.prefilling && slot.cached_tokens.size() == input.size();
      return !slot.cached_tokens.empty() && slot.cached_tokens.size() <= input.size() &&
             !whole_input_without_prediction &&
             std::equal(slot.cached_tokens.begin(), slot.cached_tokens.end(), input.begin());
    };
    std::size_t longest = 0;
    for (const SlotState &slot : slots) {
      if (reuses(slot))
        longest = std::max(longest, slot.cached_tokens.size());
    }
    const bool all_occupied = std::all_of(slots.begin(), slots.end(),
                                          [](const SlotState &slot) { return slot.occupied; });

    const auto decision = place(input, slots, idle_cache_reserve_slots);

    CHECK(std::holds_alternative<NoFreeSlot>(decision) == all_occupied);
    if (const auto *placement = std::get_if<Placement>(&decision)) {
      CHECK(!slots[static_cast<std::size_t>(placement->destination)].occupied);
      CHECK(placement->prefix_tokens == longest);
      CHECK(placement->source.has_value() == (longest > 0));
      if (placement->source)
        CHECK(reuses(slots[static_cast<std::size_t>(*placement->source)]));
    }
    if (std::holds_alternative<ReserveBlocked>(decision)) {
      const auto idle = static_cast<std::size_t>(std::count_if(
          slots.begin(), slots.end(), [](const SlotState &slot) { return !slot.occupied; }));
      CHECK(!all_occupied && idle <= idle_cache_reserve_slots);
    }
  }
}

ENGINE_TEST(admission_is_empty_without_pending_requests) {
  CHECK(!select_admission({}, slot_views({{{}, false, 0}}), {}));
}

ENGINE_TEST(admission_prefers_the_lowest_priority_value) {
  const std::vector<PendingState> pending{{{1}, 2}, {{2}, 0}, {{3}, 1}, {{4}, 0}};

  const auto admission = select_admission(pending_views(pending), slot_views({{{}, false, 0}}), {});

  REQUIRE(admission);
  CHECK(admission->pending_index == 1);
  CHECK(admission->bypassed.empty());
}

ENGINE_TEST(admission_bypasses_the_head_for_a_longer_cached_prefix) {
  const std::vector<SlotState> slots{{{7, 7}, false, 0}, {{}, false, 0}};
  const std::vector<PendingState> pending{{{1}, 0}, {{7, 7, 1}, 1}, {{7, 1}, 0}, {{7, 7, 2}, 0}};

  const auto admission =
      select_admission(pending_views(pending), slot_views(slots), {.max_bypasses = 4});

  REQUIRE(admission);
  CHECK(admission->pending_index == 3);
  CHECK(admission->bypassed == std::vector<std::size_t>({0, 2}));
  CHECK(admission->placement.source == 0);
  CHECK(admission->placement.prefix_tokens == 2);
}

ENGINE_TEST(admission_keeps_the_head_on_an_equal_prefix) {
  const std::vector<SlotState> slots{{{7}, false, 0}, {{}, false, 0}};
  const std::vector<PendingState> pending{{{7, 1}, 0}, {{7, 2}, 0}};

  const auto admission =
      select_admission(pending_views(pending), slot_views(slots), {.max_bypasses = 4});

  REQUIRE(admission);
  CHECK(admission->pending_index == 0);
  CHECK(admission->bypassed.empty());
}

ENGINE_TEST(admission_stops_bypassing_a_head_at_the_limit) {
  const std::vector<SlotState> slots{{{7}, false, 0}, {{}, false, 0}};
  const std::vector<PendingState> pending{{{1}, 0, 2}, {{7, 1}, 0}};

  const auto under_limit =
      select_admission(pending_views(pending), slot_views(slots), {.max_bypasses = 3});
  REQUIRE(under_limit);
  CHECK(under_limit->pending_index == 1);

  const auto at_limit =
      select_admission(pending_views(pending), slot_views(slots), {.max_bypasses = 2});
  REQUIRE(at_limit);
  CHECK(at_limit->pending_index == 0);
}

ENGINE_TEST(admission_without_bypasses_admits_the_head_unless_the_reserve_blocks_it) {
  const std::vector<PendingState> pending{{{1}, 0}, {{7, 1}, 0}};

  const std::vector<SlotState> free_slot{{{7}, false, 0}, {{}, false, 0}};
  const auto head = select_admission(pending_views(pending), slot_views(free_slot), {});
  REQUIRE(head);
  CHECK(head->pending_index == 0);

  const std::vector<SlotState> reserved{{{5}, true, 0}, {{7}, false, 0}};
  const AdmissionOptions reserve{.max_bypasses = 0, .idle_cache_reserve_slots = 1};
  const auto bypass = select_admission(pending_views(pending), slot_views(reserved), reserve);
  REQUIRE(bypass);
  CHECK(bypass->pending_index == 1);
  CHECK(bypass->bypassed == std::vector<std::size_t>({0}));
  CHECK(bypass->placement.destination == 1);

  const std::vector<PendingState> nothing_placeable{{{1}, 0}, {{2}, 0}};
  CHECK(!select_admission(pending_views(nothing_placeable), slot_views(reserved), reserve));
}

ENGINE_TEST(admission_is_empty_when_no_slot_is_free) {
  const std::vector<PendingState> pending{{{1}, 0}, {{7}, 0}};

  CHECK(
      !select_admission(pending_views(pending), slot_views({{{7}, true, 0}}), {.max_bypasses = 4}));
}

ENGINE_TEST(admission_bypasses_each_queued_request_at_most_the_limit) {
  constexpr std::uint32_t max_bypasses = 3;
  const std::vector<SlotState> slots{{{1, 1}, false, 0}, {{2, 2}, false, 1}, {{3, 3}, false, 2}};
  std::mt19937 random(20260923);
  std::uniform_int_distribution<int> conversation(0, 4);
  std::uniform_int_distribution<int> arrivals(0, 2);

  std::deque<PendingState> pending;
  int affinity_admissions = 0;
  int capped_heads = 0;
  for (int step = 0; step < 1000; ++step) {
    for (int arrival = arrivals(random); arrival > 0; --arrival) {
      const int prefix = conversation(random);
      pending.push_back({{prefix, prefix, 9}, 0, 0});
    }
    if (pending.empty())
      continue;

    const auto admission =
        select_admission(pending_views(pending), slot_views(slots), {.max_bypasses = max_bypasses});
    REQUIRE(admission);

    const bool head_admitted = admission->pending_index == 0;
    CHECK(admission->bypassed.empty() == head_admitted);
    CHECK(head_admitted || pending.front().bypasses < max_bypasses);
    if (pending.front().bypasses == max_bypasses) {
      CHECK(head_admitted);
      ++capped_heads;
    }

    if (!head_admitted)
      ++affinity_admissions;
    admit(pending, *admission, max_bypasses);
  }

  CHECK(affinity_admissions > 0);
  CHECK(capped_heads > 0);
}

ENGINE_TEST(charging_a_bypass_saturates_at_the_limit) {
  CHECK(charge_bypass(0, 3) == 1);
  CHECK(charge_bypass(2, 3) == 3);
  CHECK(charge_bypass(3, 3) == 3);
}

} // namespace
