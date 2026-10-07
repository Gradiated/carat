#include "runtime/slot_policy.h"

#include <algorithm>

namespace carat {
namespace {

bool reuses_cache(std::span<const int> input_ids, const SlotView &slot) {
  const std::span<const int> cached = slot.cached_tokens;
  const bool covers_input = cached.size() == input_ids.size();

  return !cached.empty() && cached.size() <= input_ids.size() &&
         !(covers_input && slot.prefilling) &&
         std::equal(cached.begin(), cached.end(), input_ids.begin());
}

const SlotView &slot_at(std::span<const SlotView> slots, int index) {
  return slots[static_cast<std::size_t>(index)];
}

} // namespace

PlacementDecision select_placement(std::span<const int> input_ids, std::span<const SlotView> slots,
                                   std::size_t idle_cache_reserve_slots) {
  std::optional<int> source;
  std::size_t prefix_tokens = 0;
  for (std::size_t index = 0; index < slots.size(); ++index) {
    const SlotView &slot = slots[index];
    if (!reuses_cache(input_ids, slot))
      continue;

    const bool longer = slot.cached_tokens.size() > prefix_tokens;
    const bool idle_replaces_occupied = slot.cached_tokens.size() == prefix_tokens && source &&
                                        slot_at(slots, *source).occupied && !slot.occupied;
    if (longer || idle_replaces_occupied) {
      source = static_cast<int>(index);
      prefix_tokens = slot.cached_tokens.size();
    }
  }

  if (source && !slot_at(slots, *source).occupied) {
    return Placement{.destination = *source, .source = source, .prefix_tokens = prefix_tokens};
  }

  for (std::size_t index = 0; index < slots.size(); ++index) {
    if (!slots[index].occupied && slots[index].cached_tokens.empty()) {
      return Placement{
          .destination = static_cast<int>(index), .source = source, .prefix_tokens = prefix_tokens};
    }
  }

  std::optional<int> victim;
  std::size_t active_slots = 0;
  std::size_t idle_cached_slots = 0;
  for (std::size_t index = 0; index < slots.size(); ++index) {
    const SlotView &slot = slots[index];
    if (slot.occupied) {
      ++active_slots;
      continue;
    }

    ++idle_cached_slots;
    if (!victim || slot.age < slot_at(slots, *victim).age)
      victim = static_cast<int>(index);
  }

  if (!victim)
    return NoFreeSlot{};

  if (active_slots > 0 && idle_cached_slots <= idle_cache_reserve_slots &&
      prefix_tokens < slot_at(slots, *victim).cached_tokens.size()) {
    return ReserveBlocked{};
  }

  return Placement{.destination = *victim, .source = source, .prefix_tokens = prefix_tokens};
}

std::optional<Admission> select_admission(std::span<const PendingView> pending,
                                          std::span<const SlotView> slots,
                                          const AdmissionOptions &options) {
  if (pending.empty())
    return std::nullopt;

  const auto head_position = std::min_element(
      pending.begin(), pending.end(), [](const PendingView &left, const PendingView &right) {
        return left.priority < right.priority;
      });
  const auto head = static_cast<std::size_t>(std::distance(pending.begin(), head_position));
  const PlacementDecision head_decision =
      select_placement(head_position->input_ids, slots, options.idle_cache_reserve_slots);

  std::optional<Admission> selected;
  if (const auto *placement = std::get_if<Placement>(&head_decision)) {
    selected = Admission{.pending_index = head, .placement = *placement, .bypassed = {}};
  }

  const bool may_bypass = std::holds_alternative<ReserveBlocked>(head_decision) ||
                          head_position->bypasses < options.max_bypasses;
  if (!may_bypass)
    return selected;

  for (std::size_t index = head + 1; index < pending.size(); ++index) {
    if (pending[index].priority != head_position->priority)
      continue;

    const PlacementDecision decision =
        select_placement(pending[index].input_ids, slots, options.idle_cache_reserve_slots);
    const auto *placement = std::get_if<Placement>(&decision);
    if (placement == nullptr)
      continue;

    if (!selected || placement->prefix_tokens > selected->placement.prefix_tokens) {
      selected = Admission{.pending_index = index, .placement = *placement, .bypassed = {}};
    }
  }

  if (selected && selected->pending_index != head) {
    for (std::size_t index = head; index < selected->pending_index; ++index) {
      if (pending[index].priority == head_position->priority)
        selected->bypassed.push_back(index);
    }
  }
  return selected;
}

std::uint32_t charge_bypass(std::uint32_t bypasses, std::uint32_t max_bypasses) {
  return std::min(bypasses + 1, max_bypasses);
}

} // namespace carat
