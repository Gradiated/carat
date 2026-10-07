#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace carat {

struct SlotView {
  std::span<const int> cached_tokens;
  bool occupied{false};
  std::uint64_t age{0};
  // A prefilling slot has no prediction for the token after its cached prefix yet.
  bool prefilling{false};
};

struct PendingView {
  std::span<const int> input_ids;
  int priority{0};
  std::uint32_t bypasses{0};
};

struct Placement {
  int destination{0};
  std::optional<int> source;
  std::size_t prefix_tokens{0};
};

struct NoFreeSlot {};

struct ReserveBlocked {};

using PlacementDecision = std::variant<Placement, NoFreeSlot, ReserveBlocked>;

// The reserve keeps idle prefix caches resident while other slots are active, so a request
// that would evict a longer cache than it reuses waits instead.
[[nodiscard]] PlacementDecision select_placement(std::span<const int> input_ids,
                                                 std::span<const SlotView> slots,
                                                 std::size_t idle_cache_reserve_slots);

struct AdmissionOptions {
  std::uint32_t max_bypasses{0};
  std::size_t idle_cache_reserve_slots{0};
};

struct Admission {
  std::size_t pending_index{0};
  Placement placement;
  // Entries of the admitted priority queued ahead of it. Empty when the priority head is admitted.
  std::vector<std::size_t> bypassed;
};

[[nodiscard]] std::optional<Admission> select_admission(std::span<const PendingView> pending,
                                                        std::span<const SlotView> slots,
                                                        const AdmissionOptions &options);

[[nodiscard]] std::uint32_t charge_bypass(std::uint32_t bypasses, std::uint32_t max_bypasses);

} // namespace carat
