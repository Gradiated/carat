#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace carat {

enum class FinishReason { stop, length };

struct CompletionResult {
  std::vector<int> output_ids;
  std::uint64_t cached_input_tokens{0};
  std::uint64_t queue_microseconds{0};
  std::uint64_t ttft_microseconds{0};
  std::uint64_t decode_microseconds{0};
  std::uint64_t maximum_tpot_microseconds{0};
  std::uint64_t total_microseconds{0};
  FinishReason finish_reason{FinishReason::stop};
};

struct CompletionUsage {
  std::size_t input_tokens{0};
  std::uint64_t cached_input_tokens{0};
  std::size_t output_tokens{0};
};

inline constexpr std::string_view stream_done_event = "data: [DONE]\n\n";

[[nodiscard]] std::string completion_json(const CompletionResult &result, std::size_t input_tokens);
[[nodiscard]] std::string token_event(int token, const CompletionUsage &usage);
[[nodiscard]] std::string finish_event(FinishReason reason, const CompletionUsage &usage);

} // namespace carat
