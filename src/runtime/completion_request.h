#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace carat {

struct CompletionLimits {
  int maximum_context{0};
  int vocabulary_size{0};
  std::span<const int> implicit_stop_ids;
};

struct CompletionRequest {
  std::vector<int> input_ids;
  int max_tokens{0};
  std::unordered_set<int> stop_ids;
  bool stream{false};
  int priority{0};
};

// Throws std::exception with a client-facing message when the body is invalid.
[[nodiscard]] CompletionRequest parse_completion_request(std::string_view body,
                                                         const CompletionLimits &limits);

class BearerAuthorization {
public:
  explicit BearerAuthorization(const std::optional<std::string> &api_key);

  // Without an API key every request is permitted. The comparison does not stop at the first
  // mismatch, so response timing does not reveal a matching prefix.
  [[nodiscard]] bool permits(std::string_view authorization_header) const;

private:
  std::optional<std::string> expected_header_;
};

} // namespace carat
