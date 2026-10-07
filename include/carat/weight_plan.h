#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace carat {

struct WeightSource {
  std::string tensor_name;
  std::uint64_t segment_offset;
  std::uint64_t byte_size;
};

struct WeightSegment {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::uint64_t arena_offset;
  std::uint64_t byte_size;
  std::vector<WeightSource> sources;
};

struct Gemma4AssistantShape {
  std::uint64_t hidden = 1024;
  std::uint64_t backbone_hidden = 5376;
  std::uint64_t intermediate = 8192;
  std::uint64_t vocabulary = 262144;
  std::size_t layers = 4;
  std::uint64_t query_heads = 32;
  std::size_t global_layer = 3;
  std::uint64_t sliding_head_dimension = 256;
  std::uint64_t global_head_dimension = 512;
};

inline constexpr Gemma4AssistantShape gemma4_assistant_shape{};

class WeightPlan {
public:
  static WeightPlan gemma4(const struct Gemma4Config &config,
                           const class ShardedSafetensors &weights);
  static WeightPlan gemma4_assistant(const class ShardedSafetensors &weights);

  [[nodiscard]] const WeightSegment &at(std::string_view name) const;
  [[nodiscard]] const std::vector<WeightSegment> &segments() const;
  [[nodiscard]] std::uint64_t arena_bytes() const;

private:
  WeightPlan(std::vector<WeightSegment> segments, std::uint64_t arena_bytes);

  std::vector<WeightSegment> segments_;
  std::map<std::string, std::size_t, std::less<>> by_name_;
  std::uint64_t arena_bytes_{0};
};

// The assistant reads the target's hidden states and shares its vocabulary.
void require_gemma4_assistant_target(const struct Gemma4Config &target);

// Equivalent to building WeightPlan::gemma4 and discarding it.
void validate_gemma4_weights(const class ShardedSafetensors &weights,
                             const struct Gemma4Config &config);

} // namespace carat
