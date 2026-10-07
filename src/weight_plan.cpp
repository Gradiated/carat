#include "carat/weight_plan.h"

#include "carat/gemma4_config.h"
#include "carat/safetensors.h"

#include <functional>
#include <set>
#include <stdexcept>

namespace carat {
namespace {

constexpr std::uint64_t alignment = 256;
constexpr std::uint64_t bf16_bytes = 2;

using Shape = std::vector<std::uint64_t>;

struct SourceSpec {
  std::string tensor_name;
  Shape shape;
};

struct SegmentSpec {
  std::string name;
  Shape shape;
  std::vector<SourceSpec> sources;
};

std::uint64_t align_up(std::uint64_t value) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

std::string layer_name(std::size_t layer, std::string_view suffix) {
  return "model.language_model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

std::vector<SegmentSpec> gemma4_segments(const Gemma4Config &config) {
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t intermediate = config.intermediate_size;
  std::vector<SegmentSpec> segments{
      {"token_embedding",
       {config.vocabulary_size, hidden},
       {{"model.language_model.embed_tokens.weight", {config.vocabulary_size, hidden}}}},
  };

  for (std::size_t layer = 0; layer < config.layers.size(); ++layer) {
    const auto &attention = config.layers[layer];
    const std::uint64_t query = attention.query_width();
    const std::uint64_t kv = attention.kv_width();
    const std::string prefix = "layer." + std::to_string(layer) + ".";
    const auto single = [&](std::string segment, std::string_view tensor, Shape shape) {
      segments.push_back({prefix + segment, shape, {{layer_name(layer, tensor), shape}}});
    };

    SegmentSpec qkv{prefix + "qkv",
                    {query + kv, hidden},
                    {{layer_name(layer, "self_attn.q_proj.weight"), {query, hidden}},
                     {layer_name(layer, "self_attn.k_proj.weight"), {kv, hidden}}}};
    if (!attention.key_equals_value) {
      qkv.shape[0] += kv;
      qkv.sources.push_back({layer_name(layer, "self_attn.v_proj.weight"), {kv, hidden}});
    }

    single("input_norm", "input_layernorm.weight", {hidden});
    single("q_norm", "self_attn.q_norm.weight", {attention.head_dimension});
    single("k_norm", "self_attn.k_norm.weight", {attention.head_dimension});
    segments.push_back(std::move(qkv));
    single("o", "self_attn.o_proj.weight", {hidden, query});
    single("post_attention_norm", "post_attention_layernorm.weight", {hidden});
    single("pre_mlp_norm", "pre_feedforward_layernorm.weight", {hidden});
    segments.push_back({prefix + "gate_up",
                        {2 * intermediate, hidden},
                        {{layer_name(layer, "mlp.gate_proj.weight"), {intermediate, hidden}},
                         {layer_name(layer, "mlp.up_proj.weight"), {intermediate, hidden}}}});
    single("down", "mlp.down_proj.weight", {hidden, intermediate});
    single("post_mlp_norm", "post_feedforward_layernorm.weight", {hidden});
    single("scalar", "layer_scalar", {1});
  }

  segments.push_back({"final_norm", {hidden}, {{"model.language_model.norm.weight", {hidden}}}});
  return segments;
}

std::vector<SegmentSpec> gemma4_assistant_segments() {
  constexpr Gemma4AssistantShape shape = gemma4_assistant_shape;
  const std::uint64_t hidden = shape.hidden;
  const std::uint64_t intermediate = shape.intermediate;
  std::vector<SegmentSpec> segments{
      {"token_embedding",
       {shape.vocabulary, hidden},
       {{"model.embed_tokens.weight", {shape.vocabulary, hidden}}}},
      {"pre_projection",
       {hidden, 2 * shape.backbone_hidden},
       {{"pre_projection.weight", {hidden, 2 * shape.backbone_hidden}}}},
      {"post_projection",
       {shape.backbone_hidden, hidden},
       {{"post_projection.weight", {shape.backbone_hidden, hidden}}}},
  };

  for (std::size_t layer = 0; layer < shape.layers; ++layer) {
    const std::uint64_t head_dimension =
        layer == shape.global_layer ? shape.global_head_dimension : shape.sliding_head_dimension;
    const std::uint64_t query = shape.query_heads * head_dimension;
    const std::string source = "model.layers." + std::to_string(layer) + ".";
    const std::string prefix = "layer." + std::to_string(layer) + ".";
    const auto single = [&](std::string segment, std::string_view tensor, Shape tensor_shape) {
      segments.push_back(
          {prefix + segment, tensor_shape, {{source + std::string(tensor), tensor_shape}}});
    };

    single("input_norm", "input_layernorm.weight", {hidden});
    single("q_norm", "self_attn.q_norm.weight", {head_dimension});
    single("q", "self_attn.q_proj.weight", {query, hidden});
    single("o", "self_attn.o_proj.weight", {hidden, query});
    single("post_attention_norm", "post_attention_layernorm.weight", {hidden});
    single("pre_mlp_norm", "pre_feedforward_layernorm.weight", {hidden});
    segments.push_back({prefix + "gate_up",
                        {2 * intermediate, hidden},
                        {{source + "mlp.gate_proj.weight", {intermediate, hidden}},
                         {source + "mlp.up_proj.weight", {intermediate, hidden}}}});
    single("down", "mlp.down_proj.weight", {hidden, intermediate});
    single("post_mlp_norm", "post_feedforward_layernorm.weight", {hidden});
    single("scalar", "layer_scalar", {1});
  }

  segments.push_back({"final_norm", {hidden}, {{"model.norm.weight", {hidden}}}});
  return segments;
}

class WeightPlanBuilder {
public:
  WeightPlanBuilder(const ShardedSafetensors &weights, std::string_view label)
      : weights_(weights), label_(label) {}

  void add(const SegmentSpec &spec) {
    WeightSegment segment{.name = spec.name,
                          .shape = spec.shape,
                          .arena_offset = align_up(offset_),
                          .byte_size = 0,
                          .sources = {}};

    for (const SourceSpec &source_spec : spec.sources) {
      const std::string &name = source_spec.tensor_name;
      const TensorLocation &source = weights_.at(name);
      if (source.dtype != SafetensorsDtype::bf16)
        fail("execution weights must be BF16: ", name);
      if (!used_.insert(name).second)
        fail("weight source used twice: ", name);

      segment.sources.push_back({name, segment.byte_size, source.byte_size});
      segment.byte_size += source.byte_size;
    }

    const std::uint64_t packed_elements = segment.byte_size / bf16_bytes;
    if (segment.byte_size % bf16_bytes != 0 || element_count(segment.shape) != packed_elements) {
      fail("packed execution shape does not match source bytes: ", segment.name);
    }
    for (const SourceSpec &source_spec : spec.sources) {
      if (weights_.at(source_spec.tensor_name).shape != source_spec.shape) {
        fail("source shape does not match execution plan: ", source_spec.tensor_name);
      }
    }

    offset_ = segment.arena_offset + segment.byte_size;
    segments_.push_back(std::move(segment));
  }

  void require_used(const std::function<bool(std::string_view)> &must_be_used) const {
    for (const auto &entry : weights_.tensors()) {
      if (must_be_used(entry.first) && !used_.contains(entry.first)) {
        fail("tensor absent from execution plan: ", entry.first);
      }
    }
  }

  [[nodiscard]] std::vector<WeightSegment> take_segments() {
    return std::move(segments_);
  }
  [[nodiscard]] std::uint64_t arena_bytes() const {
    return align_up(offset_);
  }

private:
  [[noreturn]] void fail(std::string_view message, std::string_view subject) const {
    throw std::runtime_error(std::string(label_) + std::string(message) + std::string(subject));
  }

  const ShardedSafetensors &weights_;
  std::string_view label_;
  std::set<std::string, std::less<>> used_;
  std::vector<WeightSegment> segments_;
  std::uint64_t offset_{0};
};

} // namespace

WeightPlan::WeightPlan(std::vector<WeightSegment> segments, std::uint64_t arena_bytes)
    : segments_(std::move(segments)), arena_bytes_(arena_bytes) {
  for (std::size_t index = 0; index < segments_.size(); ++index) {
    by_name_.emplace(segments_[index].name, index);
  }
}

WeightPlan WeightPlan::gemma4(const Gemma4Config &config, const ShardedSafetensors &weights) {
  require_native_gemma4_target(config);

  WeightPlanBuilder builder(weights, "");
  for (const SegmentSpec &segment : gemma4_segments(config))
    builder.add(segment);

  builder.require_used(
      [](std::string_view name) { return name.starts_with("model.language_model."); });
  const std::uint64_t arena_bytes = builder.arena_bytes();
  return WeightPlan(builder.take_segments(), arena_bytes);
}

WeightPlan WeightPlan::gemma4_assistant(const ShardedSafetensors &weights) {
  WeightPlanBuilder builder(weights, "assistant ");
  for (const SegmentSpec &segment : gemma4_assistant_segments())
    builder.add(segment);

  builder.require_used([](std::string_view) { return true; });
  const std::uint64_t arena_bytes = builder.arena_bytes();
  return WeightPlan(builder.take_segments(), arena_bytes);
}

const WeightSegment &WeightPlan::at(std::string_view name) const {
  const auto iterator = by_name_.find(name);
  if (iterator == by_name_.end())
    throw std::runtime_error("missing execution weight " + std::string(name));

  return segments_[iterator->second];
}

const std::vector<WeightSegment> &WeightPlan::segments() const {
  return segments_;
}
std::uint64_t WeightPlan::arena_bytes() const {
  return arena_bytes_;
}

void require_gemma4_assistant_target(const Gemma4Config &target) {
  constexpr Gemma4AssistantShape shape = gemma4_assistant_shape;
  if (target.hidden_size != shape.backbone_hidden || target.vocabulary_size != shape.vocabulary) {
    throw std::runtime_error("the Gemma 4 assistant requires target hidden_size " +
                             std::to_string(shape.backbone_hidden) + " and vocabulary_size " +
                             std::to_string(shape.vocabulary) + ", found " +
                             std::to_string(target.hidden_size) + " and " +
                             std::to_string(target.vocabulary_size));
  }
}

void validate_gemma4_weights(const ShardedSafetensors &weights, const Gemma4Config &config) {
  static_cast<void>(WeightPlan::gemma4(config, weights));
}

} // namespace carat
