#include "carat/gemma4_config.h"
#include "carat/gemma4_model.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include "support/model_fixtures.h"
#include "support/test_support.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using carat::Gemma4Config;
using carat::Gemma4Model;
using carat::require_gemma4_assistant_target;
using carat::ShardedSafetensors;
using carat::validate_gemma4_weights;
using carat::WeightPlan;
using carat::testing::gemma4_assistant_tensors;
using carat::testing::TemporaryDirectory;
using carat::testing::TensorSpec;
using carat::testing::tiny_gemma4_config;
using carat::testing::tiny_gemma4_tensors;
using carat::testing::write_safetensors;
using carat::testing::write_text;
using carat::testing::write_tiny_gemma4_model;

struct TinyModel {
  Gemma4Config config;
  ShardedSafetensors weights;
};

TinyModel load_tiny_model(const std::vector<TensorSpec> &tensors) {
  TemporaryDirectory temporary;
  write_text(temporary.path() / "config.json", tiny_gemma4_config());
  write_safetensors(temporary.path() / "model.safetensors", tensors);
  return {Gemma4Config::load((temporary.path() / "config.json").string()),
          ShardedSafetensors::open(temporary.path().string())};
}

ShardedSafetensors load_weights(const std::vector<TensorSpec> &tensors) {
  TemporaryDirectory temporary;
  write_safetensors(temporary.path() / "model.safetensors", tensors);
  return ShardedSafetensors::open(temporary.path().string());
}

std::vector<TensorSpec> with_tensor(std::vector<TensorSpec> tensors, std::string_view name,
                                    std::string dtype, std::vector<std::uint64_t> shape) {
  for (TensorSpec &tensor : tensors) {
    if (tensor.name == name) {
      tensor.dtype = std::move(dtype);
      tensor.shape = std::move(shape);
      return tensors;
    }
  }
  tensors.push_back({std::string(name), std::move(dtype), std::move(shape)});
  return tensors;
}

std::vector<std::string> source_names(const carat::WeightSegment &segment) {
  std::vector<std::string> names;
  for (const auto &source : segment.sources)
    names.push_back(source.tensor_name);
  return names;
}

void check_arena_layout(const WeightPlan &plan) {
  std::uint64_t previous_end = 0;

  for (const auto &segment : plan.segments()) {
    CHECK(segment.arena_offset % 256 == 0);
    CHECK(segment.arena_offset >= previous_end);
    CHECK(plan.at(segment.name).arena_offset == segment.arena_offset);

    std::uint64_t source_offset = 0;
    for (const auto &source : segment.sources) {
      CHECK(source.segment_offset == source_offset);
      source_offset += source.byte_size;
    }
    CHECK(source_offset == segment.byte_size);

    previous_end = segment.arena_offset + segment.byte_size;
  }

  CHECK(plan.arena_bytes() >= previous_end);
  CHECK(plan.arena_bytes() % 256 == 0);
}

ENGINE_TEST(weight_plan_packs_value_projection_only_without_key_value_sharing) {
  const auto model = load_tiny_model(tiny_gemma4_tensors());

  const auto plan = WeightPlan::gemma4(model.config, model.weights);

  const auto &sliding = plan.at("layer.0.qkv");
  CHECK(sliding.shape == std::vector<std::uint64_t>({1024, 8}));
  CHECK(source_names(sliding) ==
        std::vector<std::string>({"model.language_model.layers.0.self_attn.q_proj.weight",
                                  "model.language_model.layers.0.self_attn.k_proj.weight",
                                  "model.language_model.layers.0.self_attn.v_proj.weight"}));

  const auto &global = plan.at("layer.1.qkv");
  CHECK(global.shape == std::vector<std::uint64_t>({1536, 8}));
  CHECK(source_names(global) ==
        std::vector<std::string>({"model.language_model.layers.1.self_attn.q_proj.weight",
                                  "model.language_model.layers.1.self_attn.k_proj.weight"}));

  CHECK(plan.at("layer.1.gate_up").shape == std::vector<std::uint64_t>({32, 8}));
  CHECK(plan.segments().size() == 2 + 2 * 11);
  CHECK_THROWS(plan.at("layer.2.qkv"), "missing execution weight layer.2.qkv");
}

ENGINE_TEST(weight_plan_arena_segments_are_aligned_contiguous_and_disjoint) {
  const auto model = load_tiny_model(tiny_gemma4_tensors());

  const auto plan = WeightPlan::gemma4(model.config, model.weights);

  check_arena_layout(plan);
  CHECK(plan.segments().front().name == "token_embedding");
  CHECK(plan.segments().back().name == "final_norm");
}

ENGINE_TEST(weight_plan_ignores_tensors_outside_the_language_model) {
  const auto model = load_tiny_model(
      with_tensor(tiny_gemma4_tensors(), "model.vision_tower.patch.weight", "F32", {4}));

  CHECK(WeightPlan::gemma4(model.config, model.weights).segments().size() == 24);
}

ENGINE_TEST(weight_plan_rejects_invalid_sources) {
  const std::string o_proj = "model.language_model.layers.1.self_attn.o_proj.weight";

  const auto without_norm = [&] {
    auto tensors = tiny_gemma4_tensors();
    std::erase_if(tensors, [](const TensorSpec &tensor) {
      return tensor.name == "model.language_model.norm.weight";
    });
    return load_tiny_model(tensors);
  }();
  CHECK_THROWS(WeightPlan::gemma4(without_norm.config, without_norm.weights),
               "missing tensor model.language_model.norm.weight");

  const auto float_source =
      load_tiny_model(with_tensor(tiny_gemma4_tensors(), o_proj, "F32", {8, 1024}));
  CHECK_THROWS(WeightPlan::gemma4(float_source.config, float_source.weights),
               "execution weights must be BF16: " + o_proj);

  const auto wrong_shape =
      load_tiny_model(with_tensor(tiny_gemma4_tensors(), o_proj, "BF16", {8, 8}));
  CHECK_THROWS(WeightPlan::gemma4(wrong_shape.config, wrong_shape.weights),
               "packed execution shape does not match source bytes: layer.1.o");

  const auto transposed =
      load_tiny_model(with_tensor(tiny_gemma4_tensors(), o_proj, "BF16", {1024, 8}));
  CHECK_THROWS(WeightPlan::gemma4(transposed.config, transposed.weights),
               "source shape does not match execution plan: " + o_proj);

  const auto unused = load_tiny_model(
      with_tensor(tiny_gemma4_tensors(), "model.language_model.extra.weight", "BF16", {2}));
  CHECK_THROWS(WeightPlan::gemma4(unused.config, unused.weights),
               "tensor absent from execution plan: model.language_model.extra.weight");
}

ENGINE_TEST(weight_plan_rejects_a_config_the_native_kernels_do_not_support) {
  auto model = load_tiny_model(tiny_gemma4_tensors());
  model.config.layers[1].head_dimension = 256;

  CHECK_THROWS(WeightPlan::gemma4(model.config, model.weights),
               "native Gemma 4 kernels require global head_dimension");
}

ENGINE_TEST(gemma4_weight_validation_builds_the_weight_plan) {
  const auto model = load_tiny_model(tiny_gemma4_tensors());
  validate_gemma4_weights(model.weights, model.config);

  const auto unused = load_tiny_model(
      with_tensor(tiny_gemma4_tensors(), "model.language_model.extra.weight", "BF16", {2}));
  CHECK_THROWS(validate_gemma4_weights(unused.weights, unused.config),
               "tensor absent from execution plan: model.language_model.extra.weight");
}

ENGINE_TEST(gemma4_model_opens_a_valid_release) {
  TemporaryDirectory temporary;
  write_tiny_gemma4_model(temporary.path());

  const auto model = Gemma4Model::open(temporary.path().string());

  CHECK(model.config.layers.size() == 2);
  CHECK(model.plan.segments().size() == 24);
  CHECK(model.weights.parameter_count("model.language_model.") ==
        model.config.text_parameter_count());
}

ENGINE_TEST(gemma4_model_rejects_a_release_without_weights) {
  TemporaryDirectory temporary;
  write_text(temporary.path() / "config.json", tiny_gemma4_config());

  CHECK_THROWS(Gemma4Model::open(temporary.path().string()), "cannot open safetensors file");
}

ENGINE_TEST(assistant_target_must_match_the_fixed_backbone) {
  auto config = load_tiny_model(tiny_gemma4_tensors()).config;
  CHECK_THROWS(require_gemma4_assistant_target(config),
               "the Gemma 4 assistant requires target hidden_size 5376 and vocabulary_size 262144, "
               "found 8 and 32");

  config.hidden_size = 5376;
  config.vocabulary_size = 262144;
  require_gemma4_assistant_target(config);

  config.vocabulary_size = 262143;
  CHECK_THROWS(require_gemma4_assistant_target(config), "found 5376 and 262143");
}

ENGINE_TEST(assistant_weight_plan_packs_the_fixed_architecture) {
  const auto weights = load_weights(gemma4_assistant_tensors());

  const auto plan = WeightPlan::gemma4_assistant(weights);

  CHECK(plan.segments().size() == 3 + 4 * 10 + 1);
  CHECK(plan.at("layer.0.q").shape == std::vector<std::uint64_t>({8192, 1024}));
  CHECK(plan.at("layer.3.q").shape == std::vector<std::uint64_t>({16384, 1024}));
  CHECK(plan.at("layer.3.q_norm").shape == std::vector<std::uint64_t>({512}));
  CHECK(plan.at("layer.2.gate_up").sources.size() == 2);
  check_arena_layout(plan);
}

ENGINE_TEST(assistant_weight_plan_rejects_invalid_sources) {
  CHECK_THROWS(WeightPlan::gemma4_assistant(load_weights({{"unrelated", "BF16", {1}}})),
               "missing tensor model.embed_tokens.weight");

  CHECK_THROWS(
      WeightPlan::gemma4_assistant(load_weights({{"model.embed_tokens.weight", "F32", {1}}})),
      "assistant execution weights must be BF16: model.embed_tokens.weight");

  CHECK_THROWS(
      WeightPlan::gemma4_assistant(load_weights({{"model.embed_tokens.weight", "BF16", {1}}})),
      "assistant packed execution shape does not match source bytes: token_embedding");

  CHECK_THROWS(WeightPlan::gemma4_assistant(load_weights(
                   with_tensor(gemma4_assistant_tensors(), "model.layers.0.self_attn.o_proj.weight",
                               "BF16", {8192, 1024}))),
               "assistant source shape does not match execution plan: "
               "model.layers.0.self_attn.o_proj.weight");

  CHECK_THROWS(WeightPlan::gemma4_assistant(load_weights(
                   with_tensor(gemma4_assistant_tensors(), "vision.weight", "BF16", {1}))),
               "assistant tensor absent from execution plan: vision.weight");
}

} // namespace
