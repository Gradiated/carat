#include "carat/gemma4_config.h"

#include "support/model_fixtures.h"
#include "support/test_support.h"

#include <string>
#include <string_view>
#include <vector>

namespace {

using carat::AttentionKind;
using carat::Gemma4Config;
using carat::require_native_gemma4_target;
using carat::testing::replace_once;
using carat::testing::TemporaryDirectory;
using carat::testing::tiny_gemma4_config;
using carat::testing::write_text;

Gemma4Config load_config(const std::string &text) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "config.json";
  write_text(path, text);
  return Gemma4Config::load(path.string());
}

void check_mutation_rejected(std::string_view from, std::string_view to,
                             std::string_view fragment) {
  const std::string mutated = replace_once(tiny_gemma4_config(), from, to);
  CHECK_THROWS(load_config(mutated), fragment);
}

ENGINE_TEST(gemma4_config_loads_the_tiny_model) {
  const auto config = load_config(tiny_gemma4_config());

  CHECK(config.hidden_size == 8);
  CHECK(config.intermediate_size == 16);
  CHECK(config.vocabulary_size == 32);
  CHECK(config.sliding_window == 4);
  CHECK(config.maximum_positions == 128);
  CHECK(config.rms_norm_epsilon == 1e-6);
  CHECK(config.final_logit_softcap == 30.0);
  CHECK(config.eos_token_ids == std::vector<int>({1, 106}));
  REQUIRE(config.layers.size() == 2);

  const auto &sliding = config.layers[0];
  CHECK(sliding.attention == AttentionKind::sliding);
  CHECK(sliding.query_width() == 512);
  CHECK(sliding.kv_width() == 256);
  CHECK(sliding.rotary_dimension == 256);
  CHECK(sliding.rope_theta == 10000.0);
  CHECK(!sliding.key_equals_value);

  const auto &global = config.layers[1];
  CHECK(global.attention == AttentionKind::global);
  CHECK(global.query_width() == 1024);
  CHECK(global.kv_width() == 512);
  CHECK(global.rotary_dimension == 128);
  CHECK(global.rope_theta == 1000000.0);
  CHECK(global.key_equals_value);
}

ENGINE_TEST(gemma4_config_derives_parameter_and_kv_sizes) {
  const auto config = load_config(tiny_gemma4_config());

  CHECK(config.text_parameter_count() == 35402);
  CHECK(config.full_kv_bytes_per_token() == 3072);
  CHECK(config.resident_sliding_kv_bytes_per_sequence() == 4096);
}

ENGINE_TEST(gemma4_config_rejects_unsupported_architectures) {
  check_mutation_rejected(R"("model_type":"gemma4",)", R"("model_type":"gemma3",)",
                          "expected a Gemma 4 model");
  check_mutation_rejected(R"("model_type":"gemma4_text")", R"("model_type":"gemma3_text")",
                          "expected Gemma 4 text configuration");
  check_mutation_rejected(R"("tie_word_embeddings":true)", R"("tie_word_embeddings":false)",
                          "untied Gemma 4 embeddings are not supported");
  check_mutation_rejected(R"("attention_bias":false)", R"("attention_bias":true)",
                          "attention bias is not supported");
  check_mutation_rejected(R"("hidden_size_per_layer_input":0)",
                          R"("hidden_size_per_layer_input":256)",
                          "per-layer input embeddings are not supported");
  check_mutation_rejected(R"("num_kv_shared_layers":0)", R"("num_kv_shared_layers":1)",
                          "shared assistant KV layers are not valid for the target engine");
}

ENGINE_TEST(gemma4_config_rejects_inconsistent_layers) {
  check_mutation_rejected(R"("eos_token_id":[1,106])", R"("eos_token_id":[])",
                          "eos_token_id must not be empty");
  check_mutation_rejected(R"("num_hidden_layers":2)", R"("num_hidden_layers":3)",
                          "layer_types length does not match num_hidden_layers");
  check_mutation_rejected(R"(["sliding_attention","full_attention"])",
                          R"(["sliding_attention","chunked_attention"])",
                          "unsupported Gemma 4 attention type: chunked_attention");
}

ENGINE_TEST(gemma4_config_rejects_invalid_rotary_dimensions) {
  check_mutation_rejected(R"("partial_rotary_factor":0.25)", R"("partial_rotary_factor":0.3)",
                          "invalid rotary dimension");
  check_mutation_rejected(R"("partial_rotary_factor":0.25)",
                          R"("partial_rotary_factor":0.005859375)", "invalid rotary dimension");
  check_mutation_rejected(R"("partial_rotary_factor":0.25)",
                          R"("partial_rotary_factor":0.001953125)", "invalid rotary dimension");
  check_mutation_rejected(R"("partial_rotary_factor":0.25)", R"("partial_rotary_factor":2)",
                          "invalid rotary dimension");
}

ENGINE_TEST(gemma4_config_rejects_values_that_overflow_uint32) {
  check_mutation_rejected(R"("hidden_size":8)", R"("hidden_size":4294967296)",
                          "hidden_size exceeds uint32");
  check_mutation_rejected(R"("eos_token_id":[1,106])", R"("eos_token_id":[1,4294967296])",
                          "eos_token_id exceeds uint32");
  check_mutation_rejected(R"("hidden_size":8)", R"("hidden_size":-8)", "JSON integer is negative");
}

ENGINE_TEST(gemma4_config_rejects_eos_ids_beyond_int) {
  check_mutation_rejected(R"("eos_token_id":[1,106])", R"("eos_token_id":[1,2147483648])",
                          "eos_token_id exceeds int");
  CHECK(load_config(replace_once(tiny_gemma4_config(), R"("eos_token_id":[1,106])",
                                 R"("eos_token_id":[2147483647])"))
            .eos_token_ids == std::vector<int>({2147483647}));
}

ENGINE_TEST(native_target_accepts_the_gemma4_31b_attention_constants) {
  require_native_gemma4_target(load_config(tiny_gemma4_config()));
}

ENGINE_TEST(native_target_rejects_each_value_the_kernels_hard_code) {
  const auto check_rejected = [](std::string_view from, std::string_view to,
                                 std::string_view fragment) {
    const auto config = load_config(replace_once(tiny_gemma4_config(), from, to));
    CHECK_THROWS(require_native_gemma4_target(config), fragment);
  };

  check_rejected(R"("rope_theta":10000,)", R"("rope_theta":10001,)",
                 "native Gemma 4 kernels require sliding rope_theta 10000, found 10001");
  check_rejected(R"("rope_theta":1000000,)", R"("rope_theta":500000,)",
                 "native Gemma 4 kernels require global rope_theta 1000000, found 500000");
  check_rejected(R"("rope_type":"default")", R"("rope_type":"default","partial_rotary_factor":0.5)",
                 "native Gemma 4 kernels require sliding rotary_dimension 256, found 128");
  check_rejected(R"("partial_rotary_factor":0.25)", R"("partial_rotary_factor":0.5)",
                 "native Gemma 4 kernels require global rotary_dimension 128, found 256");
  check_rejected(R"("head_dim":256)", R"("head_dim":128)",
                 "native Gemma 4 kernels require sliding head_dimension 256, found 128");
  check_rejected(R"("global_head_dim":512)", R"("global_head_dim":256)",
                 "native Gemma 4 kernels require global head_dimension 512, found 256");
  check_rejected(R"("attention_k_eq_v":true)", R"("attention_k_eq_v":false)",
                 "native Gemma 4 kernels require global key_equals_value true, found false");
}

ENGINE_TEST(gemma4_config_reports_missing_fields_and_files) {
  check_mutation_rejected(R"("sliding_window":4,)", "", "missing JSON key: sliding_window");
  CHECK_THROWS(Gemma4Config::load("/nonexistent/config.json"),
               "cannot open /nonexistent/config.json");
}

} // namespace
