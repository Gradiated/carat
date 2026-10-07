#include "runtime/settings.h"

#include "support/test_support.h"

#include <map>
#include <string>
#include <string_view>

namespace {

using carat::AssistantFp8Mode;
using carat::load_runtime_settings;
using carat::RuntimeSettings;

constexpr int model_layers = 60;

RuntimeSettings load(const std::map<std::string, std::string, std::less<>> &variables) {
  return load_runtime_settings(
      [&](std::string_view name) -> std::optional<std::string> {
        const auto found = variables.find(name);
        if (found == variables.end())
          return std::nullopt;

        return found->second;
      },
      model_layers);
}

ENGINE_TEST(runtime_settings_default_to_the_documented_values) {
  const auto settings = load({});

  CHECK(settings.bind_host == "0.0.0.0");
  CHECK(settings.bind_port == 30000);
  CHECK(!settings.api_key);
  CHECK(!settings.fp8_decode);
  CHECK(!settings.assistant);
  CHECK(!settings.ignore_model_eos);
  CHECK(settings.scheduler.prefill_quantum_tokens == 1024);
  CHECK(settings.scheduler.short_prefill_batch_window_microseconds == 10000);
  CHECK(settings.scheduler.interleaved_suffix_layer_quantum == 10);
  CHECK(settings.scheduler.speculative_depth == 4);
  CHECK(settings.scheduler.admission.max_bypasses == 4);
  CHECK(settings.scheduler.admission.idle_cache_reserve_slots == 0);
}

ENGINE_TEST(runtime_settings_treat_empty_values_as_unset) {
  const auto settings = load({{"CARAT_RUNTIME_PORT", ""}, {"CARAT_FP8_DECODE", ""}});

  CHECK(settings.bind_port == 30000);
  CHECK(!settings.fp8_decode);
}

ENGINE_TEST(runtime_settings_read_every_production_value) {
  const auto settings = load({{"CARAT_RUNTIME_HOST", "127.0.0.1"},
                              {"CARAT_RUNTIME_PORT", "31000"},
                              {"CARAT_RUNTIME_API_KEY", "secret"},
                              {"CARAT_FP8_DECODE", "tensor"},
                              {"CARAT_ASSISTANT_MODEL_PATH", "/models/assistant"},
                              {"CARAT_ASSISTANT_FP8_MODE", "lm_head"},
                              {"CARAT_INTERLEAVED_SUFFIX_LAYER_QUANTUM", "60"},
                              {"CARAT_CACHE_AFFINITY_MAX_BYPASSES", "0"},
                              {"CARAT_IGNORE_MODEL_EOS", "true"}});

  CHECK(settings.bind_host == "127.0.0.1");
  CHECK(settings.bind_port == 31000);
  CHECK(settings.api_key == "secret");
  REQUIRE(settings.fp8_decode);
  CHECK(settings.fp8_decode->weight_scale_multiplier == 1.0F);
  CHECK(!settings.fp8_decode->int4_lm_head_oracle);
  REQUIRE(settings.assistant);
  CHECK(settings.assistant->model_directory == "/models/assistant");
  CHECK(settings.assistant->fp8_mode == AssistantFp8Mode::lm_head);
  CHECK(settings.scheduler.interleaved_suffix_layer_quantum == 60);
  CHECK(settings.scheduler.admission.max_bypasses == 0);
  CHECK(settings.ignore_model_eos);
}

ENGINE_TEST(runtime_settings_read_the_research_fp8_options) {
  const auto settings = load({{"CARAT_FP8_DECODE", "tensor"},
                              {"CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", "0.5"},
                              {"CARAT_INT4_LM_HEAD_ORACLE", "1"},
                              {"CARAT_INT4_LM_HEAD_ORACLE_BLOCK", "64"}});

  REQUIRE(settings.fp8_decode);
  CHECK(settings.fp8_decode->weight_scale_multiplier == 0.5F);
  REQUIRE(settings.fp8_decode->int4_lm_head_oracle);
  CHECK(settings.fp8_decode->int4_lm_head_oracle->block_width == 64);
}

ENGINE_TEST(runtime_settings_reject_malformed_integers_with_the_variable_name) {
  CHECK_THROWS(load({{"CARAT_RUNTIME_PORT", "30000abc"}}), "CARAT_RUNTIME_PORT is not an integer");
  CHECK_THROWS(load({{"CARAT_RUNTIME_PORT", "port"}}), "CARAT_RUNTIME_PORT is not an integer");
  CHECK_THROWS(load({{"CARAT_RUNTIME_PORT", "0"}}),
               "CARAT_RUNTIME_PORT must be between 1 and 65535");
  CHECK_THROWS(load({{"CARAT_RUNTIME_PORT", "99999999999999999999"}}),
               "CARAT_RUNTIME_PORT must be between 1 and 65535");
  CHECK_THROWS(load({{"CARAT_SPECULATIVE_DEPTH", "9"}}),
               "CARAT_SPECULATIVE_DEPTH must be between 1 and 8");
  CHECK_THROWS(load({{"CARAT_INTERLEAVED_SUFFIX_LAYER_QUANTUM", "61"}}),
               "CARAT_INTERLEAVED_SUFFIX_LAYER_QUANTUM must be between 0 and 60");
  CHECK_THROWS(load({{"CARAT_IDLE_CACHE_RESERVE_SLOTS", "24"}}),
               "CARAT_IDLE_CACHE_RESERVE_SLOTS must be between 0 and 23");
}

ENGINE_TEST(runtime_settings_reject_unknown_modes_and_flags) {
  CHECK_THROWS(load({{"CARAT_FP8_DECODE", "row"}}), "CARAT_FP8_DECODE must be off or tensor");
  CHECK_THROWS(load({{"CARAT_IGNORE_MODEL_EOS", "yes"}}),
               "CARAT_IGNORE_MODEL_EOS must be 0, 1, false or true");
  CHECK_THROWS(load({{"CARAT_FP8_DECODE", "tensor"}, {"CARAT_INT4_LM_HEAD_ORACLE", "true"}}),
               "CARAT_INT4_LM_HEAD_ORACLE must be 0 or 1");
  CHECK_THROWS(load({{"CARAT_FP8_DECODE", "tensor"},
                     {"CARAT_ASSISTANT_MODEL_PATH", "/assistant"},
                     {"CARAT_ASSISTANT_FP8_MODE", "on"}}),
               "CARAT_ASSISTANT_FP8_MODE must be off, lm_head or all");
}

ENGINE_TEST(runtime_settings_reject_an_empty_assistant_fp8_mode) {
  CHECK_THROWS(load({{"CARAT_FP8_DECODE", "tensor"},
                     {"CARAT_ASSISTANT_MODEL_PATH", "/assistant"},
                     {"CARAT_ASSISTANT_FP8_MODE", ""}}),
               "CARAT_ASSISTANT_FP8_MODE must not be empty");
}

ENGINE_TEST(runtime_settings_reject_invalid_fp8_research_values) {
  const auto with_fp8 = [](std::string_view name, std::string_view value) {
    return load({{"CARAT_FP8_DECODE", "tensor"}, {std::string(name), std::string(value)}});
  };

  CHECK_THROWS(with_fp8("CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", "0"),
               "CARAT_FP8_WEIGHT_SCALE_MULTIPLIER must be in (0, 1]");
  CHECK_THROWS(with_fp8("CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", "1.5"),
               "CARAT_FP8_WEIGHT_SCALE_MULTIPLIER must be in (0, 1]");
  CHECK_THROWS(with_fp8("CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", "half"),
               "CARAT_FP8_WEIGHT_SCALE_MULTIPLIER is not a number");
  CHECK_THROWS(with_fp8("CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", "nan"),
               "CARAT_FP8_WEIGHT_SCALE_MULTIPLIER is not a number");
  CHECK_THROWS(load({{"CARAT_FP8_DECODE", "tensor"},
                     {"CARAT_INT4_LM_HEAD_ORACLE", "1"},
                     {"CARAT_INT4_LM_HEAD_ORACLE_BLOCK", "96"}}),
               "CARAT_INT4_LM_HEAD_ORACLE_BLOCK must be 32, 64, 128 or 256");
}

ENGINE_TEST(runtime_settings_read_fp8_research_values_only_with_fp8_decode) {
  const auto settings =
      load({{"CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", "0"}, {"CARAT_INT4_LM_HEAD_ORACLE", "on"}});

  CHECK(!settings.fp8_decode);
}

ENGINE_TEST(runtime_settings_require_fp8_decode_for_the_assistant) {
  CHECK_THROWS(load({{"CARAT_ASSISTANT_MODEL_PATH", "/assistant"}}),
               "CARAT_ASSISTANT_MODEL_PATH requires CARAT_FP8_DECODE=tensor");
}

} // namespace
