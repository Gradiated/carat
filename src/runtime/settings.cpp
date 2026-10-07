#include "runtime/settings.h"

#include "runtime/capacity.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace carat {
namespace {

class Environment {
public:
  explicit Environment(const EnvironmentLookup &lookup) : lookup_(lookup) {}

  [[nodiscard]] std::optional<std::string> text(std::string_view name) const {
    auto value = lookup_(name);
    if (!value || value->empty())
      return std::nullopt;

    return value;
  }

  // The CUDA library also reads this variable and treats an empty value as set, so it cannot
  // count as unset here.
  [[nodiscard]] std::optional<std::string> nonempty_text(std::string_view name) const {
    auto value = lookup_(name);
    if (value && value->empty())
      fail(name, "must not be empty");

    return value;
  }

  [[nodiscard]] std::string text(std::string_view name, std::string fallback) const {
    return text(name).value_or(std::move(fallback));
  }

  [[nodiscard]] int integer(std::string_view name, int fallback, int minimum, int maximum) const {
    const auto value = text(name);
    if (!value)
      return fallback;

    int parsed = 0;
    const char *end = value->data() + value->size();
    const auto [consumed, error] = std::from_chars(value->data(), end, parsed);
    if (error == std::errc::invalid_argument || consumed != end)
      fail(name, "is not an integer");
    if (error == std::errc::result_out_of_range || parsed < minimum || parsed > maximum) {
      fail(name, "must be between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    }

    return parsed;
  }

  [[nodiscard]] float positive_fraction(std::string_view name, float fallback) const {
    const auto value = text(name);
    if (!value)
      return fallback;

    char *end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value->c_str(), &end);
    if (end != value->c_str() + value->size() || errno == ERANGE || !std::isfinite(parsed)) {
      fail(name, "is not a number");
    }
    if (parsed <= 0.0F || parsed > 1.0F)
      fail(name, "must be in (0, 1]");

    return parsed;
  }

  [[nodiscard]] bool flag(std::string_view name) const {
    const auto value = text(name);
    if (!value || *value == "0" || *value == "false")
      return false;
    if (*value == "1" || *value == "true")
      return true;

    fail(name, "must be 0, 1, false or true");
  }

  // The CUDA library reads this switch itself and enables it only for the exact value 1.
  [[nodiscard]] bool cuda_flag(std::string_view name) const {
    const auto value = text(name);
    if (!value || *value == "0")
      return false;
    if (*value == "1")
      return true;

    fail(name, "must be 0 or 1");
  }

  [[noreturn]] static void fail(std::string_view name, const std::string &problem) {
    throw std::runtime_error(std::string(name) + ' ' + problem);
  }

private:
  const EnvironmentLookup &lookup_;
};

std::optional<Int4LmHeadOracle> int4_lm_head_oracle(const Environment &environment) {
  if (!environment.cuda_flag("CARAT_INT4_LM_HEAD_ORACLE"))
    return std::nullopt;

  constexpr std::string_view block = "CARAT_INT4_LM_HEAD_ORACLE_BLOCK";
  const int width = environment.integer(block, 128, 32, 256);
  if (width != 32 && width != 64 && width != 128 && width != 256) {
    Environment::fail(block, "must be 32, 64, 128 or 256");
  }

  return Int4LmHeadOracle{.block_width = width};
}

std::optional<Fp8DecodeSettings> fp8_decode(const Environment &environment) {
  const std::string mode = environment.text("CARAT_FP8_DECODE", "off");
  if (mode == "off")
    return std::nullopt;
  if (mode != "tensor")
    Environment::fail("CARAT_FP8_DECODE", "must be off or tensor");

  return Fp8DecodeSettings{
      .weight_scale_multiplier =
          environment.positive_fraction("CARAT_FP8_WEIGHT_SCALE_MULTIPLIER", 1.0F),
      .int4_lm_head_oracle = int4_lm_head_oracle(environment),
  };
}

AssistantFp8Mode assistant_fp8_mode(const Environment &environment) {
  const std::string mode = environment.nonempty_text("CARAT_ASSISTANT_FP8_MODE").value_or("off");
  if (mode == "off")
    return AssistantFp8Mode::off;
  if (mode == "lm_head")
    return AssistantFp8Mode::lm_head;
  if (mode == "all")
    return AssistantFp8Mode::all;

  Environment::fail("CARAT_ASSISTANT_FP8_MODE", "must be off, lm_head or all");
}

std::optional<AssistantSettings> assistant(const Environment &environment, bool fp8_decode) {
  auto directory = environment.text("CARAT_ASSISTANT_MODEL_PATH");
  if (!directory)
    return std::nullopt;
  if (!fp8_decode)
    Environment::fail("CARAT_ASSISTANT_MODEL_PATH", "requires CARAT_FP8_DECODE=tensor");

  return AssistantSettings{.model_directory = std::move(*directory),
                           .fp8_mode = assistant_fp8_mode(environment)};
}

SchedulerOptions scheduler_options(const Environment &environment, int model_layers) {
  const SchedulerOptions defaults;
  return {
      .prefill_quantum_tokens = environment.integer(
          "CARAT_PREFILL_QUANTUM_TOKENS", defaults.prefill_quantum_tokens, 1, maximum_context),
      .idle_prefill_quantum_tokens =
          environment.integer("CARAT_IDLE_PREFILL_QUANTUM_TOKENS",
                              defaults.idle_prefill_quantum_tokens, 1, maximum_context),
      .short_prefill_batch_window_microseconds =
          environment.integer("CARAT_SHORT_PREFILL_BATCH_WINDOW_US",
                              defaults.short_prefill_batch_window_microseconds, 0, 20000),
      .interleaved_suffix_layer_quantum =
          environment.integer("CARAT_INTERLEAVED_SUFFIX_LAYER_QUANTUM",
                              defaults.interleaved_suffix_layer_quantum, 0, model_layers),
      .interleaved_suffix_layer_minimum_decode =
          environment.integer("CARAT_INTERLEAVED_SUFFIX_LAYER_MIN_DECODE",
                              defaults.interleaved_suffix_layer_minimum_decode, 1, maximum_slots),
      .speculative_depth =
          environment.integer("CARAT_SPECULATIVE_DEPTH", defaults.speculative_depth, 1, 8),
      .speculative_minimum_batch = environment.integer(
          "CARAT_SPECULATIVE_MIN_BATCH", defaults.speculative_minimum_batch, 1, maximum_slots),
      .admission = {.max_bypasses = static_cast<std::uint32_t>(environment.integer(
                        "CARAT_CACHE_AFFINITY_MAX_BYPASSES",
                        static_cast<int>(defaults.admission.max_bypasses), 0, 1024)),
                    .idle_cache_reserve_slots = static_cast<std::size_t>(environment.integer(
                        "CARAT_IDLE_CACHE_RESERVE_SLOTS",
                        static_cast<int>(defaults.admission.idle_cache_reserve_slots), 0,
                        maximum_slots - 1))},
  };
}

} // namespace

RuntimeSettings load_runtime_settings(const EnvironmentLookup &lookup, int model_layers) {
  const RuntimeSettings defaults;
  const Environment environment(lookup);
  auto fp8 = fp8_decode(environment);
  auto assistant_settings = assistant(environment, fp8.has_value());
  auto api_key = environment.text("CARAT_RUNTIME_API_KEY");

  return {
      .bind_host = environment.text("CARAT_RUNTIME_HOST", defaults.bind_host),
      .bind_port = static_cast<std::uint16_t>(
          environment.integer("CARAT_RUNTIME_PORT", defaults.bind_port, 1, 65535)),
      .api_key = std::move(api_key),
      .fp8_decode = std::move(fp8),
      .assistant = std::move(assistant_settings),
      .scheduler = scheduler_options(environment, model_layers),
      .ignore_model_eos = environment.flag("CARAT_IGNORE_MODEL_EOS"),
  };
}

} // namespace carat
