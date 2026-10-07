#include "runtime/metrics.h"

#include "runtime/settings.h"

#include "support/test_support.h"

#include <set>
#include <sstream>
#include <string>

namespace {

using carat::maximum_slots;
using carat::render_metrics;
using carat::RuntimeMetrics;
using carat::RuntimeSettings;

bool has_line(const std::string &body, const std::string &line) {
  return body.find(line + '\n') != std::string::npos;
}

ENGINE_TEST(metrics_render_counters_seconds_and_settings) {
  RuntimeMetrics metrics;
  metrics.requests = 7;
  metrics.invalid_requests = 2;
  metrics.failures = 1;
  metrics.inference_microseconds = 2500000;
  metrics.decode_batch_steps[3] = 4;
  metrics.suffix_prefill_batch_calls[24] = 5;
  RuntimeSettings settings;
  settings.fp8_decode.emplace();
  settings.scheduler.prefill_quantum_tokens = 256;

  const auto body = render_metrics(metrics, settings);

  CHECK(has_line(body, "carat_requests_total 7"));
  CHECK(has_line(body, "carat_invalid_requests_total 2"));
  CHECK(has_line(body, "carat_failures_total 1"));
  CHECK(has_line(body, "carat_inference_seconds_total 2.5"));
  CHECK(has_line(body, "carat_fp8_decode 1"));
  CHECK(has_line(body, "carat_prefill_quantum_tokens 256"));
  CHECK(has_line(body, "carat_decode_batch_steps_total{batch=\"3\"} 4"));
  CHECK(has_line(body, "carat_suffix_prefill_batch_calls_total{batch=\"24\"} 5"));
}

ENGINE_TEST(metrics_render_large_second_counters_exactly) {
  RuntimeMetrics metrics;
  metrics.inference_microseconds = 12345678901234;
  metrics.speculative_verify_microseconds = 7000000;

  const auto body = render_metrics(metrics, RuntimeSettings{});

  CHECK(has_line(body, "carat_inference_seconds_total 12345678.901234"));
  CHECK(has_line(body, "carat_speculative_verify_seconds_total 7"));
  CHECK(has_line(body, "carat_speculative_draft_seconds_total 0"));
}

ENGINE_TEST(metrics_render_each_series_once) {
  const auto body = render_metrics(RuntimeMetrics{}, RuntimeSettings{});

  std::istringstream lines(body);
  std::set<std::string> series;
  std::size_t count = 0;
  for (std::string line; std::getline(lines, line); ++count) {
    CHECK(line.starts_with("carat_"));
    series.insert(line.substr(0, line.rfind(' ')));
  }

  constexpr std::size_t counters = 32;
  constexpr std::size_t setting_gauges = 3;
  constexpr std::size_t series_per_batch_size = 2;
  CHECK(series.size() == count);
  CHECK(count == counters + setting_gauges +
                     series_per_batch_size * static_cast<std::size_t>(maximum_slots));
}

} // namespace
