#include "runtime/metrics.h"

#include "runtime/decimal.h"
#include "runtime/settings.h"

#include <sstream>
#include <string_view>

namespace carat {
namespace {

enum class Unit { count, seconds };

struct Series {
  std::string_view name;
  Counter RuntimeMetrics::*value;
  Unit unit{Unit::count};
};

constexpr std::array series{
    Series{"carat_requests_total", &RuntimeMetrics::requests},
    Series{"carat_invalid_requests_total", &RuntimeMetrics::invalid_requests},
    Series{"carat_failures_total", &RuntimeMetrics::failures},
    Series{"carat_cancelled_requests_total", &RuntimeMetrics::cancelled},
    Series{"carat_input_tokens_total", &RuntimeMetrics::input_tokens},
    Series{"carat_output_tokens_total", &RuntimeMetrics::output_tokens},
    Series{"carat_inference_seconds_total", &RuntimeMetrics::inference_microseconds, Unit::seconds},
    Series{"carat_queued_requests", &RuntimeMetrics::queued},
    Series{"carat_active_requests", &RuntimeMetrics::active},
    Series{"carat_decoding_requests", &RuntimeMetrics::decoding},
    Series{"carat_prefix_cache_hits_total", &RuntimeMetrics::prefix_cache_hits},
    Series{"carat_prefix_tokens_reused_total", &RuntimeMetrics::prefix_tokens_reused},
    Series{"carat_cache_affinity_admissions_total", &RuntimeMetrics::cache_affinity_admissions},
    Series{"carat_cache_affinity_bypassed_requests_total",
           &RuntimeMetrics::cache_affinity_bypassed_requests},
    Series{"carat_prefill_tokens_computed_total", &RuntimeMetrics::prefill_tokens_computed},
    Series{"carat_kv_cache_clones_total", &RuntimeMetrics::kv_cache_clones},
    Series{"carat_kv_cache_clone_bytes_total", &RuntimeMetrics::kv_cache_clone_bytes},
    Series{"carat_suffix_prefill_batches_total", &RuntimeMetrics::suffix_prefill_batches},
    Series{"carat_interleaved_suffix_prefill_batches_total",
           &RuntimeMetrics::interleaved_suffix_prefill_batches},
    Series{"carat_suffix_prefill_requests_total", &RuntimeMetrics::suffix_prefill_requests},
    Series{"carat_suffix_prefill_tokens_total", &RuntimeMetrics::suffix_prefill_tokens},
    Series{"carat_suffix_prefill_seconds_total", &RuntimeMetrics::suffix_prefill_microseconds,
           Unit::seconds},
    Series{"carat_suffix_prefill_max_seconds", &RuntimeMetrics::maximum_suffix_prefill_microseconds,
           Unit::seconds},
    Series{"carat_suffix_prefill_layer_slices_total", &RuntimeMetrics::suffix_prefill_layer_slices},
    Series{"carat_speculative_cycles_total", &RuntimeMetrics::speculative_cycles},
    Series{"carat_speculative_proposed_tokens_total", &RuntimeMetrics::speculative_proposed_tokens},
    Series{"carat_speculative_accepted_tokens_total", &RuntimeMetrics::speculative_accepted_tokens},
    Series{"carat_speculative_first_token_rejections_total",
           &RuntimeMetrics::speculative_first_token_rejections},
    Series{"carat_speculative_draft_seconds_total", &RuntimeMetrics::speculative_draft_microseconds,
           Unit::seconds},
    Series{"carat_speculative_verify_seconds_total",
           &RuntimeMetrics::speculative_verify_microseconds, Unit::seconds},
    Series{"carat_contiguous_decode_steps_total", &RuntimeMetrics::contiguous_decode_steps},
    Series{"carat_fragmented_decode_steps_total", &RuntimeMetrics::fragmented_decode_steps},
};

} // namespace

void update_maximum(Counter &target, std::uint64_t value) {
  std::uint64_t current = target.load(std::memory_order_relaxed);
  while (current < value &&
         !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
  }
}

std::string render_metrics(const RuntimeMetrics &metrics, const RuntimeSettings &settings) {
  std::ostringstream body;
  for (const Series &entry : series) {
    const std::uint64_t value = (metrics.*entry.value).load();
    body << entry.name << ' ';
    if (entry.unit == Unit::seconds) {
      write_decimal(body, value, 6);
    } else {
      body << value;
    }
    body << '\n';
  }

  body << "carat_fp8_decode " << (settings.fp8_decode ? 1 : 0) << '\n'
       << "carat_prefill_quantum_tokens " << settings.scheduler.prefill_quantum_tokens << '\n'
       << "carat_idle_prefill_quantum_tokens " << settings.scheduler.idle_prefill_quantum_tokens
       << '\n';

  for (std::size_t batch = 1; batch <= maximum_slots; ++batch) {
    body << "carat_decode_batch_steps_total{batch=\"" << batch << "\"} "
         << metrics.decode_batch_steps[batch].load() << '\n'
         << "carat_suffix_prefill_batch_calls_total{batch=\"" << batch << "\"} "
         << metrics.suffix_prefill_batch_calls[batch].load() << '\n';
  }
  return body.str();
}

} // namespace carat
