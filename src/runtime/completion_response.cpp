#include "runtime/completion_response.h"

#include "runtime/decimal.h"

#include <sstream>

namespace carat {
namespace {

std::string_view finish_reason_json(FinishReason reason) {
  return reason == FinishReason::stop ? "\"stop\"" : "\"length\"";
}

void write_usage(std::ostream &output, const CompletionUsage &usage) {
  output << "\"usage\":{\"input_tokens\":" << usage.input_tokens
         << ",\"cached_input_tokens\":" << usage.cached_input_tokens
         << ",\"output_tokens\":" << usage.output_tokens << '}';
}

struct Milliseconds {
  std::uint64_t microseconds;
};

std::ostream &operator<<(std::ostream &output, Milliseconds value) {
  write_decimal(output, value.microseconds, 3);
  return output;
}

} // namespace

std::string completion_json(const CompletionResult &result, std::size_t input_tokens) {
  const std::size_t output_tokens = result.output_ids.size();
  const Milliseconds mean_tpot{output_tokens > 1 ? result.decode_microseconds / (output_tokens - 1)
                                                 : 0};

  std::ostringstream output;
  output << "{\"output_ids\":[";
  for (std::size_t index = 0; index < output_tokens; ++index) {
    if (index != 0)
      output << ',';
    output << result.output_ids[index];
  }
  output << "],";

  write_usage(output, {input_tokens, result.cached_input_tokens, output_tokens});
  output << ",\"timing\":{\"queue_ms\":" << Milliseconds{result.queue_microseconds}
         << ",\"ttft_ms\":" << Milliseconds{result.ttft_microseconds}
         << ",\"mean_tpot_ms\":" << mean_tpot
         << ",\"max_tpot_ms\":" << Milliseconds{result.maximum_tpot_microseconds}
         << ",\"total_ms\":" << Milliseconds{result.total_microseconds}
         << "},\"finish_reason\":" << finish_reason_json(result.finish_reason) << "}\n";
  return output.str();
}

std::string token_event(int token, const CompletionUsage &usage) {
  std::ostringstream output;
  output << "data: {\"token_id\":" << token << ',';
  write_usage(output, usage);
  output << "}\n\n";
  return output.str();
}

std::string finish_event(FinishReason reason, const CompletionUsage &usage) {
  std::ostringstream output;
  output << "data: {\"finish_reason\":" << finish_reason_json(reason) << ',';
  write_usage(output, usage);
  output << "}\n\n";
  return output.str();
}

} // namespace carat
