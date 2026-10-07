#pragma once

#include "runtime/completion_job.h"
#include "runtime/completion_response.h"

#include <functional>
#include <string_view>
#include <variant>

namespace carat {

using ChunkWriter = std::function<bool(std::string_view chunk)>;

// Failed jobs omit the finish event and [DONE].
inline void stream_completion(CompletionJob &job, const ChunkWriter &write,
                              const std::function<void()> &cancel) {
  const std::size_t input_tokens = job.request.input_ids.size();
  std::size_t emitted_tokens = 0;

  while (true) {
    JobEvent event = job.next_event();

    if (const int *token = std::get_if<int>(&event)) {
      ++emitted_tokens;
      const CompletionUsage usage{.input_tokens = input_tokens,
                                  .cached_input_tokens =
                                      job.cached_input_tokens.load(std::memory_order_acquire),
                                  .output_tokens = emitted_tokens};
      if (!write(token_event(*token, usage))) {
        cancel();
        return;
      }
      continue;
    }

    const auto *result = std::get_if<CompletionResult>(&std::get<JobOutcome>(event));
    if (result == nullptr)
      return;

    const CompletionUsage usage{.input_tokens = input_tokens,
                                .cached_input_tokens = result->cached_input_tokens,
                                .output_tokens = result->output_ids.size()};
    if (!write(finish_event(result->finish_reason, usage)) || !write(stream_done_event))
      cancel();
    return;
  }
}

} // namespace carat
