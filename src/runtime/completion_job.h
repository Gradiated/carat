#pragma once

#include "runtime/completion_request.h"
#include "runtime/completion_response.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>
#include <variant>

namespace carat {

// The scheduler logs the cause with the request id; clients see only that the job failed.
struct JobFailure {};

using JobOutcome = std::variant<CompletionResult, JobFailure>;
using JobEvent = std::variant<int, JobOutcome>;

class CompletionJob {
public:
  CompletionJob(std::uint64_t id, CompletionRequest completion)
      : request_id(id), request(std::move(completion)),
        queued_at(std::chrono::steady_clock::now()) {}

  void emit(int token) {
    {
      std::lock_guard lock(mutex_);
      emitted_ids_.push_back(token);
    }
    condition_.notify_all();
  }

  void settle(JobOutcome outcome) {
    {
      std::lock_guard lock(mutex_);
      outcome_ = std::move(outcome);
    }
    condition_.notify_all();
  }

  [[nodiscard]] JobOutcome wait_for_outcome() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] { return outcome_.has_value(); });
    return std::move(*outcome_);
  }

  // Tokens drain before the outcome so a stream never ends ahead of an emitted token.
  [[nodiscard]] JobEvent next_event() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&] { return !emitted_ids_.empty() || outcome_.has_value(); });
    if (emitted_ids_.empty())
      return std::move(*outcome_);

    const int token = emitted_ids_.front();
    emitted_ids_.pop_front();
    return token;
  }

  const std::uint64_t request_id;
  const CompletionRequest request;
  const std::chrono::steady_clock::time_point queued_at;
  std::atomic<bool> cancelled{false};
  std::atomic<std::uint64_t> cached_input_tokens{0};
  std::uint32_t admission_bypasses{0};

private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<int> emitted_ids_;
  std::optional<JobOutcome> outcome_;
};

} // namespace carat
