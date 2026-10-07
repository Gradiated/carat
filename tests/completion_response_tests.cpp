#include "runtime/completion_response.h"

#include "runtime/completion_job.h"
#include "runtime/completion_stream.h"

#include "support/test_support.h"

#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace {

using carat::completion_json;
using carat::CompletionJob;
using carat::CompletionResult;
using carat::finish_event;
using carat::FinishReason;
using carat::JobEvent;
using carat::JobFailure;
using carat::JobOutcome;
using carat::stream_completion;
using carat::stream_done_event;
using carat::token_event;

struct StreamRecording {
  std::vector<std::string> chunks;
  bool cancelled{false};
};

CompletionJob make_job() {
  return CompletionJob(
      1, {.input_ids = {3, 4}, .max_tokens = 8, .stop_ids = {}, .stream = true, .priority = 0});
}

StreamRecording record_stream(CompletionJob &job,
                              std::optional<std::size_t> failing_write = std::nullopt) {
  StreamRecording recording;
  stream_completion(
      job,
      [&](std::string_view chunk) {
        if (recording.chunks.size() == failing_write)
          return false;

        recording.chunks.emplace_back(chunk);
        return true;
      },
      [&] { recording.cancelled = true; });
  return recording;
}

StreamRecording stream(const std::vector<int> &emitted, JobOutcome outcome,
                       std::optional<std::size_t> failing_write = std::nullopt) {
  auto job = make_job();
  for (const int token : emitted)
    job.emit(token);
  job.settle(std::move(outcome));
  return record_stream(job, failing_write);
}

ENGINE_TEST(completion_json_reports_tokens_usage_timing_and_finish_reason) {
  const CompletionResult result{.output_ids = {5, 6, 7},
                                .cached_input_tokens = 2,
                                .queue_microseconds = 1500,
                                .ttft_microseconds = 4000,
                                .decode_microseconds = 3000,
                                .maximum_tpot_microseconds = 2000,
                                .total_microseconds = 9000,
                                .finish_reason = FinishReason::length};

  CHECK(
      completion_json(result, 4) ==
      R"({"output_ids":[5,6,7],"usage":{"input_tokens":4,"cached_input_tokens":2,"output_tokens":3},)"
      R"("timing":{"queue_ms":1.5,"ttft_ms":4,"mean_tpot_ms":1.5,"max_tpot_ms":2,"total_ms":9},)"
      R"("finish_reason":"length"})"
      "\n");
}

ENGINE_TEST(completion_json_reports_zero_mean_tpot_for_a_single_token) {
  const CompletionResult result{.output_ids = {5}, .decode_microseconds = 700};

  const auto json = completion_json(result, 1);

  CHECK(json.find(R"("mean_tpot_ms":0,)") != std::string::npos);
  CHECK(json.find(R"("finish_reason":"stop")") != std::string::npos);
}

ENGINE_TEST(completion_json_writes_long_timings_exactly) {
  const CompletionResult result{
      .output_ids = {5, 6, 7, 8}, .decode_microseconds = 1000, .total_microseconds = 1234567800};

  const auto json = completion_json(result, 1);

  CHECK(json.find(R"("mean_tpot_ms":0.333,)") != std::string::npos);
  CHECK(json.find(R"("total_ms":1234567.8})") != std::string::npos);
}

ENGINE_TEST(stream_events_carry_cumulative_usage) {
  CHECK(token_event(42, {.input_tokens = 3, .cached_input_tokens = 1, .output_tokens = 2}) ==
        "data: "
        "{\"token_id\":42,\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_"
        "tokens\":2}}\n\n");
  CHECK(finish_event(FinishReason::stop,
                     {.input_tokens = 3, .cached_input_tokens = 1, .output_tokens = 2}) ==
        "data: {\"finish_reason\":\"stop\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,"
        "\"output_tokens\":2}}\n\n");
  CHECK(stream_done_event == "data: [DONE]\n\n");
}

ENGINE_TEST(a_completed_stream_ends_with_the_finish_event_and_done) {
  const auto recording =
      stream({5, 6}, CompletionResult{.output_ids = {5, 6}, .finish_reason = FinishReason::length});

  REQUIRE(recording.chunks.size() == 4);
  CHECK(recording.chunks[0] ==
        token_event(5, {.input_tokens = 2, .cached_input_tokens = 0, .output_tokens = 1}));
  CHECK(recording.chunks[1] ==
        token_event(6, {.input_tokens = 2, .cached_input_tokens = 0, .output_tokens = 2}));
  CHECK(recording.chunks[2] ==
        finish_event(FinishReason::length,
                     {.input_tokens = 2, .cached_input_tokens = 0, .output_tokens = 2}));
  CHECK(recording.chunks[3] == stream_done_event);
  CHECK(!recording.cancelled);
}

ENGINE_TEST(a_failed_stream_ends_without_a_finish_event_or_done) {
  const auto recording = stream({5}, JobFailure{});

  REQUIRE(recording.chunks.size() == 1);
  CHECK(recording.chunks[0].find("token_id") != std::string::npos);
  CHECK(!recording.cancelled);
}

ENGINE_TEST(a_failed_token_write_cancels_the_job) {
  const auto recording = stream({5, 6}, CompletionResult{.output_ids = {5, 6}}, 1);

  CHECK(recording.chunks.size() == 1);
  CHECK(recording.cancelled);
}

ENGINE_TEST(a_failed_finish_write_cancels_the_job) {
  const auto recording = stream({5}, CompletionResult{.output_ids = {5}}, 1);

  CHECK(recording.chunks.size() == 1);
  CHECK(recording.cancelled);
}

ENGINE_TEST(a_job_yields_emitted_tokens_before_its_outcome) {
  auto job = make_job();
  job.emit(5);
  job.settle(JobFailure{});

  CHECK(std::get<int>(job.next_event()) == 5);
  const JobEvent last = job.next_event();
  REQUIRE(std::holds_alternative<JobOutcome>(last));
  CHECK(std::holds_alternative<JobFailure>(std::get<JobOutcome>(last)));
}

ENGINE_TEST(waiting_for_an_outcome_blocks_until_the_job_settles) {
  auto job = make_job();

  std::thread producer([&] {
    job.emit(5);
    job.settle(CompletionResult{.output_ids = {5}, .finish_reason = FinishReason::length});
  });
  const JobOutcome outcome = job.wait_for_outcome();
  producer.join();

  const auto *result = std::get_if<CompletionResult>(&outcome);
  REQUIRE(result != nullptr);
  CHECK(result->output_ids == std::vector<int>{5});
}

ENGINE_TEST(a_stream_receives_every_token_from_a_concurrent_producer) {
  constexpr int token_count = 256;
  auto job = make_job();

  std::thread producer([&] {
    std::vector<int> output_ids;
    for (int token = 0; token < token_count; ++token) {
      job.emit(token);
      output_ids.push_back(token);
    }
    job.settle(CompletionResult{.output_ids = std::move(output_ids)});
  });
  const auto recording = record_stream(job);
  producer.join();

  REQUIRE(recording.chunks.size() == token_count + 2);
  for (int token = 0; token < token_count; ++token) {
    const auto usage = carat::CompletionUsage{.input_tokens = 2,
                                              .cached_input_tokens = 0,
                                              .output_tokens = static_cast<std::size_t>(token + 1)};
    CHECK(recording.chunks[static_cast<std::size_t>(token)] == token_event(token, usage));
  }
  CHECK(recording.chunks.back() == stream_done_event);
  CHECK(!recording.cancelled);
}

} // namespace
