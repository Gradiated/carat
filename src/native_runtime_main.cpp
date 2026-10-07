#include "carat/batch_model_runner.h"
#include "carat/device_weights.h"
#include "carat/gemma4_model.h"
#include "carat/http/http.h"
#include "carat/http/json.h"
#include "runtime/capacity.h"
#include "runtime/completion_job.h"
#include "runtime/completion_request.h"
#include "runtime/completion_response.h"
#include "runtime/completion_stream.h"
#include "runtime/metrics.h"
#include "runtime/settings.h"
#include "runtime/slot_policy.h"
#include "runtime/speculation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace {

using carat::CompletionJob;
using carat::CompletionRequest;
using carat::CompletionResult;
using carat::FinishReason;
using carat::JobFailure;
using carat::JobOutcome;
using carat::maximum_context;
using carat::maximum_slots;
using carat::RuntimeMetrics;
using carat::SchedulerOptions;

constexpr int default_chunk_tokens = 1024;
constexpr int short_prefill_cohort_tokens = 512;

struct RuntimeWeights {
  const carat::DeviceWeightArena &target;
  const carat::Fp8WeightArena *target_fp8{nullptr};
  const carat::DeviceWeightArena *assistant{nullptr};
  const carat::Fp8WeightArena *assistant_fp8{nullptr};
};

enum class FailureSite { admission, prefill, suffix_prefill, speculative_decode, decode };

const char *failure_site_name(FailureSite site) {
  switch (site) {
  case FailureSite::admission:
    return "admission";
  case FailureSite::prefill:
    return "prefill";
  case FailureSite::suffix_prefill:
    return "suffix_prefill";
  case FailureSite::speculative_decode:
    return "speculative_decode";
  case FailureSite::decode:
    return "decode";
  }
  return "unknown";
}

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_us(Clock::time_point from, Clock::time_point to) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(to - from).count());
}

class ContinuousScheduler {
public:
  ContinuousScheduler(const carat::Gemma4Config &config, RuntimeWeights weights,
                      const SchedulerOptions &options, RuntimeMetrics &metrics)
      : runner_(config, weights.target, maximum_slots, maximum_context, weights.target_fp8,
                weights.assistant, weights.assistant_fp8),
        slots_(maximum_slots), metrics_(metrics), options_(options),
        config_layer_count_(config.layers.size()) {
    warm_up();
    worker_ = std::thread([this] { run(); });
  }

  ~ContinuousScheduler() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    worker_.join();
  }

  JobOutcome complete(CompletionRequest request) {
    return submit(std::move(request))->wait_for_outcome();
  }

  std::shared_ptr<CompletionJob> submit(CompletionRequest request) {
    auto job = std::make_shared<CompletionJob>(
        next_request_id_.fetch_add(1, std::memory_order_relaxed), std::move(request));
    {
      std::lock_guard lock(mutex_);
      if (stopping_)
        throw std::runtime_error("runtime scheduler is stopping");

      pending_.push_back(job);
      metrics_.queued.fetch_add(1);
    }
    condition_.notify_one();
    return job;
  }

  void cancel(const std::shared_ptr<CompletionJob> &job) {
    job->cancelled.store(true, std::memory_order_release);
    condition_.notify_one();
  }

private:
  struct Slot {
    std::shared_ptr<CompletionJob> job;
    std::vector<int> cached_tokens;
    std::vector<int> output_ids;
    int token_to_append{0};
    std::uint64_t queue_microseconds{0};
    std::uint64_t ttft_microseconds{0};
    std::uint64_t decode_microseconds{0};
    std::uint64_t maximum_tpot_microseconds{0};
    std::uint64_t age{0};
    std::uint64_t admission_order{0};
    bool decode_ready{false};
    bool speculative_cohort{false};
    Clock::time_point last_token_at;
  };

  void warm_up() {
    std::vector<int> warmup(2048, 2);
    while (!runner_.prefill_slot_chunk(0, warmup, default_chunk_tokens)) {
    }
    runner_.reset_slot(0);

    // cuDNN SDPA plans are keyed by exact shape. Build the active quantum and its one-token tail
    // before readiness so a first cache extension cannot stall every decoder on graph construction.
    std::vector<int> active_prefill_warmup(
        static_cast<std::size_t>(
            std::min(maximum_context, default_chunk_tokens + options_.prefill_quantum_tokens + 1)),
        2);
    static_cast<void>(runner_.prefill_slot_chunk(0, active_prefill_warmup, default_chunk_tokens));
    while (!runner_.prefill_slot_chunk(0, active_prefill_warmup, options_.prefill_quantum_tokens)) {
    }

    runner_.seed_empty_cache_for_benchmark(1);
    for (int active = 1; active <= maximum_slots; ++active) {
      std::vector<int> tokens(static_cast<std::size_t>(active), 2);
      std::vector<int> slot_indices(static_cast<std::size_t>(active));
      for (int index = 0; index < active; ++index)
        slot_indices[static_cast<std::size_t>(index)] = index;
      static_cast<void>(runner_.append_ragged(tokens, slot_indices));
    }
    for (int index = 0; index < maximum_slots; ++index)
      runner_.reset_slot(index);
  }

  Slot &slot(int index) {
    return slots_[static_cast<std::size_t>(index)];
  }
  const Slot &slot(int index) const {
    return slots_[static_cast<std::size_t>(index)];
  }

  static void emit_token(Slot &target, int token) {
    target.output_ids.push_back(token);
    target.job->emit(token);
  }

  std::vector<carat::SlotView> slot_views() const {
    std::vector<carat::SlotView> views;
    views.reserve(slots_.size());
    for (const Slot &candidate : slots_) {
      const bool occupied = candidate.job != nullptr;
      views.push_back({.cached_tokens = candidate.cached_tokens,
                       .occupied = occupied,
                       .age = candidate.age,
                       .prefilling = occupied && !candidate.decode_ready});
    }
    return views;
  }

  std::vector<std::size_t> order_by_position(const std::vector<int> &slot_indices) const {
    std::vector<std::size_t> order(slot_indices.size());
    for (std::size_t row = 0; row < order.size(); ++row)
      order[row] = row;

    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
      return runner_.slot_position(slot_indices[left]) < runner_.slot_position(slot_indices[right]);
    });
    return order;
  }

  template <typename Value>
  static void reorder(std::vector<Value> &values, const std::vector<std::size_t> &order) {
    std::vector<Value> reordered;
    reordered.reserve(order.size());
    for (const std::size_t row : order)
      reordered.push_back(values[row]);
    values = std::move(reordered);
  }

  std::shared_ptr<CompletionJob> detach_job(int index) {
    Slot &detached = slot(index);
    metrics_.active.fetch_sub(1);
    if (detached.decode_ready)
      metrics_.decoding.fetch_sub(1);

    detached.decode_ready = false;
    return std::move(detached.job);
  }

  void start_decode(int index, int token, Clock::time_point at) {
    Slot &started = slot(index);
    started.token_to_append = token;
    emit_token(started, token);
    started.decode_ready = true;
    started.ttft_microseconds = elapsed_us(started.job->queued_at, at);
    started.last_token_at = at;
    metrics_.decoding.fetch_add(1);

    finish_if_done(index, false);
  }

  void finish_if_done(int index, bool stopped) {
    const Slot &current = slot(index);
    if (stopped || current.job->request.stop_ids.contains(current.token_to_append)) {
      complete_slot(index, FinishReason::stop);
    } else if (current.output_ids.size() >=
               static_cast<std::size_t>(current.job->request.max_tokens)) {
      complete_slot(index, FinishReason::length);
    }
  }

  void complete_slot(int index, FinishReason finish_reason) {
    Slot &finished = slot(index);
    CompletionResult result{
        .output_ids = std::move(finished.output_ids),
        .cached_input_tokens = finished.job->cached_input_tokens.load(std::memory_order_acquire),
        .queue_microseconds = finished.queue_microseconds,
        .ttft_microseconds = finished.ttft_microseconds,
        .decode_microseconds = finished.decode_microseconds,
        .maximum_tpot_microseconds = finished.maximum_tpot_microseconds,
        .total_microseconds = elapsed_us(finished.job->queued_at, Clock::now()),
        .finish_reason = finish_reason,
    };

    metrics_.input_tokens.fetch_add(finished.job->request.input_ids.size());
    metrics_.output_tokens.fetch_add(result.output_ids.size());
    metrics_.inference_microseconds.fetch_add(result.total_microseconds);

    const auto job = detach_job(index);
    finished.age = ++age_;
    job->settle(std::move(result));
  }

  void evict_slot(int index, carat::Counter &counter) {
    Slot &evicted = slot(index);

    const auto job = detach_job(index);
    evicted.cached_tokens.clear();
    evicted.output_ids.clear();
    runner_.reset_slot(index);
    counter.fetch_add(1);
    job->settle(JobFailure{});
  }

  void fail_slot(int index, FailureSite site, const std::string &message) {
    std::cerr << "native runtime: request " << slot(index).job->request_id
              << " failed site=" << failure_site_name(site) << " message=" << message << '\n';
    evict_slot(index, metrics_.failures);
  }

  void fail_slots(const std::vector<int> &slot_indices, FailureSite site,
                  const std::string &message) {
    for (const int index : slot_indices) {
      if (slot(index).job)
        fail_slot(index, site, message);
    }
  }

  void cancel_slot(int index) {
    evict_slot(index, metrics_.cancelled);
  }

  bool reap_cancelled() {
    bool reaped = false;
    {
      std::lock_guard lock(mutex_);
      for (auto current = pending_.begin(); current != pending_.end();) {
        if (!(*current)->cancelled.load(std::memory_order_acquire)) {
          ++current;
          continue;
        }

        auto job = std::move(*current);
        current = pending_.erase(current);
        metrics_.queued.fetch_sub(1);
        metrics_.cancelled.fetch_add(1);

        job->settle(JobFailure{});
        reaped = true;
      }
    }

    const bool cohort_cancelled = std::any_of(
        inflight_suffix_slots_.begin(), inflight_suffix_slots_.end(), [&](const int index) {
          return !slot(index).job || slot(index).job->cancelled.load(std::memory_order_acquire);
        });
    if (cohort_cancelled)
      abort_inflight_suffix();

    for (int index = 0; index < maximum_slots; ++index) {
      if (slot(index).job && slot(index).job->cancelled.load(std::memory_order_acquire)) {
        cancel_slot(index);
        reaped = true;
      }
    }
    return reaped;
  }

  void abort_inflight_suffix() {
    runner_.abort_suffix_layer_slices();
    inflight_suffix_slots_.clear();
    inflight_suffix_prompts_.clear();
  }

  bool admit_one() {
    std::shared_ptr<CompletionJob> job;
    carat::Placement placement;
    {
      std::lock_guard lock(mutex_);
      std::vector<carat::PendingView> pending;
      pending.reserve(pending_.size());
      for (const auto &queued : pending_) {
        pending.push_back(
            {queued->request.input_ids, queued->request.priority, queued->admission_bypasses});
      }

      const auto admission = carat::select_admission(pending, slot_views(), options_.admission);
      if (!admission)
        return false;

      if (!admission->bypassed.empty()) {
        for (const std::size_t bypassed : admission->bypassed) {
          auto &charged = pending_[bypassed]->admission_bypasses;
          charged = carat::charge_bypass(charged, options_.admission.max_bypasses);
        }
        metrics_.cache_affinity_admissions.fetch_add(1);
        metrics_.cache_affinity_bypassed_requests.fetch_add(admission->bypassed.size());
      }

      const auto selected =
          pending_.begin() + static_cast<std::ptrdiff_t>(admission->pending_index);
      placement = admission->placement;
      job = std::move(*selected);
      pending_.erase(selected);
      metrics_.queued.fetch_sub(1);
      slot(placement.destination).job = job;
      job->cached_input_tokens.store(static_cast<std::uint64_t>(placement.prefix_tokens),
                                     std::memory_order_release);
    }

    const int index = placement.destination;
    Slot &admitted = slot(index);
    admitted.output_ids.clear();
    admitted.decode_ready = false;
    admitted.speculative_cohort = false;
    admitted.queue_microseconds = elapsed_us(job->queued_at, Clock::now());
    admitted.ttft_microseconds = 0;
    admitted.decode_microseconds = 0;
    admitted.maximum_tpot_microseconds = 0;
    admitted.admission_order = ++admission_order_;
    metrics_.active.fetch_add(1);

    try {
      if (!placement.source) {
        runner_.reset_slot(index);
        admitted.cached_tokens.clear();
      } else {
        const int source = *placement.source;
        if (source != index) {
          const std::uint64_t copied_bytes = runner_.clone_slot(source, index);
          admitted.cached_tokens = slot(source).cached_tokens;
          metrics_.kv_cache_clones.fetch_add(1);
          metrics_.kv_cache_clone_bytes.fetch_add(copied_bytes);
        }
        metrics_.prefix_cache_hits.fetch_add(1);
        metrics_.prefix_tokens_reused.fetch_add(
            static_cast<std::uint64_t>(placement.prefix_tokens));

        if (placement.prefix_tokens == job->request.input_ids.size()) {
          start_decode(index, slot(source).token_to_append, Clock::now());
        }
      }
    } catch (const std::exception &error) {
      fail_slot(index, FailureSite::admission, error.what());
    }
    return true;
  }

  void record_prefill_progress(int index) {
    Slot &prefilled = slot(index);
    const int resident = runner_.slot_position(index);
    const auto &prompt = prefilled.job->request.input_ids;
    prefilled.cached_tokens.assign(prompt.begin(), prompt.begin() + resident);
  }

  bool idle_prefill_batch() {
    std::vector<int> candidates;
    for (int index = 0; index < maximum_slots; ++index) {
      if (slot(index).job && !slot(index).decode_ready)
        candidates.push_back(index);
    }
    if (candidates.size() < 2)
      return false;

    std::stable_sort(candidates.begin(), candidates.end(), [&](const int left, const int right) {
      const int left_position = runner_.slot_position(left);
      const int right_position = runner_.slot_position(right);
      if (left_position != right_position)
        return left_position < right_position;

      return slot(left).admission_order < slot(right).admission_order;
    });
    const auto maximum_requests = static_cast<std::size_t>(
        std::max(1, maximum_context / options_.idle_prefill_quantum_tokens));
    if (candidates.size() > maximum_requests)
      candidates.resize(maximum_requests);

    std::vector<const std::vector<int> *> prompts;
    std::vector<int> before;
    prompts.reserve(candidates.size());
    before.reserve(candidates.size());
    for (const int index : candidates) {
      prompts.push_back(&slot(index).job->request.input_ids);
      before.push_back(runner_.slot_position(index));
    }

    try {
      const auto predictions =
          runner_.prefill_slots_chunks(candidates, prompts, options_.idle_prefill_quantum_tokens);
      const auto end = Clock::now();

      for (std::size_t row = 0; row < candidates.size(); ++row) {
        const int index = candidates[row];
        const int resident = runner_.slot_position(index);
        const auto advanced = static_cast<std::uint64_t>(resident - before[row]);
        metrics_.prefill_tokens_computed.fetch_add(advanced);
        record_prefill_progress(index);

        if (predictions[row])
          start_decode(index, *predictions[row], end);
      }

    } catch (const std::exception &error) {
      fail_slots(candidates, FailureSite::prefill, error.what());
    }
    return true;
  }

  bool prefill_step() {
    const bool decode_is_active = decoding_count() > 0;
    if (!decode_is_active && idle_prefill_batch())
      return true;

    int selected = -1;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (int index = 0; index < maximum_slots; ++index) {
      const Slot &candidate = slot(index);
      if (candidate.job && !candidate.decode_ready && candidate.admission_order < oldest) {
        oldest = candidate.admission_order;
        selected = index;
      }
    }
    if (selected < 0)
      return false;

    try {
      const int before = runner_.slot_position(selected);
      const int quantum =
          decode_is_active ? options_.prefill_quantum_tokens : options_.idle_prefill_quantum_tokens;
      const auto prediction =
          runner_.prefill_slot_chunk(selected, slot(selected).job->request.input_ids, quantum);
      const auto end = Clock::now();
      const int resident = runner_.slot_position(selected);
      const auto advanced = static_cast<std::uint64_t>(resident - before);

      metrics_.prefill_tokens_computed.fetch_add(advanced);
      record_prefill_progress(selected);

      if (prediction)
        start_decode(selected, *prediction, end);
    } catch (const std::exception &error) {
      fail_slot(selected, FailureSite::prefill, error.what());
    }
    return true;
  }

  struct DecodeBatch {
    std::vector<int> slots;
    std::vector<int> tokens;
    bool contiguous{true};
    Clock::time_point begin;
  };

  void record_decode_timing(Slot &decoded, Clock::time_point token_time, int emitted_tokens) {
    const std::uint64_t cycle = elapsed_us(decoded.last_token_at, token_time);
    const std::uint64_t tpot = cycle / static_cast<std::uint64_t>(std::max(1, emitted_tokens));
    decoded.decode_microseconds += cycle;
    decoded.maximum_tpot_microseconds = std::max(decoded.maximum_tpot_microseconds, tpot);
    decoded.last_token_at = token_time;
  }

  bool decode_step() {
    DecodeBatch batch;
    for (int index = 0; index < maximum_slots; ++index) {
      if (!slot(index).job || !slot(index).decode_ready)
        continue;

      batch.slots.push_back(index);
      batch.tokens.push_back(slot(index).token_to_append);
    }
    if (batch.slots.empty())
      return false;

    metrics_.decode_batch_steps.at(batch.slots.size()).fetch_add(1);
    for (std::size_t row = 1; row < batch.slots.size(); ++row) {
      batch.contiguous &= batch.slots[row] == batch.slots.front() + static_cast<int>(row);
    }
    (batch.contiguous ? metrics_.contiguous_decode_steps : metrics_.fragmented_decode_steps)
        .fetch_add(1);

    batch.begin = Clock::now();

    const bool continuing_speculative_cohort =
        std::any_of(batch.slots.begin(), batch.slots.end(),
                    [&](int index) { return slot(index).speculative_cohort; });
    const bool speculate =
        batch.slots.size() >= static_cast<std::size_t>(options_.speculative_minimum_batch) ||
        continuing_speculative_cohort;
    if (runner_.has_assistant() && speculate) {
      speculative_decode(batch);
    } else {
      plain_decode(batch);
    }
    return true;
  }

  void plain_decode(const DecodeBatch &batch) {
    try {
      const auto predictions = runner_.append_ragged(batch.tokens, batch.slots);
      const auto token_time = Clock::now();

      for (std::size_t row = 0; row < batch.slots.size(); ++row) {
        const int index = batch.slots[row];
        Slot &decoded = slot(index);
        decoded.cached_tokens.push_back(decoded.token_to_append);
        decoded.token_to_append = predictions[row];
        emit_token(decoded, decoded.token_to_append);
        record_decode_timing(decoded, token_time, 1);
        finish_if_done(index, false);
      }
    } catch (const std::exception &error) {
      fail_slots(batch.slots, FailureSite::decode, error.what());
    }
  }

  void speculative_decode(DecodeBatch batch) {
    try {
      // Keep a speculative cohort active as requests finish; only its admission is batch-size
      // gated.
      for (const int index : batch.slots)
        slot(index).speculative_cohort = true;

      // Monotonic ragged contexts let the verifier split the cohort into narrower global-attention
      // GEMMs instead of padding every request to the longest one.
      const auto order = order_by_position(batch.slots);
      reorder(batch.slots, order);

      std::vector<int> last_tokens;
      std::vector<int> target_first;
      last_tokens.reserve(batch.slots.size());
      target_first.reserve(batch.slots.size());
      for (const int index : batch.slots) {
        if (slot(index).cached_tokens.empty()) {
          throw std::runtime_error("assistant request has no resident token");
        }

        last_tokens.push_back(slot(index).cached_tokens.back());
        target_first.push_back(slot(index).token_to_append);
      }

      const int depth = options_.speculative_depth;
      const auto draft_begin = Clock::now();
      const auto proposals = runner_.draft_four(last_tokens, batch.slots, target_first, depth);
      const auto draft_end = Clock::now();

      std::vector<int> acceptance_caps;
      acceptance_caps.reserve(batch.slots.size());
      for (std::size_t row = 0; row < batch.slots.size(); ++row) {
        const Slot &drafted = slot(batch.slots[row]);
        const int remaining =
            drafted.job->request.max_tokens - static_cast<int>(drafted.output_ids.size());
        acceptance_caps.push_back(
            carat::acceptance_cap(depth, remaining, proposals[row], drafted.job->request.stop_ids));
      }

      const auto verification =
          runner_.verify_greedy_proposals(proposals, target_first, batch.slots, acceptance_caps);
      const auto verify_end = Clock::now();
      metrics_.speculative_draft_microseconds.fetch_add(elapsed_us(draft_begin, draft_end));
      metrics_.speculative_verify_microseconds.fetch_add(elapsed_us(draft_end, verify_end));
      metrics_.speculative_cycles.fetch_add(1);
      metrics_.speculative_proposed_tokens.fetch_add(static_cast<std::uint64_t>(depth) *
                                                     batch.slots.size());
      for (const int accepted : verification.accepted_tokens) {
        metrics_.speculative_accepted_tokens.fetch_add(static_cast<std::uint64_t>(accepted));
        if (accepted == 0)
          metrics_.speculative_first_token_rejections.fetch_add(1);
      }

      std::vector<int> fallback_slots;
      std::vector<int> fallback_tokens;
      for (std::size_t row = 0; row < batch.slots.size(); ++row) {
        if (verification.accepted_tokens[row] != 0)
          continue;

        fallback_slots.push_back(batch.slots[row]);
        fallback_tokens.push_back(target_first[row]);
      }
      std::vector<int> fallback_next;
      if (!fallback_slots.empty())
        fallback_next = runner_.append_ragged(fallback_tokens, fallback_slots);

      const auto token_time = Clock::now();

      std::size_t fallback_index = 0;
      for (std::size_t row = 0; row < batch.slots.size(); ++row) {
        const int index = batch.slots[row];
        Slot &decoded = slot(index);
        const int accepted = verification.accepted_tokens[row];
        const std::span<const int> accepted_drafts =
            accepted > 1 ? std::span<const int>(proposals[row])
                               .subspan(1, static_cast<std::size_t>(accepted - 1))
                         : std::span<const int>{};
        const int next_token =
            accepted == 0 ? fallback_next.at(fallback_index++) : verification.next_tokens[row];
        const auto commit = carat::commit_speculation(decoded.token_to_append, accepted_drafts,
                                                      next_token, decoded.job->request.stop_ids);

        decoded.cached_tokens.insert(decoded.cached_tokens.end(), commit.resident_tokens.begin(),
                                     commit.resident_tokens.end());
        for (const int token : commit.emitted_tokens)
          emit_token(decoded, token);
        decoded.token_to_append = commit.next_token;

        record_decode_timing(decoded, token_time, static_cast<int>(commit.emitted_tokens.size()));
        finish_if_done(index, commit.stopped);
      }
    } catch (const std::exception &error) {
      fail_slots(batch.slots, FailureSite::speculative_decode, error.what());
    }
  }

  bool has_active() const {
    return std::any_of(slots_.begin(), slots_.end(),
                       [](const Slot &candidate) { return candidate.job != nullptr; });
  }

  int decoding_count() const {
    return static_cast<int>(std::count_if(slots_.begin(), slots_.end(), [](const Slot &candidate) {
      return candidate.decode_ready;
    }));
  }

  int remaining_prompt_tokens(int index) const {
    return static_cast<int>(slot(index).job->request.input_ids.size()) -
           runner_.slot_position(index);
  }

  bool drain_short_prefill_cohort() {
    const bool resuming_layer_slices = !inflight_suffix_slots_.empty();
    const int decoding = decoding_count();
    const bool has_decoding = decoding > 0;
    const bool decode_pressure = decoding >= options_.interleaved_suffix_layer_minimum_decode;
    std::vector<int> cohort_slots = inflight_suffix_slots_;
    std::vector<const std::vector<int> *> cohort_prompts = inflight_suffix_prompts_;
    std::vector<int> cohort_tokens;

    if (!resuming_layer_slices) {
      for (int index = 0; index < maximum_slots; ++index) {
        if (!slot(index).job || slot(index).decode_ready)
          continue;

        const int remaining = remaining_prompt_tokens(index);
        if (remaining <= 0 || remaining > short_prefill_cohort_tokens)
          continue;

        cohort_slots.push_back(index);
        cohort_prompts.push_back(&slot(index).job->request.input_ids);
        cohort_tokens.push_back(remaining);
      }
      if (cohort_slots.size() < 2)
        return false;

      const auto order = order_by_position(cohort_slots);
      reorder(cohort_slots, order);
      reorder(cohort_prompts, order);
      reorder(cohort_tokens, order);
    }

    try {
      const auto prefill_begin = Clock::now();
      std::vector<std::optional<int>> predictions(cohort_slots.size());
      const bool use_layer_slices =
          resuming_layer_slices ||
          (decode_pressure && options_.interleaved_suffix_layer_quantum > 0);
      bool complete = true;

      if (use_layer_slices) {
        if (!resuming_layer_slices) {
          inflight_suffix_slots_ = cohort_slots;
          inflight_suffix_prompts_ = cohort_prompts;
        }
        const int layers = decode_pressure ? options_.interleaved_suffix_layer_quantum
                                           : static_cast<int>(config_layer_count_);
        const auto result =
            runner_.prefill_slots_suffix_layer_slice(cohort_slots, cohort_prompts, layers);
        metrics_.suffix_prefill_layer_slices.fetch_add(1);
        complete = result.complete;
        if (complete)
          std::copy(result.predictions.begin(), result.predictions.end(), predictions.begin());
      } else {
        const auto full_predictions = runner_.prefill_slots_suffix(cohort_slots, cohort_prompts);
        std::copy(full_predictions.begin(), full_predictions.end(), predictions.begin());
      }

      const auto prediction_time = Clock::now();
      const std::uint64_t prefill_microseconds = elapsed_us(prefill_begin, prediction_time);
      std::uint64_t computed_tokens = 0;
      for (const int tokens : cohort_tokens)
        computed_tokens += static_cast<std::uint64_t>(tokens);

      metrics_.suffix_prefill_microseconds.fetch_add(prefill_microseconds);
      carat::update_maximum(metrics_.maximum_suffix_prefill_microseconds, prefill_microseconds);

      if (!resuming_layer_slices) {
        metrics_.suffix_prefill_batches.fetch_add(1);
        if (has_decoding)
          metrics_.interleaved_suffix_prefill_batches.fetch_add(1);
        metrics_.suffix_prefill_requests.fetch_add(cohort_slots.size());
        metrics_.suffix_prefill_batch_calls.at(cohort_slots.size()).fetch_add(1);
        metrics_.prefill_tokens_computed.fetch_add(computed_tokens);
        metrics_.suffix_prefill_tokens.fetch_add(computed_tokens);
      }
      if (!complete)
        return true;

      inflight_suffix_slots_.clear();
      inflight_suffix_prompts_.clear();
      for (std::size_t row = 0; row < cohort_slots.size(); ++row) {
        const int index = cohort_slots[row];
        record_prefill_progress(index);
        if (predictions[row])
          start_decode(index, *predictions[row], prediction_time);
      }
    } catch (const std::exception &error) {
      abort_inflight_suffix();
      fail_slots(cohort_slots, FailureSite::suffix_prefill, error.what());
      return true;
    }

    // Time to first token counts from when the whole cohort is released, not per request.
    const auto release_time = Clock::now();
    for (const int index : cohort_slots) {
      Slot &released = slot(index);
      if (!released.job || !released.decode_ready)
        continue;

      released.ttft_microseconds = elapsed_us(released.job->queued_at, release_time);
      released.last_token_at = release_time;
    }
    return true;
  }

  bool wait_for_short_prefill_cohort() {
    if (!inflight_suffix_slots_.empty())
      return false;
    if (options_.short_prefill_batch_window_microseconds <= 0)
      return false;

    std::optional<Clock::time_point> oldest;
    for (int index = 0; index < maximum_slots; ++index) {
      const Slot &candidate = slot(index);
      if (!candidate.job)
        continue;
      if (candidate.decode_ready)
        return false;

      const int remaining = remaining_prompt_tokens(index);
      if (remaining <= 0 || remaining > short_prefill_cohort_tokens)
        return false;
      if (!oldest || candidate.job->queued_at < *oldest)
        oldest = candidate.job->queued_at;
    }
    if (!oldest)
      return false;

    const auto window = std::chrono::microseconds(options_.short_prefill_batch_window_microseconds);
    const auto deadline = *oldest + window;
    if (Clock::now() >= deadline)
      return false;

    std::unique_lock lock(mutex_);
    condition_.wait_until(lock, deadline, [&] { return stopping_ || !pending_.empty(); });
    return !pending_.empty();
  }

  void run() {
    try {
      schedule();
    } catch (const std::exception &error) {
      std::cerr << "native runtime: scheduler stopped: " << error.what() << '\n';
      std::abort();
    }
  }

  void schedule() {
    while (true) {
      bool worked = false;
      if (reap_cancelled())
        worked = true;
      while (admit_one())
        worked = true;
      if (wait_for_short_prefill_cohort())
        continue;
      if (drain_short_prefill_cohort())
        worked = true;
      if (decode_step())
        worked = true;
      if (inflight_suffix_slots_.empty() && prefill_step())
        worked = true;
      if (worked)
        continue;

      std::unique_lock lock(mutex_);
      condition_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
      if (stopping_ && pending_.empty() && !has_active())
        return;
    }
  }

  carat::Gemma4BatchModelRunner runner_;
  std::vector<Slot> slots_;
  RuntimeMetrics &metrics_;
  const SchedulerOptions options_;
  const std::size_t config_layer_count_;
  std::vector<int> inflight_suffix_slots_;
  std::vector<const std::vector<int> *> inflight_suffix_prompts_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::shared_ptr<CompletionJob>> pending_;
  bool stopping_{false};
  std::uint64_t age_{0};
  std::uint64_t admission_order_{0};
  std::atomic<std::uint64_t> next_request_id_{1};
  std::thread worker_;
};

std::optional<std::string> process_environment(std::string_view name) {
  const char *value = std::getenv(std::string(name).c_str());
  if (value == nullptr)
    return std::nullopt;

  return std::string(value);
}

struct LoadedWeights {
  std::unique_ptr<carat::DeviceWeightArena> target;
  std::unique_ptr<carat::Fp8WeightArena> target_fp8;
  std::unique_ptr<carat::DeviceWeightArena> assistant;
  std::unique_ptr<carat::Fp8WeightArena> assistant_fp8;

  [[nodiscard]] RuntimeWeights view() const {
    return {.target = *target,
            .target_fp8 = target_fp8.get(),
            .assistant = assistant.get(),
            .assistant_fp8 = assistant_fp8.get()};
  }
};

LoadedWeights load_weights(const carat::Gemma4Model &model,
                           const carat::RuntimeSettings &settings) {
  using carat::DeviceWeightArena;
  using carat::Fp8Scaling;
  using carat::Fp8WeightArena;

  LoadedWeights loaded;
  loaded.target = DeviceWeightArena::load(model.plan, model.weights);

  if (const auto &fp8 = settings.fp8_decode) {
    loaded.target_fp8 = Fp8WeightArena::quantize(model.plan, *loaded.target, Fp8Scaling::tensor,
                                                 fp8->weight_scale_multiplier);
    if (const auto &oracle = fp8->int4_lm_head_oracle) {
      loaded.target_fp8->roundtrip_int4(
          "token_embedding", static_cast<int>(model.config.vocabulary_size),
          static_cast<int>(model.config.hidden_size), oracle->block_width);
    }
  }

  if (const auto &assistant = settings.assistant) {
    carat::require_gemma4_assistant_target(model.config);
    const auto host = carat::ShardedSafetensors::open(assistant->model_directory);
    const auto plan = carat::WeightPlan::gemma4_assistant(host);
    loaded.assistant = DeviceWeightArena::load(plan, host);
    if (assistant->fp8_mode != carat::AssistantFp8Mode::off) {
      loaded.assistant_fp8 = Fp8WeightArena::quantize(plan, *loaded.assistant, Fp8Scaling::tensor);
    }
  }

  return loaded;
}

carat::http::HttpResponse json_error(int status, std::string_view code,
                                     std::string_view message = {}) {
  std::string body = "{\"error\":" + carat::http::json_string(code);
  if (!message.empty())
    body += ",\"message\":" + carat::http::json_string(message);
  return {status, "application/json", body + "}\n"};
}

carat::http::HttpStream completion_stream(std::shared_ptr<CompletionJob> job,
                                          ContinuousScheduler &scheduler) {
  return [job = std::move(job), &scheduler](const carat::http::HttpChunkWriter &write) {
    carat::stream_completion(*job, write, [&] { scheduler.cancel(job); });
  };
}

carat::http::HttpResponse complete(const carat::http::HttpRequest &request,
                                   const carat::CompletionLimits &limits,
                                   ContinuousScheduler &scheduler, RuntimeMetrics &metrics) {
  metrics.requests.fetch_add(1);

  CompletionRequest completion;
  try {
    completion = carat::parse_completion_request(request.body, limits);
  } catch (const std::exception &error) {
    metrics.invalid_requests.fetch_add(1);
    return json_error(400, "invalid_request", error.what());
  }

  try {
    if (completion.stream) {
      return {200,
              "text/event-stream",
              {},
              completion_stream(scheduler.submit(std::move(completion)), scheduler)};
    }

    const std::size_t input_tokens = completion.input_ids.size();
    const auto outcome = scheduler.complete(std::move(completion));
    if (const auto *result = std::get_if<CompletionResult>(&outcome)) {
      return {200, "application/json", carat::completion_json(*result, input_tokens)};
    }
  } catch (const std::exception &error) {
    std::cerr << "native runtime: completion rejected: " << error.what() << '\n';
  }

  return json_error(503, "inference_failure");
}

void log_ready(const carat::RuntimeSettings &settings) {
  const SchedulerOptions &scheduler = settings.scheduler;
  std::cout << "carat ready on " << settings.bind_host << ':' << settings.bind_port
            << " active_prefill_quantum_tokens=" << scheduler.prefill_quantum_tokens
            << " idle_prefill_quantum_tokens=" << scheduler.idle_prefill_quantum_tokens
            << " short_prefill_batch_window_us="
            << scheduler.short_prefill_batch_window_microseconds
            << " interleaved_suffix_layer_quantum=" << scheduler.interleaved_suffix_layer_quantum
            << " interleaved_suffix_layer_min_decode="
            << scheduler.interleaved_suffix_layer_minimum_decode
            << " cache_affinity_max_bypasses=" << scheduler.admission.max_bypasses
            << " idle_cache_reserve_slots=" << scheduler.admission.idle_cache_reserve_slots
            << " fp8_decode=" << (settings.fp8_decode ? "tensor" : "off")
            << " ignore_model_eos=" << (settings.ignore_model_eos ? "true" : "false")
            << " assistant=" << (settings.assistant ? "enabled" : "off") << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: carat-runtime MODEL_DIRECTORY\n";
      return 64;
    }

    carat::require_supported_cuda_runtime();

    const auto model = carat::Gemma4Model::open(argv[1]);
    const auto settings = carat::load_runtime_settings(
        process_environment, static_cast<int>(model.config.layers.size()));
    const auto weights = load_weights(model, settings);

    RuntimeMetrics metrics;
    ContinuousScheduler scheduler(model.config, weights.view(), settings.scheduler, metrics);

    const carat::BearerAuthorization authorization(settings.api_key);
    const std::vector<int> implicit_stop_ids =
        settings.ignore_model_eos ? std::vector<int>{} : model.config.eos_token_ids;
    const carat::CompletionLimits limits{.maximum_context = maximum_context,
                                         .vocabulary_size =
                                             static_cast<int>(model.config.vocabulary_size),
                                         .implicit_stop_ids = implicit_stop_ids};

    carat::http::HttpServer server(
        settings.bind_host, settings.bind_port, [&](const carat::http::HttpRequest &request) {
          if (request.method == "GET" && request.path == "/health") {
            return carat::http::HttpResponse{200, "text/plain; charset=utf-8", "ready\n"};
          }
          if (request.method == "GET" && request.path == "/metrics") {
            return carat::http::HttpResponse{200, "text/plain; version=0.0.4; charset=utf-8",
                                             carat::render_metrics(metrics, settings)};
          }
          if (request.method != "POST" || request.path != "/v1/token-completions")
            return json_error(404, "not_found");
          if (!authorization.permits(request.authorization)) {
            return json_error(401, "unauthorized");
          }

          return complete(request, limits, scheduler, metrics);
        });

    log_ready(settings);
    server.run();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "native runtime failed: " << error.what() << '\n';
    return 70;
  }
}
