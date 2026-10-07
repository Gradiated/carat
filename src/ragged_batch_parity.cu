#include "carat/batch_model_runner.h"
#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/json.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int> integers(const carat::Json &value) {
  std::vector<int> result;
  result.reserve(value.as_array().size());
  for (const auto &item : value.as_array())
    result.push_back(static_cast<int>(item.as_i64()));
  return result;
}

void require_equal(const std::vector<int> &actual, const std::vector<int> &expected,
                   const char *stage) {
  if (actual != expected) {
    std::cerr << stage << " mismatch: actual=";
    for (const int token : actual)
      std::cerr << token << ',';
    std::cerr << " expected=";
    for (const int token : expected)
      std::cerr << token << ',';
    std::cerr << '\n';
    throw std::runtime_error("ragged parity failed");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4 && argc != 5) {
      std::cerr << "usage: carat-ragged-batch-parity MODEL_DIRECTORY FIXTURE_JSON "
                   "[profile-ragged|profile-ragged-fp8|profile-suffix|prefix-benchmark|speculative-"
                   "parity|"
                   "contiguous-parity|fp8-suffix-parity|batched-chunk-parity|assistant-benchmark] "
                   "[ASSISTANT_DIRECTORY]\n";
      return 64;
    }
    const std::string model_directory = argv[1];
    const auto fixture = carat::Json::parse(carat::read_text_file(argv[2]));
    const bool fp8_suffix_parity = argc == 4 && std::string(argv[3]) == "fp8-suffix-parity";
    const auto prompt = fp8_suffix_parity
                            ? integers(fixture.at("prompts").as_array().front().at("input_ids"))
                            : integers(fixture.at("input_ids"));
    const auto expected =
        fp8_suffix_parity ? std::vector<int>{} : integers(fixture.at("expected_output_ids"));
    if (!fp8_suffix_parity && expected.size() < 4) {
      throw std::runtime_error("fixture needs four expected tokens");
    }
    const auto config = carat::Gemma4Config::load(model_directory + "/config.json");
    const auto host_weights = carat::ShardedSafetensors::open(model_directory);
    carat::validate_gemma4_weights(host_weights, config);
    const auto plan = carat::WeightPlan::gemma4(config, host_weights);
    const auto weights = carat::DeviceWeightArena::load(plan, host_weights);
    if (argc == 4 && std::string(argv[3]) == "batched-chunk-parity") {
      constexpr int prompt_tokens = 512;
      constexpr int chunk_tokens = 64;
      std::vector<int> chunk_prompt(static_cast<std::size_t>(prompt_tokens));
      for (int index = 0; index < prompt_tokens; ++index) {
        chunk_prompt[static_cast<std::size_t>(index)] = 2 + index % 251;
      }
      carat::Gemma4BatchModelRunner runner(config, *weights, 4, prompt_tokens + 2);
      const std::vector<int> serial_predictions{runner.prefill_slot(0, chunk_prompt, chunk_tokens),
                                                runner.prefill_slot(1, chunk_prompt, chunk_tokens)};
      std::vector<std::optional<int>> batched_predictions(2);
      int chunks = 0;
      while (!batched_predictions[0] || !batched_predictions[1]) {
        batched_predictions =
            runner.prefill_slots_chunks({2, 3}, {&chunk_prompt, &chunk_prompt}, chunk_tokens);
        ++chunks;
      }
      const std::vector<int> batched_tokens{*batched_predictions[0], *batched_predictions[1]};
      require_equal(batched_tokens, serial_predictions, "batched chunk prediction");
      require_equal(runner.append_ragged(batched_tokens, {2, 3}),
                    runner.append_ragged(serial_predictions, {0, 1}),
                    "batched chunk cached decode");
      if (chunks != prompt_tokens / chunk_tokens) {
        throw std::runtime_error("batched chunk prefill advanced an unexpected number of waves");
      }
      std::cout << "batched_chunk_parity=true\n"
                << "requests=2\n"
                << "prompt_tokens=" << prompt_tokens << '\n'
                << "chunk_tokens=" << chunk_tokens << '\n'
                << "chunks=" << chunks << '\n';
      return 0;
    }
    if (argc == 5 && std::string(argv[3]) == "assistant-benchmark") {
      int batch = 16;
      if (const char *configured = std::getenv("CARAT_ASSISTANT_BENCHMARK_BATCH")) {
        batch = std::stoi(configured);
      }
      if (batch < 1 || batch > 16) {
        throw std::runtime_error("invalid assistant benchmark batch");
      }
      const auto assistant_host = carat::ShardedSafetensors::open(argv[4]);
      const auto assistant_plan = carat::WeightPlan::gemma4_assistant(assistant_host);
      const auto assistant_weights = carat::DeviceWeightArena::load(assistant_plan, assistant_host);
      std::unique_ptr<carat::Fp8WeightArena> assistant_fp8_weights;
      const char *assistant_fp8_mode = std::getenv("CARAT_ASSISTANT_FP8_MODE");
      if (assistant_fp8_mode != nullptr && std::string(assistant_fp8_mode) != "off") {
        assistant_fp8_weights = carat::Fp8WeightArena::quantize(assistant_plan, *assistant_weights,
                                                                carat::Fp8Scaling::tensor);
      }
      const auto fp8_weights =
          carat::Fp8WeightArena::quantize(plan, *weights, carat::Fp8Scaling::tensor);
      int benchmark_position = 512;
      if (const char *configured = std::getenv("CARAT_ASSISTANT_BENCHMARK_POSITION")) {
        benchmark_position = std::stoi(configured);
      }
      int draft_depth = 4;
      if (const char *configured = std::getenv("CARAT_ASSISTANT_BENCHMARK_DEPTH")) {
        draft_depth = std::stoi(configured);
      }
      if (draft_depth < 1 || draft_depth > 8 || benchmark_position < 1 ||
          benchmark_position + draft_depth >= 8192) {
        throw std::runtime_error("invalid assistant benchmark position");
      }
      carat::Gemma4BatchModelRunner runner(
          config, *weights, batch, std::max(1024, benchmark_position + 64), fp8_weights.get(),
          assistant_weights.get(), assistant_fp8_weights.get());
      std::vector<int> slots;
      for (int slot = 0; slot < batch; ++slot)
        slots.push_back(slot);
      runner.seed_empty_cache_for_benchmark(benchmark_position);
      const std::vector<int> last_tokens(static_cast<std::size_t>(batch), 2);
      std::vector<int> target_first = runner.append_ragged(last_tokens, slots);
      constexpr int draft_repetitions = 20;
      static_cast<void>(runner.draft_four(last_tokens, slots, target_first, draft_depth));
      std::vector<std::vector<int>> proposals;
      const bool profile_verify = std::getenv("CARAT_PROFILE_ASSISTANT_VERIFY") != nullptr;
      if (!profile_verify)
        cudaProfilerStart();
      const auto draft_begin = std::chrono::steady_clock::now();
      for (int repetition = 0; repetition < draft_repetitions; ++repetition) {
        proposals = runner.draft_four(last_tokens, slots, target_first, draft_depth);
      }
      const double draft_total_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - draft_begin).count();
      const double draft_seconds = draft_total_seconds / draft_repetitions;
      if (!profile_verify)
        cudaProfilerStop();
      if (std::getenv("CARAT_WARM_ASSISTANT_VERIFY") != nullptr) {
        static_cast<void>(runner.verify_greedy_proposals(proposals, target_first, slots));
        runner.seed_empty_cache_for_benchmark(benchmark_position);
        target_first = runner.append_ragged(last_tokens, slots);
      }
      if (profile_verify)
        cudaProfilerStart();
      const auto verify_begin = std::chrono::steady_clock::now();
      const auto verification = runner.verify_greedy_proposals(proposals, target_first, slots);
      const double verify_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - verify_begin).count();
      if (profile_verify)
        cudaProfilerStop();
      int accepted = 0;
      for (const int count : verification.accepted_tokens)
        accepted += count;
      std::cout << "assistant_native=true\n"
                << "batch=" << batch << '\n'
                << "benchmark_position=" << benchmark_position << '\n'
                << "draft_depth=" << draft_depth << '\n'
                << "draft_repetitions=" << draft_repetitions << '\n'
                << "draft_seconds=" << draft_seconds << '\n'
                << "verify_seconds=" << verify_seconds << '\n'
                << "mean_accepted_tokens=" << static_cast<double>(accepted) / batch << '\n'
                << "accepted_tokens=";
      for (const int count : verification.accepted_tokens)
        std::cout << count << ',';
      std::cout << '\n' << "next_tokens=";
      for (const int token : verification.next_tokens)
        std::cout << token << ',';
      std::cout << '\n'
                << "first_proposal=" << proposals.front().front() << '\n'
                << "first_proposal_sequence=";
      for (const int token : proposals.front())
        std::cout << token << ',';
      std::cout << '\n'
                << "estimated_emitted_tokens_per_second="
                << static_cast<double>(accepted) / (draft_seconds + verify_seconds) << '\n';
      return 0;
    }
    if (argc == 4 && std::string(argv[3]) == "speculative-parity") {
      if (expected.size() < 6)
        throw std::runtime_error("speculative fixture needs six tokens");
      constexpr int batch = 16;
      const auto fp8_weights =
          carat::Fp8WeightArena::quantize(plan, *weights, carat::Fp8Scaling::tensor);
      carat::Gemma4BatchModelRunner runner(config, *weights, batch, 512, fp8_weights.get());
      std::vector<int> first(static_cast<std::size_t>(batch), expected[0]);
      first[0] = runner.prefill_slot(0, prompt, 128);
      for (int slot = 1; slot < batch; ++slot)
        runner.clone_slot(0, slot);
      require_equal(first, std::vector<int>(batch, expected[0]), "speculative FP8 first token");
      int rejected = (expected[0] + 1) % static_cast<int>(config.vocabulary_size);
      if (rejected == expected[1])
        rejected = (rejected + 1) % static_cast<int>(config.vocabulary_size);
      const std::vector<std::vector<int>> proposal_pattern{
          {expected[0], expected[1], expected[2], expected[3]},
          {rejected, expected[1], expected[2], expected[3]},
          {expected[0], rejected, expected[2], expected[3]},
          {expected[0], expected[1], rejected, expected[3]},
      };
      std::vector<std::vector<int>> proposals;
      std::vector<int> slots;
      std::vector<int> expected_accepted;
      std::vector<int> expected_next;
      std::vector<int> expected_after;
      for (int request = 0; request < batch; ++request) {
        const int pattern = request % 4;
        proposals.push_back(proposal_pattern[static_cast<std::size_t>(pattern)]);
        slots.push_back(request);
        expected_accepted.push_back(pattern == 0 ? 4 : pattern - 1);
        expected_next.push_back(expected[static_cast<std::size_t>(pattern == 0 ? 4 : pattern - 1)]);
        expected_after.push_back(expected[static_cast<std::size_t>(pattern == 0 ? 5 : pattern)]);
      }
      cudaProfilerStart();
      const auto begin = std::chrono::steady_clock::now();
      const auto verification = runner.verify_greedy_proposals(proposals, first, slots);
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      cudaProfilerStop();
      require_equal(verification.accepted_tokens, expected_accepted,
                    "speculative accepted lengths");
      require_equal(verification.next_tokens, expected_next,
                    "speculative replacement and bonus tokens");
      const auto after_rollback = runner.append_ragged(verification.next_tokens, slots);
      require_equal(after_rollback, expected_after, "speculative sliding rollback decode");
      std::cout << "speculative_parity=true\n"
                << "requests=" << batch << '\n'
                << "draft_depth=4\n"
                << "verified_target_tokens=" << batch * 4 << '\n'
                << "verification_seconds=" << seconds << '\n'
                << "verification_candidate_tokens_per_second="
                << static_cast<double>(batch * 4) / seconds << '\n';
      return 0;
    }
    if (argc == 4 && std::string(argv[3]) == "contiguous-parity") {
      constexpr int maximum_slots = 16;
      constexpr int first_slot = 1;
      constexpr int batch = maximum_slots - first_slot;
      const auto fp8_weights =
          carat::Fp8WeightArena::quantize(plan, *weights, carat::Fp8Scaling::tensor);
      carat::Gemma4BatchModelRunner runner(config, *weights, maximum_slots, 512, fp8_weights.get());
      std::vector<int> prediction(static_cast<std::size_t>(batch), expected[0]);
      prediction[0] = runner.prefill_slot(first_slot, prompt, 64);
      for (int slot = first_slot + 1; slot < maximum_slots; ++slot) {
        runner.clone_slot(first_slot, slot);
      }
      require_equal(prediction, std::vector<int>(batch, expected[0]), "contiguous first token");
      std::vector<int> slots(static_cast<std::size_t>(batch));
      for (int row = 0; row < batch; ++row) {
        slots[static_cast<std::size_t>(row)] = first_slot + row;
      }
      for (std::size_t step = 1; step < expected.size(); ++step) {
        prediction = runner.append_ragged(prediction, slots);
        require_equal(prediction, std::vector<int>(batch, expected[step]), "contiguous FP8 decode");
      }
      std::cout << "contiguous_fp8_parity=true\n"
                << "batch=" << batch << '\n'
                << "first_slot=" << first_slot << '\n'
                << "verified_tokens=" << expected.size() << '\n';
      return 0;
    }
    if (argc == 4 && (std::string(argv[3]) == "profile-ragged" ||
                      std::string(argv[3]) == "profile-ragged-fp8")) {
      constexpr int batch = 16;
      constexpr int context = 8192;
      constexpr int decode_steps = 8;
      const bool fp8 = std::string(argv[3]) == "profile-ragged-fp8";
      const auto fp8_weights =
          fp8 ? carat::Fp8WeightArena::quantize(plan, *weights, carat::Fp8Scaling::tensor)
              : nullptr;
      carat::Gemma4BatchModelRunner runner(config, *weights, batch, context + decode_steps + 1,
                                           fp8_weights.get());
      runner.seed_empty_cache_for_benchmark(context - 1);
      std::vector<int> slots(static_cast<std::size_t>(batch));
      for (int index = 0; index < batch; ++index) {
        slots[static_cast<std::size_t>(index)] = fp8 ? index : batch - 1 - index;
      }
      std::vector<int> input(static_cast<std::size_t>(batch), 2);
      input = runner.append_ragged(input, slots);
      cudaProfilerStart();
      const auto begin = std::chrono::steady_clock::now();
      for (int step = 0; step < decode_steps; ++step)
        input = runner.append_ragged(input, slots);
      cudaDeviceSynchronize();
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      cudaProfilerStop();
      std::cout << "profile_ragged_fp8=" << (fp8 ? "true" : "false") << '\n'
                << "profile_ragged_tokens_per_second="
                << (static_cast<double>(batch * decode_steps) / seconds) << '\n';
      return 0;
    }
    if (argc == 4 && std::string(argv[3]) == "profile-suffix") {
      constexpr int batch = 10;
      constexpr int position = 6000;
      constexpr int suffix_tokens = 257;
      constexpr int context = 8192;
      std::vector<std::vector<int>> prompts(batch, std::vector<int>(position + suffix_tokens, 2));
      std::vector<const std::vector<int> *> prompt_pointers;
      std::vector<int> slots;
      for (int index = 0; index < batch; ++index) {
        prompt_pointers.push_back(&prompts[static_cast<std::size_t>(index)]);
        slots.push_back(index);
      }
      carat::Gemma4BatchModelRunner runner(config, *weights, batch, context);
      runner.seed_empty_cache_for_benchmark(position);
      static_cast<void>(runner.prefill_slots_suffix(slots, prompt_pointers));
      runner.seed_empty_cache_for_benchmark(position);
      cudaProfilerStart();
      const auto begin = std::chrono::steady_clock::now();
      static_cast<void>(runner.prefill_slots_suffix(slots, prompt_pointers));
      cudaDeviceSynchronize();
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      cudaProfilerStop();
      std::cout << "profile_suffix_requests=" << batch << '\n'
                << "profile_suffix_tokens=" << batch * suffix_tokens << '\n'
                << "profile_suffix_seconds=" << seconds << '\n'
                << "profile_suffix_tokens_per_second="
                << static_cast<double>(batch * suffix_tokens) / seconds << '\n';
      return 0;
    }
    if (argc == 4 && std::string(argv[3]) == "prefix-benchmark") {
      constexpr int context = 8192;
      constexpr int cached_prefix = context * 9 / 10;
      std::vector<int> long_prompt(static_cast<std::size_t>(context));
      for (int index = 0; index < context; ++index) {
        long_prompt[static_cast<std::size_t>(index)] =
            prompt[static_cast<std::size_t>(index) % prompt.size()];
      }
      const std::vector<int> prefix(long_prompt.begin(), long_prompt.begin() + cached_prefix);
      carat::Gemma4BatchModelRunner runner(config, *weights, 2, context + 2);
      static_cast<void>(runner.prefill_slot(0, prefix, 1024));
      const auto reuse_begin = std::chrono::steady_clock::now();
      const int reuse_prediction =
          runner.prefill_slot_from_prefix(0, long_prompt, cached_prefix, 1024);
      const double reuse_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - reuse_begin).count();
      const auto fresh_begin = std::chrono::steady_clock::now();
      const int fresh_prediction = runner.prefill_slot(1, long_prompt, 1024);
      const double fresh_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - fresh_begin).count();
      if (reuse_prediction != fresh_prediction) {
        throw std::runtime_error("90% prefix reuse changed the greedy prediction");
      }
      runner.reset_slot(1);
      const auto clone_begin = std::chrono::steady_clock::now();
      const std::uint64_t clone_bytes = runner.clone_slot(0, 1);
      if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("cache clone synchronization failed");
      }
      const double clone_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - clone_begin).count();
      const auto cloned_prediction =
          runner.append_ragged({reuse_prediction, reuse_prediction}, {0, 1});
      if (cloned_prediction[0] != cloned_prediction[1]) {
        throw std::runtime_error("wrapped sliding-cache clone changed decode output");
      }
      std::cout << "prefix_parity=true\n"
                << "prompt_tokens=" << context << '\n'
                << "cached_prefix_tokens=" << cached_prefix << '\n'
                << "suffix_tokens=" << (context - cached_prefix) << '\n'
                << "fresh_prefill_seconds=" << fresh_seconds << '\n'
                << "reuse_prefill_seconds=" << reuse_seconds << '\n'
                << "clone_bytes=" << clone_bytes << '\n'
                << "clone_seconds=" << clone_seconds << '\n'
                << "ttft_speedup=" << (fresh_seconds / reuse_seconds) << '\n'
                << "prefill_compute_avoided_fraction=" << (1.0 - reuse_seconds / fresh_seconds)
                << '\n';
      return 0;
    }
    if (argc == 4 && std::string(argv[3]) == "fp8-suffix-parity") {
      constexpr int batch = 10;
      constexpr int prefix_tokens = 3000;
      constexpr int suffix_tokens = 257;
      std::vector<int> full(static_cast<std::size_t>(prefix_tokens + suffix_tokens));
      for (std::size_t index = 0; index < full.size(); ++index) {
        full[index] = prompt[index % prompt.size()];
      }
      const std::vector<int> resident(full.begin(), full.begin() + prefix_tokens);
      setenv("CARAT_FP8_SUFFIX_MAX_TOKENS", "3000", 1);
      const auto fp8_weights =
          carat::Fp8WeightArena::quantize(plan, *weights, carat::Fp8Scaling::tensor);
      carat::Gemma4BatchModelRunner runner(config, *weights, batch, 4000, fp8_weights.get());
      std::vector<int> slots;
      std::vector<const std::vector<int> *> prompts;
      for (int slot = 0; slot < batch; ++slot) {
        slots.push_back(slot);
        prompts.push_back(&full);
        static_cast<void>(runner.prefill_slot(slot, resident, 1024));
      }
      const auto unsliced = runner.prefill_slots_suffix(slots, prompts);
      const auto unsliced_decode = runner.append_ragged(unsliced, slots);
      for (int slot = 0; slot < batch; ++slot) {
        static_cast<void>(runner.prefill_slot(slot, resident, 1024));
      }
      carat::LayerSlicedSuffixPrefillResult sliced;
      int slices = 0;
      while (!sliced.complete) {
        sliced = runner.prefill_slots_suffix_layer_slice(slots, prompts, 10);
        ++slices;
      }
      require_equal(sliced.predictions, unsliced, "FP8 layer-sliced suffix prediction");
      require_equal(runner.append_ragged(sliced.predictions, slots), unsliced_decode,
                    "FP8 layer-sliced suffix decode");
      if (slices != 6 || runner.has_suffix_layer_slices()) {
        throw std::runtime_error("FP8 suffix did not complete in six exact slices");
      }
      std::cout << "fp8_suffix_parity=true\n"
                << "batch=" << batch << '\n'
                << "suffix_tokens=" << suffix_tokens << '\n'
                << "slices=" << slices << '\n';
      return 0;
    }
    {
      carat::Gemma4BatchModelRunner runner(config, *weights, 4, 512);
      const int prefix_tokens = static_cast<int>(prompt.size()) - 7;
      const std::vector<int> prefix(prompt.begin(), prompt.begin() + prefix_tokens);
      static_cast<void>(runner.prefill_slot(1, prefix, 8));
      const int reused_prediction = runner.prefill_slot_from_prefix(1, prompt, prefix_tokens, 8);
      const int fresh_prediction = runner.prefill_slot(3, prompt, 8);
      require_equal({reused_prediction, fresh_prediction}, {expected[0], expected[0]},
                    "exact-prefix suffix prefill");
      require_equal(runner.append_ragged({reused_prediction, fresh_prediction}, {1, 3}),
                    {expected[1], expected[1]}, "exact-prefix cached decode");

      // Pack unequal suffix lengths into one projection batch. This covers the row-offset and
      // independent-cache invariants used by cached multi-turn cohorts in the runtime.
      const int shorter_prefix_tokens = static_cast<int>(prompt.size()) - 4;
      const std::vector<int> shorter_prefix(prompt.begin(), prompt.begin() + shorter_prefix_tokens);
      static_cast<void>(runner.prefill_slot(0, prefix, 8));
      static_cast<void>(runner.prefill_slot(2, shorter_prefix, 8));
      const auto batched_suffix_predictions =
          runner.prefill_slots_suffix({0, 2}, {&prompt, &prompt});
      require_equal(batched_suffix_predictions, {expected[0], expected[0]},
                    "unequal batched suffix prefill");
      require_equal(runner.append_ragged(batched_suffix_predictions, {0, 2}),
                    {expected[1], expected[1]}, "unequal batched suffix cached decode");
      runner.reset_slot(0);
      runner.reset_slot(2);

      std::vector<std::optional<int>> batched_chunk_predictions(2);
      int batched_chunk_count = 0;
      while (!batched_chunk_predictions[0] || !batched_chunk_predictions[1]) {
        batched_chunk_predictions = runner.prefill_slots_chunks({0, 2}, {&prompt, &prompt}, 7);
        ++batched_chunk_count;
      }
      if (batched_chunk_count < 2) {
        throw std::runtime_error("batched resumable prefill did not yield between chunks");
      }
      const std::vector<int> batched_chunk_tokens{*batched_chunk_predictions[0],
                                                  *batched_chunk_predictions[1]};
      require_equal(batched_chunk_tokens, {expected[0], expected[0]}, "batched chunk prefill");
      require_equal(runner.append_ragged(batched_chunk_tokens, {0, 2}), {expected[1], expected[1]},
                    "batched chunk cached decode");
      runner.reset_slot(0);
      runner.reset_slot(2);

      static_cast<void>(runner.prefill_slot(0, prefix, 8));
      static_cast<void>(runner.prefill_slot(2, prefix, 8));
      const auto uniform_suffix_predictions =
          runner.prefill_slots_suffix({0, 2}, {&prompt, &prompt});
      require_equal(uniform_suffix_predictions, {expected[0], expected[0]},
                    "uniform batched suffix prefill");
      require_equal(runner.append_ragged(uniform_suffix_predictions, {0, 2}),
                    {expected[1], expected[1]}, "uniform batched suffix cached decode");
      runner.reset_slot(0);
      runner.reset_slot(2);

      const int slot_zero = runner.prefill_slot(0, prompt, 16);
      std::optional<int> chunked_prediction;
      int chunk_count = 0;
      while (!chunked_prediction) {
        chunked_prediction = runner.prefill_slot_chunk(2, prompt, 7);
        ++chunk_count;
      }
      if (chunk_count < 2)
        throw std::runtime_error("resumable prefill did not yield between chunks");
      const int slot_two = *chunked_prediction;
      require_equal({slot_zero, slot_two}, {expected[0], expected[0]}, "prefill");

      auto prediction = runner.append_ragged({slot_two, slot_zero}, {2, 0});
      require_equal(prediction, {expected[1], expected[1]}, "equal-position ragged decode");
      prediction = runner.append_ragged({prediction[0]}, {2});
      require_equal(prediction, {expected[2]}, "single-slot advance");
      prediction = runner.append_ragged({expected[1], expected[2]}, {0, 2});
      require_equal(prediction, {expected[2], expected[3]}, "unequal-position ragged decode");
      if (runner.slot_position(0) != static_cast<int>(prompt.size()) + 2 ||
          runner.slot_position(2) != static_cast<int>(prompt.size()) + 3) {
        throw std::runtime_error("slot positions were not advanced independently");
      }

      constexpr int measured_steps = 16;
      std::vector<int> slots{0, 2};
      std::vector<int> inputs{prediction[0], prediction[1]};
      const auto begin = std::chrono::steady_clock::now();
      for (int step = 0; step < measured_steps; ++step)
        inputs = runner.append_ragged(inputs, slots);
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      std::cout << "ragged_parity=true\n"
                << "short_ragged_batch=2\n"
                << "short_ragged_tokens_per_second=" << (2.0 * measured_steps / seconds) << '\n';
    }

    {
      // Four sorted long contexts force the split-width global-attention path. Compare it with
      // independent suffix prefill and then compare the next cached decode token as well.
      const std::vector<int> prefix_lengths{1100, 1300, 1800, 2000};
      constexpr int suffix_tokens = 7;
      std::vector<std::vector<int>> long_prompts;
      for (const int prefix_length : prefix_lengths) {
        std::vector<int> long_prompt(static_cast<std::size_t>(prefix_length + suffix_tokens));
        for (std::size_t index = 0; index < long_prompt.size(); ++index) {
          long_prompt[index] = prompt[index % prompt.size()];
        }
        long_prompts.push_back(std::move(long_prompt));
      }
      carat::Gemma4BatchModelRunner runner(config, *weights, 8, 2048);
      for (int request = 0; request < 4; ++request) {
        const auto &full = long_prompts[static_cast<std::size_t>(request)];
        const int prefix_length = prefix_lengths[static_cast<std::size_t>(request)];
        const std::vector<int> resident(full.begin(), full.begin() + prefix_length);
        static_cast<void>(runner.prefill_slot(request, resident, 1024));
        static_cast<void>(runner.prefill_slot(request + 4, resident, 1024));
      }
      std::vector<int> serial_predictions;
      std::vector<const std::vector<int> *> prompt_pointers;
      for (int request = 0; request < 4; ++request) {
        serial_predictions.push_back(runner.prefill_slot_from_prefix(
            request, long_prompts[static_cast<std::size_t>(request)],
            prefix_lengths[static_cast<std::size_t>(request)], 16));
        prompt_pointers.push_back(&long_prompts[static_cast<std::size_t>(request)]);
      }
      const auto batched_predictions = runner.prefill_slots_suffix({4, 5, 6, 7}, prompt_pointers);
      require_equal(batched_predictions, serial_predictions,
                    "split-context batched suffix prefill");
      const auto serial_decode = runner.append_ragged(serial_predictions, {0, 1, 2, 3});
      const auto batched_decode = runner.append_ragged(batched_predictions, {4, 5, 6, 7});
      require_equal(batched_decode, serial_decode, "split-context batched suffix cached decode");
    }
    {
      // cuDNN's multi-request sliding plan is promoted only for the production-sized cohort and
      // suffix shape. Compare that exact fast path with independent full prefill and decode.
      constexpr int batch = 10;
      constexpr int prefix_tokens = 3000;
      constexpr int suffix_tokens = 257;
      std::vector<int> full(static_cast<std::size_t>(prefix_tokens + suffix_tokens));
      for (std::size_t index = 0; index < full.size(); ++index) {
        full[index] = prompt[index % prompt.size()];
      }
      const std::vector<int> resident(full.begin(), full.begin() + prefix_tokens);
      carat::Gemma4BatchModelRunner runner(config, *weights, batch, 4000);
      std::vector<int> slots;
      std::vector<const std::vector<int> *> prompts;
      for (int slot = 0; slot < batch; ++slot) {
        slots.push_back(slot);
        prompts.push_back(&full);
        static_cast<void>(runner.prefill_slot(slot, resident, 1024));
      }
      const auto batched_predictions = runner.prefill_slots_suffix(slots, prompts);
      const auto batched_decode = runner.append_ragged(batched_predictions, slots);
      std::vector<int> serial_predictions;
      for (int slot = 0; slot < batch; ++slot) {
        serial_predictions.push_back(runner.prefill_slot(slot, full, 1024));
      }
      require_equal(batched_predictions, serial_predictions,
                    "production-shape batched suffix prefill");
      require_equal(batched_decode, runner.append_ragged(serial_predictions, slots),
                    "production-shape batched suffix cached decode");

      for (int slot = 0; slot < batch; ++slot) {
        static_cast<void>(runner.prefill_slot(slot, resident, 1024));
      }
      carat::LayerSlicedSuffixPrefillResult sliced;
      int slices = 0;
      while (!sliced.complete) {
        sliced = runner.prefill_slots_suffix_layer_slice(slots, prompts, 10);
        ++slices;
      }
      if (slices != 6 || runner.has_suffix_layer_slices()) {
        throw std::runtime_error("production suffix did not use six exact layer slices");
      }
      require_equal(sliced.predictions, serial_predictions,
                    "layer-sliced production suffix prefill");
      require_equal(runner.append_ragged(sliced.predictions, slots), batched_decode,
                    "layer-sliced production suffix cached decode");
    }

    std::cout << "mode,batch,context,seconds,aggregate_tokens_per_second\n";
    for (const int batch : std::vector<int>{1, 4, 8, 16}) {
      constexpr int context = 8192;
      constexpr int decode_steps = 8;
      double homogeneous_seconds = 0.0;
      {
        carat::Gemma4BatchModelRunner runner(config, *weights, batch, context + decode_steps + 1);
        runner.seed_empty_cache_for_benchmark(context - 1);
        std::vector<int> input(static_cast<std::size_t>(batch), 2);
        auto prediction = runner.append(input);
        const auto begin = std::chrono::steady_clock::now();
        for (int step = 0; step < decode_steps; ++step) {
          input = prediction;
          prediction = runner.append(input);
        }
        homogeneous_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      }
      std::cout << "homogeneous," << batch << ',' << context << ',' << homogeneous_seconds << ','
                << (static_cast<double>(batch * decode_steps) / homogeneous_seconds) << '\n';

      double contiguous_ragged_seconds = 0.0;
      {
        carat::Gemma4BatchModelRunner runner(config, *weights, batch, context + decode_steps + 1);
        runner.seed_empty_cache_for_benchmark(context - 1);
        std::vector<int> slots(static_cast<std::size_t>(batch));
        for (int index = 0; index < batch; ++index) {
          slots[static_cast<std::size_t>(index)] = index;
        }
        std::vector<int> input(static_cast<std::size_t>(batch), 2);
        auto prediction = runner.append_ragged(input, slots);
        const auto begin = std::chrono::steady_clock::now();
        for (int step = 0; step < decode_steps; ++step) {
          input = prediction;
          prediction = runner.append_ragged(input, slots);
        }
        contiguous_ragged_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      }
      std::cout << "contiguous_ragged," << batch << ',' << context << ','
                << contiguous_ragged_seconds << ','
                << (static_cast<double>(batch * decode_steps) / contiguous_ragged_seconds) << '\n';

      double ragged_seconds = 0.0;
      {
        carat::Gemma4BatchModelRunner runner(config, *weights, batch, context + decode_steps + 1);
        runner.seed_empty_cache_for_benchmark(context - 1);
        std::vector<int> slots(static_cast<std::size_t>(batch));
        for (int index = 0; index < batch; ++index)
          slots[static_cast<std::size_t>(index)] = batch - 1 - index;
        std::vector<int> input(static_cast<std::size_t>(batch), 2);
        auto prediction = runner.append_ragged(input, slots);
        const auto begin = std::chrono::steady_clock::now();
        for (int step = 0; step < decode_steps; ++step) {
          input = prediction;
          prediction = runner.append_ragged(input, slots);
        }
        ragged_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      }
      std::cout << "ragged," << batch << ',' << context << ',' << ragged_seconds << ','
                << (static_cast<double>(batch * decode_steps) / ragged_seconds) << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ragged batch parity failed: " << error.what() << '\n';
    return 1;
  }
}
