#include "carat/batch_model_runner.h"
#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/json.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <chrono>
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

carat::Fp8Scaling parse_scaling(const std::string &value) {
  if (value == "tensor")
    return carat::Fp8Scaling::tensor;
  if (value == "channel")
    return carat::Fp8Scaling::channel;
  if (value == "block-128")
    return carat::Fp8Scaling::block_128;
  throw std::runtime_error("scaling must be tensor, channel, or block-128");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4 && argc != 5) {
      std::cerr << "usage: carat-fp8-model-parity MODEL_DIRECTORY FIXTURE_JSON "
                   "tensor|channel|block-128 [profile]\n";
      return 64;
    }
    const bool profile = argc == 5 && std::string(argv[4]) == "profile";
    if (argc == 5 && !profile)
      throw std::runtime_error("unknown benchmark mode");
    const auto scaling = parse_scaling(argv[3]);
    const std::string model_directory = argv[1];
    const auto fixture = carat::Json::parse(carat::read_text_file(argv[2]));
    const auto prompt = integers(fixture.at("input_ids"));
    const auto expected = integers(fixture.at("expected_output_ids"));
    const auto config = carat::Gemma4Config::load(model_directory + "/config.json");
    const auto host_weights = carat::ShardedSafetensors::open(model_directory);
    carat::validate_gemma4_weights(host_weights, config);
    const auto plan = carat::WeightPlan::gemma4(config, host_weights);
    const auto weights = carat::DeviceWeightArena::load(plan, host_weights);
    const auto quantize_begin = std::chrono::steady_clock::now();
    const auto fp8_weights = carat::Fp8WeightArena::quantize(plan, *weights, scaling);
    const double quantize_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - quantize_begin).count();
    std::vector<int> actual;
    actual.reserve(expected.size());
    double inference_seconds = 0.0;
    double steady_batch_one_tokens_per_second = 0.0;
    {
      carat::Gemma4BatchModelRunner runner(config, *weights, 1, 512, fp8_weights.get());
      const auto inference_begin = std::chrono::steady_clock::now();
      actual.push_back(runner.prefill_slot(0, prompt, 64));
      while (actual.size() < expected.size()) {
        actual.push_back(runner.append_ragged({actual.back()}, {0})[0]);
      }
      inference_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - inference_begin).count();
      int token = actual.back();
      for (int step = 0; step < 8; ++step)
        token = runner.append_ragged({token}, {0})[0];
      const auto steady_begin = std::chrono::steady_clock::now();
      constexpr int steady_steps = 32;
      for (int step = 0; step < steady_steps; ++step) {
        token = runner.append_ragged({token}, {0})[0];
      }
      steady_batch_one_tokens_per_second =
          steady_steps /
          std::chrono::duration<double>(std::chrono::steady_clock::now() - steady_begin).count();
    }
    std::size_t matching_prefix = 0;
    while (matching_prefix < expected.size() &&
           actual[matching_prefix] == expected[matching_prefix]) {
      ++matching_prefix;
    }
    std::cout << "scaling=" << argv[3] << '\n'
              << "fp8_weight_bytes=" << fp8_weights->allocated_bytes() << '\n'
              << "quantize_seconds=" << quantize_seconds << '\n'
              << "matching_greedy_prefix_tokens=" << matching_prefix << '\n'
              << "expected_tokens=";
    for (const int token : expected)
      std::cout << token << ',';
    std::cout << "\nactual_tokens=";
    for (const int token : actual)
      std::cout << token << ',';
    std::cout << "\ninference_seconds=" << inference_seconds << '\n'
              << "steady_batch_one_tokens_per_second=" << steady_batch_one_tokens_per_second
              << '\n';

    constexpr int benchmark_position = 8128;
    constexpr int benchmark_context = 8192;
    constexpr int benchmark_steps = 32;
    for (const int benchmark_batch : std::vector<int>{1, 4, 8, 10, 16}) {
      carat::Gemma4BatchModelRunner batch_runner(config, *weights, benchmark_batch,
                                                 benchmark_context, fp8_weights.get());
      batch_runner.seed_empty_cache_for_benchmark(benchmark_position);
      std::vector<int> slots(static_cast<std::size_t>(benchmark_batch));
      std::vector<int> inputs(static_cast<std::size_t>(benchmark_batch), 2);
      for (int index = 0; index < benchmark_batch; ++index)
        slots[index] = index;
      // The first call also captures the exact active/context graph. Keep graph construction out
      // of the steady-state throughput interval while retaining it in end-to-end runtime tests.
      inputs = batch_runner.append_ragged(inputs, slots);
      if (profile && benchmark_batch == 16)
        cudaProfilerStart();
      const auto batch_begin = std::chrono::steady_clock::now();
      for (int step = 0; step < benchmark_steps; ++step) {
        inputs = batch_runner.append_ragged(inputs, slots);
      }
      const double batch_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_begin).count();
      if (profile && benchmark_batch == 16) {
        cudaDeviceSynchronize();
        cudaProfilerStop();
      }
      std::cout << "steady_batch_" << benchmark_batch << "_tokens_per_second="
                << static_cast<double>(benchmark_batch * benchmark_steps) / batch_seconds << '\n';
    }
    return matching_prefix == expected.size() ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "FP8 model parity failed: " << error.what() << '\n';
    return 1;
  }
}
