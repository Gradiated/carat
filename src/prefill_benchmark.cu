#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/json.h"
#include "carat/model_runner.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <chrono>
#include <cuda_profiler_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int> integers(const carat::Json &value) {
  std::vector<int> result;
  for (const auto &item : value.as_array())
    result.push_back(static_cast<int>(item.as_i64()));
  return result;
}

double run(carat::Gemma4ModelRunner &runner, const std::vector<int> &prompt, int chunk_tokens) {
  const auto begin = std::chrono::steady_clock::now();
  const int prediction =
      chunk_tokens > 0 ? runner.prefill_chunked(prompt, chunk_tokens) : runner.prefill(prompt);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  if (prediction < 0)
    throw std::runtime_error("invalid greedy prediction");
  return seconds;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 5) {
      std::cerr << "usage: carat-prefill-benchmark MODEL_DIRECTORY FIXTURE_JSON [TOKENS "
                   "[CHUNK_TOKENS]]\n";
      return 64;
    }
    const std::string model_directory = argv[1];
    const auto fixture = carat::Json::parse(carat::read_text_file(argv[2]));
    const auto seed = integers(fixture.at("input_ids"));
    if (seed.empty())
      throw std::runtime_error("fixture prompt is empty");
    const auto config = carat::Gemma4Config::load(model_directory + "/config.json");
    const auto model_weights = carat::ShardedSafetensors::open(model_directory);
    carat::validate_gemma4_weights(model_weights, config);
    const auto plan = carat::WeightPlan::gemma4(config, model_weights);
    const auto weights = carat::DeviceWeightArena::load(plan, model_weights);
    constexpr int maximum_tokens = 8192;
    carat::Gemma4ModelRunner runner(config, *weights, maximum_tokens + 1);

    const std::vector<int> lengths = argc == 4 ? std::vector<int>{std::stoi(argv[3])}
                                     : argc == 5
                                         ? std::vector<int>{std::stoi(argv[3])}
                                         : std::vector<int>{27, 128, 512, 1024, 2048, 4096, 8192};
    const int chunk_tokens = argc == 5 ? std::stoi(argv[4]) : 0;
    std::cout << "tokens,chunk_tokens,cold_seconds,cold_tokens_per_second,warm_seconds,warm_tokens_"
                 "per_second\n";
    for (const int tokens : lengths) {
      if (tokens <= 0 || tokens > maximum_tokens)
        throw std::runtime_error("invalid benchmark token count");
      std::vector<int> prompt;
      prompt.reserve(tokens);
      for (int index = 0; index < tokens; ++index) {
        prompt.push_back(seed[static_cast<std::size_t>(index) % seed.size()]);
      }
      runner.reset_sequence();
      const double cold = run(runner, prompt, chunk_tokens);
      runner.reset_sequence();
      if (argc == 4)
        cudaProfilerStart();
      const double warm = run(runner, prompt, chunk_tokens);
      if (argc == 4)
        cudaProfilerStop();
      std::cout << tokens << ',' << chunk_tokens << ',' << cold << ',' << tokens / cold << ','
                << warm << ',' << tokens / warm << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "prefill benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
