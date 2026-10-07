#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/json.h"
#include "carat/model_runner.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <chrono>
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

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr
          << "usage: carat-prefill-model-parity MODEL_DIRECTORY FIXTURE_JSON [CHUNK_TOKENS]\n";
      return 64;
    }
    const std::string model_directory = argv[1];
    const auto fixture = carat::Json::parse(carat::read_text_file(argv[2]));
    const auto input_ids = integers(fixture.at("input_ids"));
    const auto expected_ids = integers(fixture.at("expected_output_ids"));
    const auto config = carat::Gemma4Config::load(model_directory + "/config.json");
    const auto model_weights = carat::ShardedSafetensors::open(model_directory);
    carat::validate_gemma4_weights(model_weights, config);
    const auto plan = carat::WeightPlan::gemma4(config, model_weights);
    const auto weights = carat::DeviceWeightArena::load(plan, model_weights);
    carat::Gemma4ModelRunner runner(config, *weights,
                                    static_cast<int>(input_ids.size() + expected_ids.size() + 1));

    const auto prefill_begin = std::chrono::steady_clock::now();
    int prediction = argc == 4 ? runner.prefill_chunked(input_ids, std::stoi(argv[3]))
                               : runner.prefill(input_ids);
    const double prefill_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_begin).count();
    int matches = 0;
    const auto decode_begin = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < expected_ids.size(); ++index) {
      const bool match = prediction == expected_ids[index];
      if (match)
        ++matches;
      std::cout << "generation_index=" << index << " expected=" << expected_ids[index]
                << " actual=" << prediction << " match=" << (match ? "true" : "false") << '\n';
      if (index + 1 < expected_ids.size())
        prediction = runner.append(expected_ids[index]);
    }
    const double decode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_begin).count();
    std::cout << "prompt_tokens=" << input_ids.size() << '\n'
              << "prefill_seconds=" << prefill_seconds << '\n'
              << "prefill_tokens_per_second=" << input_ids.size() / prefill_seconds << '\n'
              << "chunk_tokens=" << (argc == 4 ? std::stoi(argv[3]) : 0) << '\n'
              << "decode_seconds=" << decode_seconds << '\n'
              << "greedy_matches=" << matches << '/' << expected_ids.size() << '\n';
    return matches == static_cast<int>(expected_ids.size()) ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "prefill model parity failed: " << error.what() << '\n';
    return 1;
  }
}
