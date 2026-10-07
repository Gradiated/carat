#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/json.h"
#include "carat/layer_runner.h"
#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

std::vector<__nv_bfloat16> read_bf16(const std::filesystem::path &path, std::size_t elements) {
  std::vector<__nv_bfloat16> values(elements);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(elements * 2ULL));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("unexpected fixture size: " + path.string());
  }
  return values;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: carat-layer-parity MODEL_DIRECTORY FIXTURE_DIRECTORY LAYER\n";
      return 64;
    }
    const std::string model_directory = argv[1];
    const std::filesystem::path fixture_directory = argv[2];
    const int layer = std::stoi(argv[3]);
    const auto manifest =
        carat::Json::parse(carat::read_text_file((fixture_directory / "manifest.json").string()));
    if (manifest.at("layer").as_i64() != layer)
      throw std::runtime_error("fixture layer mismatch");
    const int tokens = static_cast<int>(manifest.at("tokens").as_i64());
    const auto config = carat::Gemma4Config::load(model_directory + "/config.json");
    const auto model_weights = carat::ShardedSafetensors::open(model_directory);
    carat::validate_gemma4_weights(model_weights, config);
    const auto plan = carat::WeightPlan::gemma4(config, model_weights);
    const std::string prefix = "layer." + std::to_string(layer) + ".";
    std::vector<std::string> selected;
    for (const auto &segment : plan.segments()) {
      if (segment.name.starts_with(prefix))
        selected.push_back(segment.name);
    }
    const auto weights = carat::DeviceWeightArena::load(plan, model_weights, selected);
    const std::size_t hidden = config.hidden_size;
    const auto input =
        read_bf16(fixture_directory / manifest.at("tensors").at("input").at("file").as_string(),
                  static_cast<std::size_t>(tokens) * hidden);
    const auto expected = read_bf16(
        fixture_directory / manifest.at("tensors").at("last_output").at("file").as_string(),
        hidden);
    void *device_input = nullptr;
    void *device_output = nullptr;
    check(cudaMalloc(&device_input, input.size() * 2ULL), "allocate parity input");
    check(cudaMalloc(&device_output, hidden * 2ULL), "allocate parity output");
    check(cudaMemcpy(device_input, input.data(), input.size() * 2ULL, cudaMemcpyHostToDevice),
          "copy parity input");
    carat::Gemma4LayerRunner runner(config, *weights, tokens, tokens);
    runner.run_last(layer, device_input, device_output, tokens, nullptr);
    check(cudaDeviceSynchronize(), "layer parity synchronization");
    std::vector<__nv_bfloat16> actual(hidden);
    check(cudaMemcpy(actual.data(), device_output, hidden * 2ULL, cudaMemcpyDeviceToHost),
          "copy parity output");
    cudaFree(device_output);
    cudaFree(device_input);

    double squared_error = 0.0;
    double dot = 0.0;
    double actual_square = 0.0;
    double expected_square = 0.0;
    double maximum_absolute_error = 0.0;
    std::size_t exact = 0;
    for (std::size_t index = 0; index < hidden; ++index) {
      const double actual_value = __bfloat162float(actual[index]);
      const double expected_value = __bfloat162float(expected[index]);
      const double difference = actual_value - expected_value;
      squared_error += difference * difference;
      dot += actual_value * expected_value;
      actual_square += actual_value * actual_value;
      expected_square += expected_value * expected_value;
      maximum_absolute_error = std::max(maximum_absolute_error, std::abs(difference));
      if (std::memcmp(&actual[index], &expected[index], sizeof(__nv_bfloat16)) == 0)
        ++exact;
    }
    const double cosine = dot / std::sqrt(actual_square * expected_square);
    const double rmse = std::sqrt(squared_error / hidden);
    std::cout << "layer=" << layer << '\n'
              << "tokens=" << tokens << '\n'
              << "exact_fraction=" << static_cast<double>(exact) / hidden << '\n'
              << "maximum_absolute_error=" << maximum_absolute_error << '\n'
              << "rmse=" << rmse << '\n'
              << "cosine_similarity=" << cosine << '\n';
    return cosine >= 0.999 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "layer parity failed: " << error.what() << '\n';
    return 1;
  }
}
