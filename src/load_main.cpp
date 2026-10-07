#include "carat/device_weights.h"
#include "carat/gemma4_model.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: carat-load MODEL_DIRECTORY [SEGMENT ...]\n";
      return 64;
    }

    std::vector<std::string> segments;
    for (int index = 2; index < argc; ++index)
      segments.emplace_back(argv[index]);
    const auto model = carat::Gemma4Model::open(argv[1]);

    const auto begin = std::chrono::steady_clock::now();
    const auto arena = carat::DeviceWeightArena::load(model.plan, model.weights, segments);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    std::cout << "allocated_bytes=" << arena->allocated_bytes() << '\n'
              << "payload_bytes=" << arena->payload_bytes() << '\n'
              << "load_seconds=" << seconds << '\n'
              << "payload_gb_per_second="
              << (static_cast<double>(arena->payload_bytes()) / 1e9 / seconds) << '\n'
              << "validation=ok\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "carat-load: " << error.what() << '\n';
    return 1;
  }
}
