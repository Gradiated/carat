#include "carat/gemma4_model.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: carat-inspect MODEL_DIRECTORY\n";
      return 64;
    }

    const auto model = carat::Gemma4Model::open(argv[1]);
    const auto &config = model.config;

    std::size_t sliding_layers = 0;
    for (const auto &layer : config.layers) {
      if (layer.attention == carat::AttentionKind::sliding)
        ++sliding_layers;
    }
    const std::size_t global_layers = config.layers.size() - sliding_layers;

    std::cout << "model=gemma4_text\n"
              << "layers=" << config.layers.size() << " sliding_layers=" << sliding_layers
              << " global_layers=" << global_layers << '\n'
              << "hidden_size=" << config.hidden_size
              << " intermediate_size=" << config.intermediate_size
              << " vocabulary_size=" << config.vocabulary_size << '\n'
              << "text_parameters=" << config.text_parameter_count() << '\n'
              << "text_weight_bytes=" << model.weights.byte_size("model.language_model.") << '\n'
              << "execution_segments=" << model.plan.segments().size() << '\n'
              << "execution_arena_bytes=" << model.plan.arena_bytes() << '\n'
              << "bf16_kv_bytes_per_uncapped_token=" << config.full_kv_bytes_per_token() << '\n'
              << "bf16_sliding_kv_bytes_per_sequence="
              << config.resident_sliding_kv_bytes_per_sequence() << '\n'
              << "validation=ok\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "carat-inspect: " << error.what() << '\n';
    return 1;
  }
}
