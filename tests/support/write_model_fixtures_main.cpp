#include "model_fixtures.h"

#include <exception>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: carat_model_fixtures_writer OUTPUT_DIRECTORY\n";
    return 64;
  }

  try {
    const std::filesystem::path output(argv[1]);
    std::filesystem::remove_all(output);

    const auto valid = output / "tiny";
    const auto corrupt = output / "corrupt";
    std::filesystem::create_directories(valid);
    std::filesystem::create_directories(corrupt);
    carat::testing::write_tiny_gemma4_model(valid);
    carat::testing::write_tiny_gemma4_model(corrupt);

    const auto shard = corrupt / "model-2.safetensors";
    std::filesystem::resize_file(shard, std::filesystem::file_size(shard) - 1);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "carat_model_fixtures_writer: " << error.what() << '\n';
    return 1;
  }
}
