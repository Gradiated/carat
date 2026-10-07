#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace carat::testing {

class TemporaryDirectory {
public:
  TemporaryDirectory();
  ~TemporaryDirectory();
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const;

private:
  std::filesystem::path path_;
};

struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
};

[[nodiscard]] std::uint64_t tensor_bytes(const TensorSpec &tensor);

void write_text(const std::filesystem::path &path, std::string_view text);

// Writes the 8-byte little-endian header size, the header, then extends the file to
// file_size. Extending with resize_file keeps multi-gigabyte fixtures sparse.
void write_safetensors_raw(const std::filesystem::path &path, std::uint64_t declared_header_size,
                           std::string_view header, std::uint64_t file_size);

[[nodiscard]] std::string safetensors_header(std::span<const TensorSpec> tensors);

void write_safetensors(const std::filesystem::path &path, std::span<const TensorSpec> tensors);

void write_sharded_safetensors(const std::filesystem::path &directory,
                               std::span<const std::vector<TensorSpec>> shards);

[[nodiscard]] std::string replace_once(std::string text, std::string_view from,
                                       std::string_view to);

[[nodiscard]] std::string tiny_gemma4_config();

[[nodiscard]] std::vector<TensorSpec> tiny_gemma4_tensors();

void write_tiny_gemma4_model(const std::filesystem::path &directory);

[[nodiscard]] std::vector<TensorSpec> gemma4_assistant_tensors();

} // namespace carat::testing
