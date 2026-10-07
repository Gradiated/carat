#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace carat {

enum class SafetensorsDtype { boolean, u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64 };

[[nodiscard]] std::uint64_t element_count(std::span<const std::uint64_t> shape);

struct TensorLocation {
  std::string shard_path;
  SafetensorsDtype dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_offset;
  std::uint64_t byte_size;

  [[nodiscard]] std::uint64_t element_count() const;
};

class SafetensorsFile {
public:
  static SafetensorsFile open(const std::string &path);

  [[nodiscard]] const std::map<std::string, TensorLocation, std::less<>> &tensors() const;
  [[nodiscard]] const TensorLocation &at(std::string_view name) const;

private:
  std::map<std::string, TensorLocation, std::less<>> tensors_;
};

class ShardedSafetensors {
public:
  static ShardedSafetensors open(const std::string &model_directory);

  [[nodiscard]] const TensorLocation &at(std::string_view name) const;
  [[nodiscard]] const std::map<std::string, TensorLocation, std::less<>> &tensors() const;
  [[nodiscard]] std::uint64_t parameter_count(std::string_view prefix) const;
  [[nodiscard]] std::uint64_t byte_size(std::string_view prefix) const;

private:
  std::map<std::string, TensorLocation, std::less<>> tensors_;
};

} // namespace carat
