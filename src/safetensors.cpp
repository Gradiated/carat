#include "carat/safetensors.h"

#include "carat/json.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace carat {
namespace {

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const std::string &context) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::runtime_error("integer overflow for " + context);
  }

  return left * right;
}

SafetensorsDtype parse_dtype(std::string_view name) {
  static constexpr std::array<std::pair<std::string_view, SafetensorsDtype>, 13> dtypes{{
      {"BOOL", SafetensorsDtype::boolean},
      {"U8", SafetensorsDtype::u8},
      {"I8", SafetensorsDtype::i8},
      {"U16", SafetensorsDtype::u16},
      {"I16", SafetensorsDtype::i16},
      {"F16", SafetensorsDtype::f16},
      {"BF16", SafetensorsDtype::bf16},
      {"U32", SafetensorsDtype::u32},
      {"I32", SafetensorsDtype::i32},
      {"F32", SafetensorsDtype::f32},
      {"U64", SafetensorsDtype::u64},
      {"I64", SafetensorsDtype::i64},
      {"F64", SafetensorsDtype::f64},
  }};

  for (const auto &[wire_name, dtype] : dtypes) {
    if (wire_name == name)
      return dtype;
  }
  throw std::runtime_error("unsupported safetensors dtype: " + std::string(name));
}

std::uint64_t dtype_size(SafetensorsDtype dtype) {
  switch (dtype) {
  case SafetensorsDtype::boolean:
  case SafetensorsDtype::u8:
  case SafetensorsDtype::i8:
    return 1;
  case SafetensorsDtype::u16:
  case SafetensorsDtype::i16:
  case SafetensorsDtype::f16:
  case SafetensorsDtype::bf16:
    return 2;
  case SafetensorsDtype::u32:
  case SafetensorsDtype::i32:
  case SafetensorsDtype::f32:
    return 4;
  case SafetensorsDtype::u64:
  case SafetensorsDtype::i64:
  case SafetensorsDtype::f64:
    return 8;
  }
  throw std::logic_error("unhandled safetensors dtype");
}

std::uint64_t little_endian_u64(const std::array<unsigned char, 8> &bytes) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  }
  return value;
}

} // namespace

std::uint64_t element_count(std::span<const std::uint64_t> shape) {
  std::uint64_t result = 1;
  for (const std::uint64_t dimension : shape)
    result = checked_multiply(result, dimension, "tensor shape");
  return result;
}

std::uint64_t TensorLocation::element_count() const {
  return carat::element_count(shape);
}

SafetensorsFile SafetensorsFile::open(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open safetensors file " + path);

  std::array<unsigned char, 8> header_size_bytes{};
  input.read(reinterpret_cast<char *>(header_size_bytes.data()),
             static_cast<std::streamsize>(header_size_bytes.size()));
  if (!input)
    throw std::runtime_error("cannot read safetensors header size from " + path);

  const std::uint64_t header_size = little_endian_u64(header_size_bytes);
  const std::uint64_t file_size = std::filesystem::file_size(path);
  if (header_size == 0 || header_size > file_size - 8ULL || header_size > (1ULL << 30U)) {
    throw std::runtime_error("invalid safetensors header size in " + path);
  }

  std::string header(static_cast<std::size_t>(header_size), '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input)
    throw std::runtime_error("cannot read safetensors header from " + path);

  const Json metadata = Json::parse(header);
  const std::uint64_t data_base = 8ULL + header_size;

  SafetensorsFile result;
  for (const auto &[name, value] : metadata.as_object()) {
    if (name == "__metadata__")
      continue;

    const SafetensorsDtype dtype = parse_dtype(value.at("dtype").as_string());
    std::vector<std::uint64_t> shape;
    for (const Json &dimension : value.at("shape").as_array())
      shape.push_back(dimension.as_u64());
    const auto &offsets = value.at("data_offsets").as_array();
    if (offsets.size() != 2)
      throw std::runtime_error("invalid offsets for tensor " + name);

    const std::uint64_t begin = offsets[0].as_u64();
    const std::uint64_t end = offsets[1].as_u64();
    if (end < begin || data_base > file_size || end > file_size - data_base) {
      throw std::runtime_error("out-of-range tensor " + name);
    }

    TensorLocation tensor{path, dtype, std::move(shape), data_base + begin, end - begin};
    const std::uint64_t expected_size =
        checked_multiply(tensor.element_count(), dtype_size(dtype), name);
    if (expected_size != tensor.byte_size)
      throw std::runtime_error("byte-size mismatch for tensor " + name);

    result.tensors_.emplace(name, std::move(tensor));
  }
  return result;
}

const std::map<std::string, TensorLocation, std::less<>> &SafetensorsFile::tensors() const {
  return tensors_;
}

const TensorLocation &SafetensorsFile::at(std::string_view name) const {
  const auto iterator = tensors_.find(name);
  if (iterator == tensors_.end())
    throw std::runtime_error("missing tensor " + std::string(name));

  return iterator->second;
}

ShardedSafetensors ShardedSafetensors::open(const std::string &model_directory) {
  const std::filesystem::path directory(model_directory);
  const auto index_path = directory / "model.safetensors.index.json";
  if (!std::filesystem::exists(index_path)) {
    const auto single_path = directory / "model.safetensors";
    const auto single = SafetensorsFile::open(single_path.string());
    ShardedSafetensors result;
    result.tensors_ = single.tensors();
    return result;
  }
  const Json index = Json::parse(read_text_file(index_path.string()));
  const auto &weight_map = index.at("weight_map").as_object();
  std::set<std::string> shard_names;
  for (const auto &entry : weight_map)
    shard_names.insert(entry.second.as_string());

  ShardedSafetensors result;
  std::map<std::string, SafetensorsFile, std::less<>> shards;
  for (const auto &shard_name : shard_names) {
    shards.emplace(shard_name, SafetensorsFile::open((directory / shard_name).string()));
  }
  for (const auto &[name, shard] : weight_map) {
    const auto shard_iterator = shards.find(shard.as_string());
    if (shard_iterator == shards.end())
      throw std::runtime_error("unknown shard for tensor " + name);

    const TensorLocation &tensor = shard_iterator->second.at(name);
    result.tensors_.emplace(name, tensor);
  }
  return result;
}

const TensorLocation &ShardedSafetensors::at(std::string_view name) const {
  const auto iterator = tensors_.find(name);
  if (iterator == tensors_.end())
    throw std::runtime_error("missing tensor " + std::string(name));

  return iterator->second;
}

const std::map<std::string, TensorLocation, std::less<>> &ShardedSafetensors::tensors() const {
  return tensors_;
}

std::uint64_t ShardedSafetensors::parameter_count(std::string_view prefix) const {
  std::uint64_t result = 0;
  for (const auto &[name, tensor] : tensors_) {
    if (name.starts_with(prefix))
      result += tensor.element_count();
  }
  return result;
}

std::uint64_t ShardedSafetensors::byte_size(std::string_view prefix) const {
  std::uint64_t result = 0;
  for (const auto &[name, tensor] : tensors_) {
    if (name.starts_with(prefix))
      result += tensor.byte_size;
  }
  return result;
}

} // namespace carat
