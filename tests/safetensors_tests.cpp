#include "carat/safetensors.h"

#include "support/model_fixtures.h"
#include "support/test_support.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using carat::SafetensorsFile;
using carat::ShardedSafetensors;
using carat::testing::safetensors_header;
using carat::testing::TemporaryDirectory;
using carat::testing::TensorSpec;
using carat::testing::write_safetensors;
using carat::testing::write_safetensors_raw;
using carat::testing::write_sharded_safetensors;
using carat::testing::write_text;

const std::vector<TensorSpec> three_tensors{
    {"a", "BF16", {2, 3}},
    {"b", "F32", {4}},
    {"c", "BF16", {1}},
};

std::string single_tensor_header(std::string_view tensor_json) {
  return std::string(R"({"x":)") + std::string(tensor_json) + "}";
}

void write_header_only(const std::filesystem::path &path, const std::string &header,
                       std::uint64_t data_bytes) {
  write_safetensors_raw(path, header.size(), header, 8 + header.size() + data_bytes);
}

ENGINE_TEST(safetensors_file_reports_tensor_locations) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "weights.safetensors";
  write_safetensors(path, three_tensors);
  const std::uint64_t data_base = 8 + safetensors_header(three_tensors).size();

  const auto file = SafetensorsFile::open(path.string());

  CHECK(file.tensors().size() == 3);
  CHECK(file.tensors().find("__metadata__") == file.tensors().end());
  const auto &a = file.at("a");
  CHECK(a.shard_path == path.string());
  CHECK(a.dtype == carat::SafetensorsDtype::bf16);
  CHECK(file.at("b").dtype == carat::SafetensorsDtype::f32);
  CHECK(a.shape == std::vector<std::uint64_t>({2, 3}));
  CHECK(a.element_count() == 6);
  CHECK(a.file_offset == data_base);
  CHECK(a.byte_size == 12);
  CHECK(file.at("b").file_offset == data_base + 12);
  CHECK(file.at("b").byte_size == 16);
  CHECK(file.at("c").file_offset == data_base + 28);
  CHECK_THROWS(file.at("d"), "missing tensor d");
}

ENGINE_TEST(safetensors_file_rejects_invalid_header_sizes) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "weights.safetensors";
  const std::string header = safetensors_header(three_tensors);

  write_safetensors_raw(path, 0, header, 8 + header.size() + 32);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "invalid safetensors header size");

  write_safetensors_raw(path, 8 + header.size() + 32, header, 8 + header.size() + 32);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "invalid safetensors header size");

  const std::uint64_t above_limit = (1ULL << 30U) + 8;
  write_safetensors_raw(path, above_limit, header, 8 + above_limit + 8);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "invalid safetensors header size");

  write_safetensors_raw(path, 8, "", 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "cannot read safetensors header size");

  CHECK_THROWS(SafetensorsFile::open((temporary.path() / "absent").string()),
               "cannot open safetensors file");
}

ENGINE_TEST(safetensors_file_rejects_invalid_tensor_offsets) {
  TemporaryDirectory temporary;
  const auto path = temporary.path() / "weights.safetensors";

  write_header_only(path,
                    single_tensor_header(R"({"dtype":"BF16","shape":[2],"data_offsets":[0]})"), 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "invalid offsets for tensor x");

  write_header_only(
      path, single_tensor_header(R"({"dtype":"BF16","shape":[2],"data_offsets":[4,0]})"), 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "out-of-range tensor x");

  write_header_only(
      path, single_tensor_header(R"({"dtype":"BF16","shape":[2],"data_offsets":[0,4]})"), 3);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "out-of-range tensor x");

  write_header_only(
      path, single_tensor_header(R"({"dtype":"BF16","shape":[3],"data_offsets":[0,4]})"), 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "byte-size mismatch for tensor x");

  write_header_only(
      path, single_tensor_header(R"({"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]})"), 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "unsupported safetensors dtype: F8_E4M3");

  write_header_only(
      path, single_tensor_header(R"({"dtype":"BF16","shape":[-2],"data_offsets":[0,4]})"), 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "JSON integer is negative");

  write_header_only(path, "{\"x\":", 4);
  CHECK_THROWS(SafetensorsFile::open(path.string()), "JSON byte");
}

ENGINE_TEST(safetensors_file_rejects_every_truncation) {
  TemporaryDirectory temporary;
  const auto valid = temporary.path() / "valid.safetensors";
  write_safetensors(valid, three_tensors);
  const std::uint64_t size = std::filesystem::file_size(valid);
  const auto truncated = temporary.path() / "truncated.safetensors";

  for (std::uint64_t length = 0; length < size; ++length) {
    std::filesystem::copy_file(valid, truncated, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated, length);

    bool rejected = false;
    try {
      static_cast<void>(SafetensorsFile::open(truncated.string()));
    } catch (const std::exception &) {
      rejected = true;
    }
    CHECK(rejected);
  }
}

ENGINE_TEST(sharded_safetensors_reads_every_shard_in_the_index) {
  TemporaryDirectory temporary;
  const std::vector<std::vector<TensorSpec>> shards{
      {{"model.language_model.a", "BF16", {2, 3}}, {"vision.b", "F32", {4}}},
      {{"model.language_model.c", "BF16", {5}}},
  };
  write_sharded_safetensors(temporary.path(), shards);

  const auto weights = ShardedSafetensors::open(temporary.path().string());

  CHECK(weights.tensors().size() == 3);
  CHECK(weights.at("model.language_model.c").shard_path ==
        (temporary.path() / "model-2.safetensors").string());
  CHECK(weights.parameter_count("model.language_model.") == 11);
  CHECK(weights.byte_size("model.language_model.") == 22);
  CHECK(weights.parameter_count("") == 15);
  CHECK(weights.byte_size("") == 38);
  CHECK(weights.parameter_count("absent.") == 0);
  CHECK_THROWS(weights.at("absent"), "missing tensor absent");
}

ENGINE_TEST(sharded_safetensors_falls_back_to_a_single_file) {
  TemporaryDirectory temporary;
  write_safetensors(temporary.path() / "model.safetensors", three_tensors);

  const auto weights = ShardedSafetensors::open(temporary.path().string());

  CHECK(weights.tensors().size() == 3);
  CHECK(weights.parameter_count("") == 11);
  CHECK(weights.byte_size("") == 30);
}

ENGINE_TEST(sharded_safetensors_rejects_an_inconsistent_index) {
  TemporaryDirectory temporary;
  write_safetensors(temporary.path() / "model-1.safetensors", three_tensors);

  write_text(temporary.path() / "model.safetensors.index.json",
             R"({"weight_map":{"a":"model-1.safetensors","b":"model-9.safetensors"}})");
  CHECK_THROWS(ShardedSafetensors::open(temporary.path().string()), "cannot open safetensors file");

  write_text(temporary.path() / "model.safetensors.index.json",
             R"({"weight_map":{"a":"model-1.safetensors","z":"model-1.safetensors"}})");
  CHECK_THROWS(ShardedSafetensors::open(temporary.path().string()), "missing tensor z");

  write_text(temporary.path() / "model.safetensors.index.json", R"({"metadata":{}})");
  CHECK_THROWS(ShardedSafetensors::open(temporary.path().string()), "missing JSON key: weight_map");
}

ENGINE_TEST(sharded_safetensors_requires_weights) {
  TemporaryDirectory temporary;

  CHECK_THROWS(ShardedSafetensors::open(temporary.path().string()), "cannot open safetensors file");
}

} // namespace
