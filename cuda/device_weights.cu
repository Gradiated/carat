#include "carat/device_weights.h"

#include "carat/safetensors.h"
#include "carat/weight_plan.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace carat {
namespace {

constexpr std::uint64_t alignment = 256;
constexpr std::size_t staging_bytes = 64ULL * 1024ULL * 1024ULL;

std::uint64_t align_up(std::uint64_t value) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

void cuda_check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class FileDescriptor {
public:
  explicit FileDescriptor(const std::string &path)
      : descriptor_(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
    if (descriptor_ < 0)
      throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  }
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;

  ~FileDescriptor() {
    if (descriptor_ >= 0)
      ::close(descriptor_);
  }
  int get() const {
    return descriptor_;
  }

private:
  int descriptor_;
};

void read_exact(int descriptor, void *destination, std::size_t bytes, std::uint64_t offset,
                const std::string &tensor_name) {
  auto *output = static_cast<unsigned char *>(destination);
  std::size_t complete = 0;
  while (complete < bytes) {
    const ssize_t result = ::pread(descriptor, output + complete, bytes - complete,
                                   static_cast<off_t>(offset + complete));
    if (result < 0 && errno == EINTR)
      continue;
    if (result <= 0)
      throw std::runtime_error("cannot read tensor " + tensor_name);
    complete += static_cast<std::size_t>(result);
  }
}

} // namespace

void require_supported_cuda_runtime() {
  const auto cudnn_version = cudnnGetVersion();
  if (cudnn_version < 91000) {
    throw std::runtime_error("Carat requires cuDNN 9.10 or newer; loaded version " +
                             std::to_string(cudnn_version));
  }
  int device = 0;
  cuda_check(cudaGetDevice(&device), "get CUDA device");
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
  if (properties.major != 9 || properties.minor != 0) {
    throw std::runtime_error("Carat requires a Hopper GPU (sm_90a)");
  }
}

struct DeviceWeightArena::Implementation {
  void *arena{nullptr};
  void *staging{nullptr};
  std::uint64_t allocated{0};
  std::uint64_t payload{0};
  std::map<std::string, const void *, std::less<>> pointers;

  ~Implementation() {
    if (staging != nullptr)
      cudaFreeHost(staging);
    if (arena != nullptr)
      cudaFree(arena);
  }
};

struct Fp8WeightArena::Implementation {
  void *arena{nullptr};
  void *scales{nullptr};
  std::uint64_t allocated{0};
  std::uint64_t scale_bytes{0};
  Fp8Scaling scaling{Fp8Scaling::tensor};
  std::map<std::string, const void *, std::less<>> pointers;
  std::map<std::string, const float *, std::less<>> scale_pointers;

  ~Implementation() {
    if (scales != nullptr)
      cudaFree(scales);
    if (arena != nullptr)
      cudaFree(arena);
  }
};

DeviceWeightArena::DeviceWeightArena(std::unique_ptr<Implementation> implementation)
    : implementation_(std::move(implementation)) {}
DeviceWeightArena::~DeviceWeightArena() = default;

std::unique_ptr<DeviceWeightArena>
DeviceWeightArena::load(const WeightPlan &plan, const ShardedSafetensors &weights,
                        const std::vector<std::string> &segment_names) {
  std::set<std::string, std::less<>> selected(segment_names.begin(), segment_names.end());
  for (const auto &name : selected)
    static_cast<void>(plan.at(name));
  auto implementation = std::make_unique<Implementation>();
  std::map<std::string, std::uint64_t, std::less<>> offsets;
  for (const auto &segment : plan.segments()) {
    if (!selected.empty() && !selected.contains(segment.name))
      continue;
    implementation->allocated = align_up(implementation->allocated);
    offsets.emplace(segment.name, implementation->allocated);
    implementation->allocated += segment.byte_size;
    implementation->payload += segment.byte_size;
  }
  implementation->allocated = align_up(implementation->allocated);
  if (implementation->allocated == 0)
    throw std::runtime_error("empty device weight selection");
  cuda_check(
      cudaMalloc(&implementation->arena, static_cast<std::size_t>(implementation->allocated)),
      "allocate device weight arena");
  cuda_check(cudaHostAlloc(&implementation->staging, staging_bytes, cudaHostAllocPortable),
             "allocate pinned weight staging buffer");

  std::map<std::string, std::unique_ptr<FileDescriptor>, std::less<>> files;
  for (const auto &segment : plan.segments()) {
    const auto selected_offset = offsets.find(segment.name);
    if (selected_offset == offsets.end())
      continue;
    auto *segment_destination =
        static_cast<unsigned char *>(implementation->arena) + selected_offset->second;
    implementation->pointers.emplace(segment.name, segment_destination);
    for (const auto &source : segment.sources) {
      const TensorLocation &tensor = weights.at(source.tensor_name);
      auto file = files.find(tensor.shard_path);
      if (file == files.end()) {
        file = files.emplace(tensor.shard_path, std::make_unique<FileDescriptor>(tensor.shard_path))
                   .first;
      }
      std::uint64_t copied = 0;
      while (copied < source.byte_size) {
        const auto chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(staging_bytes, source.byte_size - copied));
        read_exact(file->second->get(), implementation->staging, chunk, tensor.file_offset + copied,
                   source.tensor_name);
        cuda_check(cudaMemcpy(segment_destination + source.segment_offset + copied,
                              implementation->staging, chunk, cudaMemcpyHostToDevice),
                   "copy weight to device");
        copied += chunk;
      }
    }
  }
  return std::unique_ptr<DeviceWeightArena>(new DeviceWeightArena(std::move(implementation)));
}

const void *DeviceWeightArena::at(std::string_view name) const {
  const auto iterator = implementation_->pointers.find(name);
  if (iterator == implementation_->pointers.end()) {
    throw std::runtime_error("device weight is not loaded: " + std::string(name));
  }
  return iterator->second;
}

std::uint64_t DeviceWeightArena::allocated_bytes() const {
  return implementation_->allocated;
}
std::uint64_t DeviceWeightArena::payload_bytes() const {
  return implementation_->payload;
}

Fp8WeightArena::Fp8WeightArena(std::unique_ptr<Implementation> implementation)
    : implementation_(std::move(implementation)) {}
Fp8WeightArena::~Fp8WeightArena() = default;

std::unique_ptr<Fp8WeightArena> Fp8WeightArena::quantize(const WeightPlan &plan,
                                                         const DeviceWeightArena &bf16_weights,
                                                         Fp8Scaling scaling,
                                                         float tensor_scale_multiplier) {
  if (!std::isfinite(tensor_scale_multiplier) || tensor_scale_multiplier <= 0.0F ||
      tensor_scale_multiplier > 1.0F ||
      (scaling != Fp8Scaling::tensor && tensor_scale_multiplier != 1.0F)) {
    throw std::runtime_error("invalid FP8 tensor weight scale multiplier");
  }
  auto implementation = std::make_unique<Implementation>();
  implementation->scaling = scaling;
  std::map<std::string, std::uint64_t, std::less<>> offsets;
  std::map<std::string, std::uint64_t, std::less<>> scale_offsets;
  for (const auto &segment : plan.segments()) {
    if (segment.shape.size() != 2)
      continue;
    implementation->allocated = align_up(implementation->allocated);
    offsets.emplace(segment.name, implementation->allocated);
    implementation->allocated += segment.byte_size / 2;
    implementation->scale_bytes = align_up(implementation->scale_bytes);
    scale_offsets.emplace(segment.name, implementation->scale_bytes);
    std::size_t scale_elements = 1;
    if (scaling == Fp8Scaling::channel) {
      scale_elements = static_cast<std::size_t>(segment.shape[0]);
    } else if (scaling == Fp8Scaling::block_128) {
      scale_elements = fp8_weight_block_scale_elements(static_cast<int>(segment.shape[0]),
                                                       static_cast<int>(segment.shape[1]));
    }
    implementation->scale_bytes += scale_elements * sizeof(float);
  }
  implementation->allocated = align_up(implementation->allocated);
  implementation->scale_bytes = align_up(implementation->scale_bytes);
  if (implementation->allocated == 0 || implementation->scale_bytes == 0) {
    throw std::runtime_error("empty FP8 weight plan");
  }
  cuda_check(cudaMalloc(&implementation->arena, implementation->allocated),
             "allocate FP8 weight arena");
  cuda_check(cudaMalloc(&implementation->scales, implementation->scale_bytes),
             "allocate FP8 scale arena");
  for (const auto &segment : plan.segments()) {
    const auto offset = offsets.find(segment.name);
    if (offset == offsets.end())
      continue;
    auto *output = static_cast<unsigned char *>(implementation->arena) + offset->second;
    auto *scale = reinterpret_cast<float *>(static_cast<unsigned char *>(implementation->scales) +
                                            scale_offsets.at(segment.name));
    implementation->pointers.emplace(segment.name, output);
    implementation->scale_pointers.emplace(segment.name, scale);
    const int output_width = static_cast<int>(segment.shape[0]);
    const int input_width = static_cast<int>(segment.shape[1]);
    if (scaling == Fp8Scaling::tensor) {
      quantize_bf16_to_fp8_e4m3(bf16_weights.at(segment.name), output, scale, segment.byte_size / 2,
                                nullptr, tensor_scale_multiplier);
    } else if (scaling == Fp8Scaling::channel) {
      quantize_bf16_rows_to_fp8_e4m3(bf16_weights.at(segment.name), output, scale, output_width,
                                     input_width, nullptr);
    } else {
      quantize_bf16_weight_to_fp8_block_128(bf16_weights.at(segment.name), output, scale,
                                            output_width, input_width, nullptr);
    }
  }
  cuda_check(cudaDeviceSynchronize(), "finish FP8 weight quantization");
  return std::unique_ptr<Fp8WeightArena>(new Fp8WeightArena(std::move(implementation)));
}

const void *Fp8WeightArena::at(std::string_view name) const {
  const auto iterator = implementation_->pointers.find(name);
  if (iterator == implementation_->pointers.end()) {
    throw std::runtime_error("FP8 weight is not loaded: " + std::string(name));
  }
  return iterator->second;
}

const float *Fp8WeightArena::scale(std::string_view name) const {
  const auto iterator = implementation_->scale_pointers.find(name);
  if (iterator == implementation_->scale_pointers.end()) {
    throw std::runtime_error("FP8 weight scale is not loaded: " + std::string(name));
  }
  return iterator->second;
}

void Fp8WeightArena::roundtrip_int4(std::string_view name, int output_width, int input_width,
                                    int block_width, int scale_refinement_iterations) {
  const auto iterator = implementation_->pointers.find(name);
  if (iterator == implementation_->pointers.end()) {
    throw std::runtime_error("FP8 weight is not loaded: " + std::string(name));
  }
  roundtrip_fp8_weight_via_int4_blocks(const_cast<void *>(iterator->second), output_width,
                                       input_width, block_width, nullptr,
                                       scale_refinement_iterations);
  cuda_check(cudaDeviceSynchronize(), "finish INT4 weight oracle");
}

Fp8Scaling Fp8WeightArena::scaling() const {
  return implementation_->scaling;
}
std::uint64_t Fp8WeightArena::allocated_bytes() const {
  return implementation_->allocated + implementation_->scale_bytes;
}

} // namespace carat
