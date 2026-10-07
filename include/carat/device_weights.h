#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "carat/linear.h"

namespace carat {

void require_supported_cuda_runtime();

class DeviceWeightArena {
public:
  static std::unique_ptr<DeviceWeightArena>
  load(const class WeightPlan &plan, const class ShardedSafetensors &weights,
       const std::vector<std::string> &segment_names = {});
  ~DeviceWeightArena();

  DeviceWeightArena(const DeviceWeightArena &) = delete;
  DeviceWeightArena &operator=(const DeviceWeightArena &) = delete;

  [[nodiscard]] const void *at(std::string_view name) const;
  [[nodiscard]] std::uint64_t allocated_bytes() const;
  [[nodiscard]] std::uint64_t payload_bytes() const;

private:
  struct Implementation;
  explicit DeviceWeightArena(std::unique_ptr<Implementation> implementation);
  std::unique_ptr<Implementation> implementation_;
};

class Fp8WeightArena {
public:
  static std::unique_ptr<Fp8WeightArena> quantize(const class WeightPlan &plan,
                                                  const DeviceWeightArena &bf16_weights,
                                                  Fp8Scaling scaling,
                                                  float tensor_scale_multiplier = 1.0F);
  ~Fp8WeightArena();

  Fp8WeightArena(const Fp8WeightArena &) = delete;
  Fp8WeightArena &operator=(const Fp8WeightArena &) = delete;

  [[nodiscard]] const void *at(std::string_view name) const;
  [[nodiscard]] const float *scale(std::string_view name) const;
  void roundtrip_int4(std::string_view name, int output_width, int input_width, int block_width,
                      int scale_refinement_iterations = 0);
  [[nodiscard]] Fp8Scaling scaling() const;
  [[nodiscard]] std::uint64_t allocated_bytes() const;

private:
  struct Implementation;
  explicit Fp8WeightArena(std::unique_ptr<Implementation> implementation);
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
