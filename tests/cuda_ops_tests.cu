#include "carat/attention.h"
#include "carat/cuda_ops.h"
#include "carat/gemm_attention.h"
#include "carat/linear.h"
#include "carat/prefill_gemm_attention.h"
#include "carat/qkv.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void near(float actual, float expected, float tolerance, const char *operation) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(operation) + ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

template <std::size_t Size>
std::array<__nv_bfloat16, Size> bf16(const std::array<float, Size> &values) {
  std::array<__nv_bfloat16, Size> result{};
  for (std::size_t index = 0; index < Size; ++index)
    result[index] = __float2bfloat16(values[index]);
  return result;
}

template <typename T> class DeviceAllocation {
public:
  explicit DeviceAllocation(std::size_t elements) {
    cuda_check(cudaMalloc(&pointer_, elements * sizeof(T)), "cudaMalloc");
  }
  ~DeviceAllocation() {
    cudaFree(pointer_);
  }
  T *get() {
    return static_cast<T *>(pointer_);
  }
  const T *get() const {
    return static_cast<const T *>(pointer_);
  }

private:
  void *pointer_{nullptr};
};

void test_norm_add_and_scale() {
  constexpr std::size_t width = 8;
  const auto input = bf16<width>({1, 2, 3, 4, 5, 6, 7, 8});
  const auto weight = bf16<width>({1, 1, 1, 1, 1, 1, 1, 1});
  const auto residual = bf16<width>({0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F});
  const auto scalar = bf16<1>({0.5F});
  DeviceAllocation<__nv_bfloat16> device_input(width), device_weight(width), device_residual(width),
      device_output(width), device_fused_output(width), device_first_residual(width),
      device_fused_first_residual(width), device_normalized(width), device_fused_normalized(width),
      device_scalar(1);
  cuda_check(cudaMemcpy(device_input.get(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
             "copy input");
  cuda_check(cudaMemcpy(device_weight.get(), weight.data(), sizeof(weight), cudaMemcpyHostToDevice),
             "copy weight");
  cuda_check(
      cudaMemcpy(device_residual.get(), residual.data(), sizeof(residual), cudaMemcpyHostToDevice),
      "copy residual");
  cuda_check(cudaMemcpy(device_scalar.get(), scalar.data(), sizeof(scalar), cudaMemcpyHostToDevice),
             "copy scalar");
  carat::rms_norm_add_bf16(device_input.get(), device_weight.get(), device_residual.get(),
                           device_output.get(), 1, width, 1e-6F, nullptr);
  carat::scale_bf16(device_output.get(), device_scalar.get(), width, nullptr);
  carat::rms_norm_add_scale_bf16(device_input.get(), device_weight.get(), device_residual.get(),
                                 device_scalar.get(), device_fused_output.get(), 1, width, 1e-6F,
                                 nullptr);
  carat::rms_norm_add_bf16(device_input.get(), device_weight.get(), device_residual.get(),
                           device_first_residual.get(), 1, width, 1e-6F, nullptr);
  carat::rms_norm_bf16(device_first_residual.get(), device_weight.get(), device_normalized.get(), 1,
                       width, 1e-6F, nullptr);
  carat::rms_norm_add_and_norm_bf16(device_input.get(), device_weight.get(), device_residual.get(),
                                    device_weight.get(), device_fused_first_residual.get(),
                                    device_fused_normalized.get(), 1, width, 1e-6F, nullptr);
  cuda_check(cudaDeviceSynchronize(), "norm synchronization");
  std::array<__nv_bfloat16, width> output{};
  std::array<__nv_bfloat16, width> fused_output{}, first_residual{}, fused_first_residual{},
      normalized{}, fused_normalized{};
  cuda_check(cudaMemcpy(output.data(), device_output.get(), sizeof(output), cudaMemcpyDeviceToHost),
             "copy output");
  cuda_check(cudaMemcpy(fused_output.data(), device_fused_output.get(), sizeof(fused_output),
                        cudaMemcpyDeviceToHost),
             "copy fused output");
  cuda_check(cudaMemcpy(first_residual.data(), device_first_residual.get(), sizeof(first_residual),
                        cudaMemcpyDeviceToHost),
             "copy first residual");
  cuda_check(cudaMemcpy(fused_first_residual.data(), device_fused_first_residual.get(),
                        sizeof(fused_first_residual), cudaMemcpyDeviceToHost),
             "copy fused first residual");
  cuda_check(cudaMemcpy(normalized.data(), device_normalized.get(), sizeof(normalized),
                        cudaMemcpyDeviceToHost),
             "copy normalized");
  cuda_check(cudaMemcpy(fused_normalized.data(), device_fused_normalized.get(),
                        sizeof(fused_normalized), cudaMemcpyDeviceToHost),
             "copy fused normalized");
  float mean_square = 0.0F;
  for (const auto value : input) {
    const float converted = __bfloat162float(value);
    mean_square += converted * converted;
  }
  mean_square /= static_cast<float>(width);
  for (std::size_t index = 0; index < width; ++index) {
    const float expected =
        (__bfloat162float(input[index]) / std::sqrt(mean_square + 1e-6F) + 0.5F) * 0.5F;
    near(__bfloat162float(output[index]), expected, 0.012F, "rms_norm_add/scale");
  }
  if (std::memcmp(output.data(), fused_output.data(), sizeof(output)) != 0 ||
      std::memcmp(first_residual.data(), fused_first_residual.data(), sizeof(first_residual)) !=
          0 ||
      std::memcmp(normalized.data(), fused_normalized.data(), sizeof(normalized)) != 0) {
    throw std::runtime_error("fused normalization changed BF16 rounding");
  }
}

void test_norm_carried_fp8_amax() {
  constexpr int rows = 3;
  constexpr int width = 5376;
  constexpr float epsilon = 1e-6F;
  const std::size_t elements = static_cast<std::size_t>(rows) * width;
  std::vector<__nv_bfloat16> input(elements), residual(elements), weight(width), next_weight(width);
  for (std::size_t index = 0; index < elements; ++index) {
    input[index] = __float2bfloat16_rn(1.3F * std::sin(static_cast<float>(index) * 0.013F) +
                                       0.2F * std::cos(static_cast<float>(index) * 0.031F));
    residual[index] = __float2bfloat16_rn(0.7F * std::cos(static_cast<float>(index) * 0.017F));
  }
  for (int column = 0; column < width; ++column) {
    weight[column] =
        __float2bfloat16_rn(0.8F + 0.3F * std::sin(static_cast<float>(column) * 0.007F));
    next_weight[column] =
        __float2bfloat16_rn(1.1F + 0.2F * std::cos(static_cast<float>(column) * 0.011F));
  }

  DeviceAllocation<__nv_bfloat16> device_input(elements), device_residual(elements),
      device_weight(width), device_next_weight(width), device_regular(elements),
      device_carried(elements), device_regular_residual(elements),
      device_carried_residual(elements);
  DeviceAllocation<__nv_fp8_e4m3> device_regular_fp8(elements), device_carried_fp8(elements);
  DeviceAllocation<float> device_regular_scale(1), device_carried_scale(1), device_row_amax(rows);
  cuda_check(cudaMemcpy(device_input.get(), input.data(), elements * sizeof(input[0]),
                        cudaMemcpyHostToDevice),
             "copy carried-amax input");
  cuda_check(cudaMemcpy(device_residual.get(), residual.data(), elements * sizeof(residual[0]),
                        cudaMemcpyHostToDevice),
             "copy carried-amax residual");
  cuda_check(cudaMemcpy(device_weight.get(), weight.data(), width * sizeof(weight[0]),
                        cudaMemcpyHostToDevice),
             "copy carried-amax weight");
  cuda_check(cudaMemcpy(device_next_weight.get(), next_weight.data(),
                        width * sizeof(next_weight[0]), cudaMemcpyHostToDevice),
             "copy carried-amax next weight");

  const auto compare_quantization = [&](const char *operation) {
    carat::quantize_bf16_to_fp8_e4m3(device_regular.get(), device_regular_fp8.get(),
                                     device_regular_scale.get(), elements, nullptr);
    carat::quantize_bf16_to_fp8_e4m3_from_row_amax(device_carried.get(), device_carried_fp8.get(),
                                                   device_carried_scale.get(),
                                                   device_row_amax.get(), rows, width, nullptr);
    cuda_check(cudaDeviceSynchronize(), operation);
    std::vector<__nv_bfloat16> regular(elements), carried(elements);
    std::vector<__nv_fp8_e4m3> regular_fp8(elements), carried_fp8(elements);
    float regular_scale = 0.0F;
    float carried_scale = 0.0F;
    cuda_check(cudaMemcpy(regular.data(), device_regular.get(), elements * sizeof(regular[0]),
                          cudaMemcpyDeviceToHost),
               "copy regular normalized output");
    cuda_check(cudaMemcpy(carried.data(), device_carried.get(), elements * sizeof(carried[0]),
                          cudaMemcpyDeviceToHost),
               "copy carried-amax normalized output");
    cuda_check(cudaMemcpy(regular_fp8.data(), device_regular_fp8.get(),
                          elements * sizeof(regular_fp8[0]), cudaMemcpyDeviceToHost),
               "copy regular FP8 output");
    cuda_check(cudaMemcpy(carried_fp8.data(), device_carried_fp8.get(),
                          elements * sizeof(carried_fp8[0]), cudaMemcpyDeviceToHost),
               "copy carried-amax FP8 output");
    cuda_check(cudaMemcpy(&regular_scale, device_regular_scale.get(), sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy regular FP8 scale");
    cuda_check(cudaMemcpy(&carried_scale, device_carried_scale.get(), sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy carried-amax FP8 scale");
    if (std::memcmp(regular.data(), carried.data(), elements * sizeof(regular[0])) != 0 ||
        std::memcmp(regular_fp8.data(), carried_fp8.data(), elements * sizeof(regular_fp8[0])) !=
            0 ||
        std::memcmp(&regular_scale, &carried_scale, sizeof(float)) != 0) {
      throw std::runtime_error(std::string(operation) + " changed BF16 or FP8 representation");
    }
  };

  carat::rms_norm_bf16(device_input.get(), device_weight.get(), device_regular.get(), rows, width,
                       epsilon, nullptr);
  carat::rms_norm_bf16_with_row_amax(device_input.get(), device_weight.get(), device_carried.get(),
                                     device_row_amax.get(), rows, width, epsilon, nullptr);
  compare_quantization("synchronize RMSNorm carried amax");

  carat::rms_norm_add_and_norm_bf16(device_input.get(), device_weight.get(), device_residual.get(),
                                    device_next_weight.get(), device_regular_residual.get(),
                                    device_regular.get(), rows, width, epsilon, nullptr);
  carat::rms_norm_add_and_norm_bf16_with_row_amax(
      device_input.get(), device_weight.get(), device_residual.get(), device_next_weight.get(),
      device_carried_residual.get(), device_carried.get(), device_row_amax.get(), rows, width,
      epsilon, nullptr);
  compare_quantization("synchronize dual RMSNorm carried amax");
  std::vector<__nv_bfloat16> regular_residual(elements), carried_residual(elements);
  cuda_check(cudaMemcpy(regular_residual.data(), device_regular_residual.get(),
                        elements * sizeof(regular_residual[0]), cudaMemcpyDeviceToHost),
             "copy regular carried-amax residual");
  cuda_check(cudaMemcpy(carried_residual.data(), device_carried_residual.get(),
                        elements * sizeof(carried_residual[0]), cudaMemcpyDeviceToHost),
             "copy carried-amax residual");
  if (std::memcmp(regular_residual.data(), carried_residual.data(),
                  elements * sizeof(regular_residual[0])) != 0) {
    throw std::runtime_error("carried amax changed post-attention residual");
  }
}

void test_gelu_gate() {
  const auto gate = bf16<4>({-1.0F, 0.0F, 1.0F, 2.0F});
  const auto up = bf16<4>({2.0F, 2.0F, 2.0F, 2.0F});
  DeviceAllocation<__nv_bfloat16> device_gate(4), device_up(4), device_output(4);
  cuda_check(cudaMemcpy(device_gate.get(), gate.data(), sizeof(gate), cudaMemcpyHostToDevice),
             "copy gate");
  cuda_check(cudaMemcpy(device_up.get(), up.data(), sizeof(up), cudaMemcpyHostToDevice), "copy up");
  carat::gelu_tanh_gate_bf16(device_gate.get(), device_up.get(), device_output.get(), 4, nullptr);
  cuda_check(cudaDeviceSynchronize(), "gelu synchronization");
  std::array<__nv_bfloat16, 4> output{};
  cuda_check(cudaMemcpy(output.data(), device_output.get(), sizeof(output), cudaMemcpyDeviceToHost),
             "copy gelu");
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float value = __bfloat162float(gate[index]);
    const float expected =
        value *
        (1.0F + std::tanh(0.7978845608028654F * (value + 0.044715F * value * value * value)));
    near(__bfloat162float(output[index]), expected, 0.02F, "gelu gate");
  }
}

void test_gelu_carried_fp8_amax() {
  constexpr int rows = 3;
  constexpr int width = 21504;
  constexpr int producer_block_size = 256;
  constexpr int partial_amax_count =
      rows * ((width + producer_block_size - 1) / producer_block_size);
  constexpr int blocks_per_row = partial_amax_count / rows;
  const std::size_t elements = static_cast<std::size_t>(rows) * width;
  std::vector<__nv_bfloat16> packed(2 * elements);
  for (int row = 0; row < rows; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * 2 * width;
    for (int column = 0; column < width; ++column) {
      const float index = static_cast<float>(row * width + column);
      packed[base + column] =
          __float2bfloat16_rn(2.1F * std::sin(index * 0.007F) + 0.3F * std::cos(index * 0.019F));
      packed[base + width + column] =
          __float2bfloat16_rn(1.4F * std::cos(index * 0.011F) - 0.2F * std::sin(index * 0.023F));
    }
  }

  DeviceAllocation<__nv_bfloat16> device_packed(2 * elements), device_regular(elements),
      device_carried(elements);
  DeviceAllocation<__nv_fp8_e4m3> device_regular_fp8(elements), device_carried_fp8(elements);
  DeviceAllocation<float> device_regular_scale(1), device_carried_scale(1),
      device_partial_amax(partial_amax_count), device_row_amax(rows);
  cuda_check(cudaMemcpy(device_packed.get(), packed.data(), packed.size() * sizeof(packed[0]),
                        cudaMemcpyHostToDevice),
             "copy GELU carried-amax input");
  carat::gelu_tanh_gate_rows_bf16(device_packed.get(), device_regular.get(), rows, width, nullptr);
  carat::gelu_tanh_gate_rows_bf16_with_block_amax(device_packed.get(), device_carried.get(),
                                                  device_partial_amax.get(), rows, width, nullptr);
  carat::quantize_bf16_to_fp8_e4m3(device_regular.get(), device_regular_fp8.get(),
                                   device_regular_scale.get(), elements, nullptr);
  carat::reduce_block_amax_to_rows(device_partial_amax.get(), device_row_amax.get(), rows,
                                   blocks_per_row, nullptr);
  carat::quantize_bf16_to_fp8_e4m3_from_row_amax(device_carried.get(), device_carried_fp8.get(),
                                                 device_carried_scale.get(), device_row_amax.get(),
                                                 rows, width, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize GELU carried amax");

  std::vector<__nv_bfloat16> regular(elements), carried(elements);
  std::vector<__nv_fp8_e4m3> regular_fp8(elements), carried_fp8(elements);
  float regular_scale = 0.0F;
  float carried_scale = 0.0F;
  cuda_check(cudaMemcpy(regular.data(), device_regular.get(), elements * sizeof(regular[0]),
                        cudaMemcpyDeviceToHost),
             "copy regular GELU output");
  cuda_check(cudaMemcpy(carried.data(), device_carried.get(), elements * sizeof(carried[0]),
                        cudaMemcpyDeviceToHost),
             "copy carried GELU output");
  cuda_check(cudaMemcpy(regular_fp8.data(), device_regular_fp8.get(),
                        elements * sizeof(regular_fp8[0]), cudaMemcpyDeviceToHost),
             "copy regular GELU FP8 output");
  cuda_check(cudaMemcpy(carried_fp8.data(), device_carried_fp8.get(),
                        elements * sizeof(carried_fp8[0]), cudaMemcpyDeviceToHost),
             "copy carried GELU FP8 output");
  cuda_check(
      cudaMemcpy(&regular_scale, device_regular_scale.get(), sizeof(float), cudaMemcpyDeviceToHost),
      "copy regular GELU FP8 scale");
  cuda_check(
      cudaMemcpy(&carried_scale, device_carried_scale.get(), sizeof(float), cudaMemcpyDeviceToHost),
      "copy carried GELU FP8 scale");
  if (std::memcmp(regular.data(), carried.data(), elements * sizeof(regular[0])) != 0 ||
      std::memcmp(regular_fp8.data(), carried_fp8.data(), elements * sizeof(regular_fp8[0])) != 0 ||
      std::memcmp(&regular_scale, &carried_scale, sizeof(float)) != 0) {
    throw std::runtime_error("carried GELU amax changed BF16 or FP8 representation");
  }
}

void test_head_to_token_carried_fp8_amax() {
  constexpr int requests = 3;
  constexpr int tokens = 5;
  constexpr int heads = 2;
  constexpr int dimension = 257;
  constexpr int rows = requests * tokens;
  constexpr int width = heads * dimension;
  constexpr int blocks_per_row = (width + 255) / 256;
  constexpr int partial_amax_count = rows * blocks_per_row;
  constexpr std::size_t elements = static_cast<std::size_t>(rows) * width;
  std::vector<__nv_bfloat16> token_major(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    token_major[index] = __float2bfloat16_rn(2.3F * std::sin(static_cast<float>(index) * 0.013F) -
                                             0.4F * std::cos(static_cast<float>(index) * 0.031F));
  }
  DeviceAllocation<__nv_bfloat16> device_token_major(elements), device_head_major(elements),
      device_regular(elements), device_carried(elements);
  DeviceAllocation<__nv_fp8_e4m3> device_regular_fp8(elements), device_carried_fp8(elements);
  DeviceAllocation<float> device_regular_scale(1), device_carried_scale(1),
      device_partial_amax(partial_amax_count);
  cuda_check(cudaMemcpy(device_token_major.get(), token_major.data(),
                        elements * sizeof(token_major[0]), cudaMemcpyHostToDevice),
             "copy carried-amax transpose input");
  carat::token_to_head_batch_bf16(device_token_major.get(), device_head_major.get(), requests,
                                  tokens, heads, dimension, nullptr);
  carat::head_to_token_batch_bf16(device_head_major.get(), device_regular.get(), requests, tokens,
                                  heads, dimension, nullptr);
  carat::head_to_token_batch_bf16_with_block_amax(device_head_major.get(), device_carried.get(),
                                                  device_partial_amax.get(), requests, tokens,
                                                  heads, dimension, nullptr);
  carat::quantize_bf16_to_fp8_e4m3(device_regular.get(), device_regular_fp8.get(),
                                   device_regular_scale.get(), elements, nullptr);
  carat::quantize_bf16_to_fp8_e4m3_from_amax_values(
      device_carried.get(), device_carried_fp8.get(), device_carried_scale.get(),
      device_partial_amax.get(), partial_amax_count, rows, width, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize carried-amax transpose");

  std::vector<__nv_bfloat16> regular(elements), carried(elements);
  std::vector<__nv_fp8_e4m3> regular_fp8(elements), carried_fp8(elements);
  float regular_scale = 0.0F;
  float carried_scale = 0.0F;
  cuda_check(cudaMemcpy(regular.data(), device_regular.get(), elements * sizeof(regular[0]),
                        cudaMemcpyDeviceToHost),
             "copy regular transpose output");
  cuda_check(cudaMemcpy(carried.data(), device_carried.get(), elements * sizeof(carried[0]),
                        cudaMemcpyDeviceToHost),
             "copy carried transpose output");
  cuda_check(cudaMemcpy(regular_fp8.data(), device_regular_fp8.get(),
                        elements * sizeof(regular_fp8[0]), cudaMemcpyDeviceToHost),
             "copy regular transpose FP8 output");
  cuda_check(cudaMemcpy(carried_fp8.data(), device_carried_fp8.get(),
                        elements * sizeof(carried_fp8[0]), cudaMemcpyDeviceToHost),
             "copy carried transpose FP8 output");
  cuda_check(
      cudaMemcpy(&regular_scale, device_regular_scale.get(), sizeof(float), cudaMemcpyDeviceToHost),
      "copy regular transpose FP8 scale");
  cuda_check(
      cudaMemcpy(&carried_scale, device_carried_scale.get(), sizeof(float), cudaMemcpyDeviceToHost),
      "copy carried transpose FP8 scale");
  if (std::memcmp(regular.data(), carried.data(), elements * sizeof(regular[0])) != 0 ||
      std::memcmp(regular_fp8.data(), carried_fp8.data(), elements * sizeof(regular_fp8[0])) != 0 ||
      std::memcmp(&regular_scale, &carried_scale, sizeof(float)) != 0) {
    throw std::runtime_error("carried transpose amax changed BF16 or FP8 representation");
  }
}

void test_linear() {
  const auto input = bf16<6>({1, 2, 3, 4, 5, 6});
  const auto weight = bf16<12>({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1});
  DeviceAllocation<__nv_bfloat16> device_input(6), device_weight(12), device_output(8);
  cuda_check(cudaMemcpy(device_input.get(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
             "copy linear input");
  cuda_check(cudaMemcpy(device_weight.get(), weight.data(), sizeof(weight), cudaMemcpyHostToDevice),
             "copy linear weight");
  carat::Bf16Linear linear;
  linear.run(device_input.get(), device_weight.get(), device_output.get(), 2, 3, 4, nullptr);
  cuda_check(cudaDeviceSynchronize(), "linear synchronization");
  std::array<__nv_bfloat16, 8> output{};
  cuda_check(cudaMemcpy(output.data(), device_output.get(), sizeof(output), cudaMemcpyDeviceToHost),
             "copy linear output");
  const std::array<float, 8> expected{1, 2, 3, 6, 4, 5, 6, 15};
  for (std::size_t index = 0; index < output.size(); ++index) {
    near(__bfloat162float(output[index]), expected[index], 0.01F, "linear");
  }
}

void test_int4_weight_oracle() {
  constexpr int rows = 3;
  constexpr int width = 300;
  constexpr int block_width = 128;
  std::vector<__nv_fp8_e4m3> values(static_cast<std::size_t>(rows) * width), expected;
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = __nv_fp8_e4m3(2.75F * std::sin(static_cast<float>(index) * 0.071F) +
                                  0.31F * std::cos(static_cast<float>(index) * 0.013F));
  }
  expected = values;
  for (int row = 0; row < rows; ++row) {
    for (int begin = 0; begin < width; begin += block_width) {
      const int end = std::min(width, begin + block_width);
      float maximum = 0.0F;
      for (int column = begin; column < end; ++column) {
        maximum = std::max(maximum, std::abs(static_cast<float>(
                                        expected[static_cast<std::size_t>(row) * width + column])));
      }
      const float scale = maximum > 0.0F ? std::min(maximum / 7.0F, 56.0F) : 1.0F;
      for (int column = begin; column < end; ++column) {
        const std::size_t index = static_cast<std::size_t>(row) * width + column;
        const int quantized = std::clamp(
            static_cast<int>(std::nearbyint(static_cast<float>(expected[index]) / scale)), -8, 7);
        expected[index] = __nv_fp8_e4m3(static_cast<float>(quantized) * scale);
      }
    }
  }
  DeviceAllocation<__nv_fp8_e4m3> device(values.size());
  cuda_check(cudaMemcpy(device.get(), values.data(), values.size() * sizeof(values[0]),
                        cudaMemcpyHostToDevice),
             "copy INT4 weight oracle input");
  carat::roundtrip_fp8_weight_via_int4_blocks(device.get(), rows, width, block_width, nullptr);
  cuda_check(cudaMemcpy(values.data(), device.get(), values.size() * sizeof(values[0]),
                        cudaMemcpyDeviceToHost),
             "copy INT4 weight oracle output");
  if (std::memcmp(values.data(), expected.data(), values.size() * sizeof(values[0])) != 0) {
    throw std::runtime_error("INT4 weight oracle differs from host reference");
  }
}

void test_int4_fp8_linear() {
  constexpr int input_width = 256;
  constexpr int output_width = 256;
  constexpr int maximum_rows = 65;
  std::vector<__nv_fp8_e4m3> weights(static_cast<std::size_t>(output_width) * input_width,
                                     __nv_fp8_e4m3(0.0F));
  for (int component = 0; component < output_width; ++component) {
    weights[static_cast<std::size_t>(component) * input_width + component] = __nv_fp8_e4m3(7.0F);
  }
  std::vector<__nv_fp8_e4m3> input(static_cast<std::size_t>(maximum_rows) * input_width,
                                   __nv_fp8_e4m3(0.0F));
  for (int row = 0; row < maximum_rows; ++row) {
    input[static_cast<std::size_t>(row) * input_width + (row * 17) % input_width] =
        __nv_fp8_e4m3(1.0F);
  }

  DeviceAllocation<__nv_fp8_e4m3> device_weights(weights.size()), device_input(input.size());
  DeviceAllocation<__nv_bfloat16> device_output(static_cast<std::size_t>(maximum_rows) *
                                                output_width);
  cuda_check(cudaMemcpy(device_weights.get(), weights.data(), weights.size() * sizeof(weights[0]),
                        cudaMemcpyHostToDevice),
             "copy physical INT4 weights");
  cuda_check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(input[0]),
                        cudaMemcpyHostToDevice),
             "copy physical INT4 input");

  carat::Int4Fp8Linear linear(device_weights.get(), input_width, output_width, 128);
  for (const int rows : {1, 17, 33, 65}) {
    cuda_check(cudaMemset(device_output.get(), 0,
                          static_cast<std::size_t>(rows) * output_width * sizeof(__nv_bfloat16)),
               "clear physical INT4 output");
    linear.run(device_input.get(), device_output.get(), rows, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize physical INT4 linear");
    std::vector<__nv_bfloat16> output(static_cast<std::size_t>(rows) * output_width);
    cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(output[0]),
                          cudaMemcpyDeviceToHost),
               "copy physical INT4 output");
    for (int row = 0; row < rows; ++row) {
      int maximum_index = 0;
      float maximum = -INFINITY;
      for (int column = 0; column < output_width; ++column) {
        const float value =
            __bfloat162float(output[static_cast<std::size_t>(row) * output_width + column]);
        if (value > maximum) {
          maximum = value;
          maximum_index = column;
        }
      }
      const int expected = (row * 17) % input_width;
      if (maximum_index != expected) {
        std::string nonzero_offsets;
        for (std::size_t index = 0; index < output.size() && nonzero_offsets.size() < 200;
             ++index) {
          const float value = __bfloat162float(output[index]);
          if (value != 0.0F) {
            nonzero_offsets += std::to_string(index) + ":" + std::to_string(value) + ",";
          }
        }
        throw std::runtime_error(
            "physical INT4 linear orientation mismatch at row " + std::to_string(row) +
            ": expected " + std::to_string(expected) + ", got " + std::to_string(maximum_index) +
            ", maximum " + std::to_string(maximum) + ", expected-value " +
            std::to_string(
                __bfloat162float(output[static_cast<std::size_t>(row) * output_width + expected])) +
            ", nonzero-offsets " + nonzero_offsets);
      }
    }
  }

  constexpr int strided_rows = 16;
  constexpr int output_stride = 768;
  constexpr int output_offset = 256;
  constexpr std::uint16_t guard_pattern = 0xc1c1;
  DeviceAllocation<__nv_bfloat16> device_strided_output(static_cast<std::size_t>(strided_rows) *
                                                        output_stride);
  cuda_check(
      cudaMemset(device_strided_output.get(), 0xc1,
                 static_cast<std::size_t>(strided_rows) * output_stride * sizeof(__nv_bfloat16)),
      "initialize strided INT4 output guards");
  linear.run(device_input.get(), device_strided_output.get() + output_offset, strided_rows, nullptr,
             output_stride);
  cuda_check(cudaDeviceSynchronize(), "synchronize strided physical INT4 linear");
  std::vector<__nv_bfloat16> strided_output(static_cast<std::size_t>(strided_rows) * output_stride);
  cuda_check(cudaMemcpy(strided_output.data(), device_strided_output.get(),
                        strided_output.size() * sizeof(strided_output[0]), cudaMemcpyDeviceToHost),
             "copy strided physical INT4 output");
  for (int row = 0; row < strided_rows; ++row) {
    for (int column = 0; column < output_stride; ++column) {
      const std::size_t index = static_cast<std::size_t>(row) * output_stride + column;
      const bool is_output = column >= output_offset && column < output_offset + output_width;
      if (!is_output) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, &strided_output[index], sizeof(bits));
        if (bits != guard_pattern) {
          throw std::runtime_error("strided physical INT4 linear overwrote output guard");
        }
        continue;
      }
      const int output_column = column - output_offset;
      const int expected = (row * 17) % input_width;
      const float value = __bfloat162float(strided_output[index]);
      if ((output_column == expected && value <= 0.0F) ||
          (output_column != expected && value != 0.0F)) {
        throw std::runtime_error("strided physical INT4 linear output mismatch");
      }
    }
  }

  constexpr int comparison_rows = 17;
  constexpr int comparison_input_width = 5376;
  constexpr int comparison_output_width = 8192;
  constexpr int comparison_segment_width = 4096;
  std::vector<__nv_fp8_e4m3> comparison_weights(static_cast<std::size_t>(comparison_output_width) *
                                                comparison_input_width);
  std::vector<__nv_fp8_e4m3> comparison_input(static_cast<std::size_t>(comparison_rows) *
                                              comparison_input_width);
  for (std::size_t index = 0; index < comparison_weights.size(); ++index) {
    const int column = static_cast<int>(index % comparison_input_width);
    const int group = column / 256;
    const float amplitude =
        group + 1 == comparison_input_width / 256 ? 420.0F : 4.0F * static_cast<float>(group + 1);
    comparison_weights[index] =
        __nv_fp8_e4m3(amplitude * std::sin(static_cast<float>(index) * 0.071F) +
                      0.07F * amplitude * std::cos(static_cast<float>(index) * 0.013F));
  }
  int argmax_mismatches = 0;
  double minimum_cosine = 1.0;
  for (int row = 0; row < comparison_rows; ++row) {
    for (int column = 0; column < comparison_input_width; ++column) {
      comparison_input[static_cast<std::size_t>(row) * comparison_input_width + column] =
          __nv_fp8_e4m3(
              0.75F * std::sin(static_cast<float>(row * comparison_input_width + column) * 0.037F) -
              0.21F * std::cos(static_cast<float>(column) * 0.019F));
    }
  }
  DeviceAllocation<__nv_fp8_e4m3> device_random_weights(comparison_weights.size());
  DeviceAllocation<__nv_fp8_e4m3> device_oracle_weights(comparison_weights.size());
  DeviceAllocation<__nv_fp8_e4m3> device_refined_weights(comparison_weights.size());
  DeviceAllocation<__nv_fp8_e4m3> device_refined_oracle_weights(comparison_weights.size());
  DeviceAllocation<__nv_fp8_e4m3> device_comparison_input(comparison_input.size());
  DeviceAllocation<__nv_bfloat16> device_physical_output(static_cast<std::size_t>(comparison_rows) *
                                                         comparison_output_width);
  DeviceAllocation<__nv_bfloat16> device_oracle_output(static_cast<std::size_t>(comparison_rows) *
                                                       comparison_output_width);
  DeviceAllocation<__nv_bfloat16> device_refined_physical_output(
      static_cast<std::size_t>(comparison_rows) * comparison_output_width);
  DeviceAllocation<__nv_bfloat16> device_refined_oracle_output(
      static_cast<std::size_t>(comparison_rows) * comparison_output_width);
  DeviceAllocation<__nv_bfloat16> device_segment_output(static_cast<std::size_t>(comparison_rows) *
                                                        comparison_segment_width);
  constexpr int comparison_output_stride = 16384;
  constexpr int comparison_output_offset = 4096;
  DeviceAllocation<__nv_bfloat16> device_strided_comparison_output(
      static_cast<std::size_t>(comparison_rows) * comparison_output_stride);
  DeviceAllocation<float> unit_scale(1);
  const float one = 1.0F;
  cuda_check(cudaMemcpy(device_random_weights.get(), comparison_weights.data(),
                        comparison_weights.size() * sizeof(comparison_weights[0]),
                        cudaMemcpyHostToDevice),
             "copy random physical INT4 weights");
  cuda_check(cudaMemcpy(device_oracle_weights.get(), comparison_weights.data(),
                        comparison_weights.size() * sizeof(comparison_weights[0]),
                        cudaMemcpyHostToDevice),
             "copy random oracle INT4 weights");
  cuda_check(cudaMemcpy(device_refined_weights.get(), comparison_weights.data(),
                        comparison_weights.size() * sizeof(comparison_weights[0]),
                        cudaMemcpyHostToDevice),
             "copy refined physical INT4 weights");
  cuda_check(cudaMemcpy(device_refined_oracle_weights.get(), comparison_weights.data(),
                        comparison_weights.size() * sizeof(comparison_weights[0]),
                        cudaMemcpyHostToDevice),
             "copy refined oracle INT4 weights");
  cuda_check(cudaMemcpy(device_comparison_input.get(), comparison_input.data(),
                        comparison_input.size() * sizeof(comparison_input[0]),
                        cudaMemcpyHostToDevice),
             "copy random physical INT4 input");
  cuda_check(cudaMemcpy(unit_scale.get(), &one, sizeof(one), cudaMemcpyHostToDevice),
             "copy physical INT4 unit scale");
  carat::roundtrip_fp8_weight_via_int4_blocks(device_oracle_weights.get(), comparison_output_width,
                                              comparison_input_width, 256, nullptr);
  constexpr int refined_group_width = 128;
  constexpr int refinement_iterations = 4;
  carat::roundtrip_fp8_weight_via_int4_blocks(device_refined_oracle_weights.get(),
                                              comparison_output_width, comparison_input_width,
                                              refined_group_width, nullptr, refinement_iterations);
  carat::Int4Fp8Linear random_linear(device_random_weights.get(), comparison_input_width,
                                     comparison_output_width, 256);
  carat::Int4Fp8Linear random_strided_linear(device_random_weights.get(), comparison_input_width,
                                             comparison_output_width, 256);
  carat::Int4Fp8Linear random_segment_linear(device_random_weights.get(), comparison_input_width,
                                             comparison_segment_width, 256);
  carat::Int4Fp8Linear refined_linear(device_refined_weights.get(), comparison_input_width,
                                      comparison_output_width, refined_group_width, 256,
                                      refinement_iterations);
  carat::Fp8Linear oracle_linear;
  cuda_check(cudaMemset(device_strided_comparison_output.get(), 0xc1,
                        static_cast<std::size_t>(comparison_rows) * comparison_output_stride *
                            sizeof(__nv_bfloat16)),
             "initialize large-K strided INT4 output guards");
  random_linear.run(device_comparison_input.get(), device_physical_output.get(), comparison_rows,
                    nullptr);
  random_strided_linear.run(device_comparison_input.get(),
                            device_strided_comparison_output.get() + comparison_output_offset,
                            comparison_rows, nullptr, comparison_output_stride);
  random_segment_linear.run(device_comparison_input.get(), device_segment_output.get(),
                            comparison_rows, nullptr);
  oracle_linear.run(device_comparison_input.get(), unit_scale.get(), device_oracle_weights.get(),
                    unit_scale.get(), device_oracle_output.get(), comparison_rows,
                    comparison_input_width, comparison_output_width, nullptr);
  refined_linear.run(device_comparison_input.get(), device_refined_physical_output.get(),
                     comparison_rows, nullptr);
  oracle_linear.run(device_comparison_input.get(), unit_scale.get(),
                    device_refined_oracle_weights.get(), unit_scale.get(),
                    device_refined_oracle_output.get(), comparison_rows, comparison_input_width,
                    comparison_output_width, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize physical INT4 oracle comparison");
  std::vector<__nv_bfloat16> physical(static_cast<std::size_t>(comparison_rows) *
                                      comparison_output_width);
  std::vector<__nv_bfloat16> oracle(physical.size());
  std::vector<__nv_bfloat16> refined_physical(physical.size());
  std::vector<__nv_bfloat16> refined_oracle(physical.size());
  std::vector<__nv_bfloat16> strided_comparison(static_cast<std::size_t>(comparison_rows) *
                                                comparison_output_stride);
  std::vector<__nv_bfloat16> segment_output(static_cast<std::size_t>(comparison_rows) *
                                            comparison_segment_width);
  cuda_check(cudaMemcpy(physical.data(), device_physical_output.get(),
                        physical.size() * sizeof(physical[0]), cudaMemcpyDeviceToHost),
             "copy random physical INT4 output");
  cuda_check(cudaMemcpy(oracle.data(), device_oracle_output.get(),
                        oracle.size() * sizeof(oracle[0]), cudaMemcpyDeviceToHost),
             "copy random oracle INT4 output");
  cuda_check(cudaMemcpy(refined_physical.data(), device_refined_physical_output.get(),
                        refined_physical.size() * sizeof(refined_physical[0]),
                        cudaMemcpyDeviceToHost),
             "copy refined physical INT4 output");
  cuda_check(cudaMemcpy(refined_oracle.data(), device_refined_oracle_output.get(),
                        refined_oracle.size() * sizeof(refined_oracle[0]), cudaMemcpyDeviceToHost),
             "copy refined oracle INT4 output");
  cuda_check(cudaMemcpy(strided_comparison.data(), device_strided_comparison_output.get(),
                        strided_comparison.size() * sizeof(strided_comparison[0]),
                        cudaMemcpyDeviceToHost),
             "copy large-K strided physical INT4 output");
  cuda_check(cudaMemcpy(segment_output.data(), device_segment_output.get(),
                        segment_output.size() * sizeof(segment_output[0]), cudaMemcpyDeviceToHost),
             "copy segmented physical INT4 output");
  for (int row = 0; row < comparison_rows; ++row) {
    int physical_argmax = 0;
    int oracle_argmax = 0;
    for (int column = 1; column < comparison_output_width; ++column) {
      const std::size_t index = static_cast<std::size_t>(row) * comparison_output_width + column;
      const std::size_t physical_best =
          static_cast<std::size_t>(row) * comparison_output_width + physical_argmax;
      const std::size_t oracle_best =
          static_cast<std::size_t>(row) * comparison_output_width + oracle_argmax;
      if (__bfloat162float(physical[index]) > __bfloat162float(physical[physical_best])) {
        physical_argmax = column;
      }
      if (__bfloat162float(oracle[index]) > __bfloat162float(oracle[oracle_best])) {
        oracle_argmax = column;
      }
    }
    argmax_mismatches += physical_argmax != oracle_argmax ? 1 : 0;
    double dot = 0.0;
    double physical_squared = 0.0;
    double oracle_squared = 0.0;
    for (int column = 0; column < comparison_output_width; ++column) {
      const std::size_t index = static_cast<std::size_t>(row) * comparison_output_width + column;
      const std::size_t strided_index = static_cast<std::size_t>(row) * comparison_output_stride +
                                        comparison_output_offset + column;
      if (std::memcmp(&physical[index], &strided_comparison[strided_index],
                      sizeof(physical[index])) != 0) {
        throw std::runtime_error("large-K strided physical INT4 output mismatch");
      }
      if (column < comparison_segment_width) {
        const std::size_t segment_index =
            static_cast<std::size_t>(row) * comparison_segment_width + column;
        if (std::memcmp(&physical[index], &segment_output[segment_index],
                        sizeof(physical[index])) != 0) {
          throw std::runtime_error("physical INT4 output depends on vocabulary segment width");
        }
      }
      const double physical_value = __bfloat162float(physical[index]);
      const double oracle_value = __bfloat162float(oracle[index]);
      dot += physical_value * oracle_value;
      physical_squared += physical_value * physical_value;
      oracle_squared += oracle_value * oracle_value;
    }
    minimum_cosine = std::min(minimum_cosine, dot / std::sqrt(physical_squared * oracle_squared));
    for (int column = 0; column < comparison_output_stride; ++column) {
      if (column >= comparison_output_offset &&
          column < comparison_output_offset + comparison_output_width) {
        continue;
      }
      const std::size_t guard_index =
          static_cast<std::size_t>(row) * comparison_output_stride + column;
      std::uint16_t bits = 0;
      std::memcpy(&bits, &strided_comparison[guard_index], sizeof(bits));
      if (bits != guard_pattern) {
        throw std::runtime_error("large-K strided physical INT4 overwrote output guard");
      }
    }
  }
  std::cout << "INT4 physical/oracle K=" << comparison_input_width
            << " N=" << comparison_output_width << " minimum_cosine=" << minimum_cosine
            << " argmax_mismatches=" << argmax_mismatches << "/" << comparison_rows << '\n';
  if (minimum_cosine < 0.995 || argmax_mismatches > 6) {
    throw std::runtime_error(
        "physical INT4 linear exceeded representation-oracle tolerance: minimum cosine " +
        std::to_string(minimum_cosine) + ", argmax mismatches " +
        std::to_string(argmax_mismatches) + "/" + std::to_string(comparison_rows));
  }

  int refined_argmax_mismatches = 0;
  double refined_minimum_cosine = 1.0;
  for (int row = 0; row < comparison_rows; ++row) {
    int physical_argmax = 0;
    int oracle_argmax = 0;
    double dot = 0.0;
    double physical_squared = 0.0;
    double oracle_squared = 0.0;
    for (int column = 0; column < comparison_output_width; ++column) {
      const std::size_t index = static_cast<std::size_t>(row) * comparison_output_width + column;
      const double physical_value = __bfloat162float(refined_physical[index]);
      const double oracle_value = __bfloat162float(refined_oracle[index]);
      if (physical_value >
          __bfloat162float(
              refined_physical[static_cast<std::size_t>(row) * comparison_output_width +
                               physical_argmax])) {
        physical_argmax = column;
      }
      if (oracle_value >
          __bfloat162float(refined_oracle[static_cast<std::size_t>(row) * comparison_output_width +
                                          oracle_argmax])) {
        oracle_argmax = column;
      }
      dot += physical_value * oracle_value;
      physical_squared += physical_value * physical_value;
      oracle_squared += oracle_value * oracle_value;
    }
    refined_argmax_mismatches += physical_argmax != oracle_argmax ? 1 : 0;
    refined_minimum_cosine =
        std::min(refined_minimum_cosine, dot / std::sqrt(physical_squared * oracle_squared));
  }
  std::cout << "INT4 refined physical/oracle K=" << comparison_input_width
            << " N=" << comparison_output_width << " group=" << refined_group_width
            << " iterations=" << refinement_iterations
            << " minimum_cosine=" << refined_minimum_cosine
            << " argmax_mismatches=" << refined_argmax_mismatches << "/" << comparison_rows << '\n';
  if (refined_minimum_cosine < 0.999999 || refined_argmax_mismatches != 0) {
    throw std::runtime_error(
        "refined physical INT4 linear differs from representation oracle: minimum cosine " +
        std::to_string(refined_minimum_cosine) + ", argmax mismatches " +
        std::to_string(refined_argmax_mismatches) + "/" + std::to_string(comparison_rows));
  }
}

void test_int4_bf16_linear() {
  constexpr int rows = 17;
  constexpr int input_width = 512;
  constexpr int output_width = 256;
  std::vector<__nv_bfloat16> weights(static_cast<std::size_t>(output_width) * input_width);
  std::vector<__nv_bfloat16> oracle_weights(weights.size());
  std::vector<__nv_bfloat16> input(static_cast<std::size_t>(rows) * input_width);
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const float amplitude = index % input_width < 256 ? 1.0F : 16.0F;
    weights[index] =
        __float2bfloat16(amplitude * (2.75F * std::sin(static_cast<float>(index) * 0.071F) +
                                      0.31F * std::cos(static_cast<float>(index) * 0.013F)));
  }
  oracle_weights = weights;
  for (int row = 0; row < output_width; ++row) {
    for (int begin = 0; begin < input_width; begin += 256) {
      float maximum = 0.0F;
      for (int column = begin; column < begin + 256; ++column) {
        maximum = std::max(
            maximum, std::abs(__bfloat162float(
                         oracle_weights[static_cast<std::size_t>(row) * input_width + column])));
      }
      const __nv_bfloat16 encoded_scale = __float2bfloat16(maximum > 0.0F ? maximum / 7.0F : 1.0F);
      const float scale = __bfloat162float(encoded_scale);
      for (int column = begin; column < begin + 256; ++column) {
        const std::size_t index = static_cast<std::size_t>(row) * input_width + column;
        const int quantized = std::clamp(
            static_cast<int>(std::nearbyint(__bfloat162float(oracle_weights[index]) / scale)), -7,
            7);
        oracle_weights[index] = __float2bfloat16(static_cast<float>(quantized) * scale);
      }
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < input_width; ++column) {
      input[static_cast<std::size_t>(row) * input_width + column] = __float2bfloat16(
          0.75F * std::sin(static_cast<float>(row * input_width + column) * 0.037F) -
          0.21F * std::cos(static_cast<float>(column) * 0.019F));
    }
  }
  DeviceAllocation<__nv_bfloat16> device_weights(weights.size());
  DeviceAllocation<__nv_bfloat16> device_oracle_weights(oracle_weights.size());
  DeviceAllocation<__nv_bfloat16> device_input(input.size());
  DeviceAllocation<__nv_bfloat16> device_physical_output(static_cast<std::size_t>(rows) *
                                                         output_width);
  DeviceAllocation<__nv_bfloat16> device_oracle_output(static_cast<std::size_t>(rows) *
                                                       output_width);
  cuda_check(cudaMemcpy(device_weights.get(), weights.data(), weights.size() * sizeof(weights[0]),
                        cudaMemcpyHostToDevice),
             "copy BF16 INT4 weights");
  cuda_check(cudaMemcpy(device_oracle_weights.get(), oracle_weights.data(),
                        oracle_weights.size() * sizeof(oracle_weights[0]), cudaMemcpyHostToDevice),
             "copy BF16 INT4 oracle weights");
  cuda_check(cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(input[0]),
                        cudaMemcpyHostToDevice),
             "copy BF16 INT4 input");
  carat::Int4Bf16Linear physical_linear(device_weights.get(), input_width, output_width, 256);
  carat::Bf16Linear oracle_linear;
  physical_linear.run(device_input.get(), device_physical_output.get(), rows, nullptr);
  oracle_linear.run(device_input.get(), device_oracle_weights.get(), device_oracle_output.get(),
                    rows, input_width, output_width, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize BF16 INT4 comparison");
  std::vector<__nv_bfloat16> physical(static_cast<std::size_t>(rows) * output_width);
  std::vector<__nv_bfloat16> oracle(physical.size());
  cuda_check(cudaMemcpy(physical.data(), device_physical_output.get(),
                        physical.size() * sizeof(physical[0]), cudaMemcpyDeviceToHost),
             "copy BF16 INT4 physical output");
  cuda_check(cudaMemcpy(oracle.data(), device_oracle_output.get(),
                        oracle.size() * sizeof(oracle[0]), cudaMemcpyDeviceToHost),
             "copy BF16 INT4 oracle output");
  double minimum_cosine = 1.0;
  int argmax_mismatches = 0;
  for (int row = 0; row < rows; ++row) {
    double dot = 0.0;
    double physical_squared = 0.0;
    double oracle_squared = 0.0;
    int physical_argmax = 0;
    int oracle_argmax = 0;
    for (int column = 0; column < output_width; ++column) {
      const std::size_t index = static_cast<std::size_t>(row) * output_width + column;
      const double physical_value = __bfloat162float(physical[index]);
      const double oracle_value = __bfloat162float(oracle[index]);
      if (physical_value >
          __bfloat162float(
              physical[static_cast<std::size_t>(row) * output_width + physical_argmax])) {
        physical_argmax = column;
      }
      if (oracle_value >
          __bfloat162float(oracle[static_cast<std::size_t>(row) * output_width + oracle_argmax])) {
        oracle_argmax = column;
      }
      dot += physical_value * oracle_value;
      physical_squared += physical_value * physical_value;
      oracle_squared += oracle_value * oracle_value;
    }
    minimum_cosine = std::min(minimum_cosine, dot / std::sqrt(physical_squared * oracle_squared));
    argmax_mismatches += physical_argmax != oracle_argmax ? 1 : 0;
  }
  std::cout << "INT4 BF16 physical/oracle minimum_cosine=" << minimum_cosine
            << " argmax_mismatches=" << argmax_mismatches << '/' << rows << '\n';
  if (minimum_cosine < 0.999 || argmax_mismatches > 1) {
    throw std::runtime_error("BF16 INT4 linear exceeded representation-oracle tolerance");
  }
}

void test_grouped_attention() {
  constexpr int context = 3;
  constexpr int dimension = 256;
  constexpr int group = 2;
  std::array<float, group * dimension> query_values{};
  std::array<float, context * dimension> key_values{};
  std::array<float, context * dimension> value_values{};
  for (int token = 0; token < context; ++token) {
    for (int index = 0; index < dimension; ++index) {
      value_values[token * dimension + index] =
          static_cast<float>(token + 1) + static_cast<float>(index % 7) / 16.0F;
    }
  }
  const auto query = bf16<group * dimension>(query_values);
  const auto key = bf16<context * dimension>(key_values);
  const auto value = bf16<context * dimension>(value_values);
  DeviceAllocation<__nv_bfloat16> device_query(query.size()), device_key(key.size()),
      device_value(value.size()), device_output(query.size());
  cuda_check(cudaMemcpy(device_query.get(), query.data(), sizeof(query), cudaMemcpyHostToDevice),
             "copy attention query");
  cuda_check(cudaMemcpy(device_key.get(), key.data(), sizeof(key), cudaMemcpyHostToDevice),
             "copy attention key");
  cuda_check(cudaMemcpy(device_value.get(), value.data(), sizeof(value), cudaMemcpyHostToDevice),
             "copy attention value");
  carat::grouped_decode_attention_bf16(device_query.get(), device_key.get(), device_value.get(),
                                       device_output.get(), 1, 1, group, context, dimension,
                                       nullptr);
  cuda_check(cudaDeviceSynchronize(), "attention synchronization");
  std::array<__nv_bfloat16, group * dimension> output{};
  cuda_check(cudaMemcpy(output.data(), device_output.get(), sizeof(output), cudaMemcpyDeviceToHost),
             "copy attention output");
  for (int query_index = 0; query_index < group; ++query_index) {
    for (int index = 0; index < dimension; ++index) {
      const float expected = 2.0F + static_cast<float>(index % 7) / 16.0F;
      near(__bfloat162float(output[query_index * dimension + index]), expected, 0.02F,
           "grouped attention");
    }
  }
  carat::GemmGroupedDecodeAttention gemm_attention(1, 1, group, context);
  gemm_attention.run(device_query.get(), device_key.get(), device_value.get(), device_output.get(),
                     1, 1, group, context, dimension, nullptr);
  cuda_check(cudaDeviceSynchronize(), "GEMM attention synchronization");
  cuda_check(cudaMemcpy(output.data(), device_output.get(), sizeof(output), cudaMemcpyDeviceToHost),
             "copy GEMM attention output");
  for (int query_index = 0; query_index < group; ++query_index) {
    for (int index = 0; index < dimension; ++index) {
      const float expected = 2.0F + static_cast<float>(index % 7) / 16.0F;
      near(__bfloat162float(output[query_index * dimension + index]), expected, 0.02F,
           "GEMM grouped attention");
    }
  }
}

void test_shared_global_kv_roundtrip() {
  constexpr int tokens = 7;
  constexpr int query_heads = 32;
  constexpr int kv_heads = 4;
  constexpr int dimension = 512;
  constexpr int query_width = query_heads * dimension;
  constexpr int packed_width = query_width + kv_heads * dimension;
  constexpr int capacity = 64;
  constexpr int maximum_slots = 4;
  constexpr std::size_t packed_elements = static_cast<std::size_t>(tokens) * packed_width;
  constexpr std::size_t positioned_cache_elements =
      static_cast<std::size_t>(kv_heads) * capacity * dimension;
  constexpr std::size_t ragged_cache_elements =
      static_cast<std::size_t>(maximum_slots) * kv_heads * capacity * dimension;
  constexpr std::size_t query_elements = static_cast<std::size_t>(tokens) * query_heads * dimension;
  std::vector<__nv_bfloat16> packed(packed_elements);
  std::vector<__nv_bfloat16> norm_weight(dimension);
  for (std::size_t index = 0; index < packed.size(); ++index) {
    packed[index] = __float2bfloat16_rn(0.75F * std::sin(static_cast<float>(index) * 0.00137F) +
                                        0.2F * std::cos(static_cast<float>(index) * 0.00311F));
  }
  for (int index = 0; index < dimension; ++index) {
    norm_weight[static_cast<std::size_t>(index)] =
        __float2bfloat16_rn(0.055F + 0.01F * std::sin(static_cast<float>(index) * 0.017F));
  }
  DeviceAllocation<__nv_bfloat16> device_packed(packed_elements), device_norm(dimension),
      baseline_queries(query_elements), candidate_queries(query_elements),
      baseline_keys(ragged_cache_elements), baseline_values(ragged_cache_elements),
      candidate_keys(ragged_cache_elements), candidate_values(ragged_cache_elements);
  DeviceAllocation<int> device_positions(tokens), device_slots(tokens);
  cuda_check(cudaMemcpy(device_packed.get(), packed.data(), packed.size() * sizeof(packed[0]),
                        cudaMemcpyHostToDevice),
             "copy shared global packed input");
  cuda_check(cudaMemcpy(device_norm.get(), norm_weight.data(),
                        norm_weight.size() * sizeof(norm_weight[0]), cudaMemcpyHostToDevice),
             "copy shared global norm weight");
  unsetenv("CARAT_SHARED_GLOBAL_KV_ROUNDTRIP");
  carat::Gemma4QkvPostprocessor baseline(capacity);
  setenv("CARAT_SHARED_GLOBAL_KV_ROUNDTRIP", "1", 1);
  carat::Gemma4QkvPostprocessor candidate(capacity);
  unsetenv("CARAT_SHARED_GLOBAL_KV_ROUNDTRIP");

  auto zero_outputs = [&] {
    cuda_check(cudaMemset(baseline_keys.get(), 0, ragged_cache_elements * sizeof(__nv_bfloat16)),
               "zero baseline shared global keys");
    cuda_check(cudaMemset(baseline_values.get(), 0, ragged_cache_elements * sizeof(__nv_bfloat16)),
               "zero baseline shared global values");
    cuda_check(cudaMemset(candidate_keys.get(), 0, ragged_cache_elements * sizeof(__nv_bfloat16)),
               "zero candidate shared global keys");
    cuda_check(cudaMemset(candidate_values.get(), 0, ragged_cache_elements * sizeof(__nv_bfloat16)),
               "zero candidate shared global values");
  };
  auto require_identical = [&](std::size_t cache_elements, std::size_t compared_queries,
                               const char *operation) {
    cuda_check(cudaDeviceSynchronize(), "synchronize shared global KV roundtrip");
    std::vector<__nv_bfloat16> expected_keys(cache_elements), actual_keys(cache_elements),
        expected_values(cache_elements), actual_values(cache_elements),
        expected_queries(compared_queries), actual_queries(compared_queries);
    cuda_check(cudaMemcpy(expected_keys.data(), baseline_keys.get(),
                          cache_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "copy baseline shared global keys");
    cuda_check(cudaMemcpy(actual_keys.data(), candidate_keys.get(),
                          cache_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "copy candidate shared global keys");
    cuda_check(cudaMemcpy(expected_values.data(), baseline_values.get(),
                          cache_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "copy baseline shared global values");
    cuda_check(cudaMemcpy(actual_values.data(), candidate_values.get(),
                          cache_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "copy candidate shared global values");
    cuda_check(cudaMemcpy(expected_queries.data(), baseline_queries.get(),
                          compared_queries * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "copy baseline shared global queries");
    cuda_check(cudaMemcpy(actual_queries.data(), candidate_queries.get(),
                          compared_queries * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
               "copy candidate shared global queries");
    if (std::memcmp(expected_keys.data(), actual_keys.data(),
                    cache_elements * sizeof(__nv_bfloat16)) != 0 ||
        std::memcmp(expected_values.data(), actual_values.data(),
                    cache_elements * sizeof(__nv_bfloat16)) != 0 ||
        std::memcmp(expected_queries.data(), actual_queries.data(),
                    compared_queries * sizeof(__nv_bfloat16)) != 0) {
      throw std::runtime_error(std::string(operation) + " changed BF16 output bits");
    }
  };

  zero_outputs();
  baseline.run_positioned(device_packed.get(), device_norm.get(), device_norm.get(),
                          baseline_queries.get(), baseline_keys.get(), baseline_values.get(),
                          tokens, query_heads, kv_heads, dimension, true, 9, 9, capacity, 1e-6F,
                          nullptr);
  candidate.run_positioned(device_packed.get(), device_norm.get(), device_norm.get(),
                           candidate_queries.get(), candidate_keys.get(), candidate_values.get(),
                           tokens, query_heads, kv_heads, dimension, true, 9, 9, capacity, 1e-6F,
                           nullptr);
  require_identical(positioned_cache_elements, query_elements,
                    "positioned shared global KV roundtrip");

  const std::array<int, maximum_slots> positions{3, 17, 31, 47};
  const std::array<int, maximum_slots> slots{3, 0, 2, 1};
  cuda_check(cudaMemcpy(device_positions.get(), positions.data(), sizeof(positions),
                        cudaMemcpyHostToDevice),
             "copy shared global positions");
  cuda_check(cudaMemcpy(device_slots.get(), slots.data(), sizeof(slots), cudaMemcpyHostToDevice),
             "copy shared global slots");
  zero_outputs();
  baseline.run_decode_ragged(device_packed.get(), device_norm.get(), device_norm.get(),
                             baseline_queries.get(), baseline_keys.get(), baseline_values.get(),
                             device_positions.get(), device_slots.get(), maximum_slots,
                             maximum_slots, query_heads, kv_heads, dimension, true, capacity,
                             nullptr, nullptr, 1e-6F, nullptr);
  candidate.run_decode_ragged(device_packed.get(), device_norm.get(), device_norm.get(),
                              candidate_queries.get(), candidate_keys.get(), candidate_values.get(),
                              device_positions.get(), device_slots.get(), maximum_slots,
                              maximum_slots, query_heads, kv_heads, dimension, true, capacity,
                              nullptr, nullptr, 1e-6F, nullptr);
  require_identical(ragged_cache_elements,
                    static_cast<std::size_t>(maximum_slots) * query_heads * dimension,
                    "ragged shared global KV roundtrip");
}

void test_prefill_layouts() {
  constexpr int tokens = 5;
  constexpr int heads = 2;
  constexpr int dimension = 2;
  std::array<float, tokens * heads * dimension> values{};
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] = static_cast<float>(index);
  const auto input = bf16<values.size()>(values);
  DeviceAllocation<__nv_bfloat16> device_input(input.size()), device_head_major(input.size()),
      device_round_trip(input.size());
  cuda_check(cudaMemcpy(device_input.get(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
             "copy transpose input");
  carat::token_to_head_bf16(device_input.get(), device_head_major.get(), tokens, heads, dimension,
                            nullptr);
  carat::head_to_token_bf16(device_head_major.get(), device_round_trip.get(), tokens, heads,
                            dimension, nullptr);
  std::array<__nv_bfloat16, input.size()> round_trip{};
  cuda_check(cudaMemcpy(round_trip.data(), device_round_trip.get(), sizeof(round_trip),
                        cudaMemcpyDeviceToHost),
             "copy transpose round trip");
  for (std::size_t index = 0; index < input.size(); ++index) {
    near(__bfloat162float(round_trip[index]), values[index], 0.01F, "prefill transpose");
  }

  constexpr int capacity = 3;
  DeviceAllocation<__nv_bfloat16> device_cache(heads * capacity * dimension);
  cuda_check(
      cudaMemset(device_cache.get(), 0, heads * capacity * dimension * sizeof(__nv_bfloat16)),
      "clear ring cache");
  carat::copy_kv_to_cache_bf16(device_head_major.get(), device_cache.get(), heads, tokens,
                               dimension, capacity, 0, nullptr);
  std::array<__nv_bfloat16, heads * capacity * dimension> cache{};
  cuda_check(cudaMemcpy(cache.data(), device_cache.get(), sizeof(cache), cudaMemcpyDeviceToHost),
             "copy ring cache");
  for (int head = 0; head < heads; ++head) {
    for (int source_token = tokens - capacity; source_token < tokens; ++source_token) {
      const int slot = source_token % capacity;
      for (int component = 0; component < dimension; ++component) {
        const int source_index = (source_token * heads + head) * dimension + component;
        const int cache_index = (head * capacity + slot) * dimension + component;
        near(__bfloat162float(cache[cache_index]), values[source_index], 0.01F,
             "prefill KV ring copy");
      }
    }
  }
}

void test_batched_argmax() {
  constexpr int rows = 3;
  constexpr int elements = 513;
  std::vector<__nv_bfloat16> values(static_cast<std::size_t>(rows * elements),
                                    __float2bfloat16(-2.0F));
  const std::array<int, rows> expected{17, 0, 512};
  values[17] = __float2bfloat16(9.0F);
  values[elements + 0] = __float2bfloat16(4.0F);
  values[elements + 401] = __float2bfloat16(4.0F); // Equal values select the lower index.
  values[2 * elements + 512] = __float2bfloat16(7.0F);
  DeviceAllocation<__nv_bfloat16> device_values(values.size());
  DeviceAllocation<int> device_output(rows);
  const std::size_t workspace_bytes = rows * carat::argmax_bf16_workspace_bytes(elements);
  DeviceAllocation<unsigned char> workspace(workspace_bytes);
  cuda_check(cudaMemcpy(device_values.get(), values.data(), values.size() * sizeof(values[0]),
                        cudaMemcpyHostToDevice),
             "copy batched argmax values");
  carat::argmax_bf16_rows(device_values.get(), rows, elements, device_output.get(), workspace.get(),
                          nullptr);
  std::array<int, rows> output{};
  cuda_check(cudaMemcpy(output.data(), device_output.get(), sizeof(output), cudaMemcpyDeviceToHost),
             "copy batched argmax output");
  for (int row = 0; row < rows; ++row) {
    if (output[static_cast<std::size_t>(row)] != expected[static_cast<std::size_t>(row)]) {
      throw std::runtime_error("batched argmax mismatch at row " + std::to_string(row));
    }
  }
}

void test_int8_kv_quality_oracle() {
  constexpr int slots = 2;
  constexpr int heads = 2;
  constexpr int capacity = 5;
  constexpr int dimension = 512;
  constexpr int elements = slots * heads * capacity * dimension;
  std::vector<__nv_bfloat16> keys(elements), values(elements);
  for (int index = 0; index < elements; ++index) {
    keys[static_cast<std::size_t>(index)] =
        __float2bfloat16(static_cast<float>((index * 37) % 257 - 128) / 311.0F);
    values[static_cast<std::size_t>(index)] =
        __float2bfloat16(static_cast<float>((index * 53) % 509 - 254) / 73.0F);
  }
  const auto original_keys = keys;
  const auto original_values = values;
  DeviceAllocation<__nv_bfloat16> device_keys(elements), device_values(elements);
  cuda_check(cudaMemcpy(device_keys.get(), keys.data(), keys.size() * sizeof(keys[0]),
                        cudaMemcpyHostToDevice),
             "copy INT8 oracle keys");
  cuda_check(cudaMemcpy(device_values.get(), values.data(), values.size() * sizeof(values[0]),
                        cudaMemcpyHostToDevice),
             "copy INT8 oracle values");

  carat::roundtrip_global_kv_int8_per_token_bf16(device_keys.get(), device_values.get(), heads, 2,
                                                 dimension, capacity, 1, nullptr);
  const std::array<int, slots> positions{4, 0};
  const std::array<int, slots> slot_order{1, 0};
  DeviceAllocation<int> device_positions(slots), device_slots(slots);
  cuda_check(cudaMemcpy(device_positions.get(), positions.data(), sizeof(positions),
                        cudaMemcpyHostToDevice),
             "copy INT8 oracle positions");
  cuda_check(
      cudaMemcpy(device_slots.get(), slot_order.data(), sizeof(slot_order), cudaMemcpyHostToDevice),
      "copy INT8 oracle slots");
  carat::roundtrip_global_decode_kv_int8_per_token_bf16(
      device_keys.get(), device_values.get(), device_positions.get(), device_slots.get(), slots,
      slots, heads, dimension, capacity, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize INT8 KV oracle");
  cuda_check(cudaMemcpy(keys.data(), device_keys.get(), keys.size() * sizeof(keys[0]),
                        cudaMemcpyDeviceToHost),
             "copy INT8 oracle result keys");
  cuda_check(cudaMemcpy(values.data(), device_values.get(), values.size() * sizeof(values[0]),
                        cudaMemcpyDeviceToHost),
             "copy INT8 oracle result values");

  const auto verify_cache = [&](const std::vector<__nv_bfloat16> &original,
                                const std::vector<__nv_bfloat16> &actual, const char *operation) {
    for (int slot = 0; slot < slots; ++slot) {
      for (int head = 0; head < heads; ++head) {
        for (int token = 0; token < capacity; ++token) {
          const bool selected =
              (slot == 0 && (token == 0 || token == 1 || token == 2)) || (slot == 1 && token == 4);
          const std::size_t row =
              (static_cast<std::size_t>(slot) * heads + head) * capacity + token;
          float maximum = 0.0F;
          if (selected) {
            for (int component = 0; component < dimension; ++component) {
              maximum = std::max(
                  maximum, std::abs(__bfloat162float(
                               original[row * dimension + static_cast<std::size_t>(component)])));
            }
          }
          for (int component = 0; component < dimension; ++component) {
            const std::size_t index = row * dimension + static_cast<std::size_t>(component);
            float expected = __bfloat162float(original[index]);
            if (selected && maximum != 0.0F) {
              const int quantized = std::clamp(
                  static_cast<int>(std::nearbyint(expected * (127.0F / maximum))), -127, 127);
              expected = __bfloat162float(
                  __float2bfloat16(static_cast<float>(quantized) * (maximum / 127.0F)));
            }
            if (__bfloat162float(actual[index]) != expected) {
              throw std::runtime_error(std::string(operation) + " mismatch at row " +
                                       std::to_string(row) + ", component " +
                                       std::to_string(component));
            }
          }
        }
      }
    }
  };
  verify_cache(original_keys, keys, "INT8 key oracle");
  verify_cache(original_values, values, "INT8 value oracle");

  // Exercise the block-scaled and keys-only mapping independently. Slot 1 is a guard region.
  cuda_check(cudaMemcpy(device_keys.get(), original_keys.data(),
                        original_keys.size() * sizeof(original_keys[0]), cudaMemcpyHostToDevice),
             "reset block INT8 oracle keys");
  cuda_check(cudaMemcpy(device_values.get(), original_values.data(),
                        original_values.size() * sizeof(original_values[0]),
                        cudaMemcpyHostToDevice),
             "reset block INT8 oracle values");
  carat::roundtrip_global_kv_int8_per_token_bf16(device_keys.get(), device_values.get(), heads,
                                                 capacity, dimension, capacity, 0, nullptr, true,
                                                 false, 128);
  cuda_check(cudaMemcpy(keys.data(), device_keys.get(), keys.size() * sizeof(keys[0]),
                        cudaMemcpyDeviceToHost),
             "copy block INT8 oracle keys");
  for (int slot = 0; slot < slots; ++slot) {
    for (int head = 0; head < heads; ++head) {
      for (int token = 0; token < capacity; ++token) {
        const std::size_t row = (static_cast<std::size_t>(slot) * heads + head) * capacity + token;
        for (int component_block = 0; component_block < dimension; component_block += 128) {
          float maximum = 0.0F;
          for (int component = component_block; component < component_block + 128; ++component) {
            maximum = std::max(
                maximum,
                std::abs(__bfloat162float(
                    original_keys[row * dimension + static_cast<std::size_t>(component)])));
          }
          for (int component = component_block; component < component_block + 128; ++component) {
            const std::size_t index = row * dimension + static_cast<std::size_t>(component);
            float expected = __bfloat162float(original_keys[index]);
            if (slot == 0 && maximum != 0.0F) {
              const int quantized = std::clamp(
                  static_cast<int>(std::nearbyint(expected * (127.0F / maximum))), -127, 127);
              expected = __bfloat162float(
                  __float2bfloat16(static_cast<float>(quantized) * (maximum / 127.0F)));
            }
            if (__bfloat162float(keys[index]) != expected) {
              throw std::runtime_error("block INT8 key oracle mismatch");
            }
          }
        }
      }
    }
  }

  constexpr int materialized_rows = 3;
  DeviceAllocation<std::int8_t> materialized(materialized_rows * dimension);
  DeviceAllocation<float> materialized_scales(materialized_rows);
  carat::quantize_bf16_rows_symmetric_int8(device_keys.get(), materialized.get(),
                                           materialized_scales.get(), materialized_rows, dimension,
                                           nullptr);
  std::array<float, materialized_rows> host_scales{};
  std::vector<std::int8_t> host_materialized(materialized_rows * dimension);
  cuda_check(cudaMemcpy(host_scales.data(), materialized_scales.get(), sizeof(host_scales),
                        cudaMemcpyDeviceToHost),
             "copy materialized INT8 scales");
  cuda_check(cudaMemcpy(host_materialized.data(), materialized.get(), host_materialized.size(),
                        cudaMemcpyDeviceToHost),
             "copy materialized INT8 values");
  for (int row = 0; row < materialized_rows; ++row) {
    float maximum = 0.0F;
    for (int column = 0; column < dimension; ++column) {
      maximum = std::max(maximum, std::abs(__bfloat162float(
                                      keys[static_cast<std::size_t>(row * dimension + column)])));
    }
    const float expected_scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
    if (host_scales[static_cast<std::size_t>(row)] != expected_scale) {
      throw std::runtime_error("materialized INT8 scale mismatch");
    }
    for (int column = 0; column < dimension; ++column) {
      const float value =
          __bfloat162float(keys[static_cast<std::size_t>(row * dimension + column)]);
      const int expected =
          std::clamp(static_cast<int>(std::nearbyint(value * (1.0F / expected_scale))), -127, 127);
      if (host_materialized[static_cast<std::size_t>(row * dimension + column)] != expected) {
        throw std::runtime_error("materialized INT8 value mismatch");
      }
    }
  }
}

void test_materialized_int8_global_k_cache_writes() {
  constexpr int maximum_slots = 3;
  constexpr int heads = 2;
  constexpr int capacity = 5;
  constexpr int dimension = 512;
  constexpr int blocks_per_row = 4;
  constexpr int rows = maximum_slots * heads * capacity;
  std::vector<__nv_bfloat16> source(static_cast<std::size_t>(rows) * dimension);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = __float2bfloat16_rn(0.75F * std::sin(static_cast<float>(index) * 0.0137F) +
                                        0.125F * std::cos(static_cast<float>(index) * 0.0031F));
  }
  DeviceAllocation<__nv_bfloat16> device_source(source.size());
  DeviceAllocation<std::int8_t> device_destination(source.size());
  DeviceAllocation<float> device_scales(static_cast<std::size_t>(rows) * blocks_per_row);
  cuda_check(cudaMemcpy(device_source.get(), source.data(), source.size() * sizeof(source[0]),
                        cudaMemcpyHostToDevice),
             "copy materialized INT8 source");
  cuda_check(cudaMemset(device_destination.get(), 0, source.size()),
             "zero materialized INT8 destination");
  cuda_check(cudaMemset(device_scales.get(), 0,
                        static_cast<std::size_t>(rows) * blocks_per_row * sizeof(float)),
             "zero materialized INT8 scales");

  constexpr int span_slot = 1;
  const std::size_t span_value_offset =
      static_cast<std::size_t>(span_slot) * heads * capacity * dimension;
  const std::size_t span_scale_offset =
      static_cast<std::size_t>(span_slot) * heads * capacity * blocks_per_row;
  carat::quantize_global_k_cache_span_int8_block128(
      device_source.get() + span_value_offset, device_destination.get() + span_value_offset,
      device_scales.get() + span_scale_offset, heads, 2, dimension, capacity, 1, nullptr);

  const std::array<int, 2> positions{4, 3};
  const std::array<int, 2> slots{0, 2};
  DeviceAllocation<int> device_positions(positions.size()), device_slots(slots.size());
  cuda_check(cudaMemcpy(device_positions.get(), positions.data(), sizeof(positions),
                        cudaMemcpyHostToDevice),
             "copy materialized INT8 positions");
  cuda_check(cudaMemcpy(device_slots.get(), slots.data(), sizeof(slots), cudaMemcpyHostToDevice),
             "copy materialized INT8 slots");
  carat::quantize_global_decode_k_int8_block128(
      device_source.get(), device_destination.get(), device_scales.get(), device_positions.get(),
      device_slots.get(), 2, maximum_slots, heads, dimension, capacity, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize materialized INT8 cache writes");

  std::vector<std::int8_t> destination(source.size());
  std::vector<float> scales(static_cast<std::size_t>(rows) * blocks_per_row);
  cuda_check(cudaMemcpy(destination.data(), device_destination.get(), destination.size(),
                        cudaMemcpyDeviceToHost),
             "copy materialized INT8 cache");
  cuda_check(cudaMemcpy(scales.data(), device_scales.get(), scales.size() * sizeof(scales[0]),
                        cudaMemcpyDeviceToHost),
             "copy materialized INT8 cache scales");
  for (int slot = 0; slot < maximum_slots; ++slot) {
    for (int head = 0; head < heads; ++head) {
      for (int token = 0; token < capacity; ++token) {
        const bool touched = (slot == 1 && (token == 1 || token == 2)) ||
                             (slot == 0 && token == 4) || (slot == 2 && token == 3);
        const std::size_t row = (static_cast<std::size_t>(slot) * heads + head) * capacity + token;
        for (int block = 0; block < blocks_per_row; ++block) {
          const float scale = scales[row * blocks_per_row + block];
          if (!touched) {
            if (scale != 0.0F)
              throw std::runtime_error("INT8 cache scale guard changed");
            continue;
          }
          if (!(scale > 0.0F))
            throw std::runtime_error("INT8 cache scale was not written");
          for (int component = block * 128; component < (block + 1) * 128; ++component) {
            const std::size_t index = row * dimension + component;
            const float expected = __bfloat162float(source[index]);
            const float reconstructed = static_cast<float>(destination[index]) * scale;
            if (std::abs(expected - reconstructed) > 0.51F * scale + 1.0e-6F) {
              throw std::runtime_error("INT8 cache reconstruction exceeded half a scale step");
            }
          }
        }
      }
    }
  }
}

void test_fused_int8_global_k_cache_write() {
  constexpr int batch = 3;
  constexpr int maximum_slots = 3;
  constexpr int query_heads = 32;
  constexpr int kv_heads = 4;
  constexpr int dimension = 512;
  constexpr int capacity = 64;
  constexpr int query_width = query_heads * dimension;
  constexpr int packed_width = query_width + kv_heads * dimension;
  constexpr std::size_t packed_elements = static_cast<std::size_t>(batch) * packed_width;
  constexpr std::size_t query_elements = static_cast<std::size_t>(batch) * query_heads * dimension;
  constexpr std::size_t cache_elements =
      static_cast<std::size_t>(maximum_slots) * kv_heads * capacity * dimension;
  constexpr std::size_t scale_elements =
      static_cast<std::size_t>(maximum_slots) * kv_heads * capacity * 4;
  std::vector<__nv_bfloat16> packed(packed_elements), norm(dimension);
  for (std::size_t index = 0; index < packed.size(); ++index) {
    packed[index] = __float2bfloat16_rn(0.7F * std::sin(static_cast<float>(index) * 0.0019F) +
                                        0.3F * std::cos(static_cast<float>(index) * 0.0047F));
  }
  for (int index = 0; index < dimension; ++index) {
    norm[static_cast<std::size_t>(index)] =
        __float2bfloat16_rn(0.06F + 0.012F * std::sin(static_cast<float>(index) * 0.021F));
  }
  const std::array<int, batch> positions{3, 17, 42};
  const std::array<int, batch> slots{2, 0, 1};
  DeviceAllocation<__nv_bfloat16> device_packed(packed_elements), device_norm(dimension),
      baseline_queries(query_elements), fused_queries(query_elements),
      baseline_keys(cache_elements), fused_keys(cache_elements), baseline_values(cache_elements),
      fused_values(cache_elements);
  DeviceAllocation<std::int8_t> baseline_int8(cache_elements), fused_int8(cache_elements);
  DeviceAllocation<float> baseline_scales(scale_elements), fused_scales(scale_elements);
  DeviceAllocation<int> device_positions(batch), device_slots(batch);
  cuda_check(cudaMemcpy(device_packed.get(), packed.data(), packed.size() * sizeof(packed[0]),
                        cudaMemcpyHostToDevice),
             "copy fused INT8 QKV input");
  cuda_check(cudaMemcpy(device_norm.get(), norm.data(), norm.size() * sizeof(norm[0]),
                        cudaMemcpyHostToDevice),
             "copy fused INT8 QKV norm");
  cuda_check(cudaMemcpy(device_positions.get(), positions.data(), sizeof(positions),
                        cudaMemcpyHostToDevice),
             "copy fused INT8 positions");
  cuda_check(cudaMemcpy(device_slots.get(), slots.data(), sizeof(slots), cudaMemcpyHostToDevice),
             "copy fused INT8 slots");
  for (auto *pointer :
       {baseline_keys.get(), fused_keys.get(), baseline_values.get(), fused_values.get()}) {
    cuda_check(cudaMemset(pointer, 0, cache_elements * sizeof(__nv_bfloat16)),
               "zero fused INT8 BF16 cache");
  }
  cuda_check(cudaMemset(baseline_int8.get(), 0, cache_elements), "zero baseline fused INT8 cache");
  cuda_check(cudaMemset(fused_int8.get(), 0, cache_elements), "zero candidate fused INT8 cache");
  cuda_check(cudaMemset(baseline_scales.get(), 0, scale_elements * sizeof(float)),
             "zero baseline fused INT8 scales");
  cuda_check(cudaMemset(fused_scales.get(), 0, scale_elements * sizeof(float)),
             "zero candidate fused INT8 scales");
  unsetenv("CARAT_SHARED_GLOBAL_KV_ROUNDTRIP");
  carat::Gemma4QkvPostprocessor baseline(capacity), fused(capacity);
  baseline.run_decode_ragged(device_packed.get(), device_norm.get(), device_norm.get(),
                             baseline_queries.get(), baseline_keys.get(), baseline_values.get(),
                             device_positions.get(), device_slots.get(), batch, maximum_slots,
                             query_heads, kv_heads, dimension, true, capacity, nullptr, nullptr,
                             1e-6F, nullptr);
  carat::quantize_global_decode_k_int8_block128(
      baseline_keys.get(), baseline_int8.get(), baseline_scales.get(), device_positions.get(),
      device_slots.get(), batch, maximum_slots, kv_heads, dimension, capacity, nullptr);
  fused.run_decode_ragged(device_packed.get(), device_norm.get(), device_norm.get(),
                          fused_queries.get(), fused_keys.get(), fused_values.get(),
                          device_positions.get(), device_slots.get(), batch, maximum_slots,
                          query_heads, kv_heads, dimension, true, capacity, fused_int8.get(),
                          fused_scales.get(), 1e-6F, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize fused INT8 QKV cache write");

  std::vector<__nv_bfloat16> baseline_bf16(cache_elements), fused_bf16(cache_elements);
  const auto require_bf16_equal = [&](const __nv_bfloat16 *left, const __nv_bfloat16 *right,
                                      const char *operation) {
    cuda_check(cudaMemcpy(baseline_bf16.data(), left, cache_elements * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "copy baseline fused INT8 BF16");
    cuda_check(cudaMemcpy(fused_bf16.data(), right, cache_elements * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "copy candidate fused INT8 BF16");
    if (baseline_bf16 != fused_bf16)
      throw std::runtime_error(operation);
  };
  require_bf16_equal(baseline_keys.get(), fused_keys.get(), "fused INT8 QKV changed BF16 keys");
  require_bf16_equal(baseline_values.get(), fused_values.get(),
                     "fused INT8 QKV changed BF16 values");
  std::vector<__nv_bfloat16> baseline_query_values(query_elements),
      fused_query_values(query_elements);
  cuda_check(cudaMemcpy(baseline_query_values.data(), baseline_queries.get(),
                        query_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
             "copy baseline fused INT8 queries");
  cuda_check(cudaMemcpy(fused_query_values.data(), fused_queries.get(),
                        query_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost),
             "copy candidate fused INT8 queries");
  if (baseline_query_values != fused_query_values) {
    throw std::runtime_error("fused INT8 QKV changed queries");
  }
  std::vector<std::int8_t> baseline_int8_values(cache_elements), fused_int8_values(cache_elements);
  std::vector<float> baseline_scale_values(scale_elements), fused_scale_values(scale_elements);
  cuda_check(cudaMemcpy(baseline_int8_values.data(), baseline_int8.get(), cache_elements,
                        cudaMemcpyDeviceToHost),
             "copy baseline fused INT8 values");
  cuda_check(cudaMemcpy(fused_int8_values.data(), fused_int8.get(), cache_elements,
                        cudaMemcpyDeviceToHost),
             "copy candidate fused INT8 values");
  cuda_check(cudaMemcpy(baseline_scale_values.data(), baseline_scales.get(),
                        scale_elements * sizeof(float), cudaMemcpyDeviceToHost),
             "copy baseline fused INT8 scales");
  cuda_check(cudaMemcpy(fused_scale_values.data(), fused_scales.get(),
                        scale_elements * sizeof(float), cudaMemcpyDeviceToHost),
             "copy candidate fused INT8 scales");
  if (baseline_int8_values != fused_int8_values || baseline_scale_values != fused_scale_values) {
    throw std::runtime_error("fused INT8 QKV differs from standalone quantization");
  }
}

void test_packed_int8_global_v_cache_writes() {
  constexpr int maximum_slots = 3;
  constexpr int heads = 2;
  constexpr int capacity = 64;
  constexpr int dimension = 512;
  constexpr int blocks_per_row = 4;
  constexpr int rows = maximum_slots * heads * capacity;
  constexpr std::size_t elements = static_cast<std::size_t>(rows) * dimension;
  std::vector<__nv_bfloat16> source(elements, __float2bfloat16(0.0F));
  const auto touched = [](int slot, int token) {
    return (slot == 1 && token >= 7 && token < 10) || (slot == 0 && token == 63) ||
           (slot == 2 && token == 31);
  };
  for (int slot = 0; slot < maximum_slots; ++slot) {
    for (int head = 0; head < heads; ++head) {
      for (int token = 0; token < capacity; ++token) {
        if (!touched(slot, token))
          continue;
        const std::size_t row = (static_cast<std::size_t>(slot) * heads + head) * capacity + token;
        for (int component = 0; component < dimension; ++component) {
          source[row * dimension + component] = __float2bfloat16_rn(
              0.9F * std::sin(static_cast<float>(row * dimension + component) * 0.017F) +
              0.2F * std::cos(static_cast<float>(component) * 0.031F));
        }
      }
    }
  }
  DeviceAllocation<__nv_bfloat16> device_source(elements);
  DeviceAllocation<std::int8_t> expected_values(elements), candidate_values(elements);
  DeviceAllocation<float> expected_scales(rows * blocks_per_row),
      candidate_scales(rows * blocks_per_row);
  cuda_check(cudaMemcpy(device_source.get(), source.data(), elements * sizeof(source[0]),
                        cudaMemcpyHostToDevice),
             "copy packed INT8 V source");
  cuda_check(cudaMemset(candidate_values.get(), 0, elements), "zero packed INT8 V candidate");
  cuda_check(cudaMemset(candidate_scales.get(), 0, rows * blocks_per_row * sizeof(float)),
             "zero packed INT8 V candidate scales");
  carat::quantize_grouped_global_values_wgmma_int8(device_source.get(), expected_values.get(),
                                                   expected_scales.get(), maximum_slots, heads,
                                                   capacity, dimension, nullptr);

  constexpr int span_slot = 1;
  constexpr int span_start = 7;
  constexpr int span_tokens = 3;
  const std::size_t slot_element_offset =
      static_cast<std::size_t>(span_slot) * heads * capacity * dimension;
  const std::size_t slot_scale_offset =
      static_cast<std::size_t>(span_slot) * heads * capacity * blocks_per_row;
  carat::quantize_grouped_global_value_span_wgmma_int8(
      device_source.get() + slot_element_offset, candidate_values.get() + slot_element_offset,
      candidate_scales.get() + slot_scale_offset, heads, span_tokens, capacity, dimension,
      span_start, nullptr);
  const std::array<int, 2> positions{63, 31};
  const std::array<int, 2> slots{0, 2};
  DeviceAllocation<int> device_positions(positions.size()), device_slots(slots.size());
  cuda_check(cudaMemcpy(device_positions.get(), positions.data(), sizeof(positions),
                        cudaMemcpyHostToDevice),
             "copy packed INT8 V positions");
  cuda_check(cudaMemcpy(device_slots.get(), slots.data(), sizeof(slots), cudaMemcpyHostToDevice),
             "copy packed INT8 V slots");
  carat::quantize_grouped_global_decode_values_wgmma_int8(
      device_source.get(), candidate_values.get(), candidate_scales.get(), device_positions.get(),
      device_slots.get(), 2, maximum_slots, heads, capacity, dimension, nullptr);
  cuda_check(cudaDeviceSynchronize(), "synchronize packed INT8 V cache writes");

  std::vector<std::int8_t> expected(elements), candidate(elements);
  std::vector<float> expected_scale_values(rows * blocks_per_row),
      candidate_scale_values(rows * blocks_per_row);
  cuda_check(cudaMemcpy(expected.data(), expected_values.get(), elements, cudaMemcpyDeviceToHost),
             "copy expected packed INT8 V");
  cuda_check(cudaMemcpy(candidate.data(), candidate_values.get(), elements, cudaMemcpyDeviceToHost),
             "copy candidate packed INT8 V");
  cuda_check(cudaMemcpy(expected_scale_values.data(), expected_scales.get(),
                        rows * blocks_per_row * sizeof(float), cudaMemcpyDeviceToHost),
             "copy expected packed INT8 V scales");
  cuda_check(cudaMemcpy(candidate_scale_values.data(), candidate_scales.get(),
                        rows * blocks_per_row * sizeof(float), cudaMemcpyDeviceToHost),
             "copy candidate packed INT8 V scales");
  if (expected != candidate) {
    throw std::runtime_error("packed INT8 V span/scatter layout mismatch");
  }
  for (int slot = 0; slot < maximum_slots; ++slot) {
    for (int head = 0; head < heads; ++head) {
      for (int token = 0; token < capacity; ++token) {
        const std::size_t row = (static_cast<std::size_t>(slot) * heads + head) * capacity + token;
        for (int block = 0; block < blocks_per_row; ++block) {
          const std::size_t scale_index = row * blocks_per_row + block;
          if (touched(slot, token)) {
            if (candidate_scale_values[scale_index] != expected_scale_values[scale_index]) {
              throw std::runtime_error("packed INT8 V scale mismatch");
            }
          } else if (candidate_scale_values[scale_index] != 0.0F) {
            throw std::runtime_error("packed INT8 V scale guard changed");
          }
        }
      }
    }
  }
}

template <int query_tokens> void test_uniform_ring_attention() {
  constexpr int requests = 2;
  constexpr int query_heads = 2;
  constexpr int kv_heads = 1;
  constexpr int capacity = 16;
  constexpr int dimension = 16;
  constexpr int query_elements = requests * query_heads * query_tokens * dimension;
  constexpr int cache_elements = requests * kv_heads * capacity * dimension;
  constexpr int backup_elements = requests * kv_heads * query_tokens * dimension;
  std::vector<float> query_values(query_elements);
  std::vector<float> key_values(cache_elements);
  std::vector<float> value_values(cache_elements);
  std::vector<float> backup_key_values(backup_elements);
  std::vector<float> backup_value_values(backup_elements);
  const std::array<int, requests> positions{11, 13};
  for (int index = 0; index < query_elements; ++index) {
    query_values[static_cast<std::size_t>(index)] =
        static_cast<float>((index * 7) % 19 - 9) / 32.0F;
  }
  for (int index = 0; index < cache_elements; ++index) {
    key_values[static_cast<std::size_t>(index)] = static_cast<float>((index * 5) % 23 - 11) / 48.0F;
    value_values[static_cast<std::size_t>(index)] =
        static_cast<float>((index * 11) % 29 - 14) / 24.0F;
  }
  for (int index = 0; index < backup_elements; ++index) {
    backup_key_values[static_cast<std::size_t>(index)] =
        static_cast<float>((index * 13) % 17 - 8) / 40.0F;
    backup_value_values[static_cast<std::size_t>(index)] =
        static_cast<float>((index * 3) % 31 - 15) / 20.0F;
  }
  const auto to_bf16 = [](const std::vector<float> &values) {
    std::vector<__nv_bfloat16> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      result[index] = __float2bfloat16(values[index]);
    }
    return result;
  };
  const auto query = to_bf16(query_values);
  const auto keys = to_bf16(key_values);
  const auto values = to_bf16(value_values);
  const auto backup_keys = to_bf16(backup_key_values);
  const auto backup_values = to_bf16(backup_value_values);
  DeviceAllocation<__nv_bfloat16> device_query(query_elements), device_keys(cache_elements),
      device_values(cache_elements), device_backup_keys(backup_elements),
      device_backup_values(backup_elements), device_output(query_elements);
  DeviceAllocation<int> device_positions(requests);
  DeviceAllocation<void *> device_pointers(2 * requests);
  cuda_check(cudaMemcpy(device_query.get(), query.data(), query.size() * sizeof(query[0]),
                        cudaMemcpyHostToDevice),
             "copy ring queries");
  cuda_check(cudaMemcpy(device_keys.get(), keys.data(), keys.size() * sizeof(keys[0]),
                        cudaMemcpyHostToDevice),
             "copy ring keys");
  cuda_check(cudaMemcpy(device_values.get(), values.data(), values.size() * sizeof(values[0]),
                        cudaMemcpyHostToDevice),
             "copy ring values");
  cuda_check(cudaMemcpy(device_backup_keys.get(), backup_keys.data(),
                        backup_keys.size() * sizeof(backup_keys[0]), cudaMemcpyHostToDevice),
             "copy ring backup keys");
  cuda_check(cudaMemcpy(device_backup_values.get(), backup_values.data(),
                        backup_values.size() * sizeof(backup_values[0]), cudaMemcpyHostToDevice),
             "copy ring backup values");
  cuda_check(cudaMemcpy(device_positions.get(), positions.data(), sizeof(positions),
                        cudaMemcpyHostToDevice),
             "copy ring positions");
  std::array<void *, 2 * requests> pointers{};
  for (int request = 0; request < requests; ++request) {
    pointers[static_cast<std::size_t>(request)] =
        device_keys.get() + request * kv_heads * capacity * dimension;
    pointers[static_cast<std::size_t>(requests + request)] =
        device_values.get() + request * kv_heads * capacity * dimension;
  }
  cuda_check(
      cudaMemcpy(device_pointers.get(), pointers.data(), sizeof(pointers), cudaMemcpyHostToDevice),
      "copy ring pointers");
  carat::GemmCausalPrefillAttention attention(requests * capacity, query_heads, requests);
  attention.run_uniform_ring_batch(
      device_query.get(), reinterpret_cast<const void *const *>(device_pointers.get()),
      reinterpret_cast<const void *const *>(device_pointers.get() + requests),
      device_backup_keys.get(), device_backup_values.get(), device_output.get(),
      device_positions.get(), requests, query_heads, kv_heads, query_tokens, capacity, dimension,
      nullptr);
  cuda_check(cudaDeviceSynchronize(), "ring attention synchronization");
  std::vector<__nv_bfloat16> output(query_elements);
  cuda_check(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(output[0]),
                        cudaMemcpyDeviceToHost),
             "copy ring attention output");

  for (int request = 0; request < requests; ++request) {
    for (int head = 0; head < query_heads; ++head) {
      for (int query_token = 0; query_token < query_tokens; ++query_token) {
        std::array<float, capacity> scores{};
        float maximum = -INFINITY;
        for (int physical = 0; physical < capacity; ++physical) {
          int backup_token = -1;
          for (int proposal = query_token + 1; proposal < query_tokens; ++proposal) {
            if ((positions[static_cast<std::size_t>(request)] + proposal) % capacity == physical) {
              backup_token = proposal;
            }
          }
          float score = 0.0F;
          for (int component = 0; component < dimension; ++component) {
            const int query_index =
                ((request * query_heads + head) * query_tokens + query_token) * dimension +
                component;
            const int key_index =
                backup_token >= 0
                    ? ((request * kv_heads) * query_tokens + backup_token) * dimension + component
                    : ((request * kv_heads) * capacity + physical) * dimension + component;
            score +=
                __bfloat162float(query[static_cast<std::size_t>(query_index)]) *
                __bfloat162float(
                    (backup_token >= 0 ? backup_keys : keys)[static_cast<std::size_t>(key_index)]);
          }
          scores[static_cast<std::size_t>(physical)] = score;
          maximum = std::max(maximum, score);
        }
        float denominator = 0.0F;
        for (float &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (int component = 0; component < dimension; ++component) {
          float expected = 0.0F;
          for (int physical = 0; physical < capacity; ++physical) {
            int backup_token = -1;
            for (int proposal = query_token + 1; proposal < query_tokens; ++proposal) {
              if ((positions[static_cast<std::size_t>(request)] + proposal) % capacity ==
                  physical) {
                backup_token = proposal;
              }
            }
            const int value_index =
                backup_token >= 0
                    ? ((request * kv_heads) * query_tokens + backup_token) * dimension + component
                    : ((request * kv_heads) * capacity + physical) * dimension + component;
            expected += scores[static_cast<std::size_t>(physical)] / denominator *
                        __bfloat162float((backup_token >= 0
                                              ? backup_values
                                              : values)[static_cast<std::size_t>(value_index)]);
          }
          const int output_index =
              ((request * query_heads + head) * query_tokens + query_token) * dimension + component;
          near(__bfloat162float(output[static_cast<std::size_t>(output_index)]), expected, 0.015F,
               "uniform ring attention");
        }
      }
    }
  }
}

} // namespace

int main() {
  try {
    test_norm_add_and_scale();
    test_norm_carried_fp8_amax();
    test_gelu_gate();
    test_gelu_carried_fp8_amax();
    test_head_to_token_carried_fp8_amax();
    test_linear();
    test_int4_weight_oracle();
    test_int4_fp8_linear();
    test_int4_bf16_linear();
    test_grouped_attention();
    test_shared_global_kv_roundtrip();
    test_prefill_layouts();
    test_int8_kv_quality_oracle();
    test_materialized_int8_global_k_cache_writes();
    test_fused_int8_global_k_cache_write();
    test_packed_int8_global_v_cache_writes();
    test_batched_argmax();
    test_uniform_ring_attention<4>();
    test_uniform_ring_attention<8>();
    std::cout << "all CUDA operation tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CUDA operation test failed: " << error.what() << '\n';
    return 1;
  }
}
