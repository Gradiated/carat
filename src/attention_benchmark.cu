#include "carat/attention.h"
#include "carat/cuda_ops.h"
#include "carat/gemm_attention.h"
#include "carat/prefill_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class Allocation {
public:
  explicit Allocation(std::uint64_t bytes) {
    check(cudaMalloc(&pointer_, static_cast<std::size_t>(bytes)), "cudaMalloc attention buffer");
    check(cudaMemset(pointer_, 0, static_cast<std::size_t>(bytes)), "cudaMemset attention buffer");
  }
  ~Allocation() {
    cudaFree(pointer_);
  }
  void *get() {
    return pointer_;
  }

private:
  void *pointer_{nullptr};
};

__device__ std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  return value ^ (value >> 16U);
}

__global__ void initialize_bf16(__nv_bfloat16 *output, std::uint64_t elements, std::uint32_t seed,
                                float magnitude) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const float uniform = static_cast<float>(mix(static_cast<std::uint32_t>(index) ^ seed)) /
                        static_cast<float>(0xffffffffU);
  output[index] = __float2bfloat16_rn((2.0F * uniform - 1.0F) * magnitude);
}

__global__ void initialize_bf16_offset(__nv_bfloat16 *output, std::uint64_t elements,
                                       std::uint32_t seed, float center, float magnitude) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const float uniform = static_cast<float>(mix(static_cast<std::uint32_t>(index) ^ seed)) /
                        static_cast<float>(0xffffffffU);
  output[index] = __float2bfloat16_rn(center + (2.0F * uniform - 1.0F) * magnitude);
}

__global__ void initialize_inverse_rms(float *output, std::uint64_t elements) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = 0.85F + 0.3F * static_cast<float>(mix(static_cast<std::uint32_t>(index))) /
                                static_cast<float>(0xffffffffU);
  }
}

__global__ void initialize_rope(__nv_bfloat16 *cosine, __nv_bfloat16 *sine, std::uint64_t elements,
                                int dimension) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const int position = static_cast<int>(index / dimension);
  const int component = static_cast<int>(index % dimension);
  const int half_component = component % (dimension / 2);
  // Gemma 4 full attention uses partial_rotary_factor=0.25: 64 frequencies are repeated across
  // the two halves of a D=512 head and the remaining 384 components have identity rotation.
  const float angle =
      half_component < dimension / 8
          ? static_cast<float>(position) * expf(-0.0175F * static_cast<float>(half_component))
          : 0.0F;
  cosine[index] = __float2bfloat16_rn(cosf(angle));
  sine[index] = __float2bfloat16_rn(sinf(angle));
}

__global__ void initialize_compact_global_rope(__nv_bfloat16 *cosine, __nv_bfloat16 *sine,
                                               int positions) {
  constexpr int frequencies = 64;
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= frequencies * positions)
    return;
  const int frequency = index / positions;
  const int position = index - frequency * positions;
  const float angle = static_cast<float>(position) * expf(-0.0175F * static_cast<float>(frequency));
  cosine[index] = __float2bfloat16_rn(cosf(angle));
  sine[index] = __float2bfloat16_rn(sinf(angle));
}

__global__ void materialize_raw_global_kv(const __nv_bfloat16 *raw, const float *inverse_rms,
                                          const __nv_bfloat16 *key_norm_weight,
                                          const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                                          __nv_bfloat16 *keys, __nv_bfloat16 *values,
                                          std::uint64_t elements, int context, int dimension) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const int component = static_cast<int>(index % dimension);
  const std::uint64_t row = index / dimension;
  const int token = static_cast<int>(row % context);
  const int half = dimension / 2;
  const int partner = component < half ? component + half : component - half;
  const std::uint64_t partner_index = row * dimension + partner;
  const float scale = inverse_rms[row];
  const float normalized = __bfloat162float(__float2bfloat16_rn(
      __bfloat162float(raw[index]) * scale * __bfloat162float(key_norm_weight[component])));
  const float partner_normalized = __bfloat162float(__float2bfloat16_rn(
      __bfloat162float(raw[partner_index]) * scale * __bfloat162float(key_norm_weight[partner])));
  const float rotated = component < half ? -partner_normalized : partner_normalized;
  const std::uint64_t rotary = static_cast<std::uint64_t>(token) * dimension + component;
  keys[index] = __float2bfloat16_rn(normalized * __bfloat162float(cosine[rotary]) +
                                    rotated * __bfloat162float(sine[rotary]));
  values[index] = __float2bfloat16_rn(__bfloat162float(raw[index]) * scale);
}

__global__ void quantize_fixed_e4m3(const __nv_bfloat16 *input, __nv_fp8_e4m3 *output,
                                    std::uint64_t elements, float multiplier) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __nv_fp8_e4m3(__bfloat162float(input[index]) * multiplier);
  }
}

void fp8_parity() {
  constexpr int batch = 2;
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 8192;
  constexpr int active_context = 6144;
  constexpr int dimension = 512;
  constexpr std::uint64_t query_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * dimension;
  constexpr std::uint64_t cache_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context * dimension;
  Allocation query(2ULL * query_elements), key(2ULL * cache_elements), value(2ULL * cache_elements),
      fp8_query(query_elements), fp8_key(cache_elements), fp8_value(cache_elements),
      reference(2ULL * query_elements), candidate(2ULL * query_elements),
      fp8_kv_candidate(2ULL * query_elements), int8_key(cache_elements),
      int8_key_scales(sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads * context),
      int8_block_key(cache_elements),
      int8_block_key_scales(4ULL * sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads *
                            context),
      int8_candidate(2ULL * query_elements), int8_block_candidate(2ULL * query_elements),
      positions(sizeof(int) * batch);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x12345678U, 1.75F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key.get()), cache_elements, 0x9abcdef0U, 0.11F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(value.get()), cache_elements, 0x31415926U, 1.75F);
  quantize_fixed_e4m3<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<const __nv_bfloat16 *>(query.get()),
      static_cast<__nv_fp8_e4m3 *>(fp8_query.get()), query_elements, 64.0F);
  quantize_fixed_e4m3<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<const __nv_bfloat16 *>(key.get()), static_cast<__nv_fp8_e4m3 *>(fp8_key.get()),
      cache_elements, 1024.0F);
  carat::quantize_grouped_global_values_wgmma_fp8(value.get(), fp8_value.get(), batch, kv_heads,
                                                  context, dimension, nullptr);
  carat::quantize_bf16_rows_symmetric_int8(key.get(), int8_key.get(),
                                           static_cast<float *>(int8_key_scales.get()),
                                           batch * kv_heads * context, dimension, nullptr);
  carat::quantize_bf16_blocks_symmetric_int8(key.get(), int8_block_key.get(),
                                             static_cast<float *>(int8_block_key_scales.get()),
                                             batch * kv_heads * context, dimension, 128, nullptr);
  const std::vector<int> host_positions(batch, active_context - 1);
  check(cudaMemcpy(positions.get(), host_positions.data(), sizeof(int) * batch,
                   cudaMemcpyHostToDevice),
        "copy FP8 parity positions");
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  attention.run_ragged_contiguous(query.get(), key.get(), value.get(), reference.get(),
                                  static_cast<const int *>(positions.get()), batch, kv_heads,
                                  query_group, active_context, dimension, 0, nullptr, context);
  attention.run_ragged_contiguous_fp8_keys(
      fp8_query.get(), fp8_key.get(), value.get(), candidate.get(),
      static_cast<const int *>(positions.get()), batch, kv_heads, query_group, active_context,
      dimension, 0, nullptr, context);
  attention.run_ragged_contiguous_fp8_kv(
      fp8_query.get(), fp8_key.get(), fp8_value.get(), fp8_kv_candidate.get(),
      static_cast<const int *>(positions.get()), batch, kv_heads, query_group, active_context,
      dimension, 0, nullptr, context);
  attention.run_ragged_contiguous_int8_keys(
      query.get(), int8_key.get(), static_cast<const float *>(int8_key_scales.get()), value.get(),
      int8_candidate.get(), static_cast<const int *>(positions.get()), batch, kv_heads, query_group,
      active_context, dimension, 0, nullptr, context);
  attention.run_ragged_contiguous_int8_block128_keys(
      query.get(), int8_block_key.get(), static_cast<const float *>(int8_block_key_scales.get()),
      value.get(), int8_block_candidate.get(), static_cast<const int *>(positions.get()), batch,
      kv_heads, query_group, active_context, dimension, 0, nullptr, context);
  check(cudaDeviceSynchronize(), "synchronize FP8-key parity");
  std::vector<__nv_bfloat16> expected(query_elements), actual(query_elements),
      actual_fp8_kv(query_elements), actual_int8(query_elements), actual_int8_block(query_elements);
  check(cudaMemcpy(expected.data(), reference.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy FP8-key reference");
  check(cudaMemcpy(actual.data(), candidate.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy FP8-key candidate");
  check(cudaMemcpy(actual_fp8_kv.data(), fp8_kv_candidate.get(), 2ULL * query_elements,
                   cudaMemcpyDeviceToHost),
        "copy FP8-KV candidate");
  check(cudaMemcpy(actual_int8.data(), int8_candidate.get(), 2ULL * query_elements,
                   cudaMemcpyDeviceToHost),
        "copy INT8-key candidate");
  check(cudaMemcpy(actual_int8_block.data(), int8_block_candidate.get(), 2ULL * query_elements,
                   cudaMemcpyDeviceToHost),
        "copy INT8 block-128 key candidate");
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double squared_candidate = 0.0;
  double dot = 0.0;
  double mean = 0.0;
  float maximum = 0.0F;
  double fp8_kv_squared_error = 0.0;
  double fp8_kv_squared_candidate = 0.0;
  double fp8_kv_dot = 0.0;
  double fp8_kv_mean = 0.0;
  float fp8_kv_maximum = 0.0F;
  double int8_squared_error = 0.0;
  double int8_squared_candidate = 0.0;
  double int8_dot = 0.0;
  double int8_mean = 0.0;
  float int8_maximum = 0.0F;
  double int8_block_squared_error = 0.0;
  double int8_block_squared_candidate = 0.0;
  double int8_block_dot = 0.0;
  double int8_block_mean = 0.0;
  float int8_block_maximum = 0.0F;
  for (std::uint64_t index = 0; index < query_elements; ++index) {
    const float expected_value = __bfloat162float(expected[index]);
    const float actual_value = __bfloat162float(actual[index]);
    const float difference = std::abs(expected_value - actual_value);
    maximum = std::max(maximum, difference);
    mean += difference;
    squared_error += static_cast<double>(difference) * difference;
    squared_reference += static_cast<double>(expected_value) * expected_value;
    squared_candidate += static_cast<double>(actual_value) * actual_value;
    dot += static_cast<double>(expected_value) * actual_value;
    const float fp8_kv_value = __bfloat162float(actual_fp8_kv[index]);
    const float fp8_kv_difference = std::abs(expected_value - fp8_kv_value);
    fp8_kv_maximum = std::max(fp8_kv_maximum, fp8_kv_difference);
    fp8_kv_mean += fp8_kv_difference;
    fp8_kv_squared_error += static_cast<double>(fp8_kv_difference) * fp8_kv_difference;
    fp8_kv_squared_candidate += static_cast<double>(fp8_kv_value) * fp8_kv_value;
    fp8_kv_dot += static_cast<double>(expected_value) * fp8_kv_value;
    const float int8_value = __bfloat162float(actual_int8[index]);
    const float int8_difference = std::abs(expected_value - int8_value);
    int8_maximum = std::max(int8_maximum, int8_difference);
    int8_mean += int8_difference;
    int8_squared_error += static_cast<double>(int8_difference) * int8_difference;
    int8_squared_candidate += static_cast<double>(int8_value) * int8_value;
    int8_dot += static_cast<double>(expected_value) * int8_value;
    const float int8_block_value = __bfloat162float(actual_int8_block[index]);
    const float int8_block_difference = std::abs(expected_value - int8_block_value);
    int8_block_maximum = std::max(int8_block_maximum, int8_block_difference);
    int8_block_mean += int8_block_difference;
    int8_block_squared_error += static_cast<double>(int8_block_difference) * int8_block_difference;
    int8_block_squared_candidate += static_cast<double>(int8_block_value) * int8_block_value;
    int8_block_dot += static_cast<double>(expected_value) * int8_block_value;
  }
  std::cout << "fp8_key_parity_max_abs=" << maximum << '\n'
            << "fp8_key_parity_mean_abs=" << mean / query_elements << '\n'
            << "fp8_key_parity_relative_l2=" << std::sqrt(squared_error / squared_reference) << '\n'
            << "fp8_key_parity_cosine=" << dot / std::sqrt(squared_reference * squared_candidate)
            << '\n'
            << "fp8_kv_parity_max_abs=" << fp8_kv_maximum << '\n'
            << "fp8_kv_parity_mean_abs=" << fp8_kv_mean / query_elements << '\n'
            << "fp8_kv_parity_relative_l2=" << std::sqrt(fp8_kv_squared_error / squared_reference)
            << '\n'
            << "fp8_kv_parity_cosine="
            << fp8_kv_dot / std::sqrt(squared_reference * fp8_kv_squared_candidate) << '\n'
            << "int8_key_parity_max_abs=" << int8_maximum << '\n'
            << "int8_key_parity_mean_abs=" << int8_mean / query_elements << '\n'
            << "int8_key_parity_relative_l2=" << std::sqrt(int8_squared_error / squared_reference)
            << '\n'
            << "int8_key_parity_cosine="
            << int8_dot / std::sqrt(squared_reference * int8_squared_candidate) << '\n'
            << "int8_block128_key_parity_max_abs=" << int8_block_maximum << '\n'
            << "int8_block128_key_parity_mean_abs=" << int8_block_mean / query_elements << '\n'
            << "int8_block128_key_parity_relative_l2="
            << std::sqrt(int8_block_squared_error / squared_reference) << '\n'
            << "int8_block128_key_parity_cosine="
            << int8_block_dot / std::sqrt(squared_reference * int8_block_squared_candidate) << '\n';
}

void int8_block_qk_contract_parity() {
  constexpr int context = 64;
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  constexpr int blocks = 4;
  std::vector<__nv_bfloat16> query(query_group * dimension);
  std::vector<__nv_bfloat16> key(context * dimension);
  for (int row = 0; row < query_group; ++row) {
    for (int component = 0; component < dimension; ++component) {
      const int block = component / 128;
      query[static_cast<std::size_t>(row * dimension + component)] =
          __float2bfloat16((0.4F + 0.17F * block + 0.03F * row) *
                           std::sin(0.019F * static_cast<float>(row * dimension + component)));
    }
  }
  for (int token = 0; token < context; ++token) {
    for (int component = 0; component < dimension; ++component) {
      const int block = component / 128;
      key[static_cast<std::size_t>(token * dimension + component)] =
          __float2bfloat16((0.025F + 0.018F * block + 0.0004F * token) *
                           std::cos(0.013F * static_cast<float>(token * dimension + component)));
    }
  }
  Allocation device_query(2ULL * query.size()), device_key(2ULL * key.size()),
      int8_query(query.size()), int8_key(key.size()),
      query_scales(sizeof(float) * query_group * blocks),
      key_scales(sizeof(float) * context * blocks), scores(2ULL * context * query_group);
  check(cudaMemcpy(device_query.get(), query.data(), 2ULL * query.size(), cudaMemcpyHostToDevice),
        "copy INT8 block QK query");
  check(cudaMemcpy(device_key.get(), key.data(), 2ULL * key.size(), cudaMemcpyHostToDevice),
        "copy INT8 block QK key");
  carat::quantize_bf16_blocks_symmetric_int8(device_query.get(), int8_query.get(),
                                             static_cast<float *>(query_scales.get()), query_group,
                                             dimension, 128, nullptr);
  carat::quantize_bf16_blocks_symmetric_int8(device_key.get(), int8_key.get(),
                                             static_cast<float *>(key_scales.get()), context,
                                             dimension, 128, nullptr);
  carat::grouped_global_qk_wgmma_int8_block128(
      int8_query.get(), static_cast<const float *>(query_scales.get()), int8_key.get(),
      static_cast<const float *>(key_scales.get()), scores.get(), 1, 1, query_group, context,
      dimension, context, nullptr);
  check(cudaDeviceSynchronize(), "synchronize INT8 block QK contract");
  std::vector<std::int8_t> host_query(query.size()), host_key(key.size());
  std::array<float, query_group * blocks> host_query_scales{};
  std::array<float, context * blocks> host_key_scales{};
  std::array<__nv_bfloat16, context * query_group> actual{};
  check(cudaMemcpy(host_query.data(), int8_query.get(), host_query.size(), cudaMemcpyDeviceToHost),
        "copy INT8 block QK quantized query");
  check(cudaMemcpy(host_key.data(), int8_key.get(), host_key.size(), cudaMemcpyDeviceToHost),
        "copy INT8 block QK quantized key");
  check(cudaMemcpy(host_query_scales.data(), query_scales.get(), sizeof(host_query_scales),
                   cudaMemcpyDeviceToHost),
        "copy INT8 block QK query scales");
  check(cudaMemcpy(host_key_scales.data(), key_scales.get(), sizeof(host_key_scales),
                   cudaMemcpyDeviceToHost),
        "copy INT8 block QK key scales");
  check(cudaMemcpy(actual.data(), scores.get(), sizeof(actual), cudaMemcpyDeviceToHost),
        "copy INT8 block QK scores");
  float maximum = 0.0F;
  double mean = 0.0;
  for (int query_index = 0; query_index < query_group; ++query_index) {
    for (int token = 0; token < context; ++token) {
      float expected = 0.0F;
      for (int block = 0; block < blocks; ++block) {
        std::int32_t dot = 0;
        for (int component = 0; component < 128; ++component) {
          dot +=
              static_cast<std::int32_t>(
                  host_key[static_cast<std::size_t>(token * dimension + block * 128 + component)]) *
              static_cast<std::int32_t>(host_query[static_cast<std::size_t>(
                  query_index * dimension + block * 128 + component)]);
        }
        expected += static_cast<float>(dot) *
                    host_key_scales[static_cast<std::size_t>(token * blocks + block)] *
                    host_query_scales[static_cast<std::size_t>(query_index * blocks + block)];
      }
      expected = __bfloat162float(__float2bfloat16(expected));
      const float difference = std::abs(
          __bfloat162float(actual[static_cast<std::size_t>(query_index * context + token)]) -
          expected);
      maximum = std::max(maximum, difference);
      mean += difference;
    }
  }
  mean /= static_cast<double>(context * query_group);
  std::cout << "int8_block128_qk_contract_max_abs=" << maximum << '\n'
            << "int8_block128_qk_contract_mean_abs=" << mean << '\n';
}

void wgmma_qk_parity() {
  constexpr int batch = 2;
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 128;
  constexpr int dimension = 512;
  constexpr std::uint64_t query_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * dimension;
  constexpr std::uint64_t key_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context * dimension;
  constexpr std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * context;
  Allocation query(2ULL * query_elements), key(2ULL * key_elements),
      reference(2ULL * score_elements), candidate(2ULL * score_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x88776655U, 0.12F);
  initialize_bf16<<<(key_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key.get()), key_elements, 0x10293847U, 0.08F);
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  attention.run_qk(query.get(), key.get(), reference.get(), batch, kv_heads, query_group, context,
                   dimension, nullptr);
  carat::grouped_global_qk_wgmma_bf16(query.get(), key.get(), candidate.get(), batch, kv_heads,
                                      query_group, context, dimension, nullptr);
  check(cudaDeviceSynchronize(), "synchronize WGMMA QK parity");
  std::vector<__nv_bfloat16> expected(score_elements), actual(score_elements);
  check(cudaMemcpy(expected.data(), reference.get(), 2ULL * score_elements, cudaMemcpyDeviceToHost),
        "copy WGMMA QK reference");
  check(cudaMemcpy(actual.data(), candidate.get(), 2ULL * score_elements, cudaMemcpyDeviceToHost),
        "copy WGMMA QK candidate");
  float maximum = 0.0F;
  double mean = 0.0;
  for (std::uint64_t index = 0; index < score_elements; ++index) {
    const float difference =
        std::abs(__bfloat162float(expected[index]) - __bfloat162float(actual[index]));
    maximum = std::max(maximum, difference);
    mean += difference;
  }
  std::cout << "wgmma_qk_parity_max_abs=" << maximum << '\n'
            << "wgmma_qk_parity_mean_abs=" << mean / score_elements << '\n';
  if (maximum > 0.02F)
    throw std::runtime_error("WGMMA QK parity exceeded tolerance");
}

void wgmma_attention_tile_parity() {
  constexpr int batch = 2;
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 64;
  constexpr int dimension = 512;
  constexpr std::uint64_t query_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * dimension;
  constexpr std::uint64_t cache_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context * dimension;
  Allocation query(2ULL * query_elements), key(2ULL * cache_elements), value(2ULL * cache_elements),
      reference(2ULL * query_elements), candidate(2ULL * query_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0xa1b2c3d4U, 0.12F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key.get()), cache_elements, 0x55667788U, 0.08F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(value.get()), cache_elements, 0xabcdef01U, 1.1F);
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  attention.run(query.get(), key.get(), value.get(), reference.get(), batch, kv_heads, query_group,
                context, dimension, nullptr);
  carat::grouped_global_attention_wgmma_tile_bf16(query.get(), key.get(), value.get(),
                                                  candidate.get(), batch, kv_heads, query_group,
                                                  context, dimension, nullptr);
  check(cudaDeviceSynchronize(), "synchronize WGMMA attention tile parity");
  std::vector<__nv_bfloat16> expected(query_elements), actual(query_elements);
  check(cudaMemcpy(expected.data(), reference.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy WGMMA attention tile reference");
  check(cudaMemcpy(actual.data(), candidate.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy WGMMA attention tile candidate");
  float maximum = 0.0F;
  double mean = 0.0;
  for (std::uint64_t index = 0; index < query_elements; ++index) {
    const float difference =
        std::abs(__bfloat162float(expected[index]) - __bfloat162float(actual[index]));
    maximum = std::max(maximum, difference);
    mean += difference;
  }
  std::cout << "wgmma_attention_tile_parity_max_abs=" << maximum << '\n'
            << "wgmma_attention_tile_parity_mean_abs=" << mean / query_elements << '\n';
  if (maximum > 0.02F) {
    throw std::runtime_error("WGMMA attention tile parity exceeded tolerance");
  }
}

void wgmma_raw_attention_tile_parity() {
  constexpr int batch = 2;
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 64;
  constexpr int dimension = 512;
  constexpr std::uint64_t heads = batch * kv_heads;
  constexpr std::uint64_t query_elements = heads * query_group * dimension;
  constexpr std::uint64_t cache_elements = heads * context * dimension;
  constexpr std::uint64_t statistic_elements = heads * context;
  constexpr std::uint64_t rotary_elements = context * dimension;
  Allocation query(2ULL * query_elements), raw(2ULL * cache_elements),
      inverse_rms(sizeof(float) * statistic_elements), key_norm(2ULL * dimension),
      cosine(2ULL * rotary_elements), sine(2ULL * rotary_elements), key(2ULL * cache_elements),
      value(2ULL * cache_elements), reference(2ULL * query_elements),
      rs_candidate(2ULL * query_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x14d3a97bU, 0.12F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(raw.get()), cache_elements, 0x55ac1279U, 1.1F);
  initialize_inverse_rms<<<(statistic_elements + threads - 1) / threads, threads>>>(
      static_cast<float *>(inverse_rms.get()), statistic_elements);
  initialize_bf16_offset<<<(dimension + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key_norm.get()), dimension, 0x7f4a219dU, 1.0F, 0.2F);
  initialize_rope<<<(rotary_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(cosine.get()), static_cast<__nv_bfloat16 *>(sine.get()),
      rotary_elements, dimension);
  materialize_raw_global_kv<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<const __nv_bfloat16 *>(raw.get()), static_cast<const float *>(inverse_rms.get()),
      static_cast<const __nv_bfloat16 *>(key_norm.get()),
      static_cast<const __nv_bfloat16 *>(cosine.get()),
      static_cast<const __nv_bfloat16 *>(sine.get()), static_cast<__nv_bfloat16 *>(key.get()),
      static_cast<__nv_bfloat16 *>(value.get()), cache_elements, context, dimension);
  carat::grouped_global_attention_wgmma_tile_bf16(query.get(), key.get(), value.get(),
                                                  reference.get(), batch, kv_heads, query_group,
                                                  context, dimension, nullptr);
  carat::grouped_global_attention_wgmma_raw_rs_tile_bf16(
      query.get(), raw.get(), static_cast<const float *>(inverse_rms.get()), key_norm.get(),
      cosine.get(), sine.get(), rs_candidate.get(), batch, kv_heads, query_group, context,
      dimension, nullptr);
  check(cudaDeviceSynchronize(), "synchronize raw WGMMA attention tile parity");
  std::vector<__nv_bfloat16> expected(query_elements), rs_actual(query_elements);
  check(cudaMemcpy(expected.data(), reference.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy raw WGMMA attention tile reference");
  check(cudaMemcpy(rs_actual.data(), rs_candidate.get(), 2ULL * query_elements,
                   cudaMemcpyDeviceToHost),
        "copy RS-raw WGMMA attention tile candidate");
  float rs_maximum = 0.0F;
  double rs_mean = 0.0;
  std::uint64_t rs_unequal = 0;
  for (std::uint64_t index = 0; index < query_elements; ++index) {
    const float rs_difference =
        std::abs(__bfloat162float(expected[index]) - __bfloat162float(rs_actual[index]));
    rs_maximum = std::max(rs_maximum, rs_difference);
    rs_mean += rs_difference;
    rs_unequal += rs_difference != 0.0F;
  }
  std::cout << "wgmma_rs_raw_attention_tile_parity_max_abs=" << rs_maximum << '\n'
            << "wgmma_rs_raw_attention_tile_parity_mean_abs=" << rs_mean / query_elements << '\n'
            << "wgmma_rs_raw_attention_tile_parity_unequal=" << rs_unequal << '\n';
  if (rs_unequal != 0) {
    throw std::runtime_error("raw WGMMA attention tile was not bit-identical");
  }
}

void wgmma_raw_rs_split_attention_parity() {
  constexpr int batch = 2;
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 128;
  constexpr int dimension = 512;
  constexpr int tiles_per_segment = 1;
  constexpr int heads = batch * kv_heads;
  constexpr int segments_per_head = context / 64 / tiles_per_segment;
  constexpr std::uint64_t query_elements =
      static_cast<std::uint64_t>(heads) * query_group * dimension;
  constexpr std::uint64_t cache_elements = static_cast<std::uint64_t>(heads) * context * dimension;
  constexpr std::uint64_t inverse_elements = static_cast<std::uint64_t>(heads) * context;
  constexpr std::uint64_t rotary_elements = static_cast<std::uint64_t>(context) * dimension;
  constexpr std::uint64_t compact_rotary_elements = static_cast<std::uint64_t>(context) * 64;
  constexpr std::uint64_t partial_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group * dimension;
  constexpr std::uint64_t statistic_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group;
  Allocation query(2ULL * query_elements), raw(2ULL * cache_elements),
      inverse_rms(sizeof(float) * inverse_elements), key_norm(2ULL * dimension),
      cosine(2ULL * rotary_elements), sine(2ULL * rotary_elements),
      compact_cosine(2ULL * compact_rotary_elements), compact_sine(2ULL * compact_rotary_elements),
      key(2ULL * cache_elements), value(2ULL * cache_elements), reference(2ULL * query_elements),
      candidate(2ULL * query_elements), partial(sizeof(float) * partial_elements),
      maxima(sizeof(float) * statistic_elements), sums(sizeof(float) * statistic_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x19f47ba2U, 0.12F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(raw.get()), cache_elements, 0x82ac731fU, 1.1F);
  initialize_inverse_rms<<<(inverse_elements + threads - 1) / threads, threads>>>(
      static_cast<float *>(inverse_rms.get()), inverse_elements);
  initialize_bf16_offset<<<(dimension + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key_norm.get()), dimension, 0x3d861ae7U, 1.0F, 0.2F);
  initialize_rope<<<(rotary_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(cosine.get()), static_cast<__nv_bfloat16 *>(sine.get()),
      rotary_elements, dimension);
  initialize_compact_global_rope<<<(compact_rotary_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(compact_cosine.get()),
      static_cast<__nv_bfloat16 *>(compact_sine.get()), context);
  materialize_raw_global_kv<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<const __nv_bfloat16 *>(raw.get()), static_cast<const float *>(inverse_rms.get()),
      static_cast<const __nv_bfloat16 *>(key_norm.get()),
      static_cast<const __nv_bfloat16 *>(cosine.get()),
      static_cast<const __nv_bfloat16 *>(sine.get()), static_cast<__nv_bfloat16 *>(key.get()),
      static_cast<__nv_bfloat16 *>(value.get()), cache_elements, context, dimension);
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  attention.run(query.get(), key.get(), value.get(), reference.get(), batch, kv_heads, query_group,
                context, dimension, nullptr);
  carat::grouped_global_attention_wgmma_raw_rs_split_bf16(
      query.get(), raw.get(), static_cast<const float *>(inverse_rms.get()), key_norm.get(),
      compact_cosine.get(), compact_sine.get(), candidate.get(),
      static_cast<float *>(partial.get()), static_cast<float *>(maxima.get()),
      static_cast<float *>(sums.get()), batch, kv_heads, query_group, context, dimension,
      tiles_per_segment, nullptr);
  check(cudaDeviceSynchronize(), "synchronize split RS-raw WGMMA parity");
  std::vector<__nv_bfloat16> expected(query_elements), actual(query_elements);
  check(cudaMemcpy(expected.data(), reference.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy split RS-raw WGMMA reference");
  check(cudaMemcpy(actual.data(), candidate.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy split RS-raw WGMMA candidate");
  float maximum = 0.0F;
  double mean = 0.0;
  for (std::uint64_t index = 0; index < query_elements; ++index) {
    const float difference =
        std::abs(__bfloat162float(expected[index]) - __bfloat162float(actual[index]));
    maximum = std::max(maximum, difference);
    mean += difference;
  }
  std::cout << "wgmma_rs_raw_split_parity_max_abs=" << maximum << '\n'
            << "wgmma_rs_raw_split_parity_mean_abs=" << mean / query_elements << '\n';
  if (maximum > 0.03F) {
    throw std::runtime_error("split RS-raw WGMMA parity exceeded tolerance");
  }
}

float benchmark_raw_rs_attention_tile(int batch) {
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 64;
  constexpr int dimension = 512;
  const std::uint64_t heads = static_cast<std::uint64_t>(batch) * kv_heads;
  const std::uint64_t query_elements = heads * query_group * dimension;
  const std::uint64_t cache_elements = heads * context * dimension;
  const std::uint64_t statistic_elements = heads * context;
  const std::uint64_t rotary_elements = context * dimension;
  Allocation query(2ULL * query_elements), raw(2ULL * cache_elements),
      inverse_rms(sizeof(float) * statistic_elements), key_norm(2ULL * dimension),
      cosine(2ULL * rotary_elements), sine(2ULL * rotary_elements), output(2ULL * query_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x6a75b24cU, 0.12F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(raw.get()), cache_elements, 0x2874dc51U, 1.1F);
  initialize_inverse_rms<<<(statistic_elements + threads - 1) / threads, threads>>>(
      static_cast<float *>(inverse_rms.get()), statistic_elements);
  initialize_bf16_offset<<<(dimension + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key_norm.get()), dimension, 0x913acd75U, 1.0F, 0.2F);
  initialize_rope<<<(rotary_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(cosine.get()), static_cast<__nv_bfloat16 *>(sine.get()),
      rotary_elements, dimension);
  const auto run = [&] {
    carat::grouped_global_attention_wgmma_raw_rs_tile_bf16(
        query.get(), raw.get(), static_cast<const float *>(inverse_rms.get()), key_norm.get(),
        cosine.get(), sine.get(), output.get(), batch, kv_heads, query_group, context, dimension,
        nullptr);
  };
  for (int iteration = 0; iteration < 4; ++iteration)
    run();
  check(cudaDeviceSynchronize(), "raw WGMMA attention tile warmup");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create raw WGMMA tile begin event");
  check(cudaEventCreate(&end), "create raw WGMMA tile end event");
  constexpr int iterations = 20;
  check(cudaEventRecord(begin), "record raw WGMMA tile begin event");
  for (int iteration = 0; iteration < iterations; ++iteration)
    run();
  check(cudaEventRecord(end), "record raw WGMMA tile end event");
  check(cudaEventSynchronize(end), "synchronize raw WGMMA tile benchmark");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure raw WGMMA tile benchmark");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

void wgmma_split_attention_parity() {
  constexpr int batch = 2;
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int context = 128;
  constexpr int dimension = 512;
  constexpr int tiles_per_segment = 1;
  constexpr int segments_per_head = context / 64 / tiles_per_segment;
  constexpr int heads = batch * kv_heads;
  constexpr std::uint64_t query_elements =
      static_cast<std::uint64_t>(heads) * query_group * dimension;
  constexpr std::uint64_t cache_elements = static_cast<std::uint64_t>(heads) * context * dimension;
  constexpr std::uint64_t partial_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group * dimension;
  constexpr std::uint64_t statistic_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group;
  Allocation query(2ULL * query_elements), key(2ULL * cache_elements), value(2ULL * cache_elements),
      reference(2ULL * query_elements), candidate(2ULL * query_elements),
      partial(sizeof(float) * partial_elements), maxima(sizeof(float) * statistic_elements),
      sums(sizeof(float) * statistic_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x10203040U, 0.12F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key.get()), cache_elements, 0x50607080U, 0.08F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(value.get()), cache_elements, 0x90abcdefU, 1.1F);
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  attention.run(query.get(), key.get(), value.get(), reference.get(), batch, kv_heads, query_group,
                context, dimension, nullptr);
  carat::grouped_global_attention_wgmma_split_bf16(
      query.get(), key.get(), value.get(), candidate.get(), static_cast<float *>(partial.get()),
      static_cast<float *>(maxima.get()), static_cast<float *>(sums.get()), batch, kv_heads,
      query_group, context, dimension, tiles_per_segment, nullptr);
  check(cudaDeviceSynchronize(), "synchronize split WGMMA attention parity");
  std::vector<__nv_bfloat16> expected(query_elements), actual(query_elements);
  check(cudaMemcpy(expected.data(), reference.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy split WGMMA reference");
  check(cudaMemcpy(actual.data(), candidate.get(), 2ULL * query_elements, cudaMemcpyDeviceToHost),
        "copy split WGMMA candidate");
  float maximum = 0.0F;
  double mean = 0.0;
  for (std::uint64_t index = 0; index < query_elements; ++index) {
    const float difference =
        std::abs(__bfloat162float(expected[index]) - __bfloat162float(actual[index]));
    maximum = std::max(maximum, difference);
    mean += difference;
  }
  std::cout << "wgmma_split_attention_parity_max_abs=" << maximum << '\n'
            << "wgmma_split_attention_parity_mean_abs=" << mean / query_elements << '\n';
  if (maximum > 0.03F) {
    throw std::runtime_error("split WGMMA attention parity exceeded tolerance");
  }
}

float benchmark_split_attention(int batch, int context, int tiles_per_segment) {
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  const int heads = batch * kv_heads;
  const int segments_per_head = context / 64 / tiles_per_segment;
  const std::uint64_t query_elements = static_cast<std::uint64_t>(heads) * query_group * dimension;
  const std::uint64_t cache_elements = static_cast<std::uint64_t>(heads) * context * dimension;
  const std::uint64_t partial_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group * dimension;
  const std::uint64_t statistic_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group;
  Allocation query(2ULL * query_elements), key(2ULL * cache_elements), value(2ULL * cache_elements),
      output(2ULL * query_elements), partial(sizeof(float) * partial_elements),
      maxima(sizeof(float) * statistic_elements), sums(sizeof(float) * statistic_elements);
  const auto run = [&] {
    carat::grouped_global_attention_wgmma_split_bf16(
        query.get(), key.get(), value.get(), output.get(), static_cast<float *>(partial.get()),
        static_cast<float *>(maxima.get()), static_cast<float *>(sums.get()), batch, kv_heads,
        query_group, context, dimension, tiles_per_segment, nullptr);
  };
  for (int iteration = 0; iteration < 2; ++iteration)
    run();
  check(cudaDeviceSynchronize(), "split WGMMA attention warmup");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create split WGMMA begin event");
  check(cudaEventCreate(&end), "create split WGMMA end event");
  constexpr int iterations = 5;
  check(cudaEventRecord(begin), "record split WGMMA begin event");
  for (int iteration = 0; iteration < iterations; ++iteration)
    run();
  check(cudaEventRecord(end), "record split WGMMA end event");
  check(cudaEventSynchronize(end), "synchronize split WGMMA benchmark");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure split WGMMA benchmark");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

float benchmark_raw_rs_split_attention(int batch, int context, int tiles_per_segment) {
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  const int heads = batch * kv_heads;
  const int segments_per_head = context / 64 / tiles_per_segment;
  const std::uint64_t query_elements = static_cast<std::uint64_t>(heads) * query_group * dimension;
  const std::uint64_t cache_elements = static_cast<std::uint64_t>(heads) * context * dimension;
  const std::uint64_t inverse_elements = static_cast<std::uint64_t>(heads) * context;
  const std::uint64_t rotary_elements = static_cast<std::uint64_t>(context) * 64;
  const std::uint64_t partial_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group * dimension;
  const std::uint64_t statistic_elements =
      static_cast<std::uint64_t>(heads) * segments_per_head * query_group;
  Allocation query(2ULL * query_elements), raw(2ULL * cache_elements),
      inverse_rms(sizeof(float) * inverse_elements), key_norm(2ULL * dimension),
      cosine(2ULL * rotary_elements), sine(2ULL * rotary_elements), output(2ULL * query_elements),
      partial(sizeof(float) * partial_elements), maxima(sizeof(float) * statistic_elements),
      sums(sizeof(float) * statistic_elements);
  constexpr int threads = 256;
  initialize_bf16<<<(query_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(query.get()), query_elements, 0x471ca982U, 0.12F);
  initialize_bf16<<<(cache_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(raw.get()), cache_elements, 0xa2375bc1U, 1.1F);
  initialize_inverse_rms<<<(inverse_elements + threads - 1) / threads, threads>>>(
      static_cast<float *>(inverse_rms.get()), inverse_elements);
  initialize_bf16_offset<<<(dimension + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(key_norm.get()), dimension, 0x95ca3187U, 1.0F, 0.2F);
  initialize_compact_global_rope<<<(rotary_elements + threads - 1) / threads, threads>>>(
      static_cast<__nv_bfloat16 *>(cosine.get()), static_cast<__nv_bfloat16 *>(sine.get()),
      context);
  const auto run = [&] {
    carat::grouped_global_attention_wgmma_raw_rs_split_bf16(
        query.get(), raw.get(), static_cast<const float *>(inverse_rms.get()), key_norm.get(),
        cosine.get(), sine.get(), output.get(), static_cast<float *>(partial.get()),
        static_cast<float *>(maxima.get()), static_cast<float *>(sums.get()), batch, kv_heads,
        query_group, context, dimension, tiles_per_segment, nullptr);
  };
  for (int iteration = 0; iteration < 2; ++iteration)
    run();
  check(cudaDeviceSynchronize(), "split RS-raw WGMMA warmup");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create split RS-raw WGMMA begin event");
  check(cudaEventCreate(&end), "create split RS-raw WGMMA end event");
  constexpr int iterations = 5;
  check(cudaEventRecord(begin), "record split RS-raw WGMMA begin event");
  for (int iteration = 0; iteration < iterations; ++iteration)
    run();
  check(cudaEventRecord(end), "record split RS-raw WGMMA end event");
  check(cudaEventSynchronize(end), "synchronize split RS-raw WGMMA benchmark");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure split RS-raw WGMMA benchmark");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

float benchmark_qk(int batch, int context, bool wgmma, int tiles_per_block = 1) {
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  const std::uint64_t query_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * dimension;
  const std::uint64_t key_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context * dimension;
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * context;
  Allocation query(2ULL * query_elements), key(2ULL * key_elements), scores(2ULL * score_elements);
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  const auto run = [&] {
    if (wgmma) {
      carat::grouped_global_qk_wgmma_bf16(query.get(), key.get(), scores.get(), batch, kv_heads,
                                          query_group, context, dimension, nullptr,
                                          tiles_per_block);
    } else {
      attention.run_qk(query.get(), key.get(), scores.get(), batch, kv_heads, query_group, context,
                       dimension, nullptr);
    }
  };
  for (int iteration = 0; iteration < 4; ++iteration)
    run();
  check(cudaDeviceSynchronize(), "WGMMA QK warmup");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create WGMMA QK begin event");
  check(cudaEventCreate(&end), "create WGMMA QK end event");
  constexpr int iterations = 20;
  check(cudaEventRecord(begin), "record WGMMA QK begin event");
  for (int iteration = 0; iteration < iterations; ++iteration)
    run();
  check(cudaEventRecord(end), "record WGMMA QK end event");
  check(cudaEventSynchronize(end), "synchronize WGMMA QK benchmark");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure WGMMA QK benchmark");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

float benchmark_int8_block128_qk(int batch, int context, int tiles_per_block) {
  constexpr int kv_heads = 4;
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  const std::uint64_t query_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * dimension;
  const std::uint64_t key_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context * dimension;
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * context;
  Allocation bf16_query(2ULL * query_elements), bf16_key(2ULL * key_elements),
      int8_query(query_elements),
      int8_query_scales(sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads * query_group *
                        4ULL),
      int8_key(key_elements),
      int8_key_scales(sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads * context *
                      4ULL),
      scores(2ULL * score_elements);
  carat::quantize_bf16_blocks_symmetric_int8(
      bf16_query.get(), int8_query.get(), static_cast<float *>(int8_query_scales.get()),
      batch * kv_heads * query_group, dimension, 128, nullptr);
  carat::quantize_bf16_blocks_symmetric_int8(bf16_key.get(), int8_key.get(),
                                             static_cast<float *>(int8_key_scales.get()),
                                             batch * kv_heads * context, dimension, 128, nullptr);
  check(cudaDeviceSynchronize(), "INT8 block-128 QK quantization");
  const auto run = [&] {
    carat::grouped_global_qk_wgmma_int8_block128(
        int8_query.get(), static_cast<const float *>(int8_query_scales.get()), int8_key.get(),
        static_cast<const float *>(int8_key_scales.get()), scores.get(), batch, kv_heads,
        query_group, context, dimension, context, nullptr, tiles_per_block);
  };
  for (int iteration = 0; iteration < 4; ++iteration)
    run();
  check(cudaDeviceSynchronize(), "INT8 block-128 QK warmup");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create INT8 block-128 QK begin event");
  check(cudaEventCreate(&end), "create INT8 block-128 QK end event");
  constexpr int iterations = 20;
  check(cudaEventRecord(begin), "record INT8 block-128 QK begin event");
  for (int iteration = 0; iteration < iterations; ++iteration)
    run();
  check(cudaEventRecord(end), "record INT8 block-128 QK end event");
  check(cudaEventSynchronize(end), "synchronize INT8 block-128 QK benchmark");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure INT8 block-128 QK benchmark");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

void parity() {
  constexpr int context = 127;
  constexpr int dimension = 512;
  constexpr int group = 8;
  std::vector<__nv_bfloat16> query(static_cast<std::size_t>(group * dimension));
  std::vector<__nv_bfloat16> key(static_cast<std::size_t>(context * dimension));
  std::vector<__nv_bfloat16> value(static_cast<std::size_t>(context * dimension));
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = __float2bfloat16_rn(0.05F * std::sin(static_cast<float>(index) * 0.013F));
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = __float2bfloat16_rn(0.04F * std::cos(static_cast<float>(index) * 0.007F));
    value[index] = __float2bfloat16_rn(0.1F * std::sin(static_cast<float>(index) * 0.011F));
  }
  Allocation device_query(2ULL * query.size()), device_key(2ULL * key.size()),
      device_value(2ULL * value.size()), reference(2ULL * query.size()),
      candidate(2ULL * query.size());
  check(cudaMemcpy(device_query.get(), query.data(), 2ULL * query.size(), cudaMemcpyHostToDevice),
        "copy parity query");
  check(cudaMemcpy(device_key.get(), key.data(), 2ULL * key.size(), cudaMemcpyHostToDevice),
        "copy parity key");
  check(cudaMemcpy(device_value.get(), value.data(), 2ULL * value.size(), cudaMemcpyHostToDevice),
        "copy parity value");
  carat::GemmGroupedDecodeAttention baseline(1, 1, group, context);
  baseline.run(device_query.get(), device_key.get(), device_value.get(), reference.get(), 1, 1,
               group, context, dimension, nullptr);
  carat::grouped_decode_attention_wmma_bf16(device_query.get(), device_key.get(),
                                            device_value.get(), candidate.get(), 1, 1, group,
                                            context, dimension, nullptr);
  check(cudaDeviceSynchronize(), "synchronize WMMA parity");
  std::vector<__nv_bfloat16> expected(query.size()), actual(query.size());
  check(
      cudaMemcpy(expected.data(), reference.get(), 2ULL * expected.size(), cudaMemcpyDeviceToHost),
      "copy parity reference");
  check(cudaMemcpy(actual.data(), candidate.get(), 2ULL * actual.size(), cudaMemcpyDeviceToHost),
        "copy parity candidate");
  float maximum = 0.0F;
  double mean = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float difference =
        std::abs(__bfloat162float(expected[index]) - __bfloat162float(actual[index]));
    maximum = std::max(maximum, difference);
    mean += difference;
  }
  mean /= static_cast<double>(expected.size());
  std::cout << "wmma_parity_max_abs=" << maximum << '\n' << "wmma_parity_mean_abs=" << mean << '\n';
}

float benchmark(int batch, int kv_heads, int query_group, int context, int dimension,
                int implementation) {
  const std::uint64_t query_bytes = 2ULL * batch * kv_heads * query_group * dimension;
  const std::uint64_t cache_bytes = 2ULL * batch * kv_heads * context * dimension;
  const std::uint64_t query_elements = query_bytes / 2ULL;
  const std::uint64_t cache_elements = cache_bytes / 2ULL;
  Allocation query(query_bytes), key(cache_bytes), value(cache_bytes), output(query_bytes),
      fp8_query(query_elements), fp8_key(cache_elements), fp8_value(cache_elements),
      int8_key(cache_elements),
      int8_key_scales(sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads * context),
      int8_block_key(cache_elements),
      int8_block_key_scales(4ULL * sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads *
                            context),
      int8_block_value(cache_elements),
      int8_block_value_scales(4ULL * sizeof(float) * static_cast<std::uint64_t>(batch) * kv_heads *
                              context),
      int8_packed_value(cache_elements), positions(sizeof(int) * batch);
  if (implementation == 7) {
    carat::quantize_grouped_global_values_wgmma_fp8(value.get(), fp8_value.get(), batch, kv_heads,
                                                    context, dimension, nullptr);
  }
  if (implementation == 8) {
    carat::quantize_bf16_rows_symmetric_int8(key.get(), int8_key.get(),
                                             static_cast<float *>(int8_key_scales.get()),
                                             batch * kv_heads * context, dimension, nullptr);
  }
  if (implementation >= 9 && implementation <= 12) {
    carat::quantize_bf16_blocks_symmetric_int8(key.get(), int8_block_key.get(),
                                               static_cast<float *>(int8_block_key_scales.get()),
                                               batch * kv_heads * context, dimension, 128, nullptr);
  }
  if (implementation == 13) {
    carat::quantize_bf16_blocks_symmetric_int8(key.get(), int8_block_key.get(),
                                               static_cast<float *>(int8_block_key_scales.get()),
                                               batch * kv_heads * context, dimension, 128, nullptr);
    carat::quantize_grouped_global_values_wgmma_int8(
        value.get(), int8_packed_value.get(), static_cast<float *>(int8_block_value_scales.get()),
        batch, kv_heads, context, dimension, nullptr);
  }
  const std::vector<int> host_positions(batch, context - 1);
  check(cudaMemcpy(positions.get(), host_positions.data(), sizeof(int) * batch,
                   cudaMemcpyHostToDevice),
        "copy benchmark positions");
  carat::GemmGroupedDecodeAttention attention(batch, kv_heads, query_group, context);
  carat::CudnnPrefillAttention cudnn_attention;
  const auto run = [&] {
    if (implementation == 1) {
      carat::grouped_decode_attention_bf16(query.get(), key.get(), value.get(), output.get(), batch,
                                           kv_heads, query_group, context, dimension, nullptr);
    } else if (implementation == 2) {
      carat::grouped_decode_attention_wmma_bf16(query.get(), key.get(), value.get(), output.get(),
                                                batch, kv_heads, query_group, context, dimension,
                                                nullptr);
    } else if (implementation == 3) {
      attention.run_ragged_contiguous_fp8_keys(
          fp8_query.get(), fp8_key.get(), value.get(), output.get(),
          static_cast<const int *>(positions.get()), batch, kv_heads, query_group, context,
          dimension, 0, nullptr, context);
    } else if (implementation == 4) {
      cudnn_attention.run(query.get(), key.get(), value.get(), output.get(), batch,
                          kv_heads * query_group, kv_heads, 1, context, context, dimension, 0,
                          nullptr);
    } else if (implementation == 5) {
      attention.run_ragged_contiguous(query.get(), key.get(), value.get(), output.get(),
                                      static_cast<const int *>(positions.get()), batch, kv_heads,
                                      query_group, context, dimension, 0, nullptr, context);
    } else if (implementation == 6) {
      carat::grouped_global_attention_wgmma_tile_bf16(query.get(), key.get(), value.get(),
                                                      output.get(), batch, kv_heads, query_group,
                                                      context, dimension, nullptr);
    } else if (implementation == 7) {
      attention.run_ragged_contiguous_fp8_kv(
          fp8_query.get(), fp8_key.get(), fp8_value.get(), output.get(),
          static_cast<const int *>(positions.get()), batch, kv_heads, query_group, context,
          dimension, 0, nullptr, context);
    } else if (implementation == 8) {
      attention.run_ragged_contiguous_int8_keys(
          query.get(), int8_key.get(), static_cast<const float *>(int8_key_scales.get()),
          value.get(), output.get(), static_cast<const int *>(positions.get()), batch, kv_heads,
          query_group, context, dimension, 0, nullptr, context);
    } else if (implementation >= 9 && implementation <= 12) {
      attention.run_ragged_contiguous_int8_block128_keys(
          query.get(), int8_block_key.get(),
          static_cast<const float *>(int8_block_key_scales.get()), value.get(), output.get(),
          static_cast<const int *>(positions.get()), batch, kv_heads, query_group, context,
          dimension, 0, nullptr, context, 1 << (implementation - 9));
    } else if (implementation == 13) {
      attention.run_ragged_contiguous_int8_block128_kv(
          query.get(), int8_block_key.get(),
          static_cast<const float *>(int8_block_key_scales.get()), int8_packed_value.get(),
          static_cast<const float *>(int8_block_value_scales.get()), output.get(),
          static_cast<const int *>(positions.get()), batch, kv_heads, query_group, context,
          dimension, 0, nullptr, context, 0);
    } else {
      attention.run(query.get(), key.get(), value.get(), output.get(), batch, kv_heads, query_group,
                    context, dimension, nullptr);
    }
  };
  for (int iteration = 0; iteration < 2; ++iteration) {
    run();
  }
  check(cudaDeviceSynchronize(), "attention warmup");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create begin event");
  check(cudaEventCreate(&end), "create end event");
  constexpr int iterations = 5;
  check(cudaEventRecord(begin), "record begin event");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    run();
  }
  check(cudaEventRecord(end), "record end event");
  check(cudaEventSynchronize(end), "attention benchmark synchronization");
  float milliseconds = 0;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "attention elapsed time");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--int8-qk-only") {
      const int tiles_per_block = argc >= 3 ? std::stoi(argv[2]) : 1;
      const float milliseconds = benchmark_int8_block128_qk(16, 8192, tiles_per_block);
      std::cout << "int8_block128_qk_only," << tiles_per_block << ',' << std::fixed
                << std::setprecision(4) << milliseconds << "\n";
      return 0;
    }
    parity();
    fp8_parity();
    int8_block_qk_contract_parity();
    wgmma_qk_parity();
    wgmma_attention_tile_parity();
    wgmma_raw_attention_tile_parity();
    wgmma_raw_rs_split_attention_parity();
    wgmma_split_attention_parity();
    std::cout << "kind,batch,context,milliseconds,physical_kv_gb_s,naive_avoided_gb\n";
    for (const int batch : std::vector<int>{1, 2, 4, 8, 10, 16}) {
      const float sliding_ms = benchmark(batch, 16, 2, 1024, 256, 0);
      const float sliding_fused_ms = benchmark(batch, 16, 2, 1024, 256, 1);
      const double sliding_bytes = 4.0 * batch * 16 * 1024 * 256;
      std::cout << "sliding," << batch << ",1024," << std::fixed << std::setprecision(4)
                << sliding_ms << ',' << std::setprecision(2)
                << sliding_bytes / (sliding_ms / 1000.0) / 1e9 << ',' << sliding_bytes / 1e9
                << '\n';
      std::cout << "sliding_fused," << batch << ",1024," << std::fixed << std::setprecision(4)
                << sliding_fused_ms << ',' << std::setprecision(2)
                << sliding_bytes / (sliding_fused_ms / 1000.0) / 1e9 << ',' << sliding_bytes / 1e9
                << '\n';
      const float global_ms = benchmark(batch, 4, 8, 8192, 512, 0);
      const float global_fused_ms = benchmark(batch, 4, 8, 8192, 512, 1);
      const float global_wmma_ms = benchmark(batch, 4, 8, 8192, 512, 2);
      const float global_fp8_key_ms = benchmark(batch, 4, 8, 8192, 512, 3);
      const float global_fp8_kv_ms = benchmark(batch, 4, 8, 8192, 512, 7);
      const float global_int8_key_ms = benchmark(batch, 4, 8, 8192, 512, 8);
      const float global_int8_block128_key_ms = benchmark(batch, 4, 8, 8192, 512, 9);
      const float global_int8_block128_key_t2_ms = benchmark(batch, 4, 8, 8192, 512, 10);
      const float global_int8_block128_key_t4_ms = benchmark(batch, 4, 8, 8192, 512, 11);
      const float global_int8_block128_key_t8_ms = benchmark(batch, 4, 8, 8192, 512, 12);
      const float global_int8_block128_kv_ms = benchmark(batch, 4, 8, 8192, 512, 13);
      const float global_cudnn_ms = benchmark(batch, 4, 8, 8192, 512, 4);
      const float global_ragged_ms = benchmark(batch, 4, 8, 8192, 512, 5);
      const float global_tile_baseline_ms = benchmark(batch, 4, 8, 64, 512, 0);
      const float global_tile_wgmma_ms = benchmark(batch, 4, 8, 64, 512, 6);
      const float global_tile_rs_raw_wgmma = benchmark_raw_rs_attention_tile(batch);
      std::vector<std::pair<int, float>> global_split_wgmma;
      std::vector<std::pair<int, float>> global_rs_raw_split_wgmma;
      for (const int tiles_per_segment : std::vector<int>{4, 8, 16, 32}) {
        global_split_wgmma.emplace_back(tiles_per_segment,
                                        benchmark_split_attention(batch, 8192, tiles_per_segment));
        global_rs_raw_split_wgmma.emplace_back(
            tiles_per_segment, benchmark_raw_rs_split_attention(batch, 8192, tiles_per_segment));
      }
      const float global_qk_ms = benchmark_qk(batch, 8192, false);
      std::vector<std::pair<int, float>> global_qk_wgmma;
      for (const int tiles_per_block : std::vector<int>{1, 2, 4, 8, 16, 32}) {
        global_qk_wgmma.emplace_back(tiles_per_block,
                                     benchmark_qk(batch, 8192, true, tiles_per_block));
      }
      const double global_bytes = 4.0 * batch * 4 * 8192 * 512;
      std::cout << "global," << batch << ",8192," << std::fixed << std::setprecision(4) << global_ms
                << ',' << std::setprecision(2) << global_bytes / (global_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_fused," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_fused_ms << ',' << std::setprecision(2)
                << global_bytes / (global_fused_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_wmma," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_wmma_ms << ',' << std::setprecision(2)
                << global_bytes / (global_wmma_ms / 1000.0) / 1e9 << ',' << 7.0 * global_bytes / 1e9
                << '\n';
      std::cout << "global_fp8_key," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_fp8_key_ms << ',' << std::setprecision(2)
                << (global_bytes * 0.75) / (global_fp8_key_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_fp8_kv," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_fp8_kv_ms << ',' << std::setprecision(2)
                << (global_bytes * 0.5) / (global_fp8_kv_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_int8_key," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_int8_key_ms << ',' << std::setprecision(2)
                << (global_bytes * 0.75) / (global_int8_key_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_int8_block128_key," << batch << ",8192," << std::fixed
                << std::setprecision(4) << global_int8_block128_key_ms << ','
                << std::setprecision(2)
                << (global_bytes * 0.75) / (global_int8_block128_key_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      for (const auto &[tiles, milliseconds] :
           std::vector<std::pair<int, float>>{{2, global_int8_block128_key_t2_ms},
                                              {4, global_int8_block128_key_t4_ms},
                                              {8, global_int8_block128_key_t8_ms}}) {
        std::cout << "global_int8_block128_key_t" << tiles << ',' << batch << ",8192," << std::fixed
                  << std::setprecision(4) << milliseconds << ',' << std::setprecision(2)
                  << (global_bytes * 0.75) / (milliseconds / 1000.0) / 1e9 << ','
                  << 7.0 * global_bytes / 1e9 << '\n';
      }
      std::cout << "global_int8_block128_kv," << batch << ",8192," << std::fixed
                << std::setprecision(4) << global_int8_block128_kv_ms << ',' << std::setprecision(2)
                << (global_bytes * 0.5) / (global_int8_block128_kv_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_cudnn," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_cudnn_ms << ',' << std::setprecision(2)
                << global_bytes / (global_cudnn_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_ragged," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_ragged_ms << ',' << std::setprecision(2)
                << global_bytes / (global_ragged_ms / 1000.0) / 1e9 << ','
                << 7.0 * global_bytes / 1e9 << '\n';
      std::cout << "global_tile_baseline," << batch << ",64," << std::fixed << std::setprecision(4)
                << global_tile_baseline_ms << ",0,0\n";
      std::cout << "global_tile_wgmma," << batch << ",64," << std::fixed << std::setprecision(4)
                << global_tile_wgmma_ms << ",0,0\n";
      std::cout << "global_tile_rs_raw_wgmma," << batch << ",64," << std::fixed
                << std::setprecision(4) << global_tile_rs_raw_wgmma << ",0,0\n";
      for (const auto &[tiles_per_segment, milliseconds] : global_split_wgmma) {
        std::cout << "global_split_wgmma_t" << tiles_per_segment << ',' << batch << ",8192,"
                  << std::fixed << std::setprecision(4) << milliseconds << ",0,0\n";
      }
      for (const auto &[tiles_per_segment, milliseconds] : global_rs_raw_split_wgmma) {
        std::cout << "global_rs_raw_split_wgmma_t" << tiles_per_segment << ',' << batch << ",8192,"
                  << std::fixed << std::setprecision(4) << milliseconds << ",0,0\n";
      }
      const double global_key_bytes = global_bytes * 0.5;
      std::cout << "global_qk," << batch << ",8192," << std::fixed << std::setprecision(4)
                << global_qk_ms << ',' << std::setprecision(2)
                << global_key_bytes / (global_qk_ms / 1000.0) / 1e9 << ",0\n";
      for (const auto &[tiles_per_block, milliseconds] : global_qk_wgmma) {
        std::cout << "global_qk_wgmma_t" << tiles_per_block << ',' << batch << ",8192,"
                  << std::fixed << std::setprecision(4) << milliseconds << ','
                  << std::setprecision(2) << global_key_bytes / (milliseconds / 1000.0) / 1e9
                  << ",0\n";
      }
      const double total_ms = 50.0 * sliding_ms + 10.0 * global_ms;
      const double fused_total_ms = 50.0 * sliding_fused_ms + 10.0 * global_fused_ms;
      std::cout << "all_layers," << batch << ",8192," << std::fixed << std::setprecision(4)
                << total_ms << ",0,0\n";
      std::cout << "all_layers_fused," << batch << ",8192," << std::fixed << std::setprecision(4)
                << fused_total_ms << ",0,0\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "attention benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
