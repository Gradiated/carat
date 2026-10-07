#include "carat/qkv.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace carat {
namespace {

constexpr int threads = 256;

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

__device__ float block_sum(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  __shared__ float warp_sums[32];
  if ((threadIdx.x & 31U) == 0U)
    warp_sums[threadIdx.x >> 5U] = value;
  __syncthreads();
  value = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_sums[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32U) {
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
  }
  if (threadIdx.x == 0U)
    warp_sums[0] = value;
  __syncthreads();
  return warp_sums[0];
}

template <int dimension, bool proportional>
__global__ void rope_cache_kernel(__nv_bfloat16 *cosine, __nv_bfloat16 *sine, int positions) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= positions * dimension)
    return;
  const int position = index / dimension;
  const int half_index = index % (dimension / 2);
  float frequency = 0.0F;
  if constexpr (proportional) {
    if (half_index < dimension / 8) {
      frequency = powf(1000000.0F, -2.0F * static_cast<float>(half_index) / dimension);
    }
  } else {
    frequency = powf(10000.0F, -2.0F * static_cast<float>(half_index) / dimension);
  }
  float sin_value = 0.0F;
  float cos_value = 0.0F;
  sincosf(static_cast<float>(position) * frequency, &sin_value, &cos_value);
  cosine[index] = __float2bfloat16_rn(cos_value);
  sine[index] = __float2bfloat16_rn(sin_value);
}

template <int dimension, bool batched>
__global__ void query_kernel(const __nv_bfloat16 *packed, const __nv_bfloat16 *norm_weight,
                             const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                             __nv_bfloat16 *output, int tokens, int query_heads, int packed_width,
                             int position_start, float epsilon) {
  const int token = static_cast<int>(blockIdx.x) / query_heads;
  const int head = static_cast<int>(blockIdx.x) % query_heads;
  const __nv_bfloat16 *input =
      packed + static_cast<std::uint64_t>(token) * packed_width + head * dimension;
  float square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float value = __bfloat162float(input[index]);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum) / dimension + epsilon);
  const int position = position_start + (batched ? 0 : token);
  const int half = dimension / 2;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse_rms * __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[partner]) * inverse_rms * __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    const float cos_value = __bfloat162float(cosine[position * dimension + index]);
    const float sin_value = __bfloat162float(sine[position * dimension + index]);
    output[(static_cast<std::uint64_t>(token) * query_heads + head) * dimension + index] =
        __float2bfloat16_rn(normalized * cos_value + rotated * sin_value);
  }
}

template <int dimension, bool key_equals_value, bool batched>
__global__ void key_value_kernel(const __nv_bfloat16 *packed, const __nv_bfloat16 *norm_weight,
                                 const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                                 __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache, int tokens,
                                 int query_width, int kv_heads, int packed_width,
                                 int position_start, int cache_position_start, int cache_capacity,
                                 float epsilon) {
  const int token = static_cast<int>(blockIdx.x) / kv_heads;
  const int head = static_cast<int>(blockIdx.x) % kv_heads;
  const __nv_bfloat16 *key_input =
      packed + static_cast<std::uint64_t>(token) * packed_width + query_width + head * dimension;
  const __nv_bfloat16 *value_input =
      key_equals_value ? key_input
                       : packed + static_cast<std::uint64_t>(token) * packed_width + query_width +
                             kv_heads * dimension + head * dimension;
  float key_square_sum = 0.0F;
  float value_square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float key = __bfloat162float(key_input[index]);
    const float value = __bfloat162float(value_input[index]);
    key_square_sum += key * key;
    value_square_sum += value * value;
  }
  const float inverse_key_rms = rsqrtf(block_sum(key_square_sum) / dimension + epsilon);
  const float inverse_value_rms = key_equals_value
                                      ? inverse_key_rms
                                      : rsqrtf(block_sum(value_square_sum) / dimension + epsilon);
  const int position = position_start + (batched ? 0 : token);
  const int cache_position = batched ? position % cache_capacity : cache_position_start + token;
  const int half = dimension / 2;
  const std::uint64_t cache_head =
      (batched ? static_cast<std::uint64_t>(token) * kv_heads : 0ULL) + head;
  const std::uint64_t cache_base = (cache_head * cache_capacity + cache_position) * dimension;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized =
        __bfloat162float(__float2bfloat16_rn(__bfloat162float(key_input[index]) * inverse_key_rms *
                                             __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(
        __float2bfloat16_rn(__bfloat162float(key_input[partner]) * inverse_key_rms *
                            __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    const float cos_value = __bfloat162float(cosine[position * dimension + index]);
    const float sin_value = __bfloat162float(sine[position * dimension + index]);
    key_cache[cache_base + index] =
        __float2bfloat16_rn(normalized * cos_value + rotated * sin_value);
    value_cache[cache_base + index] =
        __float2bfloat16_rn(__bfloat162float(value_input[index]) * inverse_value_rms);
  }
}

template <int dimension>
__global__ void encode_shared_global_kv_kernel(const __nv_bfloat16 *packed,
                                               __nv_bfloat16 *raw_cache, float *inverse_rms_cache,
                                               int tokens, int query_width, int kv_heads,
                                               int packed_width, float epsilon) {
  const int token = static_cast<int>(blockIdx.x) / kv_heads;
  const int head = static_cast<int>(blockIdx.x) % kv_heads;
  if (token >= tokens)
    return;
  const __nv_bfloat16 *input =
      packed + static_cast<std::uint64_t>(token) * packed_width + query_width + head * dimension;
  float square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const __nv_bfloat16 value = input[index];
    raw_cache[(static_cast<std::uint64_t>(token) * kv_heads + head) * dimension + index] = value;
    const float fp32 = __bfloat162float(value);
    square_sum += fp32 * fp32;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum) / dimension + epsilon);
  if (threadIdx.x == 0) {
    inverse_rms_cache[static_cast<std::uint64_t>(token) * kv_heads + head] = inverse_rms;
  }
}

template <int dimension, bool batched>
__global__ void materialize_shared_global_kv_kernel(
    const __nv_bfloat16 *raw_cache, const float *inverse_rms_cache,
    const __nv_bfloat16 *norm_weight, const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
    __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache, int tokens, int kv_heads,
    int position_start, int cache_position_start, int cache_capacity) {
  const int token = static_cast<int>(blockIdx.x) / kv_heads;
  const int head = static_cast<int>(blockIdx.x) % kv_heads;
  if (token >= tokens)
    return;
  const std::uint64_t staged_head = static_cast<std::uint64_t>(token) * kv_heads + head;
  const __nv_bfloat16 *raw = raw_cache + staged_head * dimension;
  const float inverse_rms = inverse_rms_cache[staged_head];
  const int position = position_start + (batched ? 0 : token);
  const int cache_position = batched ? position % cache_capacity : cache_position_start + token;
  const std::uint64_t cache_head =
      (batched ? static_cast<std::uint64_t>(token) * kv_heads : 0ULL) + head;
  const std::uint64_t cache_base = (cache_head * cache_capacity + cache_position) * dimension;
  const int half = dimension / 2;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(raw[index]) * inverse_rms * __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(raw[partner]) * inverse_rms * __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    key_cache[cache_base + index] =
        __float2bfloat16_rn(normalized * __bfloat162float(cosine[position * dimension + index]) +
                            rotated * __bfloat162float(sine[position * dimension + index]));
    value_cache[cache_base + index] =
        __float2bfloat16_rn(__bfloat162float(raw[index]) * inverse_rms);
  }
}

template <int dimension>
__global__ void ragged_query_kernel(const __nv_bfloat16 *packed, const __nv_bfloat16 *norm_weight,
                                    const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                                    __nv_bfloat16 *output, const int *positions, int batch,
                                    int query_heads, int packed_width, float epsilon) {
  const int row = static_cast<int>(blockIdx.x) / query_heads;
  const int head = static_cast<int>(blockIdx.x) % query_heads;
  if (row >= batch)
    return;
  const __nv_bfloat16 *input =
      packed + static_cast<std::uint64_t>(row) * packed_width + head * dimension;
  float square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float value = __bfloat162float(input[index]);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum) / dimension + epsilon);
  const int position = positions[row];
  const int half = dimension / 2;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse_rms * __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[partner]) * inverse_rms * __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    const float cos_value = __bfloat162float(cosine[position * dimension + index]);
    const float sin_value = __bfloat162float(sine[position * dimension + index]);
    output[(static_cast<std::uint64_t>(row) * query_heads + head) * dimension + index] =
        __float2bfloat16_rn(normalized * cos_value + rotated * sin_value);
  }
}

template <int dimension, bool key_equals_value>
__global__ void
ragged_key_value_kernel(const __nv_bfloat16 *packed, const __nv_bfloat16 *norm_weight,
                        const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                        __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache, const int *positions,
                        const int *slots, int batch, int maximum_slots, int query_width,
                        int kv_heads, int packed_width, int cache_capacity,
                        std::int8_t *int8_key_cache, float *int8_key_scales, float epsilon) {
  const int row = static_cast<int>(blockIdx.x) / kv_heads;
  const int head = static_cast<int>(blockIdx.x) % kv_heads;
  if (row >= batch)
    return;
  const int position = positions[row];
  const int slot = slots[row];
  if (position < 0 || slot < 0 || slot >= maximum_slots)
    return;
  const __nv_bfloat16 *key_input =
      packed + static_cast<std::uint64_t>(row) * packed_width + query_width + head * dimension;
  const __nv_bfloat16 *value_input =
      key_equals_value ? key_input
                       : packed + static_cast<std::uint64_t>(row) * packed_width + query_width +
                             kv_heads * dimension + head * dimension;
  float key_square_sum = 0.0F;
  float value_square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float key = __bfloat162float(key_input[index]);
    const float value = __bfloat162float(value_input[index]);
    key_square_sum += key * key;
    value_square_sum += value * value;
  }
  const float inverse_key_rms = rsqrtf(block_sum(key_square_sum) / dimension + epsilon);
  const float inverse_value_rms = key_equals_value
                                      ? inverse_key_rms
                                      : rsqrtf(block_sum(value_square_sum) / dimension + epsilon);
  const int cache_position = position % cache_capacity;
  const int half = dimension / 2;
  const std::uint64_t cache_head = static_cast<std::uint64_t>(slot) * kv_heads + head;
  const std::uint64_t cache_base = (cache_head * cache_capacity + cache_position) * dimension;
  if constexpr (dimension == 512) {
    if (int8_key_cache != nullptr) {
      constexpr int block_width = 128;
      constexpr int blocks_per_row = dimension / block_width;
      __shared__ float warp_maxima[8];
      __shared__ float block_maxima[blocks_per_row];
      const int lane_in_block = static_cast<int>(threadIdx.x) % block_width;
      const int block_group = static_cast<int>(threadIdx.x) / block_width;
      const int warp = static_cast<int>(threadIdx.x) / 32;
      for (int block_pair = 0; block_pair < 2; ++block_pair) {
        const int component_block = block_pair * 2 + block_group;
        const int index = component_block * block_width + lane_in_block;
        const int partner = index < half ? index + half : index - half;
        const float normalized = __bfloat162float(
            __float2bfloat16_rn(__bfloat162float(key_input[index]) * inverse_key_rms *
                                __bfloat162float(norm_weight[index])));
        const float partner_normalized = __bfloat162float(
            __float2bfloat16_rn(__bfloat162float(key_input[partner]) * inverse_key_rms *
                                __bfloat162float(norm_weight[partner])));
        const float rotated = index < half ? -partner_normalized : partner_normalized;
        const float cos_value = __bfloat162float(cosine[position * dimension + index]);
        const float sin_value = __bfloat162float(sine[position * dimension + index]);
        const __nv_bfloat16 key = __float2bfloat16_rn(normalized * cos_value + rotated * sin_value);
        key_cache[cache_base + index] = key;
        value_cache[cache_base + index] =
            __float2bfloat16_rn(__bfloat162float(value_input[index]) * inverse_value_rms);

        float maximum = fabsf(__bfloat162float(key));
        for (unsigned offset = 16; offset > 0; offset >>= 1U) {
          maximum = max(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        }
        if ((threadIdx.x & 31U) == 0U)
          warp_maxima[warp] = maximum;
        __syncthreads();
        if (lane_in_block < 32) {
          const int first_warp = block_group * 4;
          maximum = lane_in_block < 4 ? warp_maxima[first_warp + lane_in_block] : 0.0F;
          for (unsigned offset = 16; offset > 0; offset >>= 1U) {
            maximum = max(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
          }
          if ((threadIdx.x & 31U) == 0U)
            block_maxima[component_block] = maximum;
        }
        __syncthreads();
        const float maximum_block = block_maxima[component_block];
        const float scale = maximum_block == 0.0F ? 1.0F : maximum_block / 127.0F;
        if (lane_in_block == 0) {
          int8_key_scales[(cache_head * cache_capacity + cache_position) * blocks_per_row +
                          component_block] = scale;
        }
        const int quantized = max(-127, min(127, __float2int_rn(__bfloat162float(key) / scale)));
        int8_key_cache[cache_base + index] = static_cast<std::int8_t>(quantized);
      }
      return;
    }
  }
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized =
        __bfloat162float(__float2bfloat16_rn(__bfloat162float(key_input[index]) * inverse_key_rms *
                                             __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(
        __float2bfloat16_rn(__bfloat162float(key_input[partner]) * inverse_key_rms *
                            __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    const float cos_value = __bfloat162float(cosine[position * dimension + index]);
    const float sin_value = __bfloat162float(sine[position * dimension + index]);
    key_cache[cache_base + index] =
        __float2bfloat16_rn(normalized * cos_value + rotated * sin_value);
    value_cache[cache_base + index] =
        __float2bfloat16_rn(__bfloat162float(value_input[index]) * inverse_value_rms);
  }
}

template <int dimension>
__global__ void materialize_shared_global_kv_ragged_kernel(
    const __nv_bfloat16 *raw_cache, const float *inverse_rms_cache,
    const __nv_bfloat16 *norm_weight, const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
    __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache, const int *positions, const int *slots,
    int batch, int maximum_slots, int kv_heads, int cache_capacity) {
  const int row = static_cast<int>(blockIdx.x) / kv_heads;
  const int head = static_cast<int>(blockIdx.x) % kv_heads;
  if (row >= batch)
    return;
  const int position = positions[row];
  const int slot = slots[row];
  if (position < 0 || slot < 0 || slot >= maximum_slots)
    return;
  const std::uint64_t staged_head = static_cast<std::uint64_t>(row) * kv_heads + head;
  const __nv_bfloat16 *raw = raw_cache + staged_head * dimension;
  const float inverse_rms = inverse_rms_cache[staged_head];
  const int cache_position = position % cache_capacity;
  const std::uint64_t cache_head = static_cast<std::uint64_t>(slot) * kv_heads + head;
  const std::uint64_t cache_base = (cache_head * cache_capacity + cache_position) * dimension;
  const int half = dimension / 2;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(raw[index]) * inverse_rms * __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(raw[partner]) * inverse_rms * __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    key_cache[cache_base + index] =
        __float2bfloat16_rn(normalized * __bfloat162float(cosine[position * dimension + index]) +
                            rotated * __bfloat162float(sine[position * dimension + index]));
    value_cache[cache_base + index] =
        __float2bfloat16_rn(__bfloat162float(raw[index]) * inverse_rms);
  }
}

template <int dimension>
__global__ void uniform_prefill_query_kernel(const __nv_bfloat16 *packed,
                                             const __nv_bfloat16 *norm_weight,
                                             const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine,
                                             __nv_bfloat16 *output, const int *position_starts,
                                             int requests, int tokens_per_request, int query_heads,
                                             int packed_width, float epsilon) {
  const int packed_head = static_cast<int>(blockIdx.x);
  const int row = packed_head / query_heads;
  const int head = packed_head % query_heads;
  const int request = row / tokens_per_request;
  const int token = row % tokens_per_request;
  if (request >= requests)
    return;
  const __nv_bfloat16 *input =
      packed + static_cast<std::uint64_t>(row) * packed_width + head * dimension;
  float square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float value = __bfloat162float(input[index]);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum) / dimension + epsilon);
  const int position = position_starts[request] + token;
  const int half = dimension / 2;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse_rms * __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(input[partner]) * inverse_rms * __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    output[(static_cast<std::uint64_t>(row) * query_heads + head) * dimension + index] =
        __float2bfloat16_rn(normalized * __bfloat162float(cosine[position * dimension + index]) +
                            rotated * __bfloat162float(sine[position * dimension + index]));
  }
}

template <int dimension, bool key_equals_value>
__global__ void uniform_prefill_key_value_kernel(
    const __nv_bfloat16 *packed, const __nv_bfloat16 *norm_weight, const __nv_bfloat16 *cosine,
    const __nv_bfloat16 *sine, __nv_bfloat16 *key_cache, __nv_bfloat16 *value_cache,
    const int *position_starts, int requests, int tokens_per_request, int query_width, int kv_heads,
    int packed_width, int cache_position_start, int cache_capacity, float epsilon) {
  const int packed_head = static_cast<int>(blockIdx.x);
  const int row = packed_head / kv_heads;
  const int head = packed_head % kv_heads;
  const int request = row / tokens_per_request;
  const int token = row % tokens_per_request;
  if (request >= requests)
    return;
  const __nv_bfloat16 *key_input =
      packed + static_cast<std::uint64_t>(row) * packed_width + query_width + head * dimension;
  const __nv_bfloat16 *value_input =
      key_equals_value ? key_input
                       : packed + static_cast<std::uint64_t>(row) * packed_width + query_width +
                             kv_heads * dimension + head * dimension;
  float key_square_sum = 0.0F;
  float value_square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float key = __bfloat162float(key_input[index]);
    const float value = __bfloat162float(value_input[index]);
    key_square_sum += key * key;
    value_square_sum += value * value;
  }
  const float inverse_key_rms = rsqrtf(block_sum(key_square_sum) / dimension + epsilon);
  const float inverse_value_rms = key_equals_value
                                      ? inverse_key_rms
                                      : rsqrtf(block_sum(value_square_sum) / dimension + epsilon);
  const int position = position_starts[request] + token;
  const int cache_position = cache_position_start + token;
  const int half = dimension / 2;
  const std::uint64_t cache_head = static_cast<std::uint64_t>(request) * kv_heads + head;
  const std::uint64_t cache_base = (cache_head * cache_capacity + cache_position) * dimension;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized =
        __bfloat162float(__float2bfloat16_rn(__bfloat162float(key_input[index]) * inverse_key_rms *
                                             __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(
        __float2bfloat16_rn(__bfloat162float(key_input[partner]) * inverse_key_rms *
                            __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    key_cache[cache_base + index] =
        __float2bfloat16_rn(normalized * __bfloat162float(cosine[position * dimension + index]) +
                            rotated * __bfloat162float(sine[position * dimension + index]));
    value_cache[cache_base + index] =
        __float2bfloat16_rn(__bfloat162float(value_input[index]) * inverse_value_rms);
  }
}

template <int dimension, bool key_equals_value>
__global__ void uniform_prefill_direct_key_value_kernel(
    const __nv_bfloat16 *packed, const __nv_bfloat16 *norm_weight, const __nv_bfloat16 *cosine,
    const __nv_bfloat16 *sine, __nv_bfloat16 *const *key_caches, __nv_bfloat16 *const *value_caches,
    const int *position_starts, int requests, int tokens_per_request, int query_width, int kv_heads,
    int packed_width, int cache_capacity, float epsilon) {
  const int packed_head = static_cast<int>(blockIdx.x);
  const int row = packed_head / kv_heads;
  const int head = packed_head % kv_heads;
  const int request = row / tokens_per_request;
  const int token = row % tokens_per_request;
  if (request >= requests)
    return;
  const __nv_bfloat16 *key_input =
      packed + static_cast<std::uint64_t>(row) * packed_width + query_width + head * dimension;
  const __nv_bfloat16 *value_input =
      key_equals_value ? key_input
                       : packed + static_cast<std::uint64_t>(row) * packed_width + query_width +
                             kv_heads * dimension + head * dimension;
  float key_square_sum = 0.0F;
  float value_square_sum = 0.0F;
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const float key = __bfloat162float(key_input[index]);
    const float value = __bfloat162float(value_input[index]);
    key_square_sum += key * key;
    value_square_sum += value * value;
  }
  const float inverse_key_rms = rsqrtf(block_sum(key_square_sum) / dimension + epsilon);
  const float inverse_value_rms = key_equals_value
                                      ? inverse_key_rms
                                      : rsqrtf(block_sum(value_square_sum) / dimension + epsilon);
  const int position = position_starts[request] + token;
  const int cache_position = position % cache_capacity;
  const int half = dimension / 2;
  const std::uint64_t cache_base =
      (static_cast<std::uint64_t>(head) * cache_capacity + cache_position) * dimension;
  __nv_bfloat16 *key_cache = key_caches[request];
  __nv_bfloat16 *value_cache = value_caches[request];
  for (int index = threadIdx.x; index < dimension; index += threads) {
    const int partner = index < half ? index + half : index - half;
    const float normalized =
        __bfloat162float(__float2bfloat16_rn(__bfloat162float(key_input[index]) * inverse_key_rms *
                                             __bfloat162float(norm_weight[index])));
    const float partner_normalized = __bfloat162float(
        __float2bfloat16_rn(__bfloat162float(key_input[partner]) * inverse_key_rms *
                            __bfloat162float(norm_weight[partner])));
    const float rotated = index < half ? -partner_normalized : partner_normalized;
    key_cache[cache_base + index] =
        __float2bfloat16_rn(normalized * __bfloat162float(cosine[position * dimension + index]) +
                            rotated * __bfloat162float(sine[position * dimension + index]));
    value_cache[cache_base + index] =
        __float2bfloat16_rn(__bfloat162float(value_input[index]) * inverse_value_rms);
  }
}

} // namespace

struct Gemma4QkvPostprocessor::Implementation {
  int maximum_positions;
  bool shared_global_kv_roundtrip;
  __nv_bfloat16 *sliding_cosine{nullptr};
  __nv_bfloat16 *sliding_sine{nullptr};
  __nv_bfloat16 *global_cosine{nullptr};
  __nv_bfloat16 *global_sine{nullptr};
  __nv_bfloat16 *shared_global_kv{nullptr};
  float *shared_global_inverse_rms{nullptr};

  explicit Implementation(int positions)
      : maximum_positions(positions), shared_global_kv_roundtrip([] {
          const char *value = std::getenv("CARAT_SHARED_GLOBAL_KV_ROUNDTRIP");
          return value != nullptr && std::string_view(value) == "1";
        }()) {
    if (positions <= 0)
      throw std::runtime_error("invalid RoPE cache size");
    check(cudaMalloc(&sliding_cosine, static_cast<std::size_t>(positions) * 256 * 2),
          "allocate sliding cosine");
    check(cudaMalloc(&sliding_sine, static_cast<std::size_t>(positions) * 256 * 2),
          "allocate sliding sine");
    check(cudaMalloc(&global_cosine, static_cast<std::size_t>(positions) * 512 * 2),
          "allocate global cosine");
    check(cudaMalloc(&global_sine, static_cast<std::size_t>(positions) * 512 * 2),
          "allocate global sine");
    const int sliding_elements = positions * 256;
    const int global_elements = positions * 512;
    rope_cache_kernel<256, false><<<(sliding_elements + threads - 1) / threads, threads>>>(
        sliding_cosine, sliding_sine, positions);
    rope_cache_kernel<512, true><<<(global_elements + threads - 1) / threads, threads>>>(
        global_cosine, global_sine, positions);
    if (shared_global_kv_roundtrip) {
      constexpr int maximum_global_kv_heads = 4;
      check(cudaMalloc(&shared_global_kv,
                       static_cast<std::size_t>(positions) * maximum_global_kv_heads * 512 * 2),
            "allocate shared global KV staging");
      check(cudaMalloc(&shared_global_inverse_rms, static_cast<std::size_t>(positions) *
                                                       maximum_global_kv_heads * sizeof(float)),
            "allocate shared global inverse RMS staging");
    }
    check(cudaGetLastError(), "initialize RoPE cache");
  }
  ~Implementation() {
    cudaFree(shared_global_inverse_rms);
    cudaFree(shared_global_kv);
    cudaFree(global_sine);
    cudaFree(global_cosine);
    cudaFree(sliding_sine);
    cudaFree(sliding_cosine);
  }
};

Gemma4QkvPostprocessor::Gemma4QkvPostprocessor(int maximum_positions)
    : implementation_(std::make_unique<Implementation>(maximum_positions)) {}
Gemma4QkvPostprocessor::~Gemma4QkvPostprocessor() = default;

void Gemma4QkvPostprocessor::run(const void *packed_qkv, const void *query_norm_weight,
                                 const void *key_norm_weight, void *queries, void *key_cache,
                                 void *value_cache, int tokens, int query_heads, int kv_heads,
                                 int head_dimension, bool key_equals_value, int position_start,
                                 int cache_capacity, float epsilon, cudaStream_t stream) {
  run_positioned(packed_qkv, query_norm_weight, key_norm_weight, queries, key_cache, value_cache,
                 tokens, query_heads, kv_heads, head_dimension, key_equals_value, position_start,
                 position_start, cache_capacity, epsilon, stream);
}

void Gemma4QkvPostprocessor::run_positioned(const void *packed_qkv, const void *query_norm_weight,
                                            const void *key_norm_weight, void *queries,
                                            void *key_cache, void *value_cache, int tokens,
                                            int query_heads, int kv_heads, int head_dimension,
                                            bool key_equals_value, int position_start,
                                            int cache_position_start, int cache_capacity,
                                            float epsilon, cudaStream_t stream) {
  if (tokens <= 0 || position_start < 0 || cache_position_start < 0 ||
      cache_position_start + tokens > cache_capacity ||
      position_start + tokens > implementation_->maximum_positions) {
    throw std::runtime_error("invalid QKV positions");
  }
  const int query_width = query_heads * head_dimension;
  const int kv_width = kv_heads * head_dimension;
  const int packed_width = query_width + kv_width * (key_equals_value ? 1 : 2);
  if (head_dimension == 256 && !key_equals_value) {
    query_kernel<256, false><<<tokens * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(queries), tokens, query_heads,
        packed_width, position_start, epsilon);
    key_value_kernel<256, false, false><<<tokens * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(key_cache),
        static_cast<__nv_bfloat16 *>(value_cache), tokens, query_width, kv_heads, packed_width,
        position_start, cache_position_start, cache_capacity, epsilon);
  } else if (head_dimension == 512 && key_equals_value) {
    query_kernel<512, false><<<tokens * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(queries), tokens, query_heads,
        packed_width, position_start, epsilon);
    if (implementation_->shared_global_kv_roundtrip) {
      encode_shared_global_kv_kernel<512><<<tokens * kv_heads, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(packed_qkv), implementation_->shared_global_kv,
          implementation_->shared_global_inverse_rms, tokens, query_width, kv_heads, packed_width,
          epsilon);
      materialize_shared_global_kv_kernel<512, false><<<tokens * kv_heads, threads, 0, stream>>>(
          implementation_->shared_global_kv, implementation_->shared_global_inverse_rms,
          static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
          implementation_->global_sine, static_cast<__nv_bfloat16 *>(key_cache),
          static_cast<__nv_bfloat16 *>(value_cache), tokens, kv_heads, position_start,
          cache_position_start, cache_capacity);
    } else {
      key_value_kernel<512, true, false><<<tokens * kv_heads, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(packed_qkv),
          static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
          implementation_->global_sine, static_cast<__nv_bfloat16 *>(key_cache),
          static_cast<__nv_bfloat16 *>(value_cache), tokens, query_width, kv_heads, packed_width,
          position_start, cache_position_start, cache_capacity, epsilon);
    }
  } else {
    throw std::runtime_error("unsupported Gemma 4 QKV shape");
  }
  check(cudaPeekAtLastError(), "launch QKV postprocessing");
}

void Gemma4QkvPostprocessor::run_decode_batch(const void *packed_qkv, const void *query_norm_weight,
                                              const void *key_norm_weight, void *queries,
                                              void *key_cache, void *value_cache, int batch,
                                              int query_heads, int kv_heads, int head_dimension,
                                              bool key_equals_value, int position,
                                              int cache_capacity, float epsilon,
                                              cudaStream_t stream) {
  if (batch <= 0 || position < 0 || position >= implementation_->maximum_positions) {
    throw std::runtime_error("invalid batched QKV position");
  }
  const int query_width = query_heads * head_dimension;
  const int kv_width = kv_heads * head_dimension;
  const int packed_width = query_width + kv_width * (key_equals_value ? 1 : 2);
  if (head_dimension == 256 && !key_equals_value) {
    query_kernel<256, true><<<batch * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(queries), batch, query_heads,
        packed_width, position, epsilon);
    key_value_kernel<256, false, true><<<batch * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(key_cache),
        static_cast<__nv_bfloat16 *>(value_cache), batch, query_width, kv_heads, packed_width,
        position, 0, cache_capacity, epsilon);
  } else if (head_dimension == 512 && key_equals_value) {
    query_kernel<512, true><<<batch * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(queries), batch, query_heads,
        packed_width, position, epsilon);
    if (implementation_->shared_global_kv_roundtrip) {
      encode_shared_global_kv_kernel<512><<<batch * kv_heads, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(packed_qkv), implementation_->shared_global_kv,
          implementation_->shared_global_inverse_rms, batch, query_width, kv_heads, packed_width,
          epsilon);
      materialize_shared_global_kv_kernel<512, true><<<batch * kv_heads, threads, 0, stream>>>(
          implementation_->shared_global_kv, implementation_->shared_global_inverse_rms,
          static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
          implementation_->global_sine, static_cast<__nv_bfloat16 *>(key_cache),
          static_cast<__nv_bfloat16 *>(value_cache), batch, kv_heads, position, 0, cache_capacity);
    } else {
      key_value_kernel<512, true, true><<<batch * kv_heads, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(packed_qkv),
          static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
          implementation_->global_sine, static_cast<__nv_bfloat16 *>(key_cache),
          static_cast<__nv_bfloat16 *>(value_cache), batch, query_width, kv_heads, packed_width,
          position, 0, cache_capacity, epsilon);
    }
  } else {
    throw std::runtime_error("unsupported batched Gemma 4 QKV shape");
  }
  check(cudaPeekAtLastError(), "launch batched QKV postprocessing");
}

void Gemma4QkvPostprocessor::run_decode_ragged(
    const void *packed_qkv, const void *query_norm_weight, const void *key_norm_weight,
    void *queries, void *key_cache, void *value_cache, const int *positions, const int *slots,
    int batch, int maximum_slots, int query_heads, int kv_heads, int head_dimension,
    bool key_equals_value, int cache_capacity, void *int8_key_cache, float *int8_key_scales,
    float epsilon, cudaStream_t stream) {
  if (batch <= 0 || maximum_slots <= 0 || batch > maximum_slots || positions == nullptr ||
      slots == nullptr || cache_capacity <= 0 ||
      cache_capacity > implementation_->maximum_positions ||
      ((int8_key_cache == nullptr) != (int8_key_scales == nullptr))) {
    throw std::runtime_error("invalid ragged QKV shape");
  }
  if (int8_key_cache != nullptr && implementation_->shared_global_kv_roundtrip) {
    throw std::runtime_error("fused INT8 K cannot be combined with shared global KV roundtrip");
  }
  const int query_width = query_heads * head_dimension;
  const int kv_width = kv_heads * head_dimension;
  const int packed_width = query_width + kv_width * (key_equals_value ? 1 : 2);
  if (head_dimension == 256 && !key_equals_value) {
    ragged_query_kernel<256><<<batch * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(queries), positions, batch,
        query_heads, packed_width, epsilon);
    ragged_key_value_kernel<256, false><<<batch * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(key_cache),
        static_cast<__nv_bfloat16 *>(value_cache), positions, slots, batch, maximum_slots,
        query_width, kv_heads, packed_width, cache_capacity, nullptr, nullptr, epsilon);
  } else if (head_dimension == 512 && key_equals_value) {
    ragged_query_kernel<512><<<batch * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(queries), positions, batch,
        query_heads, packed_width, epsilon);
    if (implementation_->shared_global_kv_roundtrip) {
      encode_shared_global_kv_kernel<512><<<batch * kv_heads, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(packed_qkv), implementation_->shared_global_kv,
          implementation_->shared_global_inverse_rms, batch, query_width, kv_heads, packed_width,
          epsilon);
      materialize_shared_global_kv_ragged_kernel<512><<<batch * kv_heads, threads, 0, stream>>>(
          implementation_->shared_global_kv, implementation_->shared_global_inverse_rms,
          static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
          implementation_->global_sine, static_cast<__nv_bfloat16 *>(key_cache),
          static_cast<__nv_bfloat16 *>(value_cache), positions, slots, batch, maximum_slots,
          kv_heads, cache_capacity);
    } else {
      ragged_key_value_kernel<512, true><<<batch * kv_heads, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(packed_qkv),
          static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
          implementation_->global_sine, static_cast<__nv_bfloat16 *>(key_cache),
          static_cast<__nv_bfloat16 *>(value_cache), positions, slots, batch, maximum_slots,
          query_width, kv_heads, packed_width, cache_capacity,
          static_cast<std::int8_t *>(int8_key_cache), int8_key_scales, epsilon);
    }
  } else {
    throw std::runtime_error("unsupported ragged Gemma 4 QKV shape");
  }
  check(cudaPeekAtLastError(), "launch ragged QKV postprocessing");
}

void Gemma4QkvPostprocessor::run_prefill_uniform(
    const void *packed_qkv, const void *query_norm_weight, const void *key_norm_weight,
    void *queries, void *staged_keys, void *staged_values, const int *device_position_starts,
    int requests, int tokens_per_request, int query_heads, int kv_heads, int head_dimension,
    bool key_equals_value, int staged_position_start, int staged_capacity, float epsilon,
    cudaStream_t stream) {
  if (packed_qkv == nullptr || query_norm_weight == nullptr || key_norm_weight == nullptr ||
      queries == nullptr || staged_keys == nullptr || staged_values == nullptr ||
      device_position_starts == nullptr || requests <= 0 || tokens_per_request <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || staged_position_start < 0 ||
      staged_position_start + tokens_per_request > staged_capacity) {
    throw std::runtime_error("invalid uniform prefill QKV shape");
  }
  const int rows = requests * tokens_per_request;
  const int query_width = query_heads * head_dimension;
  const int kv_width = kv_heads * head_dimension;
  const int packed_width = query_width + kv_width * (key_equals_value ? 1 : 2);
  if (head_dimension == 256 && !key_equals_value) {
    uniform_prefill_query_kernel<256><<<rows * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(queries),
        device_position_starts, requests, tokens_per_request, query_heads, packed_width, epsilon);
    uniform_prefill_key_value_kernel<256, false><<<rows * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(staged_keys),
        static_cast<__nv_bfloat16 *>(staged_values), device_position_starts, requests,
        tokens_per_request, query_width, kv_heads, packed_width, staged_position_start,
        staged_capacity, epsilon);
  } else if (head_dimension == 512 && key_equals_value) {
    uniform_prefill_query_kernel<512><<<rows * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(queries), device_position_starts,
        requests, tokens_per_request, query_heads, packed_width, epsilon);
    uniform_prefill_key_value_kernel<512, true><<<rows * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(staged_keys),
        static_cast<__nv_bfloat16 *>(staged_values), device_position_starts, requests,
        tokens_per_request, query_width, kv_heads, packed_width, staged_position_start,
        staged_capacity, epsilon);
  } else {
    throw std::runtime_error("unsupported uniform prefill Gemma 4 QKV shape");
  }
  check(cudaPeekAtLastError(), "launch uniform prefill QKV postprocessing");
}

void Gemma4QkvPostprocessor::run_prefill_uniform_direct(
    const void *packed_qkv, const void *query_norm_weight, const void *key_norm_weight,
    void *queries, void *const *device_key_caches, void *const *device_value_caches,
    const int *device_position_starts, int requests, int tokens_per_request, int query_heads,
    int kv_heads, int head_dimension, bool key_equals_value, int cache_capacity, float epsilon,
    cudaStream_t stream) {
  if (packed_qkv == nullptr || query_norm_weight == nullptr || key_norm_weight == nullptr ||
      queries == nullptr || device_key_caches == nullptr || device_value_caches == nullptr ||
      device_position_starts == nullptr || requests <= 0 || tokens_per_request <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || cache_capacity <= 0 ||
      cache_capacity > implementation_->maximum_positions) {
    throw std::runtime_error("invalid direct uniform prefill QKV shape");
  }
  const int rows = requests * tokens_per_request;
  const int query_width = query_heads * head_dimension;
  const int kv_width = kv_heads * head_dimension;
  const int packed_width = query_width + kv_width * (key_equals_value ? 1 : 2);
  if (head_dimension == 256 && !key_equals_value) {
    uniform_prefill_query_kernel<256><<<rows * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(queries),
        device_position_starts, requests, tokens_per_request, query_heads, packed_width, epsilon);
    uniform_prefill_direct_key_value_kernel<256, false><<<rows * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, reinterpret_cast<__nv_bfloat16 *const *>(device_key_caches),
        reinterpret_cast<__nv_bfloat16 *const *>(device_value_caches), device_position_starts,
        requests, tokens_per_request, query_width, kv_heads, packed_width, cache_capacity, epsilon);
  } else if (head_dimension == 512 && key_equals_value) {
    uniform_prefill_query_kernel<512><<<rows * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(queries), device_position_starts,
        requests, tokens_per_request, query_heads, packed_width, epsilon);
    uniform_prefill_direct_key_value_kernel<512, true><<<rows * kv_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed_qkv),
        static_cast<const __nv_bfloat16 *>(key_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, reinterpret_cast<__nv_bfloat16 *const *>(device_key_caches),
        reinterpret_cast<__nv_bfloat16 *const *>(device_value_caches), device_position_starts,
        requests, tokens_per_request, query_width, kv_heads, packed_width, cache_capacity, epsilon);
  } else {
    throw std::runtime_error("unsupported direct uniform prefill Gemma 4 QKV shape");
  }
  check(cudaPeekAtLastError(), "launch direct uniform prefill QKV postprocessing");
}

void Gemma4QkvPostprocessor::run_query_ragged(const void *projected_queries,
                                              const void *query_norm_weight, void *queries,
                                              const int *device_positions, int batch,
                                              int query_heads, int head_dimension,
                                              bool global_attention, float epsilon,
                                              cudaStream_t stream) {
  if (projected_queries == nullptr || query_norm_weight == nullptr || queries == nullptr ||
      device_positions == nullptr || batch <= 0 || query_heads <= 0 ||
      (head_dimension != 256 && head_dimension != 512) ||
      global_attention != (head_dimension == 512)) {
    throw std::runtime_error("invalid assistant query shape");
  }
  const int query_width = query_heads * head_dimension;
  if (global_attention) {
    ragged_query_kernel<512><<<batch * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(projected_queries),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->global_cosine,
        implementation_->global_sine, static_cast<__nv_bfloat16 *>(queries), device_positions,
        batch, query_heads, query_width, epsilon);
  } else {
    ragged_query_kernel<256><<<batch * query_heads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(projected_queries),
        static_cast<const __nv_bfloat16 *>(query_norm_weight), implementation_->sliding_cosine,
        implementation_->sliding_sine, static_cast<__nv_bfloat16 *>(queries), device_positions,
        batch, query_heads, query_width, epsilon);
  }
  check(cudaPeekAtLastError(), "launch assistant query postprocessing");
}

} // namespace carat
