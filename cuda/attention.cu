#include "carat/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace carat {
namespace {

constexpr int threads = 256;
constexpr int global_wmma_threads = 512;

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

template <int head_dimension, int query_group, int tile_tokens>
__global__ void grouped_decode_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key_cache,
                                      const __nv_bfloat16 *value_cache, __nv_bfloat16 *output,
                                      int kv_heads, int context_length) {
  const int batch_index = static_cast<int>(blockIdx.x) / kv_heads;
  const int kv_head = static_cast<int>(blockIdx.x) % kv_heads;
  const std::uint64_t cache_base = (static_cast<std::uint64_t>(batch_index) * kv_heads + kv_head) *
                                   context_length * head_dimension;
  const std::uint64_t query_base =
      (static_cast<std::uint64_t>(batch_index) * kv_heads + kv_head) * query_group * head_dimension;

  extern __shared__ __align__(16) unsigned char storage[];
  auto *shared_query = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *shared_key = shared_query + query_group * head_dimension;
  auto *shared_value = shared_key + tile_tokens * head_dimension;
  auto *scores = reinterpret_cast<float *>(shared_value + tile_tokens * head_dimension);
  __shared__ float running_max[query_group];
  __shared__ float running_sum[query_group];
  __shared__ float old_scale[query_group];

  for (int index = threadIdx.x; index < query_group * head_dimension; index += threads) {
    shared_query[index] = query[query_base + index];
  }
  if (threadIdx.x < query_group) {
    running_max[threadIdx.x] = -CUDART_INF_F;
    running_sum[threadIdx.x] = 0.0F;
  }
  constexpr int dimensions_per_thread = (head_dimension + threads - 1) / threads;
  float accumulator[query_group][dimensions_per_thread]{};
  __syncthreads();

  for (int tile_begin = 0; tile_begin < context_length; tile_begin += tile_tokens) {
    const int valid_tokens = min(tile_tokens, context_length - tile_begin);
    const int tile_elements = valid_tokens * head_dimension;
    for (int index = threadIdx.x; index < tile_elements; index += threads) {
      const int token = index / head_dimension;
      const int dimension = index % head_dimension;
      const std::uint64_t cache_index =
          cache_base + static_cast<std::uint64_t>(tile_begin + token) * head_dimension + dimension;
      shared_key[index] = key_cache[cache_index];
      shared_value[index] = value_cache[cache_index];
    }
    __syncthreads();

    const int score_count = query_group * valid_tokens;
    if (threadIdx.x < score_count) {
      const int query_index = threadIdx.x / valid_tokens;
      const int token = threadIdx.x % valid_tokens;
      float sum0 = 0.0F, sum1 = 0.0F, sum2 = 0.0F, sum3 = 0.0F;
      int dimension = 0;
      for (; dimension + 3 < head_dimension; dimension += 4) {
        sum0 += __bfloat162float(shared_query[query_index * head_dimension + dimension]) *
                __bfloat162float(shared_key[token * head_dimension + dimension]);
        sum1 += __bfloat162float(shared_query[query_index * head_dimension + dimension + 1]) *
                __bfloat162float(shared_key[token * head_dimension + dimension + 1]);
        sum2 += __bfloat162float(shared_query[query_index * head_dimension + dimension + 2]) *
                __bfloat162float(shared_key[token * head_dimension + dimension + 2]);
        sum3 += __bfloat162float(shared_query[query_index * head_dimension + dimension + 3]) *
                __bfloat162float(shared_key[token * head_dimension + dimension + 3]);
      }
      scores[query_index * tile_tokens + token] = (sum0 + sum1) + (sum2 + sum3);
    }
    __syncthreads();

    if (threadIdx.x < query_group) {
      const int query_index = threadIdx.x;
      float tile_max = -CUDART_INF_F;
      for (int token = 0; token < valid_tokens; ++token) {
        tile_max = max(tile_max, scores[query_index * tile_tokens + token]);
      }
      const float next_max = max(running_max[query_index], tile_max);
      const float alpha = expf(running_max[query_index] - next_max);
      float tile_sum = 0.0F;
      for (int token = 0; token < valid_tokens; ++token) {
        const float probability = expf(scores[query_index * tile_tokens + token] - next_max);
        scores[query_index * tile_tokens + token] = probability;
        tile_sum += probability;
      }
      old_scale[query_index] = alpha;
      running_sum[query_index] = running_sum[query_index] * alpha + tile_sum;
      running_max[query_index] = next_max;
    }
    __syncthreads();

#pragma unroll
    for (int query_index = 0; query_index < query_group; ++query_index) {
#pragma unroll
      for (int dimension_slot = 0; dimension_slot < dimensions_per_thread; ++dimension_slot) {
        const int dimension = threadIdx.x + dimension_slot * threads;
        if (dimension >= head_dimension)
          continue;
        float value = accumulator[query_index][dimension_slot] * old_scale[query_index];
        for (int token = 0; token < valid_tokens; ++token) {
          value += scores[query_index * tile_tokens + token] *
                   __bfloat162float(shared_value[token * head_dimension + dimension]);
        }
        accumulator[query_index][dimension_slot] = value;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int query_index = 0; query_index < query_group; ++query_index) {
#pragma unroll
    for (int dimension_slot = 0; dimension_slot < dimensions_per_thread; ++dimension_slot) {
      const int dimension = threadIdx.x + dimension_slot * threads;
      if (dimension < head_dimension) {
        output[query_base + query_index * head_dimension + dimension] = __float2bfloat16_rn(
            accumulator[query_index][dimension_slot] / running_sum[query_index]);
      }
    }
  }
}

__global__ void grouped_global_wmma_kernel(const __nv_bfloat16 *query,
                                           const __nv_bfloat16 *key_cache,
                                           const __nv_bfloat16 *value_cache, __nv_bfloat16 *output,
                                           int kv_heads, int context_length) {
  constexpr int dimension = 512;
  constexpr int group = 8;
  constexpr int tile_tokens = 64;
  constexpr int padded_group = 16;
  const int batch_index = static_cast<int>(blockIdx.x) / kv_heads;
  const int kv_head = static_cast<int>(blockIdx.x) % kv_heads;
  const std::uint64_t cache_base =
      (static_cast<std::uint64_t>(batch_index) * kv_heads + kv_head) * context_length * dimension;
  const std::uint64_t query_base =
      (static_cast<std::uint64_t>(batch_index) * kv_heads + kv_head) * group * dimension;

  extern __shared__ __align__(16) unsigned char storage[];
  auto *shared_query = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *shared_key = shared_query + padded_group * dimension;
  auto *shared_value = shared_key + tile_tokens * dimension;
  auto *probabilities = shared_value + tile_tokens * dimension;
  auto *scores = reinterpret_cast<float *>(probabilities + padded_group * tile_tokens);
  auto *partial_output = scores + padded_group * tile_tokens;
  __shared__ float running_max[group];
  __shared__ float running_sum[group];
  __shared__ float previous_scale[group];

  for (int index = threadIdx.x; index < padded_group * dimension; index += global_wmma_threads) {
    const int row = index / dimension;
    shared_query[index] = row < group ? query[query_base + index] : __float2bfloat16_rn(0.0F);
  }
  if (threadIdx.x < group) {
    running_max[threadIdx.x] = -CUDART_INF_F;
    running_sum[threadIdx.x] = 0.0F;
  }
  float accumulator[group]{};
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / 32;
  for (int tile_begin = 0; tile_begin < context_length; tile_begin += tile_tokens) {
    const int valid_tokens = min(tile_tokens, context_length - tile_begin);
    for (int index = threadIdx.x; index < tile_tokens * dimension; index += global_wmma_threads) {
      const int token = index / dimension;
      const int component = index % dimension;
      if (token < valid_tokens) {
        const auto source =
            cache_base + static_cast<std::uint64_t>(tile_begin + token) * dimension + component;
        shared_key[index] = key_cache[source];
        shared_value[index] = value_cache[source];
      } else {
        shared_key[index] = __float2bfloat16_rn(0.0F);
        shared_value[index] = __float2bfloat16_rn(0.0F);
      }
    }
    __syncthreads();

    if (warp < 4) {
      using namespace nvcuda;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragment;
      wmma::fill_fragment(score_fragment, 0.0F);
      for (int component = 0; component < dimension; component += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> query_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> key_fragment;
        wmma::load_matrix_sync(query_fragment, shared_query + component, dimension);
        wmma::load_matrix_sync(key_fragment, shared_key + warp * 16 * dimension + component,
                               dimension);
        wmma::mma_sync(score_fragment, query_fragment, key_fragment, score_fragment);
      }
      wmma::store_matrix_sync(scores + warp * 16, score_fragment, tile_tokens, wmma::mem_row_major);
    }
    __syncthreads();

    if (threadIdx.x < padded_group * tile_tokens) {
      probabilities[threadIdx.x] = __float2bfloat16_rn(0.0F);
    }
    __syncthreads();
    if (threadIdx.x < group) {
      const int row = static_cast<int>(threadIdx.x);
      float tile_maximum = -CUDART_INF_F;
      for (int token = 0; token < valid_tokens; ++token) {
        tile_maximum = max(tile_maximum, scores[row * tile_tokens + token]);
      }
      const float next_maximum = max(running_max[row], tile_maximum);
      const float old_scale = expf(running_max[row] - next_maximum);
      float tile_sum = 0.0F;
      for (int token = 0; token < valid_tokens; ++token) {
        const float probability = expf(scores[row * tile_tokens + token] - next_maximum);
        probabilities[row * tile_tokens + token] = __float2bfloat16_rn(probability);
        tile_sum += probability;
      }
      previous_scale[row] = old_scale;
      running_sum[row] = running_sum[row] * old_scale + tile_sum;
      running_max[row] = next_maximum;
    }
    __syncthreads();

    if (warp < 16) {
      using namespace nvcuda;
      const int dimension_base = warp * 32;
#pragma unroll
      for (int dimension_half = 0; dimension_half < 2; ++dimension_half) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> output_fragment;
        wmma::fill_fragment(output_fragment, 0.0F);
#pragma unroll
        for (int token = 0; token < tile_tokens; token += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
              probability_fragment;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> value_fragment;
          wmma::load_matrix_sync(probability_fragment, probabilities + token, tile_tokens);
          wmma::load_matrix_sync(
              value_fragment,
              shared_value + token * dimension + dimension_base + dimension_half * 16, dimension);
          wmma::mma_sync(output_fragment, probability_fragment, value_fragment, output_fragment);
        }
        wmma::store_matrix_sync(partial_output + dimension_base + dimension_half * 16,
                                output_fragment, dimension, wmma::mem_row_major);
      }
    }
    __syncthreads();

    const int component = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int row = 0; row < group; ++row) {
      accumulator[row] =
          accumulator[row] * previous_scale[row] + partial_output[row * dimension + component];
    }
    __syncthreads();
  }

  const int component = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int row = 0; row < group; ++row) {
    output[query_base + row * dimension + component] =
        __float2bfloat16_rn(accumulator[row] / running_sum[row]);
  }
}

template <int head_dimension, int query_group, int tile_tokens>
void launch(const void *query, const void *key_cache, const void *value_cache, void *output,
            int batch, int kv_heads, int context_length, cudaStream_t stream) {
  const std::size_t shared_bytes =
      sizeof(__nv_bfloat16) * (query_group * head_dimension + 2 * tile_tokens * head_dimension) +
      sizeof(float) * query_group * tile_tokens;
  check(cudaFuncSetAttribute(grouped_decode_kernel<head_dimension, query_group, tile_tokens>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set grouped attention shared memory");
  grouped_decode_kernel<head_dimension, query_group, tile_tokens>
      <<<batch * kv_heads, threads, shared_bytes, stream>>>(
          static_cast<const __nv_bfloat16 *>(query), static_cast<const __nv_bfloat16 *>(key_cache),
          static_cast<const __nv_bfloat16 *>(value_cache), static_cast<__nv_bfloat16 *>(output),
          kv_heads, context_length);
  check(cudaPeekAtLastError(), "launch grouped decode attention");
}

} // namespace

void grouped_decode_attention_bf16(const void *query, const void *key_cache,
                                   const void *value_cache, void *output, int batch, int kv_heads,
                                   int query_group, int context_length, int head_dimension,
                                   cudaStream_t stream) {
  if (batch <= 0 || kv_heads <= 0 || context_length <= 0) {
    throw std::runtime_error("invalid grouped attention shape");
  }
  if (head_dimension == 256 && query_group == 2) {
    launch<256, 2, 64>(query, key_cache, value_cache, output, batch, kv_heads, context_length,
                       stream);
  } else if (head_dimension == 512 && query_group == 8) {
    launch<512, 8, 32>(query, key_cache, value_cache, output, batch, kv_heads, context_length,
                       stream);
  } else {
    throw std::runtime_error("unsupported grouped attention head shape");
  }
}

void grouped_decode_attention_wmma_bf16(const void *query, const void *key_cache,
                                        const void *value_cache, void *output, int batch,
                                        int kv_heads, int query_group, int context_length,
                                        int head_dimension, cudaStream_t stream) {
  if (batch <= 0 || kv_heads <= 0 || context_length <= 0 || query_group != 8 ||
      head_dimension != 512) {
    throw std::runtime_error("invalid WMMA global attention shape");
  }
  constexpr int padded_group = 16;
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr std::size_t shared_bytes =
      sizeof(__nv_bfloat16) *
          (padded_group * dimension + 2 * tile_tokens * dimension + padded_group * tile_tokens) +
      sizeof(float) * (padded_group * tile_tokens + padded_group * dimension);
  check(cudaFuncSetAttribute(grouped_global_wmma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set WMMA global attention shared memory");
  grouped_global_wmma_kernel<<<batch * kv_heads, global_wmma_threads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16 *>(query), static_cast<const __nv_bfloat16 *>(key_cache),
      static_cast<const __nv_bfloat16 *>(value_cache), static_cast<__nv_bfloat16 *>(output),
      kv_heads, context_length);
  check(cudaPeekAtLastError(), "launch WMMA global attention");
}

} // namespace carat
