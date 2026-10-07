#include "carat/attention.h"
#include "carat/cuda_ops.h"
#include "carat/gemm_attention.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace carat {
namespace {

void cuda_check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void cublas_check(cublasStatus_t result, const char *operation) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(result));
  }
}

__device__ float warp_max(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value = max(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ float warp_sum(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void softmax_bf16_kernel(__nv_bfloat16 *scores, int width) {
  __shared__ float warp_values[32];
  __nv_bfloat16 *row = scores + static_cast<std::uint64_t>(blockIdx.x) * width;
  float maximum = -CUDART_INF_F;
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    maximum = max(maximum, __bfloat162float(row[index]));
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  maximum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : -CUDART_INF_F;
  if (threadIdx.x < 32U)
    maximum = warp_max(maximum);
  if (threadIdx.x == 0U)
    warp_values[0] = maximum;
  __syncthreads();
  maximum = warp_values[0];

  float sum = 0.0F;
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    sum += expf(__bfloat162float(row[index]) - maximum);
  }
  sum = warp_sum(sum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = sum;
  __syncthreads();
  sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32U)
    sum = warp_sum(sum);
  if (threadIdx.x == 0U)
    warp_values[0] = 1.0F / sum;
  __syncthreads();
  const float inverse_sum = warp_values[0];
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    row[index] = __float2bfloat16_rn(expf(__bfloat162float(row[index]) - maximum) * inverse_sum);
  }
}

__global__ void ragged_softmax_bf16_kernel(__nv_bfloat16 *scores, const int *positions,
                                           int kv_heads, int query_group, int width,
                                           int cache_capacity) {
  __shared__ float warp_values[32];
  const int request = static_cast<int>(blockIdx.x) / (kv_heads * query_group);
  const int valid = min(positions[request] + 1, cache_capacity);
  __nv_bfloat16 *row = scores + static_cast<std::uint64_t>(blockIdx.x) * width;
  float maximum = -CUDART_INF_F;
  for (int index = threadIdx.x; index < valid; index += blockDim.x) {
    maximum = max(maximum, __bfloat162float(row[index]));
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  maximum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : -CUDART_INF_F;
  if (threadIdx.x < 32U)
    maximum = warp_max(maximum);
  if (threadIdx.x == 0U)
    warp_values[0] = maximum;
  __syncthreads();
  maximum = warp_values[0];

  float sum = 0.0F;
  for (int index = threadIdx.x; index < valid; index += blockDim.x) {
    sum += expf(__bfloat162float(row[index]) - maximum);
  }
  sum = warp_sum(sum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = sum;
  __syncthreads();
  sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32U)
    sum = warp_sum(sum);
  if (threadIdx.x == 0U)
    warp_values[0] = 1.0F / sum;
  __syncthreads();
  const float inverse_sum = warp_values[0];
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    row[index] =
        index < valid
            ? __float2bfloat16_rn(expf(__bfloat162float(row[index]) - maximum) * inverse_sum)
            : __float2bfloat16_rn(0.0F);
  }
}

__global__ void ragged_softmax_quantize_scaled_probabilities_int8_row_kernel(
    const __nv_bfloat16 *scores, const float *value_scales, std::int8_t *quantized_probabilities,
    float *probability_scales, const int *positions, int kv_heads, int query_group, int width,
    int cache_capacity) {
  __shared__ float warp_maxima[32];
  __shared__ float warp_sums[32];
  const int score_row = static_cast<int>(blockIdx.x);
  const int request = score_row / (kv_heads * query_group);
  const int compact_head = score_row / query_group;
  const int valid = min(positions[request] + 1, cache_capacity);
  const auto *row_scores = scores + static_cast<std::uint64_t>(score_row) * width;
  const auto *row_value_scales =
      value_scales + static_cast<std::uint64_t>(compact_head) * cache_capacity;
  auto *row_output = quantized_probabilities + static_cast<std::uint64_t>(score_row) * width;

  float maximum = -CUDART_INF_F;
  for (int token = static_cast<int>(threadIdx.x); token < valid;
       token += static_cast<int>(blockDim.x)) {
    maximum = max(maximum, __bfloat162float(row_scores[token]));
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_maxima[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  maximum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_maxima[threadIdx.x] : -CUDART_INF_F;
  if (threadIdx.x < 32U)
    maximum = warp_max(maximum);
  if (threadIdx.x == 0U)
    warp_maxima[0] = maximum;
  __syncthreads();
  maximum = warp_maxima[0];

  float sum = 0.0F;
  float scaled_maximum = 0.0F;
  for (int token = static_cast<int>(threadIdx.x); token < valid;
       token += static_cast<int>(blockDim.x)) {
    const float exponential = expf(__bfloat162float(row_scores[token]) - maximum);
    sum += exponential;
    scaled_maximum = max(scaled_maximum, exponential * row_value_scales[token]);
  }
  sum = warp_sum(sum);
  scaled_maximum = warp_max(scaled_maximum);
  if ((threadIdx.x & 31U) == 0U) {
    const int warp = threadIdx.x >> 5U;
    warp_sums[warp] = sum;
    warp_maxima[warp] = scaled_maximum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    const bool active_warp = threadIdx.x < (blockDim.x + 31U) / 32U;
    sum = active_warp ? warp_sums[threadIdx.x] : 0.0F;
    scaled_maximum = active_warp ? warp_maxima[threadIdx.x] : 0.0F;
    sum = warp_sum(sum);
    scaled_maximum = warp_max(scaled_maximum);
    if (threadIdx.x == 0U) {
      warp_sums[0] = sum;
      warp_maxima[0] = scaled_maximum;
      probability_scales[score_row] =
          scaled_maximum == 0.0F ? 1.0F : scaled_maximum / (sum * 127.0F);
    }
  }
  __syncthreads();
  scaled_maximum = warp_maxima[0];
  const float quantization_scale = scaled_maximum == 0.0F ? 0.0F : 127.0F / scaled_maximum;
  for (int token = static_cast<int>(threadIdx.x); token < width;
       token += static_cast<int>(blockDim.x)) {
    if (token < valid) {
      const float exponential = expf(__bfloat162float(row_scores[token]) - maximum);
      row_output[token] = static_cast<std::int8_t>(max(
          0, min(127, __float2int_rn(exponential * row_value_scales[token] * quantization_scale))));
    } else {
      row_output[token] = 0;
    }
  }
}

__global__ void ragged_softmax_quantize_scaled_probabilities_int8_block128_kernel(
    const __nv_bfloat16 *scores, const float *value_scales, std::int8_t *quantized_probabilities,
    float *probability_scales, const int *positions, int kv_heads, int query_group, int width,
    int cache_capacity, int head_batches) {
  constexpr int value_blocks = 4;
  __shared__ float warp_maxima[value_blocks][8];
  __shared__ float warp_sums[8];
  const int score_row = static_cast<int>(blockIdx.x);
  const int request = score_row / (kv_heads * query_group);
  const int compact_head = score_row / query_group;
  const int valid = min(positions[request] + 1, cache_capacity);
  const auto *row_scores = scores + static_cast<std::uint64_t>(score_row) * width;
  const auto *row_value_scales =
      value_scales + static_cast<std::uint64_t>(compact_head) * cache_capacity * value_blocks;

  float score_maximum = -CUDART_INF_F;
  for (int token = static_cast<int>(threadIdx.x); token < valid;
       token += static_cast<int>(blockDim.x)) {
    score_maximum = max(score_maximum, __bfloat162float(row_scores[token]));
  }
  score_maximum = warp_max(score_maximum);
  if ((threadIdx.x & 31U) == 0U) {
    warp_maxima[0][threadIdx.x >> 5U] = score_maximum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    score_maximum = threadIdx.x < 8U ? warp_maxima[0][threadIdx.x] : -CUDART_INF_F;
    score_maximum = warp_max(score_maximum);
    if (threadIdx.x == 0U)
      warp_maxima[0][0] = score_maximum;
  }
  __syncthreads();
  score_maximum = warp_maxima[0][0];

  float sum = 0.0F;
  float scaled_maxima[value_blocks] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (int token = static_cast<int>(threadIdx.x); token < valid;
       token += static_cast<int>(blockDim.x)) {
    const float exponential = expf(__bfloat162float(row_scores[token]) - score_maximum);
    sum += exponential;
#pragma unroll
    for (int block = 0; block < value_blocks; ++block) {
      scaled_maxima[block] =
          max(scaled_maxima[block], exponential * row_value_scales[token * value_blocks + block]);
    }
  }
  sum = warp_sum(sum);
#pragma unroll
  for (int block = 0; block < value_blocks; ++block) {
    scaled_maxima[block] = warp_max(scaled_maxima[block]);
  }
  if ((threadIdx.x & 31U) == 0U) {
    const int warp = threadIdx.x >> 5U;
    warp_sums[warp] = sum;
#pragma unroll
    for (int block = 0; block < value_blocks; ++block) {
      warp_maxima[block][warp] = scaled_maxima[block];
    }
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    sum = threadIdx.x < 8U ? warp_sums[threadIdx.x] : 0.0F;
    sum = warp_sum(sum);
#pragma unroll
    for (int block = 0; block < value_blocks; ++block) {
      scaled_maxima[block] = threadIdx.x < 8U ? warp_maxima[block][threadIdx.x] : 0.0F;
      scaled_maxima[block] = warp_max(scaled_maxima[block]);
      if (threadIdx.x == 0U) {
        warp_maxima[block][0] = scaled_maxima[block];
        probability_scales[(static_cast<std::uint64_t>(block) * head_batches * query_group) +
                           score_row] =
            scaled_maxima[block] == 0.0F ? 1.0F : scaled_maxima[block] / (sum * 127.0F);
      }
    }
  }
  __syncthreads();
  for (int token = static_cast<int>(threadIdx.x); token < width;
       token += static_cast<int>(blockDim.x)) {
    const float exponential =
        token < valid ? expf(__bfloat162float(row_scores[token]) - score_maximum) : 0.0F;
#pragma unroll
    for (int block = 0; block < value_blocks; ++block) {
      const float scaled_maximum = warp_maxima[block][0];
      const float inverse_scale = scaled_maximum == 0.0F ? 0.0F : 127.0F / scaled_maximum;
      const std::uint64_t output_row =
          static_cast<std::uint64_t>(block) * head_batches * query_group + score_row;
      quantized_probabilities[output_row * width + token] = static_cast<std::int8_t>(max(
          0, min(127, __float2int_rn(exponential * row_value_scales[token * value_blocks + block] *
                                     inverse_scale))));
    }
  }
}

__global__ void
scaled_int32_ragged_softmax_bf16_kernel(const std::int32_t *scores, __nv_bfloat16 *probabilities,
                                        const float *key_scales, const float *query_scales,
                                        const int *positions, int kv_heads, int query_group,
                                        int width, int cache_capacity) {
  __shared__ float warp_values[32];
  const int score_row = static_cast<int>(blockIdx.x);
  const int request = score_row / (kv_heads * query_group);
  const int compact_head = score_row / query_group;
  const int valid = min(positions[request] + 1, cache_capacity);
  const std::int32_t *row_scores = scores + static_cast<std::uint64_t>(score_row) * width;
  __nv_bfloat16 *row_probabilities = probabilities + static_cast<std::uint64_t>(score_row) * width;
  const float *row_key_scales =
      key_scales + static_cast<std::uint64_t>(compact_head) * cache_capacity;
  const float query_scale = query_scales[score_row];
  float maximum = -CUDART_INF_F;
  for (int index = threadIdx.x; index < valid; index += blockDim.x) {
    maximum =
        max(maximum, static_cast<float>(row_scores[index]) * query_scale * row_key_scales[index]);
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  maximum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : -CUDART_INF_F;
  if (threadIdx.x < 32U)
    maximum = warp_max(maximum);
  if (threadIdx.x == 0U)
    warp_values[0] = maximum;
  __syncthreads();
  maximum = warp_values[0];

  float sum = 0.0F;
  for (int index = threadIdx.x; index < valid; index += blockDim.x) {
    const float score = static_cast<float>(row_scores[index]) * query_scale * row_key_scales[index];
    sum += expf(score - maximum);
  }
  sum = warp_sum(sum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = sum;
  __syncthreads();
  sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32U)
    sum = warp_sum(sum);
  if (threadIdx.x == 0U)
    warp_values[0] = 1.0F / sum;
  __syncthreads();
  const float inverse_sum = warp_values[0];
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    if (index < valid) {
      const float score =
          static_cast<float>(row_scores[index]) * query_scale * row_key_scales[index];
      row_probabilities[index] = __float2bfloat16_rn(expf(score - maximum) * inverse_sum);
    } else {
      row_probabilities[index] = __float2bfloat16_rn(0.0F);
    }
  }
}

__global__ void quantize_scaled_probabilities_int8_block128_kernel(
    const __nv_bfloat16 *probabilities, const float *value_scales,
    std::int8_t *quantized_probabilities, float *probability_scales, int head_batches,
    int query_group, int context_length, int cache_capacity) {
  constexpr int value_blocks = 4;
  __shared__ float warp_values[32];
  const int output_row = static_cast<int>(blockIdx.x);
  const int query = output_row % query_group;
  const int head_block = output_row / query_group;
  const int head = head_block % head_batches;
  const int value_block = head_block / head_batches;
  const auto *row_probabilities =
      probabilities + (static_cast<std::uint64_t>(head) * query_group + query) * context_length;
  const auto *row_value_scales =
      value_scales + static_cast<std::uint64_t>(head) * cache_capacity * value_blocks + value_block;
  float maximum = 0.0F;
  for (int token = static_cast<int>(threadIdx.x); token < context_length;
       token += static_cast<int>(blockDim.x)) {
    maximum = max(maximum, __bfloat162float(row_probabilities[token]) *
                               row_value_scales[token * value_blocks]);
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  maximum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32U)
    maximum = warp_max(maximum);
  if (threadIdx.x == 0U)
    warp_values[0] = maximum;
  __syncthreads();
  maximum = warp_values[0];
  const float scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
  if (threadIdx.x == 0U)
    probability_scales[output_row] = scale;
  const float inverse_scale = 1.0F / scale;
  auto *row_output =
      quantized_probabilities + static_cast<std::uint64_t>(output_row) * context_length;
  for (int token = static_cast<int>(threadIdx.x); token < context_length;
       token += static_cast<int>(blockDim.x)) {
    const float scaled_probability =
        __bfloat162float(row_probabilities[token]) * row_value_scales[token * value_blocks];
    row_output[token] = static_cast<std::int8_t>(
        max(0, min(127, __float2int_rn(scaled_probability * inverse_scale))));
  }
}

__global__ void scale_int32_pv_output_bf16_kernel(const std::int32_t *source, const float *scales,
                                                  __nv_bfloat16 *output, int head_batches,
                                                  int query_group) {
  constexpr int dimension = 512;
  constexpr int block_width = 128;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t elements = static_cast<std::uint64_t>(head_batches) * query_group * dimension;
  if (index >= elements)
    return;
  const int component = static_cast<int>(index % dimension);
  const int query = static_cast<int>((index / dimension) % query_group);
  const int head = static_cast<int>(index / (dimension * query_group));
  const int value_block = component / block_width;
  const int block_component = component - value_block * block_width;
  const std::uint64_t source_row =
      (static_cast<std::uint64_t>(value_block) * head_batches + head) * query_group + query;
  output[index] = __float2bfloat16_rn(
      static_cast<float>(source[source_row * block_width + block_component]) * scales[source_row]);
}

__global__ void quantize_scaled_probabilities_int8_row_kernel(
    const __nv_bfloat16 *probabilities, const float *value_scales,
    std::int8_t *quantized_probabilities, float *probability_scales, int head_batches,
    int query_group, int context_length, int cache_capacity) {
  __shared__ float warp_values[32];
  const int output_row = static_cast<int>(blockIdx.x);
  const int query = output_row % query_group;
  const int head = output_row / query_group;
  const auto *row_probabilities =
      probabilities + (static_cast<std::uint64_t>(head) * query_group + query) * context_length;
  const auto *row_value_scales = value_scales + static_cast<std::uint64_t>(head) * cache_capacity;
  float maximum = 0.0F;
  for (int token = static_cast<int>(threadIdx.x); token < context_length;
       token += static_cast<int>(blockDim.x)) {
    maximum = max(maximum, __bfloat162float(row_probabilities[token]) * row_value_scales[token]);
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_values[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  maximum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_values[threadIdx.x] : 0.0F;
  if (threadIdx.x < 32U)
    maximum = warp_max(maximum);
  if (threadIdx.x == 0U)
    warp_values[0] = maximum;
  __syncthreads();
  maximum = warp_values[0];
  const float scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
  if (threadIdx.x == 0U)
    probability_scales[output_row] = scale;
  const float inverse_scale = 1.0F / scale;
  auto *row_output =
      quantized_probabilities + static_cast<std::uint64_t>(output_row) * context_length;
  for (int token = static_cast<int>(threadIdx.x); token < context_length;
       token += static_cast<int>(blockDim.x)) {
    const float scaled_probability =
        __bfloat162float(row_probabilities[token]) * row_value_scales[token];
    row_output[token] = static_cast<std::int8_t>(
        max(0, min(127, __float2int_rn(scaled_probability * inverse_scale))));
  }
}

__global__ void scale_int32_pv_row_output_bf16_kernel(const std::int32_t *source,
                                                      const float *scales, __nv_bfloat16 *output,
                                                      std::uint64_t elements, int query_group) {
  constexpr int dimension = 512;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const std::uint64_t output_row = index / dimension;
  output[index] = __float2bfloat16_rn(static_cast<float>(source[index]) * scales[output_row]);
}

__global__ void ragged_pointer_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *key_cache,
                                      const __nv_bfloat16 *value_cache, __nv_bfloat16 *scores,
                                      __nv_bfloat16 *output, const int *slots, int batch,
                                      int maximum_slots, int kv_heads, int query_group, int width,
                                      int cache_capacity, int head_dimension,
                                      const void **key_pointers, const void **query_pointers,
                                      const void **value_pointers, void **score_pointers,
                                      void **output_pointers) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int head_batches = batch * kv_heads;
  if (index >= head_batches)
    return;
  const int row = index / kv_heads;
  const int kv_head = index % kv_heads;
  const int slot = slots[row];
  if (slot < 0 || slot >= maximum_slots)
    return;
  const std::uint64_t cache_head = static_cast<std::uint64_t>(slot) * kv_heads + kv_head;
  const std::uint64_t compact_head = static_cast<std::uint64_t>(row) * kv_heads + kv_head;
  key_pointers[index] = key_cache + cache_head * cache_capacity * head_dimension;
  value_pointers[index] = value_cache + cache_head * cache_capacity * head_dimension;
  query_pointers[index] = query + compact_head * query_group * head_dimension;
  score_pointers[index] = scores + compact_head * query_group * width;
  output_pointers[index] = output + compact_head * query_group * head_dimension;
}

} // namespace

struct GemmGroupedDecodeAttention::Implementation {
  struct Fp8QkPlan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t key_layout{};
    cublasLtMatrixLayout_t query_layout{};
    cublasLtMatrixLayout_t score_layout{};
    cublasLtMatmulAlgo_t algorithm{};

    ~Fp8QkPlan() {
      if (score_layout != nullptr)
        cublasLtMatrixLayoutDestroy(score_layout);
      if (query_layout != nullptr)
        cublasLtMatrixLayoutDestroy(query_layout);
      if (key_layout != nullptr)
        cublasLtMatrixLayoutDestroy(key_layout);
      if (operation != nullptr)
        cublasLtMatmulDescDestroy(operation);
    }
  };

  cublasHandle_t handle{};
  cublasLtHandle_t lt_handle{};
  void *scores{nullptr};
  void *fp8_scores{nullptr};
  void *fp8_pv_partials{nullptr};
  void *int8_scores{nullptr};
  void *int8_queries{nullptr};
  float *int8_query_scales{nullptr};
  void *int8_scaled_probabilities{nullptr};
  float *int8_probability_scales{nullptr};
  void *int8_pv_accumulators{nullptr};
  void *pointers{nullptr};
  void *lt_workspace{nullptr};
  float *fp8_key_scale{nullptr};
  float *fp8_query_scale{nullptr};
  std::uint64_t maximum_score_elements;
  int maximum_batch;
  int maximum_head_batches;
  int maximum_query_group;
  static constexpr std::size_t lt_workspace_bytes = 32ULL * 1024 * 1024;
  static constexpr int maximum_fp8_pv_segments = 32;
  static constexpr int maximum_int8_pv_segments = 32;
  std::map<std::tuple<int, int, int, int, int, int>, std::unique_ptr<Fp8QkPlan>> fp8_plans;

  Implementation(int batch, int kv_heads, int query_group, int context)
      : maximum_score_elements(static_cast<std::uint64_t>(batch) * kv_heads * query_group *
                               context),
        maximum_batch(batch), maximum_head_batches(batch * kv_heads),
        maximum_query_group(query_group) {
    if (batch <= 0 || kv_heads <= 0 || query_group <= 0 || context <= 0) {
      throw std::runtime_error("invalid maximum attention shape");
    }
    cublas_check(cublasCreate(&handle), "cublasCreate attention");
    cuda_check(cudaMalloc(&scores, static_cast<std::size_t>(maximum_score_elements * 2ULL)),
               "allocate attention score workspace");
    cuda_check(cudaMalloc(&pointers,
                          static_cast<std::size_t>(5ULL * maximum_head_batches * sizeof(void *))),
               "allocate ragged attention pointers");
  }
  ~Implementation() {
    fp8_plans.clear();
    if (int8_pv_accumulators != nullptr)
      cudaFree(int8_pv_accumulators);
    if (int8_probability_scales != nullptr)
      cudaFree(int8_probability_scales);
    if (int8_scaled_probabilities != nullptr)
      cudaFree(int8_scaled_probabilities);
    if (int8_query_scales != nullptr)
      cudaFree(int8_query_scales);
    if (int8_queries != nullptr)
      cudaFree(int8_queries);
    if (int8_scores != nullptr)
      cudaFree(int8_scores);
    if (fp8_pv_partials != nullptr)
      cudaFree(fp8_pv_partials);
    if (fp8_query_scale != nullptr)
      cudaFree(fp8_query_scale);
    if (fp8_key_scale != nullptr)
      cudaFree(fp8_key_scale);
    if (fp8_scores != nullptr)
      cudaFree(fp8_scores);
    if (lt_workspace != nullptr)
      cudaFree(lt_workspace);
    if (pointers != nullptr)
      cudaFree(pointers);
    if (scores != nullptr)
      cudaFree(scores);
    if (lt_handle != nullptr)
      cublasLtDestroy(lt_handle);
    if (handle != nullptr)
      cublasDestroy(handle);
  }

  void ensure_fp8_resources() {
    if (lt_handle != nullptr)
      return;
    cublas_check(cublasLtCreate(&lt_handle), "cublasLtCreate attention");
    cuda_check(cudaMalloc(&lt_workspace, lt_workspace_bytes), "allocate FP8 QK workspace");
    cuda_check(cudaMalloc(&fp8_key_scale, sizeof(float)), "allocate FP8 key scale");
    cuda_check(cudaMalloc(&fp8_query_scale, sizeof(float)), "allocate FP8 query scale");
    constexpr float fixed_key_scale = 1.0F / 1024.0F;
    constexpr float fixed_query_scale = 1.0F / 64.0F;
    cuda_check(cudaMemcpy(fp8_key_scale, &fixed_key_scale, sizeof(fixed_key_scale),
                          cudaMemcpyHostToDevice),
               "initialize FP8 key scale");
    cuda_check(cudaMemcpy(fp8_query_scale, &fixed_query_scale, sizeof(fixed_query_scale),
                          cudaMemcpyHostToDevice),
               "initialize FP8 query scale");
  }

  void ensure_fp8_pv_resources() {
    if (fp8_scores != nullptr)
      return;
    cuda_check(cudaMalloc(&fp8_scores, static_cast<std::size_t>(maximum_score_elements)),
               "allocate FP8 probability workspace");
    cuda_check(cudaMalloc(&fp8_pv_partials, static_cast<std::size_t>(maximum_head_batches) * 8ULL *
                                                512ULL * maximum_fp8_pv_segments * sizeof(float)),
               "allocate segmented FP8 PV workspace");
  }

  void ensure_int8_resources() {
    if (int8_scores != nullptr)
      return;
    cuda_check(cudaMalloc(&int8_scores,
                          static_cast<std::size_t>(maximum_score_elements) * sizeof(std::int32_t)),
               "allocate INT8 QK score workspace");
    const std::uint64_t maximum_query_rows =
        static_cast<std::uint64_t>(maximum_head_batches) * maximum_query_group;
    cuda_check(cudaMalloc(&int8_queries, static_cast<std::size_t>(maximum_query_rows * 512ULL)),
               "allocate INT8 query workspace");
    cuda_check(cudaMalloc(&int8_query_scales,
                          static_cast<std::size_t>(maximum_query_rows * 4ULL * sizeof(float))),
               "allocate INT8 query scales");
    cuda_check(cudaMalloc(&int8_scaled_probabilities,
                          static_cast<std::size_t>(maximum_score_elements * 4ULL)),
               "allocate INT8 scaled probability workspace");
    cuda_check(cudaMalloc(&int8_probability_scales,
                          static_cast<std::size_t>(maximum_query_rows * 4ULL * sizeof(float))),
               "allocate INT8 probability scales");
    cuda_check(
        cudaMalloc(&int8_pv_accumulators,
                   static_cast<std::size_t>(maximum_query_rows * 512ULL * maximum_int8_pv_segments *
                                            sizeof(std::int32_t))),
        "allocate INT8 PV accumulator workspace");
  }

  Fp8QkPlan &fp8_plan(int batch, int kv_heads, int query_group, int context, int dimension,
                      int cache_capacity) {
    ensure_fp8_resources();
    const auto key =
        std::make_tuple(batch, kv_heads, query_group, context, dimension, cache_capacity);
    const auto existing = fp8_plans.find(key);
    if (existing != fp8_plans.end())
      return *existing->second;
    auto plan = std::make_unique<Fp8QkPlan>();
    cublas_check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "create FP8 QK descriptor");
    constexpr cublasOperation_t transpose = CUBLAS_OP_T;
    cublas_check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transpose, sizeof(transpose)),
                 "transpose FP8 keys");
    constexpr cublasLtMatmulMatrixScale_t scalar_scale = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublas_check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                &scalar_scale, sizeof(scalar_scale)),
                 "set FP8 key scale mode");
    cublas_check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                &scalar_scale, sizeof(scalar_scale)),
                 "set FP8 query scale mode");
    const void *key_scale_pointer = fp8_key_scale;
    const void *query_scale_pointer = fp8_query_scale;
    cublas_check(cublasLtMatmulDescSetAttribute(plan->operation,
                                                CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &key_scale_pointer, sizeof(key_scale_pointer)),
                 "set FP8 key scale");
    cublas_check(cublasLtMatmulDescSetAttribute(plan->operation,
                                                CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &query_scale_pointer, sizeof(query_scale_pointer)),
                 "set FP8 query scale");
    cublas_check(cublasLtMatrixLayoutCreate(&plan->key_layout, CUDA_R_8F_E4M3, dimension, context,
                                            dimension),
                 "create FP8 key layout");
    cublas_check(cublasLtMatrixLayoutCreate(&plan->query_layout, CUDA_R_8F_E4M3, dimension,
                                            query_group, dimension),
                 "create FP8 query layout");
    cublas_check(
        cublasLtMatrixLayoutCreate(&plan->score_layout, CUDA_R_16BF, context, query_group, context),
        "create FP8 QK score layout");
    const std::int32_t head_batches = batch * kv_heads;
    const std::int64_t key_stride = static_cast<std::int64_t>(cache_capacity) * dimension;
    const std::int64_t query_stride = static_cast<std::int64_t>(query_group) * dimension;
    const std::int64_t score_stride = static_cast<std::int64_t>(query_group) * context;
    for (const auto layout : {plan->key_layout, plan->query_layout, plan->score_layout}) {
      cublas_check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                    &head_batches, sizeof(head_batches)),
                   "set FP8 QK batch count");
    }
    cublas_check(cublasLtMatrixLayoutSetAttribute(plan->key_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &key_stride, sizeof(key_stride)),
                 "set FP8 key stride");
    cublas_check(cublasLtMatrixLayoutSetAttribute(plan->query_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &query_stride, sizeof(query_stride)),
                 "set FP8 query stride");
    cublas_check(cublasLtMatrixLayoutSetAttribute(plan->score_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &score_stride, sizeof(score_stride)),
                 "set FP8 score stride");
    cublasLtMatmulPreference_t preference{};
    cublas_check(cublasLtMatmulPreferenceCreate(&preference), "create FP8 QK preference");
    cublas_check(
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &lt_workspace_bytes, sizeof(lt_workspace_bytes)),
        "set FP8 QK workspace limit");
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, plan->operation, plan->key_layout, plan->query_layout, plan->score_layout,
        plan->score_layout, preference, 1, &heuristic, &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    cublas_check(heuristic_status, "select FP8 QK algorithm");
    if (returned == 0)
      throw std::runtime_error("no FP8 QK algorithm");
    plan->algorithm = heuristic.algo;
    Fp8QkPlan &result = *plan;
    fp8_plans.emplace(key, std::move(plan));
    return result;
  }
};

GemmGroupedDecodeAttention::GemmGroupedDecodeAttention(int maximum_batch, int maximum_kv_heads,
                                                       int maximum_query_group, int maximum_context)
    : implementation_(std::make_unique<Implementation>(maximum_batch, maximum_kv_heads,
                                                       maximum_query_group, maximum_context)) {}
GemmGroupedDecodeAttention::~GemmGroupedDecodeAttention() = default;

void GemmGroupedDecodeAttention::run_qk(const void *query, const void *key_cache, void *scores,
                                        int batch, int kv_heads, int query_group,
                                        int context_length, int head_dimension, cudaStream_t stream,
                                        int cache_capacity) {
  if (cache_capacity == 0)
    cache_capacity = context_length;
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * context_length;
  if (query == nullptr || key_cache == nullptr || scores == nullptr || batch <= 0 ||
      kv_heads <= 0 || query_group <= 0 || context_length <= 0 || cache_capacity < context_length ||
      head_dimension <= 0 || score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("QK shape exceeds workspace");
  }
  cublas_check(cublasSetStream(implementation_->handle, stream), "set QK stream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int head_batches = batch * kv_heads;
  const long long cache_stride = static_cast<long long>(cache_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(query_group) * head_dimension;
  const long long score_stride = static_cast<long long>(query_group) * context_length;
  cublas_check(cublasGemmStridedBatchedEx(
                   implementation_->handle, CUBLAS_OP_T, CUBLAS_OP_N, context_length, query_group,
                   head_dimension, &alpha, key_cache, CUDA_R_16BF, head_dimension, cache_stride,
                   query, CUDA_R_16BF, head_dimension, query_stride, &beta, scores, CUDA_R_16BF,
                   context_length, score_stride, head_batches, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "benchmark grouped QK GEMM");
}

void GemmGroupedDecodeAttention::run(const void *query, const void *key_cache,
                                     const void *value_cache, void *output, int batch, int kv_heads,
                                     int query_group, int context_length, int head_dimension,
                                     cudaStream_t stream, int cache_capacity) {
  if (cache_capacity == 0)
    cache_capacity = context_length;
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * context_length;
  if (batch <= 0 || kv_heads <= 0 || query_group <= 0 || context_length <= 0 ||
      cache_capacity < context_length || head_dimension <= 0 ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("attention shape exceeds workspace");
  }
  cublas_check(cublasSetStream(implementation_->handle, stream), "set attention stream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int head_batches = batch * kv_heads;
  const long long cache_stride = static_cast<long long>(cache_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(query_group) * head_dimension;
  const long long score_stride = static_cast<long long>(query_group) * context_length;
  cublas_check(cublasGemmStridedBatchedEx(
                   implementation_->handle, CUBLAS_OP_T, CUBLAS_OP_N, context_length, query_group,
                   head_dimension, &alpha, key_cache, CUDA_R_16BF, head_dimension, cache_stride,
                   query, CUDA_R_16BF, head_dimension, query_stride, &beta, implementation_->scores,
                   CUDA_R_16BF, context_length, score_stride, head_batches, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "grouped QK GEMM");
  softmax_bf16_kernel<<<head_batches * query_group, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(implementation_->scores), context_length);
  cuda_check(cudaPeekAtLastError(), "launch attention softmax");
  cublas_check(cublasGemmStridedBatchedEx(
                   implementation_->handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_group,
                   context_length, &alpha, value_cache, CUDA_R_16BF, head_dimension, cache_stride,
                   implementation_->scores, CUDA_R_16BF, context_length, score_stride, &beta,
                   output, CUDA_R_16BF, head_dimension, query_stride, head_batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "grouped PV GEMM");
}

void GemmGroupedDecodeAttention::run_ragged(const void *query, const void *key_cache,
                                            const void *value_cache, void *output,
                                            const int *positions, const int *slots, int batch,
                                            int maximum_slots, int kv_heads, int query_group,
                                            int maximum_context_length, int head_dimension,
                                            cudaStream_t stream, int cache_capacity) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
      positions == nullptr || slots == nullptr || batch <= 0 || maximum_slots < batch ||
      kv_heads <= 0 || query_group <= 0 || maximum_context_length <= 0 ||
      maximum_context_length > cache_capacity || head_dimension <= 0 ||
      batch * kv_heads > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("ragged attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  cublas_check(cublasSetStream(state.handle, stream), "set ragged attention stream");
  const int head_batches = batch * kv_heads;
  auto **pointer_base = static_cast<void **>(state.pointers);
  auto **key_pointers = const_cast<void **>(pointer_base);
  auto **query_pointers = pointer_base + state.maximum_head_batches;
  auto **value_pointers = pointer_base + 2 * state.maximum_head_batches;
  auto **score_pointers = pointer_base + 3 * state.maximum_head_batches;
  auto **output_pointers = pointer_base + 4 * state.maximum_head_batches;
  constexpr int pointer_threads = 256;
  ragged_pointer_kernel<<<(head_batches + pointer_threads - 1) / pointer_threads, pointer_threads,
                          0, stream>>>(
      static_cast<const __nv_bfloat16 *>(query), static_cast<const __nv_bfloat16 *>(key_cache),
      static_cast<const __nv_bfloat16 *>(value_cache), static_cast<__nv_bfloat16 *>(state.scores),
      static_cast<__nv_bfloat16 *>(output), slots, batch, maximum_slots, kv_heads, query_group,
      maximum_context_length, cache_capacity, head_dimension,
      const_cast<const void **>(key_pointers), const_cast<const void **>(query_pointers),
      const_cast<const void **>(value_pointers), score_pointers, output_pointers);
  cuda_check(cudaPeekAtLastError(), "build ragged attention pointers");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  cublas_check(cublasGemmBatchedEx(
                   state.handle, CUBLAS_OP_T, CUBLAS_OP_N, maximum_context_length, query_group,
                   head_dimension, &alpha, reinterpret_cast<const void *const *>(key_pointers),
                   CUDA_R_16BF, head_dimension,
                   reinterpret_cast<const void *const *>(query_pointers), CUDA_R_16BF,
                   head_dimension, &beta, score_pointers, CUDA_R_16BF, maximum_context_length,
                   head_batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "ragged grouped QK GEMM");
  ragged_softmax_bf16_kernel<<<batch * kv_heads * query_group, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), positions, kv_heads, query_group,
      maximum_context_length, cache_capacity);
  cuda_check(cudaPeekAtLastError(), "launch ragged attention softmax");
  cublas_check(cublasGemmBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_group,
                   maximum_context_length, &alpha,
                   reinterpret_cast<const void *const *>(value_pointers), CUDA_R_16BF,
                   head_dimension, reinterpret_cast<const void *const *>(score_pointers),
                   CUDA_R_16BF, maximum_context_length, &beta, output_pointers, CUDA_R_16BF,
                   head_dimension, head_batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "ragged grouped PV GEMM");
}

void GemmGroupedDecodeAttention::run_ragged_contiguous(
    const void *query, const void *key_cache, const void *value_cache, void *output,
    const int *positions, int batch, int kv_heads, int query_group, int maximum_context_length,
    int head_dimension, int first_slot, cudaStream_t stream, int cache_capacity) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
      positions == nullptr || batch <= 0 || kv_heads <= 0 || query_group <= 0 || first_slot < 0 ||
      first_slot + batch > implementation_->maximum_batch || maximum_context_length <= 0 ||
      maximum_context_length > cache_capacity || head_dimension <= 0 ||
      batch * kv_heads > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("contiguous ragged attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  cublas_check(cublasSetStream(state.handle, stream), "set contiguous ragged attention stream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int head_batches = batch * kv_heads;
  const long long cache_stride = static_cast<long long>(cache_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(query_group) * head_dimension;
  const long long score_stride = static_cast<long long>(query_group) * maximum_context_length;
  const auto cache_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  const auto *contiguous_keys = static_cast<const __nv_bfloat16 *>(key_cache) + cache_offset;
  const auto *contiguous_values = static_cast<const __nv_bfloat16 *>(value_cache) + cache_offset;
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_T, CUBLAS_OP_N, maximum_context_length, query_group,
                   head_dimension, &alpha, contiguous_keys, CUDA_R_16BF, head_dimension,
                   cache_stride, query, CUDA_R_16BF, head_dimension, query_stride, &beta,
                   state.scores, CUDA_R_16BF, maximum_context_length, score_stride, head_batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "contiguous ragged grouped QK GEMM");
  ragged_softmax_bf16_kernel<<<batch * kv_heads * query_group, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), positions, kv_heads, query_group,
      maximum_context_length, cache_capacity);
  cuda_check(cudaPeekAtLastError(), "launch contiguous ragged attention softmax");
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_group,
                   maximum_context_length, &alpha, contiguous_values, CUDA_R_16BF, head_dimension,
                   cache_stride, state.scores, CUDA_R_16BF, maximum_context_length, score_stride,
                   &beta, output, CUDA_R_16BF, head_dimension, query_stride, head_batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "contiguous ragged grouped PV GEMM");
}

void GemmGroupedDecodeAttention::run_ragged_contiguous_fp8_keys(
    const void *fp8_query, const void *fp8_key_cache, const void *value_cache, void *output,
    const int *positions, int batch, int kv_heads, int query_group, int maximum_context_length,
    int head_dimension, int first_slot, cudaStream_t stream, int cache_capacity) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (fp8_query == nullptr || fp8_key_cache == nullptr || value_cache == nullptr ||
      output == nullptr || positions == nullptr || batch <= 0 || kv_heads <= 0 ||
      query_group <= 0 || first_slot < 0 || first_slot + batch > implementation_->maximum_batch ||
      maximum_context_length <= 0 || maximum_context_length > cache_capacity ||
      head_dimension <= 0 || batch * kv_heads > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("FP8 contiguous ragged attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  auto &plan = state.fp8_plan(batch, kv_heads, query_group, maximum_context_length, head_dimension,
                              cache_capacity);
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const auto cache_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  const auto *contiguous_keys = static_cast<const unsigned char *>(fp8_key_cache) + cache_offset;
  cublas_check(cublasLtMatmul(state.lt_handle, plan.operation, &alpha, contiguous_keys,
                              plan.key_layout, fp8_query, plan.query_layout, &beta, state.scores,
                              plan.score_layout, state.scores, plan.score_layout, &plan.algorithm,
                              state.lt_workspace, state.lt_workspace_bytes, stream),
               "run FP8 contiguous ragged QK");
  ragged_softmax_bf16_kernel<<<batch * kv_heads * query_group, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), positions, kv_heads, query_group,
      maximum_context_length, cache_capacity);
  cuda_check(cudaPeekAtLastError(), "launch FP8 contiguous ragged softmax");
  cublas_check(cublasSetStream(state.handle, stream), "set FP8 PV stream");
  const int head_batches = batch * kv_heads;
  const long long cache_stride = static_cast<long long>(cache_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(query_group) * head_dimension;
  const long long score_stride = static_cast<long long>(query_group) * maximum_context_length;
  const auto *contiguous_values =
      static_cast<const __nv_bfloat16 *>(value_cache) +
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_group,
                   maximum_context_length, &alpha, contiguous_values, CUDA_R_16BF, head_dimension,
                   cache_stride, state.scores, CUDA_R_16BF, maximum_context_length, score_stride,
                   &beta, output, CUDA_R_16BF, head_dimension, query_stride, head_batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "FP8-key contiguous ragged PV GEMM");
}

void GemmGroupedDecodeAttention::run_ragged_contiguous_fp8_kv(
    const void *fp8_query, const void *fp8_key_cache, const void *fp8_value_cache, void *output,
    const int *positions, int batch, int kv_heads, int query_group, int maximum_context_length,
    int head_dimension, int first_slot, cudaStream_t stream, int cache_capacity) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (fp8_query == nullptr || fp8_key_cache == nullptr || fp8_value_cache == nullptr ||
      output == nullptr || positions == nullptr || batch <= 0 || kv_heads <= 0 ||
      query_group <= 0 || first_slot < 0 || first_slot + batch > implementation_->maximum_batch ||
      maximum_context_length <= 0 || maximum_context_length > cache_capacity ||
      head_dimension <= 0 || batch * kv_heads > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("FP8-KV contiguous ragged attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  auto &plan = state.fp8_plan(batch, kv_heads, query_group, maximum_context_length, head_dimension,
                              cache_capacity);
  state.ensure_fp8_pv_resources();
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const auto cache_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  const auto *contiguous_keys = static_cast<const unsigned char *>(fp8_key_cache) + cache_offset;
  cublas_check(cublasLtMatmul(state.lt_handle, plan.operation, &alpha, contiguous_keys,
                              plan.key_layout, fp8_query, plan.query_layout, &beta, state.scores,
                              plan.score_layout, state.scores, plan.score_layout, &plan.algorithm,
                              state.lt_workspace, state.lt_workspace_bytes, stream),
               "run FP8-KV contiguous ragged QK");
  ragged_softmax_bf16_kernel<<<batch * kv_heads * query_group, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), positions, kv_heads, query_group,
      maximum_context_length, cache_capacity);
  cuda_check(cudaPeekAtLastError(), "launch FP8-KV contiguous ragged softmax");
  quantize_grouped_global_probabilities_wgmma_fp8(state.scores, state.fp8_scores, batch, kv_heads,
                                                  query_group, maximum_context_length, stream);
  const auto *contiguous_values =
      static_cast<const unsigned char *>(fp8_value_cache) + cache_offset;
  int pv_segments = 1;
  const int context_tiles = maximum_context_length / 64;
  while (pv_segments < Implementation::maximum_fp8_pv_segments &&
         context_tiles % (pv_segments * 2) == 0 && batch * kv_heads * 8 * pv_segments < 512) {
    pv_segments *= 2;
  }
  grouped_global_pv_wgmma_fp8_rs(contiguous_values, state.fp8_scores, state.fp8_pv_partials, output,
                                 batch, kv_heads, query_group, maximum_context_length,
                                 head_dimension, cache_capacity, pv_segments, stream);
}

void GemmGroupedDecodeAttention::run_ragged_contiguous_int8_keys(
    const void *query, const void *int8_key_cache, const float *key_scales, const void *value_cache,
    void *output, const int *positions, int batch, int kv_heads, int query_group,
    int maximum_context_length, int head_dimension, int first_slot, cudaStream_t stream,
    int cache_capacity) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (query == nullptr || int8_key_cache == nullptr || key_scales == nullptr ||
      value_cache == nullptr || output == nullptr || positions == nullptr || batch <= 0 ||
      kv_heads <= 0 || query_group <= 0 || query_group > implementation_->maximum_query_group ||
      first_slot < 0 || first_slot + batch > implementation_->maximum_batch ||
      maximum_context_length <= 0 || maximum_context_length > cache_capacity ||
      head_dimension != 512 || batch * kv_heads > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("INT8-key contiguous ragged attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  state.ensure_int8_resources();
  cublas_check(cublasSetStream(state.handle, stream), "set INT8-key attention stream");
  const int head_batches = batch * kv_heads;
  const int query_rows = head_batches * query_group;
  quantize_bf16_rows_symmetric_int8(query, state.int8_queries, state.int8_query_scales, query_rows,
                                    head_dimension, stream);
  const auto cache_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  const auto scale_offset = static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity;
  const auto *contiguous_keys = static_cast<const std::int8_t *>(int8_key_cache) + cache_offset;
  const auto *contiguous_key_scales = key_scales + scale_offset;
  constexpr std::int32_t alpha_int = 1;
  constexpr std::int32_t beta_int = 0;
  const long long cache_stride = static_cast<long long>(cache_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(query_group) * head_dimension;
  const long long score_stride = static_cast<long long>(query_group) * maximum_context_length;
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_T, CUBLAS_OP_N, maximum_context_length, query_group,
                   head_dimension, &alpha_int, contiguous_keys, CUDA_R_8I, head_dimension,
                   cache_stride, state.int8_queries, CUDA_R_8I, head_dimension, query_stride,
                   &beta_int, state.int8_scores, CUDA_R_32I, maximum_context_length, score_stride,
                   head_batches, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "INT8-key contiguous ragged QK GEMM");
  scaled_int32_ragged_softmax_bf16_kernel<<<query_rows, 256, 0, stream>>>(
      static_cast<const std::int32_t *>(state.int8_scores),
      static_cast<__nv_bfloat16 *>(state.scores), contiguous_key_scales, state.int8_query_scales,
      positions, kv_heads, query_group, maximum_context_length, cache_capacity);
  cuda_check(cudaPeekAtLastError(), "launch INT8-key scaled ragged softmax");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const auto *contiguous_values = static_cast<const __nv_bfloat16 *>(value_cache) + cache_offset;
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_group,
                   maximum_context_length, &alpha, contiguous_values, CUDA_R_16BF, head_dimension,
                   cache_stride, state.scores, CUDA_R_16BF, maximum_context_length, score_stride,
                   &beta, output, CUDA_R_16BF, head_dimension, query_stride, head_batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "INT8-key contiguous ragged PV GEMM");
}

void GemmGroupedDecodeAttention::run_ragged_contiguous_int8_block128_keys(
    const void *query, const void *int8_key_cache, const float *key_scales, const void *value_cache,
    void *output, const int *positions, int batch, int kv_heads, int query_group,
    int maximum_context_length, int head_dimension, int first_slot, cudaStream_t stream,
    int cache_capacity, int tiles_per_block) {
  const int head_batches = batch * kv_heads;
  int selected_tiles = tiles_per_block;
  if (selected_tiles == 0) {
    selected_tiles = head_batches <= 4 ? 1 : head_batches <= 16 ? 2 : head_batches <= 32 ? 4 : 8;
    while (selected_tiles > 1 && (maximum_context_length / 64) % selected_tiles != 0) {
      selected_tiles /= 2;
    }
  }
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (query == nullptr || int8_key_cache == nullptr || key_scales == nullptr ||
      value_cache == nullptr || output == nullptr || positions == nullptr || batch <= 0 ||
      kv_heads <= 0 || query_group != 8 || first_slot < 0 ||
      first_slot + batch > implementation_->maximum_batch || maximum_context_length <= 0 ||
      maximum_context_length > cache_capacity || maximum_context_length % 64 != 0 ||
      selected_tiles <= 0 || selected_tiles > 32 ||
      (maximum_context_length / 64) % selected_tiles != 0 || head_dimension != 512 ||
      batch * kv_heads > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("INT8 block-128 contiguous ragged attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  state.ensure_int8_resources();
  cublas_check(cublasSetStream(state.handle, stream), "set INT8 block-128 attention stream");
  const int query_rows = head_batches * query_group;
  quantize_bf16_blocks_symmetric_int8(query, state.int8_queries, state.int8_query_scales,
                                      query_rows, head_dimension, 128, stream);
  const auto cache_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  const auto scale_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * 4ULL;
  const auto *contiguous_keys = static_cast<const std::int8_t *>(int8_key_cache) + cache_offset;
  const auto *contiguous_key_scales = key_scales + scale_offset;
  grouped_global_qk_wgmma_int8_block128(state.int8_queries, state.int8_query_scales,
                                        contiguous_keys, contiguous_key_scales, state.scores, batch,
                                        kv_heads, query_group, maximum_context_length,
                                        head_dimension, cache_capacity, stream, selected_tiles);
  ragged_softmax_bf16_kernel<<<query_rows, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), positions, kv_heads, query_group,
      maximum_context_length, cache_capacity);
  cuda_check(cudaPeekAtLastError(), "launch INT8 block-128 ragged softmax");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const long long cache_stride = static_cast<long long>(cache_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(query_group) * head_dimension;
  const long long score_stride = static_cast<long long>(query_group) * maximum_context_length;
  const auto *contiguous_values = static_cast<const __nv_bfloat16 *>(value_cache) + cache_offset;
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_group,
                   maximum_context_length, &alpha, contiguous_values, CUDA_R_16BF, head_dimension,
                   cache_stride, state.scores, CUDA_R_16BF, maximum_context_length, score_stride,
                   &beta, output, CUDA_R_16BF, head_dimension, query_stride, head_batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "INT8 block-128 contiguous ragged PV GEMM");
}

void GemmGroupedDecodeAttention::run_ragged_contiguous_int8_block128_kv(
    const void *query, const void *int8_key_cache, const float *key_scales,
    const void *int8_value_cache, const float *value_scales, void *output, const int *positions,
    int batch, int kv_heads, int query_group, int maximum_context_length, int head_dimension,
    int first_slot, cudaStream_t stream, int cache_capacity, int tiles_per_block) {
  const int head_batches = batch * kv_heads;
  int selected_tiles = tiles_per_block;
  if (selected_tiles == 0) {
    selected_tiles = head_batches <= 4 ? 1 : head_batches <= 16 ? 2 : head_batches <= 32 ? 4 : 8;
    while (selected_tiles > 1 && (maximum_context_length / 64) % selected_tiles != 0) {
      selected_tiles /= 2;
    }
  }
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * maximum_context_length;
  if (query == nullptr || int8_key_cache == nullptr || key_scales == nullptr ||
      int8_value_cache == nullptr || value_scales == nullptr || output == nullptr ||
      positions == nullptr || batch <= 0 || kv_heads <= 0 || query_group != 8 || first_slot < 0 ||
      first_slot + batch > implementation_->maximum_batch || maximum_context_length <= 0 ||
      maximum_context_length > cache_capacity || maximum_context_length % 64 != 0 ||
      selected_tiles <= 0 || selected_tiles > 32 ||
      (maximum_context_length / 64) % selected_tiles != 0 || head_dimension != 512 ||
      head_batches > implementation_->maximum_head_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error(
        "INT8 block-128 contiguous ragged KV attention shape exceeds workspace");
  }
  auto &state = *implementation_;
  state.ensure_int8_resources();
  cublas_check(cublasSetStream(state.handle, stream), "set INT8 block-128 KV attention stream");
  const int query_rows = head_batches * query_group;
  quantize_bf16_blocks_symmetric_int8(query, state.int8_queries, state.int8_query_scales,
                                      query_rows, head_dimension, 128, stream);
  const auto cache_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * head_dimension;
  const auto scale_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * 4ULL;
  const auto *contiguous_keys = static_cast<const std::int8_t *>(int8_key_cache) + cache_offset;
  const auto *contiguous_key_scales = key_scales + scale_offset;
  grouped_global_qk_wgmma_int8_block128(state.int8_queries, state.int8_query_scales,
                                        contiguous_keys, contiguous_key_scales, state.scores, batch,
                                        kv_heads, query_group, maximum_context_length,
                                        head_dimension, cache_capacity, stream, selected_tiles);
  const auto *contiguous_values = static_cast<const std::int8_t *>(int8_value_cache) + cache_offset;
  const auto value_scale_offset =
      static_cast<std::uint64_t>(first_slot) * kv_heads * cache_capacity * 4ULL;
  const auto *contiguous_value_scales = value_scales + value_scale_offset;
  ragged_softmax_quantize_scaled_probabilities_int8_block128_kernel<<<query_rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(state.scores), contiguous_value_scales,
      static_cast<std::int8_t *>(state.int8_scaled_probabilities), state.int8_probability_scales,
      positions, kv_heads, query_group, maximum_context_length, cache_capacity, head_batches);
  cuda_check(cudaPeekAtLastError(), "launch fused block-128 INT8 softmax quantization");
  int pv_segments = 1;
  const int context_tiles = maximum_context_length / 64;
  while (pv_segments < Implementation::maximum_int8_pv_segments &&
         context_tiles % (pv_segments * 2) == 0 && head_batches * pv_segments < 512) {
    pv_segments *= 2;
  }
  grouped_global_pv_wgmma_int8_rs(contiguous_values, state.int8_scaled_probabilities,
                                  state.int8_probability_scales, state.int8_pv_accumulators, output,
                                  batch, kv_heads, query_group, maximum_context_length,
                                  head_dimension, cache_capacity, pv_segments, stream);
}

} // namespace carat
