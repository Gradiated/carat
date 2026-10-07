#include "carat/cuda_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace carat {
namespace {

// Gemma's rows are long enough that a full-SM block beats smaller CTAs in both decode and prefill.
// At B16 decode there are only sixteen independent rows, making within-row parallelism essential.
constexpr int rms_threads = 1024;

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

template <bool emit_row_amax>
__global__ void rms_norm_kernel(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                                const __nv_bfloat16 *residual, __nv_bfloat16 *output,
                                float *row_amax, std::size_t width, float epsilon) {
  const std::size_t row = blockIdx.x;
  const __nv_bfloat16 *row_input = input + row * width;
  __nv_bfloat16 *row_output = output + row * width;
  float square_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(row_input[column]);
    square_sum += value * value;
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
  }
  __shared__ float warp_sums[32];
  if ((threadIdx.x & 31U) == 0U)
    warp_sums[threadIdx.x >> 5U] = square_sum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    square_sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_sums[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (threadIdx.x == 0U)
      warp_sums[0] = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
  }
  __syncthreads();
  const float inverse_rms = warp_sums[0];
  float maximum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(row_input[column]);
    const float scale = __bfloat162float(weight[column]);
    float result = value * inverse_rms * scale;
    if (residual != nullptr)
      result += __bfloat162float(residual[row * width + column]);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(result);
    row_output[column] = rounded;
    if constexpr (emit_row_amax) {
      maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
    }
  }
  if constexpr (emit_row_amax) {
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if ((threadIdx.x & 31U) == 0U)
      warp_sums[threadIdx.x >> 5U] = maximum;
    __syncthreads();
    if (threadIdx.x < 32U) {
      maximum = warp_sums[threadIdx.x];
      for (unsigned offset = 16; offset > 0; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
      }
      if (threadIdx.x == 0U)
        row_amax[row] = maximum;
    }
  }
}

template <bool emit_row_amax>
__global__ void
rms_norm_add_and_norm_kernel(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                             const __nv_bfloat16 *residual, const __nv_bfloat16 *next_weight,
                             __nv_bfloat16 *residual_output, __nv_bfloat16 *normalized_output,
                             float *normalized_row_amax, std::size_t width, float epsilon) {
  const std::size_t row = blockIdx.x;
  const __nv_bfloat16 *row_input = input + row * width;
  const __nv_bfloat16 *row_residual = residual + row * width;
  __nv_bfloat16 *row_residual_output = residual_output + row * width;
  __nv_bfloat16 *row_normalized_output = normalized_output + row * width;
  extern __shared__ __align__(16) unsigned char storage[];
  auto *first_residual = reinterpret_cast<__nv_bfloat16 *>(storage);
  __shared__ float warp_sums[32];

  float square_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(row_input[column]);
    square_sum += value * value;
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
  }
  if ((threadIdx.x & 31U) == 0U)
    warp_sums[threadIdx.x >> 5U] = square_sum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    square_sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_sums[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (threadIdx.x == 0U)
      warp_sums[0] = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
  }
  __syncthreads();
  const float first_inverse_rms = warp_sums[0];
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const __nv_bfloat16 value = __float2bfloat16_rn(
        __bfloat162float(row_input[column]) * first_inverse_rms * __bfloat162float(weight[column]) +
        __bfloat162float(row_residual[column]));
    first_residual[column] = value;
    row_residual_output[column] = value;
  }
  __syncthreads();

  square_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(first_residual[column]);
    square_sum += value * value;
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
  }
  if ((threadIdx.x & 31U) == 0U)
    warp_sums[threadIdx.x >> 5U] = square_sum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    square_sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_sums[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (threadIdx.x == 0U)
      warp_sums[0] = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
  }
  __syncthreads();
  const float second_inverse_rms = warp_sums[0];
  float maximum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const __nv_bfloat16 rounded =
        __float2bfloat16_rn(__bfloat162float(first_residual[column]) * second_inverse_rms *
                            __bfloat162float(next_weight[column]));
    row_normalized_output[column] = rounded;
    if constexpr (emit_row_amax) {
      maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
    }
  }
  if constexpr (emit_row_amax) {
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if ((threadIdx.x & 31U) == 0U)
      warp_sums[threadIdx.x >> 5U] = maximum;
    __syncthreads();
    if (threadIdx.x < 32U) {
      maximum = warp_sums[threadIdx.x];
      for (unsigned offset = 16; offset > 0; offset >>= 1U) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
      }
      if (threadIdx.x == 0U)
        normalized_row_amax[row] = maximum;
    }
  }
}

__global__ void rms_norm_add_scale_kernel(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                                          const __nv_bfloat16 *residual,
                                          const __nv_bfloat16 *scalar, __nv_bfloat16 *output,
                                          std::size_t width, float epsilon) {
  const std::size_t row = blockIdx.x;
  const __nv_bfloat16 *row_input = input + row * width;
  __nv_bfloat16 *row_output = output + row * width;
  float square_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(row_input[column]);
    square_sum += value * value;
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
  }
  __shared__ float warp_sums[32];
  if ((threadIdx.x & 31U) == 0U)
    warp_sums[threadIdx.x >> 5U] = square_sum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    square_sum = threadIdx.x < (blockDim.x + 31U) / 32U ? warp_sums[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      square_sum += __shfl_down_sync(0xffffffffU, square_sum, offset);
    }
    if (threadIdx.x == 0U)
      warp_sums[0] = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
  }
  __syncthreads();
  const float inverse_rms = warp_sums[0];
  const float layer_scale = __bfloat162float(*scalar);
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const __nv_bfloat16 residual_value = __float2bfloat16_rn(
        __bfloat162float(row_input[column]) * inverse_rms * __bfloat162float(weight[column]) +
        __bfloat162float(residual[row * width + column]));
    row_output[column] = __float2bfloat16_rn(__bfloat162float(residual_value) * layer_scale);
  }
}

__global__ void scale_kernel(__nv_bfloat16 *values, const __nv_bfloat16 *scalar,
                             std::size_t elements) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] =
        __float2bfloat16_rn(__bfloat162float(values[index]) * __bfloat162float(*scalar));
  }
}

__device__ float gelu_tanh(float value) {
  constexpr float coefficient = 0.7978845608028654F;
  return 0.5F * value * (1.0F + tanhf(coefficient * (value + 0.044715F * value * value * value)));
}

__global__ void gelu_tanh_gate_kernel(const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
                                      __nv_bfloat16 *output, std::size_t elements) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] =
        __float2bfloat16_rn(gelu_tanh(__bfloat162float(gate[index])) * __bfloat162float(up[index]));
  }
}

__global__ void gelu_tanh_gate_rows_kernel(const __nv_bfloat16 *packed, __nv_bfloat16 *output,
                                           int rows, int width) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * width) {
    const int row = index / width;
    const int column = index % width;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * 2 * width;
    output[index] = __float2bfloat16_rn(gelu_tanh(__bfloat162float(packed[base + column])) *
                                        __bfloat162float(packed[base + width + column]));
  }
}

__global__ void gelu_tanh_gate_rows_block_amax_kernel(const __nv_bfloat16 *packed,
                                                      __nv_bfloat16 *output, float *block_amax,
                                                      int width, int chunks_per_row) {
  const int row = static_cast<int>(blockIdx.x) / chunks_per_row;
  const int chunk = static_cast<int>(blockIdx.x) % chunks_per_row;
  const int column = chunk * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  float maximum = 0.0F;
  if (column < width) {
    const std::uint64_t base = static_cast<std::uint64_t>(row) * 2 * width;
    const __nv_bfloat16 rounded =
        __float2bfloat16_rn(gelu_tanh(__bfloat162float(packed[base + column])) *
                            __bfloat162float(packed[base + width + column]));
    output[static_cast<std::uint64_t>(row) * width + column] = rounded;
    maximum = fabsf(__bfloat162float(rounded));
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U)
    warp_maxima[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    maximum = threadIdx.x < 8U ? warp_maxima[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (threadIdx.x == 0U)
      block_amax[blockIdx.x] = maximum;
  }
}

__global__ void reduce_block_amax_to_rows_kernel(const float *block_amax, float *row_amax,
                                                 int blocks_per_row) {
  const int row = static_cast<int>(blockIdx.x);
  float maximum = 0.0F;
  for (int block = static_cast<int>(threadIdx.x); block < blocks_per_row;
       block += static_cast<int>(blockDim.x)) {
    maximum = fmaxf(maximum, block_amax[row * blocks_per_row + block]);
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U)
    warp_maxima[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    maximum = threadIdx.x < 8U ? warp_maxima[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (threadIdx.x == 0U)
      row_amax[row] = maximum;
  }
}

__global__ void embedding_kernel(const __nv_bfloat16 *weight, int token_id, __nv_bfloat16 *output,
                                 int hidden_size, __nv_bfloat16 scale) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < hidden_size) {
    output[index] = __float2bfloat16_rn(
        __bfloat162float(weight[static_cast<std::size_t>(token_id) * hidden_size + index]) *
        __bfloat162float(scale));
  }
}

__global__ void embedding_batch_kernel(const __nv_bfloat16 *weight, const int *token_ids,
                                       __nv_bfloat16 *output, int batch, int vocabulary_size,
                                       int hidden_size, __nv_bfloat16 scale) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < batch * hidden_size) {
    const int row = index / hidden_size;
    const int column = index % hidden_size;
    const int token_id = token_ids[row];
    if (token_id >= 0 && token_id < vocabulary_size) {
      output[index] = __float2bfloat16_rn(
          __bfloat162float(weight[static_cast<std::size_t>(token_id) * hidden_size + column]) *
          __bfloat162float(scale));
    }
  }
}

__global__ void gather_rows_kernel(const __nv_bfloat16 *input, const int *row_indices,
                                   __nv_bfloat16 *output, int rows, int width) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * width) {
    const int output_row = index / width;
    const int column = index % width;
    output[index] = input[static_cast<std::size_t>(row_indices[output_row]) * width + column];
  }
}

__global__ void scatter_rows_kernel(const __nv_bfloat16 *input, const int *row_indices,
                                    __nv_bfloat16 *output, int rows, int width) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * width) {
    const int input_row = index / width;
    const int column = index % width;
    output[static_cast<std::size_t>(row_indices[input_row]) * width + column] = input[index];
  }
}

__global__ void concatenate_rows_kernel(const __nv_bfloat16 *left, int left_width,
                                        const __nv_bfloat16 *right, int right_width,
                                        __nv_bfloat16 *output, int rows) {
  const int width = left_width + right_width;
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * width)
    return;
  const int row = index / width;
  const int column = index % width;
  output[index] = column < left_width
                      ? left[static_cast<std::size_t>(row) * left_width + column]
                      : right[static_cast<std::size_t>(row) * right_width + column - left_width];
}

__global__ void token_head_transpose_kernel(const __nv_bfloat16 *input, __nv_bfloat16 *output,
                                            int tokens, int heads, int dimension,
                                            bool token_to_head) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = tokens * heads * dimension;
  if (index < elements) {
    const int component = index % dimension;
    const int head = (index / dimension) % heads;
    const int token = index / (dimension * heads);
    const std::uint64_t token_major =
        (static_cast<std::uint64_t>(token) * heads + head) * dimension + component;
    const std::uint64_t head_major =
        (static_cast<std::uint64_t>(head) * tokens + token) * dimension + component;
    if (token_to_head) {
      output[head_major] = input[token_major];
    } else {
      output[token_major] = input[head_major];
    }
  }
}

__global__ void token_head_batch_transpose_kernel(const __nv_bfloat16 *input, __nv_bfloat16 *output,
                                                  int requests, int tokens, int heads,
                                                  int dimension, bool token_to_head) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int request_elements = tokens * heads * dimension;
  if (index >= requests * request_elements)
    return;
  const int request = index / request_elements;
  const int local = index % request_elements;
  const int component = local % dimension;
  const int head = (local / dimension) % heads;
  const int token = local / (dimension * heads);
  const std::uint64_t token_major =
      (static_cast<std::uint64_t>(request) * tokens * heads + token * heads + head) * dimension +
      component;
  const std::uint64_t head_major =
      (static_cast<std::uint64_t>(request) * heads * tokens + head * tokens + token) * dimension +
      component;
  if (token_to_head) {
    output[head_major] = input[token_major];
  } else {
    output[token_major] = input[head_major];
  }
}

__global__ void head_to_token_batch_block_amax_kernel(const __nv_bfloat16 *input,
                                                      __nv_bfloat16 *output, float *block_amax,
                                                      int tokens, int heads, int dimension,
                                                      int blocks_per_row) {
  const int row = static_cast<int>(blockIdx.x) / blocks_per_row;
  const int row_block = static_cast<int>(blockIdx.x) % blocks_per_row;
  const int width = heads * dimension;
  const int column = row_block * blockDim.x + threadIdx.x;
  const int request = row / tokens;
  const int token = row % tokens;
  float maximum = 0.0F;
  if (column < width) {
    const int head = column / dimension;
    const int component = column % dimension;
    const std::uint64_t source_index =
        (static_cast<std::uint64_t>(request) * heads * tokens + head * tokens + token) * dimension +
        component;
    const __nv_bfloat16 value = input[source_index];
    output[static_cast<std::uint64_t>(row) * width + column] = value;
    maximum = fabsf(__bfloat162float(value));
  }
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U)
    warp_maxima[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    maximum = threadIdx.x < 8U ? warp_maxima[threadIdx.x] : 0.0F;
    for (unsigned offset = 16; offset > 0; offset >>= 1U) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (threadIdx.x == 0U)
      block_amax[blockIdx.x] = maximum;
  }
}

__global__ void copy_kv_to_cache_kernel(const __nv_bfloat16 *source, __nv_bfloat16 *destination,
                                        int heads, int tokens, int dimension,
                                        int destination_capacity, int position_start) {
  const int retained = min(tokens, destination_capacity);
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = heads * retained * dimension;
  if (index < elements) {
    const int component = index % dimension;
    const int retained_token = (index / dimension) % retained;
    const int head = index / (dimension * retained);
    const int source_token = tokens - retained + retained_token;
    const int logical_position = position_start + source_token;
    const int destination_token = logical_position % destination_capacity;
    const std::uint64_t source_index =
        (static_cast<std::uint64_t>(head) * tokens + source_token) * dimension + component;
    const std::uint64_t destination_index =
        (static_cast<std::uint64_t>(head) * destination_capacity + destination_token) * dimension +
        component;
    destination[destination_index] = source[source_index];
  }
}

__global__ void gather_kv_ring_kernel(const __nv_bfloat16 *source, __nv_bfloat16 *destination,
                                      int heads, int retained_tokens, int dimension,
                                      int source_capacity, int destination_capacity,
                                      int logical_position_start) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = heads * retained_tokens * dimension;
  if (index < elements) {
    const int component = index % dimension;
    const int token = (index / dimension) % retained_tokens;
    const int head = index / (dimension * retained_tokens);
    const int source_token = (logical_position_start + token) % source_capacity;
    const std::uint64_t source_index =
        (static_cast<std::uint64_t>(head) * source_capacity + source_token) * dimension + component;
    const std::uint64_t destination_index =
        (static_cast<std::uint64_t>(head) * destination_capacity + token) * dimension + component;
    destination[destination_index] = source[source_index];
  }
}

__global__ void copy_kv_span_to_ring_kernel(const __nv_bfloat16 *source, __nv_bfloat16 *destination,
                                            int heads, int tokens, int dimension,
                                            int source_capacity, int source_position_start,
                                            int destination_capacity, int logical_position_start) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = heads * tokens * dimension;
  if (index < elements) {
    const int component = index % dimension;
    const int token = (index / dimension) % tokens;
    const int head = index / (dimension * tokens);
    const int destination_token = (logical_position_start + token) % destination_capacity;
    const std::uint64_t source_index =
        (static_cast<std::uint64_t>(head) * source_capacity + source_position_start + token) *
            dimension +
        component;
    const std::uint64_t destination_index =
        (static_cast<std::uint64_t>(head) * destination_capacity + destination_token) * dimension +
        component;
    destination[destination_index] = source[source_index];
  }
}

__global__ void quantize_global_k_cache_span_kernel(const __nv_bfloat16 *source,
                                                    __nv_fp8_e4m3 *destination, int heads,
                                                    int tokens, int dimension, int cache_capacity,
                                                    int position_start) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = heads * tokens * dimension;
  if (index >= elements)
    return;
  const int component = index % dimension;
  const int token = (index / dimension) % tokens;
  const int head = index / (dimension * tokens);
  const std::uint64_t cache_index =
      (static_cast<std::uint64_t>(head) * cache_capacity + position_start + token) * dimension +
      component;
  destination[cache_index] = __nv_fp8_e4m3(__bfloat162float(source[cache_index]) * 1024.0F);
}

__global__ void quantize_global_decode_qk_kernel(
    const __nv_bfloat16 *queries, const __nv_bfloat16 *key_cache, __nv_fp8_e4m3 *fp8_queries,
    __nv_fp8_e4m3 *fp8_key_cache, const int *positions, const int *slots, int batch,
    int maximum_slots, int query_heads, int kv_heads, int dimension, int cache_capacity) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int query_elements = batch * query_heads * dimension;
  const int key_elements = batch * kv_heads * dimension;
  if (index < query_elements) {
    fp8_queries[index] = __nv_fp8_e4m3(__bfloat162float(queries[index]) * 64.0F);
    return;
  }
  const int key_index = index - query_elements;
  if (key_index >= key_elements)
    return;
  const int component = key_index % dimension;
  const int head = (key_index / dimension) % kv_heads;
  const int row = key_index / (dimension * kv_heads);
  const int slot = slots[row];
  if (slot < 0 || slot >= maximum_slots)
    return;
  const int position = positions[row];
  if (position < 0 || position >= cache_capacity)
    return;
  const std::uint64_t cache_index =
      ((static_cast<std::uint64_t>(slot) * kv_heads + head) * cache_capacity + position) *
          dimension +
      component;
  fp8_key_cache[cache_index] = __nv_fp8_e4m3(__bfloat162float(key_cache[cache_index]) * 1024.0F);
}

__device__ float warp_maximum(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ void roundtrip_symmetric_int8_row(__nv_bfloat16 *row, int dimension) {
  float maximum = 0.0F;
  for (int component = static_cast<int>(threadIdx.x); component < dimension;
       component += static_cast<int>(blockDim.x)) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(row[component])));
  }
  maximum = warp_maximum(maximum);
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U) {
    warp_maxima[threadIdx.x >> 5U] = maximum;
  }
  __syncthreads();
  if (threadIdx.x < 32U) {
    maximum = threadIdx.x < blockDim.x / 32 ? warp_maxima[threadIdx.x] : 0.0F;
    maximum = warp_maximum(maximum);
    if (threadIdx.x == 0U)
      warp_maxima[0] = maximum;
  }
  __syncthreads();
  maximum = warp_maxima[0];
  if (maximum == 0.0F)
    return;
  const float quantization_scale = 127.0F / maximum;
  const float dequantization_scale = maximum / 127.0F;
  for (int component = static_cast<int>(threadIdx.x); component < dimension;
       component += static_cast<int>(blockDim.x)) {
    const float value = __bfloat162float(row[component]);
    const int quantized = max(-127, min(127, __float2int_rn(value * quantization_scale)));
    row[component] = __float2bfloat16_rn(static_cast<float>(quantized) * dequantization_scale);
  }
}

__global__ void quantize_bf16_rows_symmetric_int8_kernel(const __nv_bfloat16 *source,
                                                         std::int8_t *destination,
                                                         float *block_scales, int width,
                                                         int block_width) {
  const int blocks_per_row = width / block_width;
  const int block = static_cast<int>(blockIdx.x);
  const int row = block / blocks_per_row;
  const int component_block = block - row * blocks_per_row;
  const std::uint64_t offset =
      static_cast<std::uint64_t>(row) * width + component_block * block_width;
  const __nv_bfloat16 *row_source = source + offset;
  std::int8_t *row_destination = destination + offset;
  float maximum = 0.0F;
  for (int column = static_cast<int>(threadIdx.x); column < block_width;
       column += static_cast<int>(blockDim.x)) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(row_source[column])));
  }
  maximum = warp_maximum(maximum);
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31U) == 0U)
    warp_maxima[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    maximum = threadIdx.x < blockDim.x / 32 ? warp_maxima[threadIdx.x] : 0.0F;
    maximum = warp_maximum(maximum);
    if (threadIdx.x == 0U)
      warp_maxima[0] = maximum;
  }
  __syncthreads();
  maximum = warp_maxima[0];
  const float dequantization_scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
  if (threadIdx.x == 0U)
    block_scales[block] = dequantization_scale;
  const float quantization_scale = 1.0F / dequantization_scale;
  for (int column = static_cast<int>(threadIdx.x); column < block_width;
       column += static_cast<int>(blockDim.x)) {
    const int quantized = max(
        -127, min(127, __float2int_rn(__bfloat162float(row_source[column]) * quantization_scale)));
    row_destination[column] = static_cast<std::int8_t>(quantized);
  }
}

__device__ void quantize_bf16_block_symmetric_int8(const __nv_bfloat16 *source,
                                                   std::int8_t *destination, float *scale,
                                                   int block_width, float *warp_maxima) {
  float maximum = 0.0F;
  for (int column = static_cast<int>(threadIdx.x); column < block_width;
       column += static_cast<int>(blockDim.x)) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(source[column])));
  }
  maximum = warp_maximum(maximum);
  if ((threadIdx.x & 31U) == 0U)
    warp_maxima[threadIdx.x >> 5U] = maximum;
  __syncthreads();
  if (threadIdx.x < 32U) {
    maximum = threadIdx.x < blockDim.x / 32 ? warp_maxima[threadIdx.x] : 0.0F;
    maximum = warp_maximum(maximum);
    if (threadIdx.x == 0U)
      warp_maxima[0] = maximum;
  }
  __syncthreads();
  maximum = warp_maxima[0];
  const float dequantization_scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
  if (threadIdx.x == 0U)
    *scale = dequantization_scale;
  const float quantization_scale = 1.0F / dequantization_scale;
  for (int column = static_cast<int>(threadIdx.x); column < block_width;
       column += static_cast<int>(blockDim.x)) {
    const int quantized =
        max(-127, min(127, __float2int_rn(__bfloat162float(source[column]) * quantization_scale)));
    destination[column] = static_cast<std::int8_t>(quantized);
  }
}

__global__ void quantize_global_k_cache_span_int8_block128_kernel(const __nv_bfloat16 *source,
                                                                  std::int8_t *destination,
                                                                  float *scales, int heads,
                                                                  int tokens, int cache_capacity,
                                                                  int position_start) {
  constexpr int dimension = 512;
  constexpr int block_width = 128;
  constexpr int blocks_per_row = dimension / block_width;
  const int combined_block = static_cast<int>(blockIdx.x);
  const int component_block = combined_block % blocks_per_row;
  const int row = combined_block / blocks_per_row;
  const int token = row % tokens;
  const int head = row / tokens;
  const std::uint64_t cache_row =
      static_cast<std::uint64_t>(head) * cache_capacity + position_start + token;
  const std::uint64_t value_offset = cache_row * dimension + component_block * block_width;
  __shared__ float warp_maxima[4];
  quantize_bf16_block_symmetric_int8(source + value_offset, destination + value_offset,
                                     scales + cache_row * blocks_per_row + component_block,
                                     block_width, warp_maxima);
}

__global__ void quantize_global_decode_k_int8_block128_kernel(
    const __nv_bfloat16 *source, std::int8_t *destination, float *scales, const int *positions,
    const int *slots, int batch, int maximum_slots, int heads, int cache_capacity) {
  constexpr int dimension = 512;
  constexpr int block_width = 128;
  constexpr int blocks_per_row = dimension / block_width;
  const int combined_block = static_cast<int>(blockIdx.x);
  const int component_block = combined_block % blocks_per_row;
  const int head_row = combined_block / blocks_per_row;
  const int head = head_row % heads;
  const int row = head_row / heads;
  if (row >= batch)
    return;
  const int slot = slots[row];
  const int position = positions[row];
  if (slot < 0 || slot >= maximum_slots || position < 0 || position >= cache_capacity)
    return;
  const std::uint64_t cache_row =
      (static_cast<std::uint64_t>(slot) * heads + head) * cache_capacity + position;
  const std::uint64_t value_offset = cache_row * dimension + component_block * block_width;
  __shared__ float warp_maxima[4];
  quantize_bf16_block_symmetric_int8(source + value_offset, destination + value_offset,
                                     scales + cache_row * blocks_per_row + component_block,
                                     block_width, warp_maxima);
}

__global__ void roundtrip_global_kv_int8_per_token_kernel(__nv_bfloat16 *keys,
                                                          __nv_bfloat16 *values, int heads,
                                                          int tokens, int dimension,
                                                          int cache_capacity, int position_start,
                                                          bool quantize_keys, bool quantize_values,
                                                          int block_width) {
  const int rows = heads * tokens;
  const int blocks_per_row = dimension / block_width;
  const int tensor_blocks = rows * blocks_per_row;
  const int combined_block = static_cast<int>(blockIdx.x);
  const bool is_value = quantize_values && (!quantize_keys || combined_block >= tensor_blocks);
  const int local_block = combined_block - (is_value && quantize_keys ? tensor_blocks : 0);
  const int row_index = local_block / blocks_per_row;
  const int component_block = local_block - row_index * blocks_per_row;
  const int head = row_index / tokens;
  const int token = row_index - head * tokens;
  __nv_bfloat16 *cache = is_value ? values : keys;
  cache +=
      (static_cast<std::uint64_t>(head) * cache_capacity + position_start + token) * dimension +
      component_block * block_width;
  roundtrip_symmetric_int8_row(cache, block_width);
}

__global__ void roundtrip_global_decode_kv_int8_per_token_kernel(
    __nv_bfloat16 *keys, __nv_bfloat16 *values, const int *positions, const int *slots, int batch,
    int maximum_slots, int heads, int dimension, int cache_capacity, bool quantize_keys,
    bool quantize_values, int block_width) {
  const int rows = batch * heads;
  const int blocks_per_row = dimension / block_width;
  const int tensor_blocks = rows * blocks_per_row;
  const int combined_block = static_cast<int>(blockIdx.x);
  const bool is_value = quantize_values && (!quantize_keys || combined_block >= tensor_blocks);
  const int local_block = combined_block - (is_value && quantize_keys ? tensor_blocks : 0);
  const int row_index = local_block / blocks_per_row;
  const int component_block = local_block - row_index * blocks_per_row;
  const int request = row_index / heads;
  const int head = row_index - request * heads;
  const int slot = slots[request];
  const int position = positions[request];
  if (slot < 0 || slot >= maximum_slots || position < 0 || position >= cache_capacity)
    return;
  __nv_bfloat16 *cache = is_value ? values : keys;
  cache +=
      ((static_cast<std::uint64_t>(slot) * heads + head) * cache_capacity + position) * dimension +
      component_block * block_width;
  roundtrip_symmetric_int8_row(cache, block_width);
}

__global__ void gather_kv_rings_kernel(const __nv_bfloat16 *const *key_sources,
                                       const __nv_bfloat16 *const *value_sources,
                                       __nv_bfloat16 *staged_keys, __nv_bfloat16 *staged_values,
                                       const int *position_starts, int requests, int heads,
                                       int retained_tokens, int dimension, int source_capacity,
                                       int staging_capacity) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int request_elements = heads * retained_tokens * dimension;
  if (index >= requests * request_elements)
    return;
  const int request = index / request_elements;
  const int local = index % request_elements;
  const int component = local % dimension;
  const int token = (local / dimension) % retained_tokens;
  const int head = local / (dimension * retained_tokens);
  const int source_token = (position_starts[request] - retained_tokens + token) % source_capacity;
  const std::uint64_t source_index =
      (static_cast<std::uint64_t>(head) * source_capacity + source_token) * dimension + component;
  const std::uint64_t destination_index =
      ((static_cast<std::uint64_t>(request) * heads + head) * staging_capacity + token) *
          dimension +
      component;
  staged_keys[destination_index] = key_sources[request][source_index];
  staged_values[destination_index] = value_sources[request][source_index];
}

__global__ void copy_kv_spans_to_rings_kernel(
    const __nv_bfloat16 *staged_keys, const __nv_bfloat16 *staged_values,
    __nv_bfloat16 *const *key_destinations, __nv_bfloat16 *const *value_destinations,
    const int *position_starts, int requests, int heads, int tokens, int dimension,
    int staging_capacity, int staging_position_start, int destination_capacity) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int request_elements = heads * tokens * dimension;
  if (index >= requests * request_elements)
    return;
  const int request = index / request_elements;
  const int local = index % request_elements;
  const int component = local % dimension;
  const int token = (local / dimension) % tokens;
  const int head = local / (dimension * tokens);
  const int destination_token = (position_starts[request] + token) % destination_capacity;
  const std::uint64_t source_index =
      ((static_cast<std::uint64_t>(request) * heads + head) * staging_capacity +
       staging_position_start + token) *
          dimension +
      component;
  const std::uint64_t destination_index =
      (static_cast<std::uint64_t>(head) * destination_capacity + destination_token) * dimension +
      component;
  key_destinations[request][destination_index] = staged_keys[source_index];
  value_destinations[request][destination_index] = staged_values[source_index];
}

__global__ void backup_kv_ring_spans_kernel(const __nv_bfloat16 *const *key_sources,
                                            const __nv_bfloat16 *const *value_sources,
                                            __nv_bfloat16 *backup_keys,
                                            __nv_bfloat16 *backup_values,
                                            const int *position_starts, int requests, int heads,
                                            int tokens, int dimension, int source_capacity) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int request_elements = heads * tokens * dimension;
  if (index >= requests * request_elements)
    return;
  const int request = index / request_elements;
  const int local = index % request_elements;
  const int component = local % dimension;
  const int token = (local / dimension) % tokens;
  const int head = local / (dimension * tokens);
  const int source_token = (position_starts[request] + token) % source_capacity;
  const std::uint64_t source_index =
      (static_cast<std::uint64_t>(head) * source_capacity + source_token) * dimension + component;
  backup_keys[index] = key_sources[request][source_index];
  backup_values[index] = value_sources[request][source_index];
}

__global__ void restore_kv_ring_suffixes_kernel(
    const __nv_bfloat16 *backup_keys, const __nv_bfloat16 *backup_values,
    __nv_bfloat16 *const *key_destinations, __nv_bfloat16 *const *value_destinations,
    const int *position_starts, const int *accepted_tokens, int requests, int heads, int tokens,
    int dimension, int destination_capacity) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int request_elements = heads * tokens * dimension;
  if (index >= requests * request_elements)
    return;
  const int request = index / request_elements;
  const int local = index % request_elements;
  const int component = local % dimension;
  const int token = (local / dimension) % tokens;
  if (token < accepted_tokens[request])
    return;
  const int head = local / (dimension * tokens);
  const int destination_token = (position_starts[request] + token) % destination_capacity;
  const std::uint64_t destination_index =
      (static_cast<std::uint64_t>(head) * destination_capacity + destination_token) * dimension +
      component;
  key_destinations[request][destination_index] = backup_keys[index];
  value_destinations[request][destination_index] = backup_values[index];
}

__global__ void
clone_kv_cache_prefix_kernel(const __nv_bfloat16 *source_keys, const __nv_bfloat16 *source_values,
                             __nv_bfloat16 *destination_keys, __nv_bfloat16 *destination_values,
                             int heads, int resident_tokens, int dimension, int cache_capacity) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = heads * resident_tokens * dimension;
  if (index >= elements)
    return;
  const int component = index % dimension;
  const int token = (index / dimension) % resident_tokens;
  const int head = index / (dimension * resident_tokens);
  const std::uint64_t cache_index =
      (static_cast<std::uint64_t>(head) * cache_capacity + token) * dimension + component;
  destination_keys[cache_index] = source_keys[cache_index];
  destination_values[cache_index] = source_values[cache_index];
}

struct Maximum {
  float value;
  int index;
};

__device__ Maximum better(Maximum left, Maximum right) {
  return right.value > left.value || (right.value == left.value && right.index < left.index) ? right
                                                                                             : left;
}

__global__ void argmax_partials_kernel(const __nv_bfloat16 *values, int elements,
                                       Maximum *partials) {
  Maximum current{-CUDART_INF_F, 0};
  for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    current = better(current, {__bfloat162float(values[index]), index});
  }
  __shared__ Maximum reduction[256];
  reduction[threadIdx.x] = current;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] = better(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partials[blockIdx.x] = reduction[0];
}

__global__ void argmax_final_kernel(const Maximum *partials, int count, int *output) {
  Maximum current{-CUDART_INF_F, 0};
  for (int index = threadIdx.x; index < count; index += blockDim.x)
    current = better(current, partials[index]);
  __shared__ Maximum reduction[256];
  reduction[threadIdx.x] = current;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] = better(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    *output = reduction[0].index;
}

__global__ void argmax_rows_partials_kernel(const __nv_bfloat16 *values, int rows, int elements,
                                            int partial_count, Maximum *partials) {
  const int row = static_cast<int>(blockIdx.y);
  if (row >= rows)
    return;
  const __nv_bfloat16 *row_values = values + static_cast<std::uint64_t>(row) * elements;
  Maximum current{-CUDART_INF_F, 0};
  for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    current = better(current, {__bfloat162float(row_values[index]), index});
  }
  __shared__ Maximum reduction[256];
  reduction[threadIdx.x] = current;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = better(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[static_cast<std::uint64_t>(row) * partial_count + blockIdx.x] = reduction[0];
  }
}

__global__ void argmax_rows_final_kernel(const Maximum *partials, int rows, int partial_count,
                                         int *output) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows)
    return;
  const Maximum *row_partials = partials + static_cast<std::uint64_t>(row) * partial_count;
  Maximum current{-CUDART_INF_F, 0};
  for (int index = threadIdx.x; index < partial_count; index += blockDim.x) {
    current = better(current, row_partials[index]);
  }
  __shared__ Maximum reduction[256];
  reduction[threadIdx.x] = current;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = better(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[row] = reduction[0].index;
}

} // namespace

void rms_norm_bf16(const void *input, const void *weight, void *output, std::size_t rows,
                   std::size_t width, float epsilon, cudaStream_t stream) {
  if (rows == 0 || width == 0)
    return;
  rms_norm_kernel<false><<<static_cast<unsigned>(rows), rms_threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<const __nv_bfloat16 *>(weight),
      nullptr, static_cast<__nv_bfloat16 *>(output), nullptr, width, epsilon);
  check(cudaPeekAtLastError(), "launch rms_norm_kernel");
}

void rms_norm_bf16_with_row_amax(const void *input, const void *weight, void *output,
                                 float *row_amax, std::size_t rows, std::size_t width,
                                 float epsilon, cudaStream_t stream) {
  if (rows == 0 || width == 0)
    return;
  if (row_amax == nullptr)
    throw std::runtime_error("null RMSNorm row-amax output");
  rms_norm_kernel<true><<<static_cast<unsigned>(rows), rms_threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<const __nv_bfloat16 *>(weight),
      nullptr, static_cast<__nv_bfloat16 *>(output), row_amax, width, epsilon);
  check(cudaPeekAtLastError(), "launch RMSNorm with row amax");
}

void rms_norm_add_bf16(const void *input, const void *weight, const void *residual, void *output,
                       std::size_t rows, std::size_t width, float epsilon, cudaStream_t stream) {
  if (rows == 0 || width == 0)
    return;
  rms_norm_kernel<false><<<static_cast<unsigned>(rows), rms_threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<const __nv_bfloat16 *>(weight),
      static_cast<const __nv_bfloat16 *>(residual), static_cast<__nv_bfloat16 *>(output), nullptr,
      width, epsilon);
  check(cudaPeekAtLastError(), "launch rms_norm_add_kernel");
}

void rms_norm_add_and_norm_bf16(const void *input, const void *weight, const void *residual,
                                const void *next_weight, void *residual_output,
                                void *normalized_output, std::size_t rows, std::size_t width,
                                float epsilon, cudaStream_t stream) {
  if (rows == 0 || width == 0)
    return;
  rms_norm_add_and_norm_kernel<false>
      <<<static_cast<unsigned>(rows), rms_threads, width * sizeof(__nv_bfloat16), stream>>>(
          static_cast<const __nv_bfloat16 *>(input), static_cast<const __nv_bfloat16 *>(weight),
          static_cast<const __nv_bfloat16 *>(residual),
          static_cast<const __nv_bfloat16 *>(next_weight),
          static_cast<__nv_bfloat16 *>(residual_output),
          static_cast<__nv_bfloat16 *>(normalized_output), nullptr, width, epsilon);
  check(cudaPeekAtLastError(), "launch fused post-attention normalization");
}

void rms_norm_add_and_norm_bf16_with_row_amax(const void *input, const void *weight,
                                              const void *residual, const void *next_weight,
                                              void *residual_output, void *normalized_output,
                                              float *normalized_row_amax, std::size_t rows,
                                              std::size_t width, float epsilon,
                                              cudaStream_t stream) {
  if (rows == 0 || width == 0)
    return;
  if (normalized_row_amax == nullptr) {
    throw std::runtime_error("null fused normalization row-amax output");
  }
  rms_norm_add_and_norm_kernel<true>
      <<<static_cast<unsigned>(rows), rms_threads, width * sizeof(__nv_bfloat16), stream>>>(
          static_cast<const __nv_bfloat16 *>(input), static_cast<const __nv_bfloat16 *>(weight),
          static_cast<const __nv_bfloat16 *>(residual),
          static_cast<const __nv_bfloat16 *>(next_weight),
          static_cast<__nv_bfloat16 *>(residual_output),
          static_cast<__nv_bfloat16 *>(normalized_output), normalized_row_amax, width, epsilon);
  check(cudaPeekAtLastError(), "launch fused normalization with row amax");
}

void rms_norm_add_scale_bf16(const void *input, const void *weight, const void *residual,
                             const void *scalar, void *output, std::size_t rows, std::size_t width,
                             float epsilon, cudaStream_t stream) {
  if (rows == 0 || width == 0)
    return;
  rms_norm_add_scale_kernel<<<static_cast<unsigned>(rows), rms_threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<const __nv_bfloat16 *>(weight),
      static_cast<const __nv_bfloat16 *>(residual), static_cast<const __nv_bfloat16 *>(scalar),
      static_cast<__nv_bfloat16 *>(output), width, epsilon);
  check(cudaPeekAtLastError(), "launch fused post-MLP normalization and scale");
}

void scale_bf16(void *values, const void *scalar, std::size_t elements, cudaStream_t stream) {
  if (elements == 0)
    return;
  constexpr unsigned threads = 256;
  const auto blocks = static_cast<unsigned>((elements + threads - 1U) / threads);
  scale_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(values), static_cast<const __nv_bfloat16 *>(scalar), elements);
  check(cudaPeekAtLastError(), "launch scale_kernel");
}

void gelu_tanh_gate_bf16(const void *gate, const void *up, void *output, std::size_t elements,
                         cudaStream_t stream) {
  if (elements == 0)
    return;
  constexpr unsigned threads = 256;
  const auto blocks = static_cast<unsigned>((elements + threads - 1U) / threads);
  gelu_tanh_gate_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(gate), static_cast<const __nv_bfloat16 *>(up),
      static_cast<__nv_bfloat16 *>(output), elements);
  check(cudaPeekAtLastError(), "launch gelu_tanh_gate_kernel");
}

void gelu_tanh_gate_rows_bf16(const void *packed_gate_up, void *output, int rows, int width,
                              cudaStream_t stream) {
  if (rows <= 0 || width <= 0)
    throw std::runtime_error("invalid packed gate/up shape");
  constexpr int block_size = 256;
  const int elements = rows * width;
  gelu_tanh_gate_rows_kernel<<<(elements + block_size - 1) / block_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(packed_gate_up), static_cast<__nv_bfloat16 *>(output),
      rows, width);
  check(cudaPeekAtLastError(), "launch row-wise GELU gate kernel");
}

void gelu_tanh_gate_rows_bf16_with_block_amax(const void *packed_gate_up, void *output,
                                              float *block_amax, int rows, int width,
                                              cudaStream_t stream) {
  if (packed_gate_up == nullptr || output == nullptr || block_amax == nullptr || rows <= 0 ||
      width <= 0) {
    throw std::runtime_error("invalid packed gate/up block-amax shape");
  }
  constexpr int block_size = 256;
  const int chunks_per_row = (width + block_size - 1) / block_size;
  gelu_tanh_gate_rows_block_amax_kernel<<<rows * chunks_per_row, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(packed_gate_up), static_cast<__nv_bfloat16 *>(output),
      block_amax, width, chunks_per_row);
  check(cudaPeekAtLastError(), "launch row-wise GELU gate with block amax");
}

void reduce_block_amax_to_rows(const float *block_amax, float *row_amax, int rows,
                               int blocks_per_row, cudaStream_t stream) {
  if (block_amax == nullptr || row_amax == nullptr || rows <= 0 || blocks_per_row <= 0) {
    throw std::runtime_error("invalid block-amax reduction shape");
  }
  constexpr int block_size = 256;
  reduce_block_amax_to_rows_kernel<<<rows, block_size, 0, stream>>>(block_amax, row_amax,
                                                                    blocks_per_row);
  check(cudaPeekAtLastError(), "launch block-amax row reduction");
}

void embedding_bf16(const void *embedding_weight, int token_id, void *output, int vocabulary_size,
                    int hidden_size, float scale, cudaStream_t stream) {
  if (token_id < 0 || token_id >= vocabulary_size || hidden_size <= 0) {
    throw std::runtime_error("invalid embedding lookup");
  }
  constexpr int block_size = 256;
  embedding_kernel<<<(hidden_size + block_size - 1) / block_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(embedding_weight), token_id,
      static_cast<__nv_bfloat16 *>(output), hidden_size, __float2bfloat16_rn(scale));
  check(cudaPeekAtLastError(), "launch embedding kernel");
}

void embedding_batch_bf16(const void *embedding_weight, const int *device_token_ids, void *output,
                          int batch, int vocabulary_size, int hidden_size, float scale,
                          cudaStream_t stream) {
  if (batch <= 0 || device_token_ids == nullptr || vocabulary_size <= 0 || hidden_size <= 0) {
    throw std::runtime_error("invalid batched embedding lookup");
  }
  constexpr int block_size = 256;
  const int elements = batch * hidden_size;
  embedding_batch_kernel<<<(elements + block_size - 1) / block_size, block_size, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(embedding_weight), device_token_ids,
      static_cast<__nv_bfloat16 *>(output), batch, vocabulary_size, hidden_size,
      __float2bfloat16_rn(scale));
  check(cudaPeekAtLastError(), "launch batched embedding kernel");
}

void gather_rows_bf16(const void *input, const int *device_row_indices, void *output, int rows,
                      int width, cudaStream_t stream) {
  if (input == nullptr || device_row_indices == nullptr || output == nullptr || rows <= 0 ||
      width <= 0) {
    throw std::runtime_error("invalid gather rows shape");
  }
  constexpr int threads = 256;
  const int elements = rows * width;
  gather_rows_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), device_row_indices,
      static_cast<__nv_bfloat16 *>(output), rows, width);
  check(cudaPeekAtLastError(), "launch gather rows");
}

void scatter_rows_bf16(const void *input, const int *device_row_indices, void *output, int rows,
                       int width, cudaStream_t stream) {
  if (input == nullptr || device_row_indices == nullptr || output == nullptr || rows <= 0 ||
      width <= 0) {
    throw std::runtime_error("invalid scatter rows shape");
  }
  const int elements = rows * width;
  scatter_rows_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), device_row_indices,
      static_cast<__nv_bfloat16 *>(output), rows, width);
  check(cudaPeekAtLastError(), "launch scatter rows");
}

void concatenate_rows_bf16(const void *left, int left_width, const void *right, int right_width,
                           void *output, int rows, cudaStream_t stream) {
  if (left == nullptr || right == nullptr || output == nullptr || left_width <= 0 ||
      right_width <= 0 || rows <= 0) {
    throw std::runtime_error("invalid concatenate rows shape");
  }
  const int elements = rows * (left_width + right_width);
  concatenate_rows_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(left), left_width,
      static_cast<const __nv_bfloat16 *>(right), right_width, static_cast<__nv_bfloat16 *>(output),
      rows);
  check(cudaPeekAtLastError(), "launch concatenate rows");
}

void token_to_head_bf16(const void *input, void *output, int tokens, int heads, int head_dimension,
                        cudaStream_t stream) {
  if (input == nullptr || output == nullptr || tokens <= 0 || heads <= 0 || head_dimension <= 0) {
    throw std::runtime_error("invalid token-to-head transpose shape");
  }
  const int elements = tokens * heads * head_dimension;
  token_head_transpose_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_bfloat16 *>(output), tokens,
      heads, head_dimension, true);
  check(cudaPeekAtLastError(), "launch token-to-head transpose");
}

void head_to_token_bf16(const void *input, void *output, int tokens, int heads, int head_dimension,
                        cudaStream_t stream) {
  if (input == nullptr || output == nullptr || tokens <= 0 || heads <= 0 || head_dimension <= 0) {
    throw std::runtime_error("invalid head-to-token transpose shape");
  }
  const int elements = tokens * heads * head_dimension;
  token_head_transpose_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_bfloat16 *>(output), tokens,
      heads, head_dimension, false);
  check(cudaPeekAtLastError(), "launch head-to-token transpose");
}

void token_to_head_batch_bf16(const void *input, void *output, int requests, int tokens, int heads,
                              int head_dimension, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || requests <= 0 || tokens <= 0 || heads <= 0 ||
      head_dimension <= 0) {
    throw std::runtime_error("invalid batched token-to-head transpose shape");
  }
  const int elements = requests * tokens * heads * head_dimension;
  token_head_batch_transpose_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_bfloat16 *>(output), requests,
      tokens, heads, head_dimension, true);
  check(cudaPeekAtLastError(), "launch batched token-to-head transpose");
}

void head_to_token_batch_bf16(const void *input, void *output, int requests, int tokens, int heads,
                              int head_dimension, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || requests <= 0 || tokens <= 0 || heads <= 0 ||
      head_dimension <= 0) {
    throw std::runtime_error("invalid batched head-to-token transpose shape");
  }
  const int elements = requests * tokens * heads * head_dimension;
  token_head_batch_transpose_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_bfloat16 *>(output), requests,
      tokens, heads, head_dimension, false);
  check(cudaPeekAtLastError(), "launch batched head-to-token transpose");
}

void head_to_token_batch_bf16_with_block_amax(const void *input, void *output, float *block_amax,
                                              int requests, int tokens, int heads,
                                              int head_dimension, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || block_amax == nullptr || requests <= 0 ||
      tokens <= 0 || heads <= 0 || head_dimension <= 0) {
    throw std::runtime_error("invalid carried-amax batched head-to-token transpose shape");
  }
  constexpr int threads = 256;
  const int blocks_per_row = (heads * head_dimension + threads - 1) / threads;
  head_to_token_batch_block_amax_kernel<<<requests * tokens * blocks_per_row, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_bfloat16 *>(output), block_amax,
      tokens, heads, head_dimension, blocks_per_row);
  check(cudaPeekAtLastError(), "launch carried-amax batched head-to-token transpose");
}

void copy_kv_to_cache_bf16(const void *source, void *destination, int heads, int tokens,
                           int head_dimension, int destination_capacity, int position_start,
                           cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || heads <= 0 || tokens <= 0 ||
      head_dimension <= 0 || destination_capacity <= 0 || position_start < 0) {
    throw std::runtime_error("invalid KV cache copy shape");
  }
  const int elements = heads * std::min(tokens, destination_capacity) * head_dimension;
  copy_kv_to_cache_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<__nv_bfloat16 *>(destination), heads,
      tokens, head_dimension, destination_capacity, position_start);
  check(cudaPeekAtLastError(), "launch KV cache copy");
}

void quantize_global_k_cache_span_fp8(const void *source, void *destination, int heads, int tokens,
                                      int head_dimension, int cache_capacity, int position_start,
                                      cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || heads <= 0 || tokens <= 0 ||
      head_dimension != 512 || cache_capacity < position_start + tokens || position_start < 0) {
    throw std::runtime_error("invalid global FP8 K span shape");
  }
  const int elements = heads * tokens * head_dimension;
  quantize_global_k_cache_span_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<__nv_fp8_e4m3 *>(destination), heads,
      tokens, head_dimension, cache_capacity, position_start);
  check(cudaPeekAtLastError(), "launch global FP8 K span quantization");
}

void quantize_global_decode_qk_fp8(const void *queries, const void *key_cache, void *fp8_queries,
                                   void *fp8_key_cache, const int *positions, const int *slots,
                                   int batch, int maximum_slots, int query_heads, int kv_heads,
                                   int head_dimension, int cache_capacity, cudaStream_t stream) {
  if (queries == nullptr || key_cache == nullptr || fp8_queries == nullptr ||
      fp8_key_cache == nullptr || positions == nullptr || slots == nullptr || batch <= 0 ||
      maximum_slots < batch || query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
      head_dimension != 512 || cache_capacity <= 0) {
    throw std::runtime_error("invalid global FP8 decode QK shape");
  }
  const int elements = batch * (query_heads + kv_heads) * head_dimension;
  quantize_global_decode_qk_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(queries), static_cast<const __nv_bfloat16 *>(key_cache),
      static_cast<__nv_fp8_e4m3 *>(fp8_queries), static_cast<__nv_fp8_e4m3 *>(fp8_key_cache),
      positions, slots, batch, maximum_slots, query_heads, kv_heads, head_dimension,
      cache_capacity);
  check(cudaPeekAtLastError(), "launch global FP8 decode QK quantization");
}

void roundtrip_global_kv_int8_per_token_bf16(void *key_cache, void *value_cache, int heads,
                                             int tokens, int head_dimension, int cache_capacity,
                                             int position_start, cudaStream_t stream,
                                             bool quantize_keys, bool quantize_values,
                                             int block_width) {
  if (key_cache == nullptr || value_cache == nullptr || heads <= 0 || tokens <= 0 ||
      head_dimension != 512 || cache_capacity < position_start + tokens || position_start < 0 ||
      (!quantize_keys && !quantize_values) || block_width < 32 ||
      head_dimension % block_width != 0) {
    throw std::runtime_error("invalid global INT8 KV oracle span shape");
  }
  constexpr int threads = 256;
  const int tensors = static_cast<int>(quantize_keys) + static_cast<int>(quantize_values);
  const int blocks_per_row = head_dimension / block_width;
  roundtrip_global_kv_int8_per_token_kernel<<<tensors * heads * tokens * blocks_per_row, threads, 0,
                                              stream>>>(
      static_cast<__nv_bfloat16 *>(key_cache), static_cast<__nv_bfloat16 *>(value_cache), heads,
      tokens, head_dimension, cache_capacity, position_start, quantize_keys, quantize_values,
      block_width);
  check(cudaPeekAtLastError(), "launch global INT8 KV oracle span");
}

void roundtrip_global_decode_kv_int8_per_token_bf16(void *key_cache, void *value_cache,
                                                    const int *positions, const int *slots,
                                                    int batch, int maximum_slots, int heads,
                                                    int head_dimension, int cache_capacity,
                                                    cudaStream_t stream, bool quantize_keys,
                                                    bool quantize_values, int block_width) {
  if (key_cache == nullptr || value_cache == nullptr || positions == nullptr || slots == nullptr ||
      batch <= 0 || maximum_slots < batch || heads <= 0 || head_dimension != 512 ||
      cache_capacity <= 0 || (!quantize_keys && !quantize_values) || block_width < 32 ||
      head_dimension % block_width != 0) {
    throw std::runtime_error("invalid global INT8 KV oracle decode shape");
  }
  constexpr int threads = 256;
  const int tensors = static_cast<int>(quantize_keys) + static_cast<int>(quantize_values);
  const int blocks_per_row = head_dimension / block_width;
  roundtrip_global_decode_kv_int8_per_token_kernel<<<tensors * batch * heads * blocks_per_row,
                                                     threads, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(key_cache), static_cast<__nv_bfloat16 *>(value_cache), positions,
      slots, batch, maximum_slots, heads, head_dimension, cache_capacity, quantize_keys,
      quantize_values, block_width);
  check(cudaPeekAtLastError(), "launch global INT8 KV oracle decode");
}

void quantize_bf16_rows_symmetric_int8(const void *source, void *destination, float *row_scales,
                                       int rows, int width, cudaStream_t stream) {
  quantize_bf16_blocks_symmetric_int8(source, destination, row_scales, rows, width, width, stream);
}

void quantize_bf16_blocks_symmetric_int8(const void *source, void *destination, float *block_scales,
                                         int rows, int width, int block_width,
                                         cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || block_scales == nullptr || rows <= 0 ||
      width <= 0 || block_width <= 0 || width % block_width != 0) {
    throw std::runtime_error("invalid symmetric INT8 block quantization shape");
  }
  constexpr int threads = 256;
  quantize_bf16_rows_symmetric_int8_kernel<<<rows *(width / block_width), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<std::int8_t *>(destination),
      block_scales, width, block_width);
  check(cudaPeekAtLastError(), "launch symmetric INT8 block quantization");
}

void quantize_global_k_cache_span_int8_block128(const void *source, void *destination,
                                                float *scales, int heads, int tokens,
                                                int head_dimension, int cache_capacity,
                                                int position_start, cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || scales == nullptr || heads <= 0 ||
      tokens <= 0 || head_dimension != 512 || cache_capacity <= 0 || position_start < 0 ||
      position_start + tokens > cache_capacity) {
    throw std::runtime_error("invalid global INT8 K span shape");
  }
  constexpr int blocks_per_row = 4;
  quantize_global_k_cache_span_int8_block128_kernel<<<heads * tokens * blocks_per_row, 128, 0,
                                                      stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<std::int8_t *>(destination), scales,
      heads, tokens, cache_capacity, position_start);
  check(cudaPeekAtLastError(), "launch global INT8 K span quantization");
}

void quantize_global_decode_k_int8_block128(const void *source, void *destination, float *scales,
                                            const int *positions, const int *slots, int batch,
                                            int maximum_slots, int heads, int head_dimension,
                                            int cache_capacity, cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || scales == nullptr || positions == nullptr ||
      slots == nullptr || batch <= 0 || maximum_slots < batch || heads <= 0 ||
      head_dimension != 512 || cache_capacity <= 0) {
    throw std::runtime_error("invalid global INT8 decode K shape");
  }
  constexpr int blocks_per_row = 4;
  quantize_global_decode_k_int8_block128_kernel<<<batch * heads * blocks_per_row, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<std::int8_t *>(destination), scales,
      positions, slots, batch, maximum_slots, heads, cache_capacity);
  check(cudaPeekAtLastError(), "launch global INT8 decode K quantization");
}

void gather_kv_ring_bf16(const void *source, void *destination, int heads, int retained_tokens,
                         int head_dimension, int source_capacity, int destination_capacity,
                         int logical_position_start, cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || heads <= 0 || retained_tokens < 0 ||
      head_dimension <= 0 || source_capacity <= 0 || destination_capacity < retained_tokens ||
      logical_position_start < 0) {
    throw std::runtime_error("invalid KV ring gather shape");
  }
  if (retained_tokens == 0)
    return;
  const int elements = heads * retained_tokens * head_dimension;
  gather_kv_ring_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<__nv_bfloat16 *>(destination), heads,
      retained_tokens, head_dimension, source_capacity, destination_capacity,
      logical_position_start);
  check(cudaPeekAtLastError(), "launch KV ring gather");
}

void copy_kv_span_to_ring_bf16(const void *source, void *destination, int heads, int tokens,
                               int head_dimension, int source_capacity, int source_position_start,
                               int destination_capacity, int logical_position_start,
                               cudaStream_t stream) {
  if (source == nullptr || destination == nullptr || heads <= 0 || tokens <= 0 ||
      head_dimension <= 0 || source_capacity < source_position_start + tokens ||
      source_position_start < 0 || destination_capacity <= 0 || logical_position_start < 0) {
    throw std::runtime_error("invalid KV ring span copy shape");
  }
  const int elements = heads * tokens * head_dimension;
  copy_kv_span_to_ring_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source), static_cast<__nv_bfloat16 *>(destination), heads,
      tokens, head_dimension, source_capacity, source_position_start, destination_capacity,
      logical_position_start);
  check(cudaPeekAtLastError(), "launch KV ring span copy");
}

void gather_kv_rings_bf16(const void *const *device_key_sources,
                          const void *const *device_value_sources, void *staged_keys,
                          void *staged_values, const int *device_position_starts, int requests,
                          int heads, int retained_tokens, int head_dimension, int source_capacity,
                          int staging_capacity, cudaStream_t stream) {
  if (device_key_sources == nullptr || device_value_sources == nullptr || staged_keys == nullptr ||
      staged_values == nullptr || device_position_starts == nullptr || requests <= 0 ||
      heads <= 0 || retained_tokens <= 0 || head_dimension <= 0 || source_capacity <= 0 ||
      staging_capacity < retained_tokens) {
    throw std::runtime_error("invalid batched KV ring gather shape");
  }
  const int elements = requests * heads * retained_tokens * head_dimension;
  gather_kv_rings_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *const *>(device_key_sources),
      reinterpret_cast<const __nv_bfloat16 *const *>(device_value_sources),
      static_cast<__nv_bfloat16 *>(staged_keys), static_cast<__nv_bfloat16 *>(staged_values),
      device_position_starts, requests, heads, retained_tokens, head_dimension, source_capacity,
      staging_capacity);
  check(cudaPeekAtLastError(), "launch batched KV ring gather");
}

void copy_kv_spans_to_rings_bf16(const void *staged_keys, const void *staged_values,
                                 void *const *device_key_destinations,
                                 void *const *device_value_destinations,
                                 const int *device_position_starts, int requests, int heads,
                                 int tokens, int head_dimension, int staging_capacity,
                                 int staging_position_start, int destination_capacity,
                                 cudaStream_t stream) {
  if (staged_keys == nullptr || staged_values == nullptr || device_key_destinations == nullptr ||
      device_value_destinations == nullptr || device_position_starts == nullptr || requests <= 0 ||
      heads <= 0 || tokens <= 0 || head_dimension <= 0 || staging_position_start < 0 ||
      staging_capacity < staging_position_start + tokens || destination_capacity <= 0) {
    throw std::runtime_error("invalid batched KV ring copy shape");
  }
  const int elements = requests * heads * tokens * head_dimension;
  copy_kv_spans_to_rings_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(staged_keys),
      static_cast<const __nv_bfloat16 *>(staged_values),
      reinterpret_cast<__nv_bfloat16 *const *>(device_key_destinations),
      reinterpret_cast<__nv_bfloat16 *const *>(device_value_destinations), device_position_starts,
      requests, heads, tokens, head_dimension, staging_capacity, staging_position_start,
      destination_capacity);
  check(cudaPeekAtLastError(), "launch batched KV ring copy");
}

void backup_kv_ring_spans_bf16(const void *const *device_key_sources,
                               const void *const *device_value_sources, void *backup_keys,
                               void *backup_values, const int *device_position_starts, int requests,
                               int heads, int tokens, int head_dimension, int source_capacity,
                               cudaStream_t stream) {
  if (device_key_sources == nullptr || device_value_sources == nullptr || backup_keys == nullptr ||
      backup_values == nullptr || device_position_starts == nullptr || requests <= 0 ||
      heads <= 0 || tokens <= 0 || head_dimension <= 0 || source_capacity <= 0) {
    throw std::runtime_error("invalid speculative KV backup shape");
  }
  const int elements = requests * heads * tokens * head_dimension;
  backup_kv_ring_spans_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *const *>(device_key_sources),
      reinterpret_cast<const __nv_bfloat16 *const *>(device_value_sources),
      static_cast<__nv_bfloat16 *>(backup_keys), static_cast<__nv_bfloat16 *>(backup_values),
      device_position_starts, requests, heads, tokens, head_dimension, source_capacity);
  check(cudaPeekAtLastError(), "launch speculative KV backup");
}

void restore_kv_ring_suffixes_bf16(const void *backup_keys, const void *backup_values,
                                   void *const *device_key_destinations,
                                   void *const *device_value_destinations,
                                   const int *device_position_starts,
                                   const int *device_accepted_tokens, int requests, int heads,
                                   int tokens, int head_dimension, int destination_capacity,
                                   cudaStream_t stream) {
  if (backup_keys == nullptr || backup_values == nullptr || device_key_destinations == nullptr ||
      device_value_destinations == nullptr || device_position_starts == nullptr ||
      device_accepted_tokens == nullptr || requests <= 0 || heads <= 0 || tokens <= 0 ||
      head_dimension <= 0 || destination_capacity <= 0) {
    throw std::runtime_error("invalid speculative KV restore shape");
  }
  const int elements = requests * heads * tokens * head_dimension;
  restore_kv_ring_suffixes_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(backup_keys),
      static_cast<const __nv_bfloat16 *>(backup_values),
      reinterpret_cast<__nv_bfloat16 *const *>(device_key_destinations),
      reinterpret_cast<__nv_bfloat16 *const *>(device_value_destinations), device_position_starts,
      device_accepted_tokens, requests, heads, tokens, head_dimension, destination_capacity);
  check(cudaPeekAtLastError(), "launch speculative KV restore");
}

void clone_kv_cache_prefix_bf16(const void *source_keys, const void *source_values,
                                void *destination_keys, void *destination_values, int heads,
                                int resident_tokens, int head_dimension, int cache_capacity,
                                cudaStream_t stream) {
  if (source_keys == nullptr || source_values == nullptr || destination_keys == nullptr ||
      destination_values == nullptr || heads <= 0 || resident_tokens <= 0 ||
      resident_tokens > cache_capacity || head_dimension <= 0 || cache_capacity <= 0) {
    throw std::runtime_error("invalid KV cache prefix clone shape");
  }
  const int elements = heads * resident_tokens * head_dimension;
  clone_kv_cache_prefix_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(source_keys),
      static_cast<const __nv_bfloat16 *>(source_values),
      static_cast<__nv_bfloat16 *>(destination_keys),
      static_cast<__nv_bfloat16 *>(destination_values), heads, resident_tokens, head_dimension,
      cache_capacity);
  check(cudaPeekAtLastError(), "launch KV cache prefix clone");
}

std::size_t argmax_bf16_workspace_bytes(int elements) {
  if (elements <= 0)
    throw std::runtime_error("invalid argmax size");
  const int blocks = std::min(256, (elements + 255) / 256);
  return static_cast<std::size_t>(blocks) * sizeof(Maximum);
}

void argmax_bf16(const void *values, int elements, int *device_output, void *workspace,
                 cudaStream_t stream) {
  if (elements <= 0 || device_output == nullptr || workspace == nullptr) {
    throw std::runtime_error("invalid argmax arguments");
  }
  const int blocks = std::min(256, (elements + 255) / 256);
  auto *partials = static_cast<Maximum *>(workspace);
  argmax_partials_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(values),
                                                     elements, partials);
  argmax_final_kernel<<<1, 256, 0, stream>>>(partials, blocks, device_output);
  check(cudaPeekAtLastError(), "launch argmax kernels");
}

void argmax_bf16_rows(const void *values, int rows, int elements, int *device_output,
                      void *workspace, cudaStream_t stream) {
  if (values == nullptr || rows <= 0 || elements <= 0 || device_output == nullptr ||
      workspace == nullptr) {
    throw std::runtime_error("invalid batched argmax arguments");
  }
  const int blocks = std::min(256, (elements + 255) / 256);
  auto *partials = static_cast<Maximum *>(workspace);
  const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(rows));
  argmax_rows_partials_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(values),
                                                        rows, elements, blocks, partials);
  argmax_rows_final_kernel<<<rows, 256, 0, stream>>>(partials, rows, blocks, device_output);
  check(cudaPeekAtLastError(), "launch batched argmax kernels");
}

} // namespace carat
