#include "carat/linear.h"

#include <cooperative_groups.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

namespace carat {
namespace {

namespace cg = cooperative_groups;

void cuda_check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

void cublas_check(cublasStatus_t result, const char *operation) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(result));
  }
}

int configured_fp8_heuristic_index() {
  const char *configured = std::getenv("CARAT_FP8_HEURISTIC_INDEX");
  if (configured == nullptr || *configured == '\0')
    return -1;
  try {
    const std::string text(configured);
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < 0 || value > 63) {
      throw std::runtime_error("out of range");
    }
    return value;
  } catch (const std::exception &) {
    throw std::runtime_error("CARAT_FP8_HEURISTIC_INDEX must be in [0, 63]");
  }
}

int hopper_fp8_heuristic_index(int rows, int input_width, int output_width) {
  // These are retained H200 measurements for the pinned CUDA/cuBLASLt image. The index selects
  // one result from the ordered heuristic list, so this policy must be requalified when that
  // image changes. Unlisted shapes and row counts deliberately retain cuBLASLt's first choice.
  if (input_width == 21504 && output_width == 5376) {
    switch (rows) {
    case 64:
    case 72:
    case 80:
    case 96:
    case 112:
      return 1;
    default:
      return 0;
    }
  }
  if (input_width == 5376 && output_width == 16384 && rows == 64)
    return 1;
  return 0;
}

} // namespace

__global__ void quantize_bf16_fp8_cooperative_kernel(const __nv_bfloat16 *input,
                                                     __nv_fp8_e4m3 *output, float *scale,
                                                     std::size_t elements, float scale_multiplier) {
  const cg::grid_group grid = cg::this_grid();
  if (grid.thread_rank() == 0)
    *scale = 0.0F;
  grid.sync();

  float maximum = 0.0F;
  for (std::size_t index = grid.thread_rank(); index < elements; index += grid.size()) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(input[index])));
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  __shared__ float warp_maxima[16];
  if ((threadIdx.x & 31) == 0)
    warp_maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  if (threadIdx.x < 32) {
    maximum = threadIdx.x < blockDim.x / 32 ? warp_maxima[threadIdx.x] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (threadIdx.x == 0) {
      atomicMax(reinterpret_cast<unsigned int *>(scale), __float_as_uint(maximum));
    }
  }
  grid.sync();
  if (grid.thread_rank() == 0) {
    const float observed_maximum = *scale;
    *scale = observed_maximum > 0.0F ? observed_maximum * scale_multiplier / 448.0F : 1.0F;
  }
  grid.sync();

  const float inverse_scale = 1.0F / *scale;
  for (std::size_t index = grid.thread_rank(); index < elements; index += grid.size()) {
    output[index] = __nv_fp8_e4m3(__bfloat162float(input[index]) * inverse_scale);
  }
}

__global__ void bf16_amax_kernel(const __nv_bfloat16 *input, std::size_t elements,
                                 unsigned int *maximum_bits) {
  float maximum = 0.0F;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    maximum = fmaxf(maximum, fabsf(__bfloat162float(input[index])));
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  __shared__ float warp_maxima[8];
  if ((threadIdx.x & 31) == 0)
    warp_maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  if (threadIdx.x < 32) {
    maximum = threadIdx.x < blockDim.x / 32 ? warp_maxima[threadIdx.x] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (threadIdx.x == 0)
      atomicMax(maximum_bits, __float_as_uint(maximum));
  }
}

__global__ void finalize_fp8_scale_kernel(float *scale, float scale_multiplier) {
  const float maximum = *scale;
  *scale = maximum > 0.0F ? maximum * scale_multiplier / 448.0F : 1.0F;
}

__global__ void quantize_bf16_fp8_kernel(const __nv_bfloat16 *input, __nv_fp8_e4m3 *output,
                                         const float *scale, std::size_t elements) {
  const float inverse_scale = 1.0F / *scale;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = __nv_fp8_e4m3(__bfloat162float(input[index]) * inverse_scale);
  }
}

__global__ void quantize_bf16_fp8_from_row_amax_kernel(const __nv_bfloat16 *input,
                                                       __nv_fp8_e4m3 *output, float *output_scale,
                                                       const float *amax_values, int amax_count,
                                                       std::size_t elements,
                                                       float scale_multiplier) {
  __shared__ float tensor_scale;
  float maximum = 0.0F;
  for (int index = threadIdx.x; index < amax_count; index += blockDim.x) {
    maximum = fmaxf(maximum, amax_values[index]);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  __shared__ float warp_maxima[16];
  if ((threadIdx.x & 31) == 0)
    warp_maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  if (threadIdx.x < 32) {
    maximum = threadIdx.x < blockDim.x / 32 ? warp_maxima[threadIdx.x] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (threadIdx.x == 0) {
      tensor_scale = maximum > 0.0F ? maximum * scale_multiplier / 448.0F : 1.0F;
      if (blockIdx.x == 0)
        *output_scale = tensor_scale;
    }
  }
  __syncthreads();
  const float inverse_scale = 1.0F / tensor_scale;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = __nv_fp8_e4m3(__bfloat162float(input[index]) * inverse_scale);
  }
}

void quantize_bf16_to_fp8_e4m3(const void *input, void *output, float *scale, std::size_t elements,
                               cudaStream_t stream, float scale_multiplier) {
  if (input == nullptr || output == nullptr || scale == nullptr || elements == 0 ||
      !std::isfinite(scale_multiplier) || scale_multiplier <= 0.0F) {
    throw std::runtime_error("invalid FP8 quantization shape");
  }
  // Wider CTAs reduce grid-barrier and streaming-pass cost for large decode matrices;
  // smaller activations retain 256 threads for lower launch and reduction overhead.
  const int threads = elements >= 65536 ? 512 : 256;
  struct CooperativeBlockLimits {
    int threads_256;
    int threads_512;
  };
  static const CooperativeBlockLimits cooperative_block_limits = [] {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "get FP8 quantization device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device),
               "get FP8 quantization device properties");
    if (properties.cooperativeLaunch == 0)
      return CooperativeBlockLimits{0, 0};
    int blocks_256_per_multiprocessor = 0;
    int blocks_512_per_multiprocessor = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_256_per_multiprocessor, quantize_bf16_fp8_cooperative_kernel, 256, 0),
               "measure 256-thread cooperative FP8 quantization occupancy");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &blocks_512_per_multiprocessor, quantize_bf16_fp8_cooperative_kernel, 512, 0),
               "measure 512-thread cooperative FP8 quantization occupancy");
    // Two CTAs per SM retain enough latency hiding for the two streaming passes while keeping
    // grid-wide barriers cheap. Larger resident grids made these small decode tensors slower.
    return CooperativeBlockLimits{
        properties.multiProcessorCount * std::min(blocks_256_per_multiprocessor, 2),
        properties.multiProcessorCount * std::min(blocks_512_per_multiprocessor, 2)};
  }();
  const int cooperative_block_limit =
      threads == 512 ? cooperative_block_limits.threads_512 : cooperative_block_limits.threads_256;
  const int required_blocks = static_cast<int>((elements + threads - 1) / threads);
  // Decode activations are sub-megabyte-scale tensors where launch collapse dominates. Large
  // model weights quantize faster with the conventional unconstrained grid during startup.
  static const std::size_t cooperative_maximum_elements = [] {
    const char *configured = std::getenv("CARAT_FP8_COOPERATIVE_MAX_ELEMENTS");
    if (configured == nullptr || *configured == '\0') {
      return static_cast<std::size_t>(2ULL * 1024ULL * 1024ULL);
    }
    try {
      const std::string text(configured);
      std::size_t consumed = 0;
      const auto value = std::stoull(text, &consumed);
      if (consumed != text.size() || value == 0) {
        throw std::runtime_error("invalid cooperative FP8 quantization limit");
      }
      return static_cast<std::size_t>(value);
    } catch (const std::exception &) {
      throw std::runtime_error("CARAT_FP8_COOPERATIVE_MAX_ELEMENTS must be a positive integer");
    }
  }();
  if (cooperative_block_limit > 0 && elements <= cooperative_maximum_elements) {
    const int blocks = std::min(required_blocks, cooperative_block_limit);
    const auto *typed_input = static_cast<const __nv_bfloat16 *>(input);
    auto *typed_output = static_cast<__nv_fp8_e4m3 *>(output);
    void *arguments[] = {&typed_input, &typed_output, &scale, &elements, &scale_multiplier};
    cuda_check(cudaLaunchCooperativeKernel(
                   reinterpret_cast<const void *>(quantize_bf16_fp8_cooperative_kernel), blocks,
                   threads, arguments, 0, stream),
               "launch cooperative E4M3 quantization");
    return;
  }

  cuda_check(cudaMemsetAsync(scale, 0, sizeof(float), stream), "clear FP8 amax");
  const int blocks = std::min(required_blocks, 4096);
  bf16_amax_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), elements, reinterpret_cast<unsigned int *>(scale));
  cuda_check(cudaPeekAtLastError(), "launch BF16 amax");
  finalize_fp8_scale_kernel<<<1, 1, 0, stream>>>(scale, scale_multiplier);
  cuda_check(cudaPeekAtLastError(), "finalize FP8 scale");
  quantize_bf16_fp8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output), scale,
      elements);
  cuda_check(cudaPeekAtLastError(), "launch E4M3 quantization");
}

void quantize_bf16_to_fp8_e4m3_from_row_amax(const void *input, void *output, float *scale,
                                             const float *row_amax, int rows, int width,
                                             cudaStream_t stream, float scale_multiplier) {
  quantize_bf16_to_fp8_e4m3_from_amax_values(input, output, scale, row_amax, rows, rows, width,
                                             stream, scale_multiplier);
}

void quantize_bf16_to_fp8_e4m3_from_amax_values(const void *input, void *output, float *scale,
                                                const float *amax_values, int amax_count, int rows,
                                                int width, cudaStream_t stream,
                                                float scale_multiplier) {
  if (input == nullptr || output == nullptr || scale == nullptr || amax_values == nullptr ||
      amax_count <= 0 || rows <= 0 || width <= 0 || !std::isfinite(scale_multiplier) ||
      scale_multiplier <= 0.0F) {
    throw std::runtime_error("invalid producer-amax FP8 quantization shape");
  }
  const std::size_t elements = static_cast<std::size_t>(rows) * width;
  const int threads = elements >= 65536 ? 512 : 256;
  const int blocks = std::min(static_cast<int>((elements + threads - 1) / threads), 4096);
  quantize_bf16_fp8_from_row_amax_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output), scale,
      amax_values, amax_count, elements, scale_multiplier);
  cuda_check(cudaPeekAtLastError(), "launch producer-amax E4M3 quantization");
}

__device__ float reduce_block_max(float maximum, float *warp_maxima);

__global__ void quantize_bf16_rows_kernel(const __nv_bfloat16 *input, __nv_fp8_e4m3 *output,
                                          float *scales, int rows, int width) {
  const int row = static_cast<int>(blockIdx.x);
  float maximum = 0.0F;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    maximum = fmaxf(maximum,
                    fabsf(__bfloat162float(input[static_cast<std::size_t>(row) * width + column])));
  }
  __shared__ float warp_maxima[8];
  const float scale = fmaxf(reduce_block_max(maximum, warp_maxima) / 448.0F, 0x1p-24F);
  if (threadIdx.x == 0)
    scales[row] = scale;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const std::size_t index = static_cast<std::size_t>(row) * width + column;
    output[index] = __nv_fp8_e4m3(__bfloat162float(input[index]) / scale);
  }
}

void quantize_bf16_rows_to_fp8_e4m3(const void *input, void *output, float *scales, int rows,
                                    int width, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 || width <= 0) {
    throw std::runtime_error("invalid row-scaled FP8 quantization shape");
  }
  quantize_bf16_rows_kernel<<<rows, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(input),
                                                      static_cast<__nv_fp8_e4m3 *>(output), scales,
                                                      rows, width);
  cuda_check(cudaPeekAtLastError(), "launch row-scaled E4M3 quantization");
}

__device__ float reduce_block_max(float maximum, float *warp_maxima) {
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  if (lane == 0)
    warp_maxima[warp] = maximum;
  __syncthreads();
  if (warp == 0) {
    maximum = lane < blockDim.x / 32 ? warp_maxima[lane] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    }
    if (lane == 0)
      warp_maxima[0] = maximum;
  }
  __syncthreads();
  return warp_maxima[0];
}

__global__ void quantize_weight_block_128_kernel(const __nv_bfloat16 *input, __nv_fp8_e4m3 *output,
                                                 float *scales, int output_width, int input_width,
                                                 int scale_k_blocks) {
  const int k_begin = static_cast<int>(blockIdx.x) * 128;
  const int output_begin = static_cast<int>(blockIdx.y) * 128;
  float maximum = 0.0F;
  for (int element = threadIdx.x; element < 128 * 128; element += blockDim.x) {
    const int output_row = output_begin + element / 128;
    const int input_column = k_begin + element % 128;
    if (output_row < output_width && input_column < input_width) {
      maximum = fmaxf(
          maximum, fabsf(__bfloat162float(
                       input[static_cast<std::size_t>(output_row) * input_width + input_column])));
    }
  }
  __shared__ float warp_maxima[8];
  const float scale = fmaxf(reduce_block_max(maximum, warp_maxima) / 448.0F, 0x1p-24F);
  if (threadIdx.x == 0) {
    scales[static_cast<std::size_t>(blockIdx.y) * scale_k_blocks + blockIdx.x] = scale;
  }
  for (int element = threadIdx.x; element < 128 * 128; element += blockDim.x) {
    const int output_row = output_begin + element / 128;
    const int input_column = k_begin + element % 128;
    if (output_row < output_width && input_column < input_width) {
      const std::size_t index = static_cast<std::size_t>(output_row) * input_width + input_column;
      output[index] = __nv_fp8_e4m3(__bfloat162float(input[index]) / scale);
    }
  }
}

__global__ void quantize_activation_block_128_kernel(const __nv_bfloat16 *input,
                                                     __nv_fp8_e4m3 *output, float *scales, int rows,
                                                     int input_width) {
  const int row = static_cast<int>(blockIdx.y);
  const int k_begin = static_cast<int>(blockIdx.x) * 128;
  const int column = k_begin + threadIdx.x;
  float value = 0.0F;
  if (row < rows && column < input_width) {
    value = __bfloat162float(input[static_cast<std::size_t>(row) * input_width + column]);
  }
  __shared__ float warp_maxima[4];
  const float scale = fmaxf(reduce_block_max(fabsf(value), warp_maxima) / 448.0F, 0x1p-24F);
  if (threadIdx.x == 0) {
    scales[static_cast<std::size_t>(blockIdx.x) * rows + row] = scale;
  }
  if (row < rows && column < input_width) {
    output[static_cast<std::size_t>(row) * input_width + column] = __nv_fp8_e4m3(value / scale);
  }
}

std::size_t fp8_weight_block_scale_elements(int output_width, int input_width) {
  if (output_width <= 0 || input_width <= 0)
    throw std::runtime_error("invalid FP8 weight shape");
  const std::size_t k_blocks = (static_cast<std::size_t>(input_width) + 127) / 128;
  const std::size_t padded_k_blocks = (k_blocks + 3) & ~std::size_t{3};
  return padded_k_blocks * ((static_cast<std::size_t>(output_width) + 127) / 128);
}

std::size_t fp8_activation_block_scale_elements(int rows, int input_width) {
  if (rows <= 0 || input_width <= 0)
    throw std::runtime_error("invalid FP8 activation shape");
  return static_cast<std::size_t>(rows) * ((static_cast<std::size_t>(input_width) + 127) / 128);
}

void quantize_bf16_weight_to_fp8_block_128(const void *input, void *output, float *scales,
                                           int output_width, int input_width, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || output_width <= 0 ||
      input_width <= 0) {
    throw std::runtime_error("invalid FP8 block weight quantization shape");
  }
  const int k_blocks = (input_width + 127) / 128;
  const int padded_k_blocks = (k_blocks + 3) & ~3;
  cuda_check(
      cudaMemsetAsync(scales, 0,
                      fp8_weight_block_scale_elements(output_width, input_width) * sizeof(float),
                      stream),
      "clear FP8 block weight scales");
  const dim3 grid(static_cast<unsigned>(k_blocks),
                  static_cast<unsigned>((output_width + 127) / 128));
  quantize_weight_block_128_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output), scales,
      output_width, input_width, padded_k_blocks);
  cuda_check(cudaPeekAtLastError(), "launch block-scaled FP8 weight quantization");
}

void quantize_bf16_activation_to_fp8_block_128(const void *input, void *output, float *scales,
                                               int rows, int input_width, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 || input_width <= 0) {
    throw std::runtime_error("invalid FP8 block activation quantization shape");
  }
  const dim3 grid(static_cast<unsigned>((input_width + 127) / 128), static_cast<unsigned>(rows));
  quantize_activation_block_128_kernel<<<grid, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output), scales, rows,
      input_width);
  cuda_check(cudaPeekAtLastError(), "launch block-scaled FP8 activation quantization");
}

__global__ void roundtrip_fp8_int4_blocks_kernel(__nv_fp8_e4m3 *weight, int output_width,
                                                 int input_width, int block_width,
                                                 int scale_refinement_iterations) {
  const int output_row = static_cast<int>(blockIdx.x);
  if (output_row >= output_width)
    return;
  __shared__ float warp_maxima[8];
  __shared__ float optimized_scale;
  const std::size_t row_base = static_cast<std::size_t>(output_row) * input_width;
  for (int block_begin = 0; block_begin < input_width; block_begin += block_width) {
    const int column = block_begin + static_cast<int>(threadIdx.x);
    float value = 0.0F;
    if (threadIdx.x < block_width && column < input_width) {
      value = static_cast<float>(weight[row_base + column]);
    }
    const float maximum = reduce_block_max(fabsf(value), warp_maxima);
    if (threadIdx.x == 0) {
      optimized_scale = maximum > 0.0F ? fminf(maximum / 7.0F, 56.0F) : 1.0F;
    }
    __syncthreads();
    for (int iteration = 0; iteration < scale_refinement_iterations; ++iteration) {
      const int quantized = max(-8, min(7, __float2int_rn(value / optimized_scale)));
      float numerator = value * static_cast<float>(quantized);
      float denominator = static_cast<float>(quantized * quantized);
      for (int offset = 16; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(0xffffffffU, numerator, offset);
        denominator += __shfl_down_sync(0xffffffffU, denominator, offset);
      }
      const int lane = static_cast<int>(threadIdx.x) & 31;
      const int warp = static_cast<int>(threadIdx.x) / 32;
      if (lane == 0) {
        warp_maxima[warp] = numerator;
      }
      __syncthreads();
      if (warp == 0) {
        numerator = lane < static_cast<int>(blockDim.x) / 32 ? warp_maxima[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset /= 2) {
          numerator += __shfl_down_sync(0xffffffffU, numerator, offset);
        }
        if (lane == 0)
          warp_maxima[0] = numerator;
      }
      __syncthreads();
      numerator = warp_maxima[0];
      __syncthreads();
      if (lane == 0) {
        warp_maxima[warp] = denominator;
      }
      __syncthreads();
      if (warp == 0) {
        denominator = lane < static_cast<int>(blockDim.x) / 32 ? warp_maxima[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset /= 2) {
          denominator += __shfl_down_sync(0xffffffffU, denominator, offset);
        }
        if (lane == 0)
          warp_maxima[0] = denominator;
      }
      __syncthreads();
      denominator = warp_maxima[0];
      if (threadIdx.x == 0 && denominator > 0.0F) {
        optimized_scale = fminf(numerator / denominator, 56.0F);
      }
      __syncthreads();
    }
    const float scale = optimized_scale;
    if (threadIdx.x < block_width && column < input_width) {
      const int quantized = max(-8, min(7, __float2int_rn(value / scale)));
      weight[row_base + column] = __nv_fp8_e4m3(static_cast<float>(quantized) * scale);
    }
    __syncthreads();
  }
}

void roundtrip_fp8_weight_via_int4_blocks(void *fp8_weight, int output_width, int input_width,
                                          int block_width, cudaStream_t stream,
                                          int scale_refinement_iterations) {
  if (fp8_weight == nullptr || output_width <= 0 || input_width <= 0 ||
      (block_width != 32 && block_width != 64 && block_width != 128 && block_width != 256) ||
      scale_refinement_iterations < 0 || scale_refinement_iterations > 8) {
    throw std::runtime_error("invalid INT4 weight oracle shape");
  }
  roundtrip_fp8_int4_blocks_kernel<<<output_width, 256, 0, stream>>>(
      static_cast<__nv_fp8_e4m3 *>(fp8_weight), output_width, input_width, block_width,
      scale_refinement_iterations);
  cuda_check(cudaPeekAtLastError(), "launch INT4 weight oracle");
}

struct Bf16Linear::Implementation {
  struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t input_layout{};
    cublasLtMatrixLayout_t weight_layout{};
    cublasLtMatrixLayout_t output_layout{};
    cublasLtMatmulHeuristicResult_t heuristic{};

    ~Plan() {
      if (output_layout != nullptr)
        cublasLtMatrixLayoutDestroy(output_layout);
      if (weight_layout != nullptr)
        cublasLtMatrixLayoutDestroy(weight_layout);
      if (input_layout != nullptr)
        cublasLtMatrixLayoutDestroy(input_layout);
      if (operation != nullptr)
        cublasLtMatmulDescDestroy(operation);
    }
  };

  cublasLtHandle_t handle{};
  void *workspace{nullptr};
  std::size_t workspace_bytes;
  std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;

  explicit Implementation(std::size_t bytes) : workspace_bytes(bytes) {
    cublas_check(cublasLtCreate(&handle), "cublasLtCreate");
    if (workspace_bytes != 0)
      cuda_check(cudaMalloc(&workspace, workspace_bytes), "allocate linear workspace");
  }
  ~Implementation() {
    plans.clear();
    if (workspace != nullptr)
      cudaFree(workspace);
    if (handle != nullptr)
      cublasLtDestroy(handle);
  }

  Plan &plan(int rows, int input_width, int output_width) {
    const auto key = std::make_tuple(rows, input_width, output_width);
    const auto existing = plans.find(key);
    if (existing != plans.end())
      return *existing->second;
    auto result = std::make_unique<Plan>();
    cublas_check(cublasLtMatmulDescCreate(&result->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "create matmul descriptor");
    const cublasOperation_t transpose_weight = CUBLAS_OP_T;
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transpose_weight, sizeof(transpose_weight)),
                 "set weight transpose");
    cublas_check(cublasLtMatrixLayoutCreate(&result->input_layout, CUDA_R_16BF, rows, input_width,
                                            input_width),
                 "create input layout");
    cublas_check(cublasLtMatrixLayoutCreate(&result->weight_layout, CUDA_R_16BF, output_width,
                                            input_width, input_width),
                 "create weight layout");
    cublas_check(cublasLtMatrixLayoutCreate(&result->output_layout, CUDA_R_16BF, rows, output_width,
                                            output_width),
                 "create output layout");
    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublas_check(cublasLtMatrixLayoutSetAttribute(result->input_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                                                  sizeof(row_order)),
                 "set input order");
    cublas_check(cublasLtMatrixLayoutSetAttribute(result->weight_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                                                  sizeof(row_order)),
                 "set weight order");
    cublas_check(cublasLtMatrixLayoutSetAttribute(result->output_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                                                  sizeof(row_order)),
                 "set output order");
    cublasLtMatmulPreference_t preference{};
    cublas_check(cublasLtMatmulPreferenceCreate(&preference), "create matmul preference");
    cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_bytes, sizeof(workspace_bytes)),
                 "set matmul workspace");
    int returned = 0;
    const cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle, result->operation, result->input_layout, result->weight_layout,
        result->output_layout, result->output_layout, preference, 1, &result->heuristic, &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    cublas_check(heuristic_status, "select matmul algorithm");
    if (returned == 0)
      throw std::runtime_error("no cublasLt algorithm for linear shape");
    Plan &reference = *result;
    plans.emplace(key, std::move(result));
    return reference;
  }
};

Bf16Linear::Bf16Linear(std::size_t workspace_bytes)
    : implementation_(std::make_unique<Implementation>(workspace_bytes)) {}
Bf16Linear::~Bf16Linear() = default;

void Bf16Linear::run(const void *input, const void *row_major_weight, void *output, int rows,
                     int input_width, int output_width, cudaStream_t stream) {
  if (rows <= 0 || input_width <= 0 || output_width <= 0)
    throw std::runtime_error("invalid linear shape");
  auto &plan = implementation_->plan(rows, input_width, output_width);
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  cublas_check(cublasLtMatmul(implementation_->handle, plan.operation, &alpha, input,
                              plan.input_layout, row_major_weight, plan.weight_layout, &beta,
                              output, plan.output_layout, output, plan.output_layout,
                              &plan.heuristic.algo, implementation_->workspace,
                              implementation_->workspace_bytes, stream),
               "run BF16 linear");
}

struct Fp8Linear::Implementation {
  struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight_layout{};
    cublasLtMatrixLayout_t input_layout{};
    cublasLtMatrixLayout_t output_layout{};
    cublasLtMatmulHeuristicResult_t heuristic{};

    ~Plan() {
      if (output_layout != nullptr)
        cublasLtMatrixLayoutDestroy(output_layout);
      if (input_layout != nullptr)
        cublasLtMatrixLayoutDestroy(input_layout);
      if (weight_layout != nullptr)
        cublasLtMatrixLayoutDestroy(weight_layout);
      if (operation != nullptr)
        cublasLtMatmulDescDestroy(operation);
    }
  };

  cublasLtHandle_t handle{};
  void *workspace{nullptr};
  std::size_t workspace_bytes;
  Fp8Scaling scaling;
  int heuristic_override;
  bool use_hopper_heuristic_policy;
  std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;

  Implementation(Fp8Scaling requested_scaling, std::size_t bytes)
      : workspace_bytes(bytes), scaling(requested_scaling),
        heuristic_override(configured_fp8_heuristic_index()), use_hopper_heuristic_policy(false) {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "get FP8 linear device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "get FP8 linear device properties");
    use_hopper_heuristic_policy = properties.major == 9 && properties.minor == 0 &&
                                  std::getenv("CARAT_DISABLE_FP8_HEURISTIC_POLICY") == nullptr;
    cublas_check(cublasLtCreate(&handle), "cublasLtCreate FP8");
    if (workspace_bytes != 0) {
      cuda_check(cudaMalloc(&workspace, workspace_bytes), "allocate FP8 linear workspace");
    }
  }
  ~Implementation() {
    plans.clear();
    if (workspace != nullptr)
      cudaFree(workspace);
    if (handle != nullptr)
      cublasLtDestroy(handle);
  }

  Plan &plan(int rows, int input_width, int output_width) {
    const auto key = std::make_tuple(rows, input_width, output_width);
    const auto existing = plans.find(key);
    if (existing != plans.end())
      return *existing->second;
    auto result = std::make_unique<Plan>();
    cublas_check(cublasLtMatmulDescCreate(&result->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "create FP8 matmul descriptor");
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t identity = CUBLAS_OP_N;
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transpose, sizeof(transpose)),
                 "set FP8 weight transpose");
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &identity, sizeof(identity)),
                 "set FP8 input orientation");
    const cublasLtMatmulMatrixScale_t weight_scale_mode =
        scaling == Fp8Scaling::block_128 ? CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F
        : scaling == Fp8Scaling::channel ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
                                         : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    const cublasLtMatmulMatrixScale_t input_scale_mode =
        scaling == Fp8Scaling::block_128 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
        : scaling == Fp8Scaling::channel ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
                                         : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation,
                                                CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                &weight_scale_mode, sizeof(weight_scale_mode)),
                 "set FP8 weight scale mode");
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation,
                                                CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                &input_scale_mode, sizeof(input_scale_mode)),
                 "set FP8 input scale mode");

    // Row-major W[N,K], X[M,K], and Y[M,N] are column-major KxN, KxM, and NxM views.
    cublas_check(cublasLtMatrixLayoutCreate(&result->weight_layout, CUDA_R_8F_E4M3, input_width,
                                            output_width, input_width),
                 "create FP8 weight layout");
    cublas_check(cublasLtMatrixLayoutCreate(&result->input_layout, CUDA_R_8F_E4M3, input_width,
                                            rows, input_width),
                 "create FP8 input layout");
    cublas_check(cublasLtMatrixLayoutCreate(&result->output_layout, CUDA_R_16BF, output_width, rows,
                                            output_width),
                 "create FP8 output layout");
    const float *planning_scale = static_cast<const float *>(workspace);
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation,
                                                CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &planning_scale, sizeof(planning_scale)),
                 "set FP8 planning weight scale");
    cublas_check(cublasLtMatmulDescSetAttribute(result->operation,
                                                CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &planning_scale, sizeof(planning_scale)),
                 "set FP8 planning input scale");
    cublasLtMatmulPreference_t preference{};
    cublas_check(cublasLtMatmulPreferenceCreate(&preference), "create FP8 matmul preference");
    cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_bytes, sizeof(workspace_bytes)),
                 "set FP8 matmul workspace");
    const int heuristic_index =
        heuristic_override >= 0 ? heuristic_override
                                : (use_hopper_heuristic_policy
                                       ? hopper_fp8_heuristic_index(rows, input_width, output_width)
                                       : 0);
    std::array<cublasLtMatmulHeuristicResult_t, 64> heuristics{};
    int returned = 0;
    const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        handle, result->operation, result->weight_layout, result->input_layout,
        result->output_layout, result->output_layout, preference, heuristic_index + 1,
        heuristics.data(), &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    cublas_check(status, "select FP8 matmul algorithm");
    if (returned <= heuristic_index) {
      throw std::runtime_error("requested FP8 heuristic index is unavailable for linear shape");
    }
    result->heuristic = heuristics.at(static_cast<std::size_t>(heuristic_index));
    Plan &reference = *result;
    plans.emplace(key, std::move(result));
    return reference;
  }
};

Fp8Linear::Fp8Linear(Fp8Scaling scaling, std::size_t workspace_bytes)
    : implementation_(std::make_unique<Implementation>(scaling, workspace_bytes)) {}
Fp8Linear::~Fp8Linear() = default;

void Fp8Linear::run(const void *fp8_input, const float *input_scale,
                    const void *fp8_row_major_weight, const float *weight_scale, void *bf16_output,
                    int rows, int input_width, int output_width, cudaStream_t stream) {
  if (rows <= 0 || input_width <= 0 || output_width <= 0 || fp8_input == nullptr ||
      input_scale == nullptr || fp8_row_major_weight == nullptr || weight_scale == nullptr ||
      bf16_output == nullptr) {
    throw std::runtime_error("invalid FP8 linear shape");
  }
  auto &state = *implementation_;
  auto &plan = state.plan(rows, input_width, output_width);
  cublas_check(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                              &weight_scale, sizeof(weight_scale)),
               "set FP8 weight scale");
  cublas_check(cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                              &input_scale, sizeof(input_scale)),
               "set FP8 input scale");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  cublas_check(cublasLtMatmul(state.handle, plan.operation, &alpha, fp8_row_major_weight,
                              plan.weight_layout, fp8_input, plan.input_layout, &beta, bf16_output,
                              plan.output_layout, bf16_output, plan.output_layout,
                              &plan.heuristic.algo, state.workspace, state.workspace_bytes, stream),
               "run FP8 linear");
}

} // namespace carat
