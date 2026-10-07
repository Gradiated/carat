#include "carat/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace carat {
namespace {

using namespace cute;
using Element = cutlass::bfloat16_t;
using KeyLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},
                                         make_shape(Int<64>{}, Int<512>{})));
using QueryLayout =
    decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, make_shape(Int<8>{}, Int<512>{})));
using GlobalQkMma =
    decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
using GlobalKeyCopy =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
using ValueTileLayout = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{},
                                               make_shape(Int<64>{}, Int<64>{})));
using ProbabilityLayout =
    decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, make_shape(Int<8>{}, Int<64>{})));
using ScoreLayout = Layout<Shape<_64, _8>, Stride<_8, _1>>;
using GlobalPvMma =
    decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::K>{}));
using GlobalRsMma =
    decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}));
using Fp8Element = cutlass::float_e4m3_t;
using Fp8ValueLayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Fp8Element>{},
                                              make_shape(Int<64>{}, Int<64>{})));
using Fp8ProbabilityLayout = decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<Fp8Element>{},
                                                    make_shape(Int<8>{}, Int<64>{})));
using Fp8PvRsMma = decltype(make_tiled_mma(SM90_64x8x32_F32E4M3E4M3_RS_TN<>{}));
using Int8Element = std::int8_t;
using Int8KeyBlockLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Int8Element>{},
                                                  make_shape(Int<64>{}, Int<128>{})));
using Int8QueryBlockLayout = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Int8Element>{},
                                                    make_shape(Int<8>{}, Int<128>{})));
using Int8QkMma = decltype(make_tiled_mma(SM90_64x8x32_S32S8S8_SS_TN{}));
using Int8PvValueLayout = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Int8Element>{},
                                                 make_shape(Int<64>{}, Int<64>{})));
using Int8PvProbabilityLayout = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Int8Element>{},
                                                       make_shape(Int<8>{}, Int<64>{})));
using Int8KeyCopy =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Int8Element>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _16>>{}));

struct alignas(128) GlobalQkSharedStorage {
  ArrayEngine<Element, cosize_v<KeyLayout>> keys;
  ArrayEngine<Element, cosize_v<QueryLayout>> queries;
};

struct alignas(128) Int8GlobalQkSharedStorage {
  ArrayEngine<Int8Element, cosize_v<Int8KeyBlockLayout>> keys[2];
  ArrayEngine<Int8Element, cosize_v<Int8QueryBlockLayout>> queries[2];
  float key_scales[64 * 4];
  float query_scales[8 * 4];
};

struct alignas(128) Int8GlobalQkTileOuterSharedStorage {
  ArrayEngine<Int8Element, cosize_v<Int8KeyBlockLayout>> keys[2];
  ArrayEngine<Int8Element, cosize_v<Int8QueryBlockLayout>> queries[4];
  float key_scales[64 * 4];
  float query_scales[8 * 4];
};

struct alignas(128) GlobalAttentionTileSharedStorage {
  ArrayEngine<Element, cosize_v<KeyLayout>> keys;
  ArrayEngine<Element, cosize_v<QueryLayout>> queries;
  ArrayEngine<Element, cosize_v<ValueTileLayout>> values[8];
  ArrayEngine<Element, cosize_v<ProbabilityLayout>> probabilities;
  ArrayEngine<Element, cosize_v<ScoreLayout>> scores;
  float running_maximum[8];
  float running_sum[8];
  float old_scale[8];
};

struct alignas(128) Fp8PvSharedStorage {
  ArrayEngine<Fp8Element, cosize_v<Fp8ValueLayout>> values;
  ArrayEngine<Fp8Element, cosize_v<Fp8ProbabilityLayout>> probabilities;
};

struct alignas(128) Int8GlobalPvSharedStorage {
  ArrayEngine<Element, cosize_v<ValueTileLayout>> values;
  ArrayEngine<Element, cosize_v<ProbabilityLayout>> probabilities;
};

struct alignas(128) Int8GlobalPvRsSharedStorage {
  ArrayEngine<Int8Element, cosize_v<Int8PvValueLayout>> values[8];
  ArrayEngine<Int8Element, cosize_v<Int8PvProbabilityLayout>> probabilities[4];
};

__device__ std::uint64_t packed_fp8_value_index(std::uint64_t head, int token, int component,
                                                int context_length) {
  constexpr int tile_tokens = 64;
  constexpr int dimension_tiles = 8;
  const int token_tile = token / tile_tokens;
  const int token_in_tile = token - token_tile * tile_tokens;
  const int dimension_tile = component / 64;
  const int component_in_tile = component - dimension_tile * 64;
  const std::uint64_t tile =
      (head * (context_length / tile_tokens) + token_tile) * dimension_tiles + dimension_tile;
  return tile * cosize_v<Fp8ValueLayout> + Fp8ValueLayout{}(component_in_tile, token_in_tile);
}

__host__ __device__ std::uint64_t packed_int8_value_index(std::uint64_t head, int token,
                                                          int component, int context_length) {
  constexpr int tile_tokens = 64;
  constexpr int dimension_tiles = 8;
  const int token_tile = token / tile_tokens;
  const int token_in_tile = token - token_tile * tile_tokens;
  const int dimension_tile = component / 64;
  const int component_in_tile = component - dimension_tile * 64;
  const std::uint64_t tile =
      (head * (context_length / tile_tokens) + token_tile) * dimension_tiles + dimension_tile;
  return tile * cosize_v<Int8PvValueLayout> + Int8PvValueLayout{}(component_in_tile, token_in_tile);
}

__global__ void pack_global_values_wgmma_int8_kernel(const Int8Element *values,
                                                     Int8Element *packed_values,
                                                     std::uint64_t elements, int context_length) {
  constexpr int dimension = 512;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const std::uint64_t head = index / (static_cast<std::uint64_t>(context_length) * dimension);
  const int head_index =
      static_cast<int>(index - head * static_cast<std::uint64_t>(context_length) * dimension);
  const int token = head_index / dimension;
  const int component = head_index - token * dimension;
  packed_values[packed_int8_value_index(head, token, component, context_length)] = values[index];
}

__global__ void quantize_global_values_wgmma_fp8_kernel(const Element *values,
                                                        Fp8Element *packed_values,
                                                        std::uint64_t elements,
                                                        int context_length) {
  constexpr int dimension = 512;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const std::uint64_t head = index / (static_cast<std::uint64_t>(context_length) * dimension);
  const int head_index =
      static_cast<int>(index - head * static_cast<std::uint64_t>(context_length) * dimension);
  const int token = head_index / dimension;
  const int component = head_index - token * dimension;
  const std::uint64_t destination = packed_fp8_value_index(head, token, component, context_length);
  packed_values[destination] = Fp8Element(static_cast<float>(values[index]) * 64.0F);
}

__global__ void quantize_global_value_span_wgmma_fp8_kernel(const Element *values,
                                                            Fp8Element *packed_values, int kv_heads,
                                                            int tokens, int context_length,
                                                            int position_start) {
  constexpr int dimension = 512;
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = kv_heads * tokens * dimension;
  if (index >= elements)
    return;
  const int component = index % dimension;
  const int token_offset = (index / dimension) % tokens;
  const int head = index / (dimension * tokens);
  const int token = position_start + token_offset;
  const std::uint64_t source =
      (static_cast<std::uint64_t>(head) * context_length + token) * dimension + component;
  packed_values[packed_fp8_value_index(head, token, component, context_length)] =
      Fp8Element(static_cast<float>(values[source]) * 64.0F);
}

__global__ void quantize_global_decode_values_wgmma_fp8_kernel(
    const Element *values, Fp8Element *packed_values, const int *positions, const int *slots,
    int batch, int maximum_slots, int kv_heads, int context_length) {
  constexpr int dimension = 512;
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = batch * kv_heads * dimension;
  if (index >= elements)
    return;
  const int component = index % dimension;
  const int head = (index / dimension) % kv_heads;
  const int row = index / (dimension * kv_heads);
  const int slot = slots[row];
  const int token = positions[row];
  if (slot < 0 || slot >= maximum_slots || token < 0 || token >= context_length)
    return;
  const std::uint64_t physical_head = static_cast<std::uint64_t>(slot) * kv_heads + head;
  const std::uint64_t source = (physical_head * context_length + token) * dimension + component;
  packed_values[packed_fp8_value_index(physical_head, token, component, context_length)] =
      Fp8Element(static_cast<float>(values[source]) * 64.0F);
}

__global__ void quantize_global_probabilities_wgmma_fp8_kernel(const Element *probabilities,
                                                               Fp8Element *packed_probabilities,
                                                               std::uint64_t elements,
                                                               int query_group,
                                                               int context_length) {
  constexpr int tile_tokens = 64;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const std::uint64_t head = index / (static_cast<std::uint64_t>(query_group) * context_length);
  const int head_index =
      static_cast<int>(index - head * static_cast<std::uint64_t>(query_group) * context_length);
  const int query = head_index / context_length;
  const int token = head_index - query * context_length;
  const int token_tile = token / tile_tokens;
  const int token_in_tile = token - token_tile * tile_tokens;
  const std::uint64_t tile = head * (context_length / tile_tokens) + token_tile;
  const std::uint64_t destination =
      tile * cosize_v<Fp8ProbabilityLayout> + Fp8ProbabilityLayout{}(query, token_in_tile);
  packed_probabilities[destination] = Fp8Element(static_cast<float>(probabilities[index]) * 448.0F);
}

struct alignas(128) RsRawGlobalAttentionTileSharedStorage {
  ArrayEngine<Element, cosize_v<KeyLayout>> raw;
  ArrayEngine<Element, cosize_v<QueryLayout>> queries;
  ArrayEngine<Element, cosize_v<ProbabilityLayout>> probabilities;
  ArrayEngine<Element, cosize_v<ScoreLayout>> scores;
  float inverse_rms[64];
  float running_maximum[8];
  float running_sum[8];
  float old_scale[8];
};

struct alignas(128) RsRawGlobalAttentionSegmentSharedStorage {
  ArrayEngine<Element, cosize_v<KeyLayout>> raw;
  ArrayEngine<Element, cosize_v<QueryLayout>> queries;
  ArrayEngine<Element, cosize_v<ProbabilityLayout>> probabilities;
  ArrayEngine<Element, cosize_v<ScoreLayout>> scores;
  float inverse_rms[64];
  float running_maximum[8];
  float running_sum[8];
  float old_scale[8];
};

static_assert(cosize_v<KeyLayout> == 8 * cosize_v<ValueTileLayout>);

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

__device__ float warp_maximum(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value = max(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

__device__ float warp_total(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

__device__ void quantize_packed_int8_value_row(const Element *source, Int8Element *destination,
                                               float *scales, std::uint64_t physical_head,
                                               int token, int cache_capacity) {
  constexpr int dimension = 512;
  constexpr int block_width = 128;
  constexpr int blocks_per_row = dimension / block_width;
  const std::uint64_t source_row = (physical_head * cache_capacity + token) * dimension;
  __shared__ float warp_maxima[8];
  for (int block = 0; block < blocks_per_row; ++block) {
    float maximum = 0.0F;
    for (int component = block * block_width + static_cast<int>(threadIdx.x);
         component < (block + 1) * block_width; component += static_cast<int>(blockDim.x)) {
      maximum = max(maximum, fabsf(static_cast<float>(source[source_row + component])));
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
    const float scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
    if (threadIdx.x == 0U) {
      scales[(physical_head * cache_capacity + token) * blocks_per_row + block] = scale;
    }
    const float inverse_scale = 1.0F / scale;
    for (int component = block * block_width + static_cast<int>(threadIdx.x);
         component < (block + 1) * block_width; component += static_cast<int>(blockDim.x)) {
      const int quantized =
          max(-127, min(127, __float2int_rn(static_cast<float>(source[source_row + component]) *
                                            inverse_scale)));
      destination[packed_int8_value_index(physical_head, token, component, cache_capacity)] =
          static_cast<Int8Element>(quantized);
    }
    __syncthreads();
  }
}

__global__ void quantize_global_values_wgmma_int8_kernel(const Element *values,
                                                         Int8Element *packed_values, float *scales,
                                                         int rows, int cache_capacity) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows)
    return;
  const std::uint64_t head = static_cast<std::uint64_t>(row / cache_capacity);
  const int token = row - static_cast<int>(head) * cache_capacity;
  quantize_packed_int8_value_row(values, packed_values, scales, head, token, cache_capacity);
}

__global__ void quantize_global_value_span_wgmma_int8_kernel(const Element *values,
                                                             Int8Element *packed_values,
                                                             float *scales, int heads, int tokens,
                                                             int cache_capacity,
                                                             int position_start) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= heads * tokens)
    return;
  const std::uint64_t head = static_cast<std::uint64_t>(row / tokens);
  const int token = position_start + row - static_cast<int>(head) * tokens;
  quantize_packed_int8_value_row(values, packed_values, scales, head, token, cache_capacity);
}

__global__ void quantize_global_decode_values_wgmma_int8_kernel(
    const Element *values, Int8Element *packed_values, float *scales, const int *positions,
    const int *slots, int batch, int maximum_slots, int heads, int cache_capacity) {
  const int row_head = static_cast<int>(blockIdx.x);
  const int row = row_head / heads;
  const int head = row_head - row * heads;
  if (row >= batch)
    return;
  const int slot = slots[row];
  const int token = positions[row];
  if (slot < 0 || slot >= maximum_slots || token < 0 || token >= cache_capacity)
    return;
  const std::uint64_t physical_head = static_cast<std::uint64_t>(slot) * heads + head;
  quantize_packed_int8_value_row(values, packed_values, scales, physical_head, token,
                                 cache_capacity);
}

__global__ __launch_bounds__(128) void global_qk_wgmma_kernel(const Element *queries,
                                                              const Element *keys, Element *scores,
                                                              int kv_heads, int context_length,
                                                              int tiles_per_block) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  const int tiles_per_head = context_length / tile_tokens;
  const int segments_per_head = tiles_per_head / tiles_per_block;
  const int head_segment = static_cast<int>(blockIdx.x);
  const int head = head_segment / segments_per_head;
  const int segment = head_segment - head * segments_per_head;
  const int first_tile = segment * tiles_per_block;
  const std::uint64_t query_base = static_cast<std::uint64_t>(head) * query_group * dimension;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<GlobalQkSharedStorage *>(storage_bytes);
  Tensor shared_keys = make_tensor(make_smem_ptr(storage.keys.begin()), KeyLayout{});
  Tensor shared_queries = make_tensor(make_smem_ptr(storage.queries.begin()), QueryLayout{});

  for (int index = static_cast<int>(threadIdx.x); index < query_group * dimension;
       index += static_cast<int>(blockDim.x)) {
    const int query = index / dimension;
    const int component = index - query * dimension;
    shared_queries(query, component) = queries[query_base + index];
  }
  GlobalKeyCopy key_copy;
  ThrCopy thread_copy = key_copy.get_slice(threadIdx.x);
  Tensor position_independent_keys = as_position_independent_swizzle_tensor(shared_keys);
  Tensor thread_shared_keys = thread_copy.partition_D(position_independent_keys);
  GlobalQkMma mma;
  ThrMMA thread_mma = mma.get_slice(threadIdx.x);
  Tensor fragment_keys = thread_mma.partition_A(shared_keys);
  Tensor fragment_queries = thread_mma.partition_B(shared_queries);
  for (int tile_offset = 0; tile_offset < tiles_per_block; ++tile_offset) {
    const int tile = first_tile + tile_offset;
    const std::uint64_t key_base =
        (static_cast<std::uint64_t>(head) * context_length + tile * tile_tokens) * dimension;
    Tensor global_keys =
        make_tensor(make_gmem_ptr(keys + key_base), make_shape(Int<64>{}, Int<512>{}),
                    make_stride(Int<512>{}, Int<1>{}));
    Tensor thread_global_keys = thread_copy.partition_S(global_keys);
    copy(key_copy, thread_global_keys, thread_shared_keys);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    const std::uint64_t score_base =
        static_cast<std::uint64_t>(head) * query_group * context_length + tile * tile_tokens;
    Tensor global_scores =
        make_tensor(make_gmem_ptr(scores + score_base), make_shape(Int<64>{}, Int<8>{}),
                    make_stride(Int<1>{}, context_length));
    Tensor fragment_global_scores = thread_mma.partition_C(global_scores);
    Tensor accumulators = thread_mma.make_fragment_C(fragment_global_scores);
    clear(accumulators);
    warpgroup_fence_operand(accumulators);
    warpgroup_arrive();
    cute::gemm(mma, fragment_keys, fragment_queries, accumulators);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accumulators);
    copy(accumulators, fragment_global_scores);
    __syncthreads();
  }
}

template <int tiles_per_block>
__global__ __launch_bounds__(128, 2) void global_qk_wgmma_int8_block128_kernel(
    const Int8Element *queries, const float *query_scales, const Int8Element *keys,
    const float *key_scales, Element *scores, int kv_heads, int context_length,
    int cache_capacity) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  constexpr int block_width = 128;
  constexpr int blocks = dimension / block_width;
  const int tiles_per_head = context_length / tile_tokens;
  const int segments_per_head = tiles_per_head / tiles_per_block;
  const int head_segment = static_cast<int>(blockIdx.x);
  const int head = head_segment / segments_per_head;
  const int segment = head_segment - head * segments_per_head;
  const int first_tile = segment * tiles_per_block;
  const std::uint64_t query_base = static_cast<std::uint64_t>(head) * query_group * dimension;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<Int8GlobalQkSharedStorage *>(storage_bytes);
  Tensor shared_keys = make_tensor(make_smem_ptr(storage.keys[0].begin()), Int8KeyBlockLayout{});
  Tensor shared_queries =
      make_tensor(make_smem_ptr(storage.queries[0].begin()), Int8QueryBlockLayout{});
  Int8QkMma mma;
  ThrMMA thread_mma = mma.get_slice(static_cast<int>(threadIdx.x));
  Tensor fragment_keys = thread_mma.partition_A(shared_keys);
  Tensor fragment_queries = thread_mma.partition_B(shared_queries);
  Int8KeyCopy key_copy;
  ThrCopy key_thread_copy = key_copy.get_slice(static_cast<int>(threadIdx.x));
  Tensor independent_shared_keys = as_position_independent_swizzle_tensor(shared_keys);
  Tensor thread_shared_keys = key_thread_copy.partition_D(independent_shared_keys);
  Tensor output_coordinates =
      thread_mma.partition_C(make_identity_tensor(make_shape(Int<64>{}, Int<8>{})));
  Tensor output_template = make_tensor(make_gmem_ptr(scores), make_shape(Int<64>{}, Int<8>{}),
                                       make_stride(Int<1>{}, Int<64>{}));
  Tensor fragment_output = thread_mma.partition_C(output_template);
  Tensor integer_accumulators = thread_mma.make_fragment_C(fragment_output);
  using ScaledAccumulator = decltype(make_fragment_like<float>(integer_accumulators));
  ScaledAccumulator scaled_accumulators[tiles_per_block];
#pragma unroll
  for (int tile_offset = 0; tile_offset < tiles_per_block; ++tile_offset) {
    clear(scaled_accumulators[tile_offset]);
  }

  if constexpr (tiles_per_block == 1) {
    const int tile = first_tile;
    const std::uint64_t key_scale_base =
        (static_cast<std::uint64_t>(head) * cache_capacity + tile * tile_tokens) * blocks;
    for (int index = static_cast<int>(threadIdx.x); index < tile_tokens * blocks;
         index += static_cast<int>(blockDim.x)) {
      storage.key_scales[index] = key_scales[key_scale_base + index];
    }
    for (int index = static_cast<int>(threadIdx.x); index < query_group * blocks;
         index += static_cast<int>(blockDim.x)) {
      storage.query_scales[index] =
          query_scales[static_cast<std::uint64_t>(head) * query_group * blocks + index];
    }
    __syncthreads();
  }

  if constexpr (tiles_per_block == 1) {
    constexpr int blocks_per_stage = 2;
#pragma unroll
    for (int first_block = 0; first_block < blocks; first_block += blocks_per_stage) {
#pragma unroll
      for (int stage = 0; stage < blocks_per_stage; ++stage) {
        const int block = first_block + stage;
        Tensor staged_keys =
            make_tensor(make_smem_ptr(storage.keys[stage].begin()), Int8KeyBlockLayout{});
        Tensor staged_queries =
            make_tensor(make_smem_ptr(storage.queries[stage].begin()), Int8QueryBlockLayout{});
        for (int index = static_cast<int>(threadIdx.x); index < query_group * block_width;
             index += static_cast<int>(blockDim.x)) {
          const int query = index / block_width;
          const int component = index - query * block_width;
          staged_queries(query, component) =
              queries[query_base + static_cast<std::uint64_t>(query) * dimension +
                      block * block_width + component];
        }
        const std::uint64_t key_base =
            (static_cast<std::uint64_t>(head) * cache_capacity + first_tile * tile_tokens) *
            dimension;
        Tensor global_keys =
            make_tensor(make_gmem_ptr(keys + key_base + block * block_width),
                        make_shape(Int<64>{}, Int<128>{}), make_stride(Int<512>{}, Int<1>{}));
        Tensor independent_staged_keys = as_position_independent_swizzle_tensor(staged_keys);
        copy(key_copy, key_thread_copy.partition_S(global_keys),
             key_thread_copy.partition_D(independent_staged_keys));
        cp_async_fence();
      }
      cp_async_wait<0>();
      __syncthreads();
#pragma unroll
      for (int stage = 0; stage < blocks_per_stage; ++stage) {
        const int block = first_block + stage;
        Tensor staged_keys =
            make_tensor(make_smem_ptr(storage.keys[stage].begin()), Int8KeyBlockLayout{});
        Tensor staged_queries =
            make_tensor(make_smem_ptr(storage.queries[stage].begin()), Int8QueryBlockLayout{});
        Tensor staged_fragment_keys = thread_mma.partition_A(staged_keys);
        Tensor staged_fragment_queries = thread_mma.partition_B(staged_queries);
        clear(integer_accumulators);
        warpgroup_fence_operand(integer_accumulators);
        warpgroup_arrive();
        cute::gemm(mma, staged_fragment_keys, staged_fragment_queries, integer_accumulators);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(integer_accumulators);
        CUTE_UNROLL
        for (int index = 0; index < size(integer_accumulators); ++index) {
          const auto coordinate = output_coordinates(index);
          const int token = static_cast<int>(get<0>(coordinate));
          const int query = static_cast<int>(get<1>(coordinate));
          scaled_accumulators[0](index) += static_cast<float>(integer_accumulators(index)) *
                                           storage.key_scales[token * blocks + block] *
                                           storage.query_scales[query * blocks + block];
        }
      }
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int block = 0; block < blocks; ++block) {
      for (int index = static_cast<int>(threadIdx.x); index < query_group * block_width;
           index += static_cast<int>(blockDim.x)) {
        const int query = index / block_width;
        const int component = index - query * block_width;
        shared_queries(query, component) =
            queries[query_base + static_cast<std::uint64_t>(query) * dimension +
                    block * block_width + component];
      }
#pragma unroll
      for (int tile_offset = 0; tile_offset < tiles_per_block; ++tile_offset) {
        const int tile = first_tile + tile_offset;
        const std::uint64_t key_base =
            (static_cast<std::uint64_t>(head) * cache_capacity + tile * tile_tokens) * dimension;
        Tensor global_keys =
            make_tensor(make_gmem_ptr(keys + key_base + block * block_width),
                        make_shape(Int<64>{}, Int<128>{}), make_stride(Int<512>{}, Int<1>{}));
        copy(key_copy, key_thread_copy.partition_S(global_keys), thread_shared_keys);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        clear(integer_accumulators);
        warpgroup_fence_operand(integer_accumulators);
        warpgroup_arrive();
        cute::gemm(mma, fragment_keys, fragment_queries, integer_accumulators);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(integer_accumulators);
        CUTE_UNROLL
        for (int index = 0; index < size(integer_accumulators); ++index) {
          const auto coordinate = output_coordinates(index);
          const int token = static_cast<int>(get<0>(coordinate));
          const int query = static_cast<int>(get<1>(coordinate));
          const float key_scale = key_scales[((static_cast<std::uint64_t>(head) * cache_capacity +
                                               tile * tile_tokens + token) *
                                              blocks) +
                                             block];
          const float query_scale =
              query_scales[(static_cast<std::uint64_t>(head) * query_group + query) * blocks +
                           block];
          scaled_accumulators[tile_offset](index) +=
              static_cast<float>(integer_accumulators(index)) * key_scale * query_scale;
        }
        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int tile_offset = 0; tile_offset < tiles_per_block; ++tile_offset) {
    const int tile = first_tile + tile_offset;
    CUTE_UNROLL
    for (int index = 0; index < size(scaled_accumulators[tile_offset]); ++index) {
      const auto coordinate = output_coordinates(index);
      const int token = static_cast<int>(get<0>(coordinate));
      const int query = static_cast<int>(get<1>(coordinate));
      const std::uint64_t output = static_cast<std::uint64_t>(head) * query_group * context_length +
                                   static_cast<std::uint64_t>(query) * context_length +
                                   tile * tile_tokens + token;
      scores[output] = Element(scaled_accumulators[tile_offset](index));
    }
  }
}

template <int tiles_per_block>
__global__ __launch_bounds__(128, 2) void global_qk_wgmma_int8_block128_tile_outer_kernel(
    const Int8Element *queries, const float *query_scales, const Int8Element *keys,
    const float *key_scales, Element *scores, int kv_heads, int context_length,
    int cache_capacity) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  constexpr int block_width = 128;
  constexpr int blocks = dimension / block_width;
  constexpr int blocks_per_stage = 2;
  const int tiles_per_head = context_length / tile_tokens;
  const int segments_per_head = tiles_per_head / tiles_per_block;
  const int head_segment = static_cast<int>(blockIdx.x);
  const int head = head_segment / segments_per_head;
  const int segment = head_segment - head * segments_per_head;
  const int first_tile = segment * tiles_per_block;
  const std::uint64_t query_base = static_cast<std::uint64_t>(head) * query_group * dimension;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<Int8GlobalQkTileOuterSharedStorage *>(storage_bytes);
  Int8QkMma mma;
  ThrMMA thread_mma = mma.get_slice(static_cast<int>(threadIdx.x));
  Int8KeyCopy key_copy;
  ThrCopy key_thread_copy = key_copy.get_slice(static_cast<int>(threadIdx.x));
  Tensor output_coordinates =
      thread_mma.partition_C(make_identity_tensor(make_shape(Int<64>{}, Int<8>{})));
  Tensor output_template = make_tensor(make_gmem_ptr(scores), make_shape(Int<64>{}, Int<8>{}),
                                       make_stride(Int<1>{}, Int<64>{}));
  Tensor fragment_output = thread_mma.partition_C(output_template);
  Tensor integer_accumulators = thread_mma.make_fragment_C(fragment_output);
  auto scaled_accumulator = make_fragment_like<float>(integer_accumulators);

#pragma unroll
  for (int block = 0; block < blocks; ++block) {
    Tensor staged_queries =
        make_tensor(make_smem_ptr(storage.queries[block].begin()), Int8QueryBlockLayout{});
    for (int index = static_cast<int>(threadIdx.x); index < query_group * block_width;
         index += static_cast<int>(blockDim.x)) {
      const int query = index / block_width;
      const int component = index - query * block_width;
      staged_queries(query, component) =
          queries[query_base + static_cast<std::uint64_t>(query) * dimension + block * block_width +
                  component];
    }
  }
  for (int index = static_cast<int>(threadIdx.x); index < query_group * blocks;
       index += static_cast<int>(blockDim.x)) {
    storage.query_scales[index] =
        query_scales[static_cast<std::uint64_t>(head) * query_group * blocks + index];
  }
  __syncthreads();

#pragma unroll 1
  for (int tile_offset = 0; tile_offset < tiles_per_block; ++tile_offset) {
    const int tile = first_tile + tile_offset;
    const std::uint64_t key_scale_base =
        (static_cast<std::uint64_t>(head) * cache_capacity + tile * tile_tokens) * blocks;
    for (int index = static_cast<int>(threadIdx.x); index < tile_tokens * blocks;
         index += static_cast<int>(blockDim.x)) {
      storage.key_scales[index] = key_scales[key_scale_base + index];
    }
    clear(scaled_accumulator);
#pragma unroll
    for (int first_block = 0; first_block < blocks; first_block += blocks_per_stage) {
#pragma unroll
      for (int stage = 0; stage < blocks_per_stage; ++stage) {
        const int block = first_block + stage;
        Tensor staged_keys =
            make_tensor(make_smem_ptr(storage.keys[stage].begin()), Int8KeyBlockLayout{});
        const std::uint64_t key_base =
            (static_cast<std::uint64_t>(head) * cache_capacity + tile * tile_tokens) * dimension;
        Tensor global_keys =
            make_tensor(make_gmem_ptr(keys + key_base + block * block_width),
                        make_shape(Int<64>{}, Int<128>{}), make_stride(Int<512>{}, Int<1>{}));
        Tensor independent_staged_keys = as_position_independent_swizzle_tensor(staged_keys);
        copy(key_copy, key_thread_copy.partition_S(global_keys),
             key_thread_copy.partition_D(independent_staged_keys));
        cp_async_fence();
      }
      cp_async_wait<0>();
      __syncthreads();
#pragma unroll
      for (int stage = 0; stage < blocks_per_stage; ++stage) {
        const int block = first_block + stage;
        Tensor staged_keys =
            make_tensor(make_smem_ptr(storage.keys[stage].begin()), Int8KeyBlockLayout{});
        Tensor staged_queries =
            make_tensor(make_smem_ptr(storage.queries[block].begin()), Int8QueryBlockLayout{});
        Tensor staged_fragment_keys = thread_mma.partition_A(staged_keys);
        Tensor staged_fragment_queries = thread_mma.partition_B(staged_queries);
        clear(integer_accumulators);
        warpgroup_fence_operand(integer_accumulators);
        warpgroup_arrive();
        cute::gemm(mma, staged_fragment_keys, staged_fragment_queries, integer_accumulators);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(integer_accumulators);
        CUTE_UNROLL
        for (int index = 0; index < size(integer_accumulators); ++index) {
          const auto coordinate = output_coordinates(index);
          const int token = static_cast<int>(get<0>(coordinate));
          const int query = static_cast<int>(get<1>(coordinate));
          scaled_accumulator(index) += static_cast<float>(integer_accumulators(index)) *
                                       storage.key_scales[token * blocks + block] *
                                       storage.query_scales[query * blocks + block];
        }
      }
      __syncthreads();
    }
    CUTE_UNROLL
    for (int index = 0; index < size(scaled_accumulator); ++index) {
      const auto coordinate = output_coordinates(index);
      const int token = static_cast<int>(get<0>(coordinate));
      const int query = static_cast<int>(get<1>(coordinate));
      const std::uint64_t output = static_cast<std::uint64_t>(head) * query_group * context_length +
                                   static_cast<std::uint64_t>(query) * context_length +
                                   tile * tile_tokens + token;
      scores[output] = Element(scaled_accumulator(index));
    }
  }
}

__global__ __launch_bounds__(128, 2) void global_pv_wgmma_int8_block128_kernel(
    const Int8Element *values, const float *value_scales, const Element *probabilities,
    Element *output, int context_length, int cache_capacity) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  constexpr int dimension_tiles = dimension / 64;
  constexpr int blocks_per_row = dimension / 128;
  const int head_dimension_tile = static_cast<int>(blockIdx.x);
  const int head = head_dimension_tile / dimension_tiles;
  const int dimension_tile = head_dimension_tile - head * dimension_tiles;
  const int component_start = dimension_tile * 64;
  const int scale_block = dimension_tile / 2;
  const std::uint64_t head_value_base =
      static_cast<std::uint64_t>(head) * cache_capacity * dimension;
  const std::uint64_t head_scale_base =
      static_cast<std::uint64_t>(head) * cache_capacity * blocks_per_row;
  const std::uint64_t head_probability_base =
      static_cast<std::uint64_t>(head) * query_group * context_length;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<Int8GlobalPvSharedStorage *>(storage_bytes);
  Tensor shared_values = make_tensor(make_smem_ptr(storage.values.begin()), ValueTileLayout{});
  Tensor shared_probabilities =
      make_tensor(make_smem_ptr(storage.probabilities.begin()), ProbabilityLayout{});
  GlobalPvMma mma;
  ThrMMA thread_mma = mma.get_slice(static_cast<int>(threadIdx.x));
  Tensor fragment_values = thread_mma.partition_A(shared_values);
  Tensor fragment_probabilities = thread_mma.partition_B(shared_probabilities);
  const std::uint64_t output_base =
      static_cast<std::uint64_t>(head) * query_group * dimension + component_start;
  Tensor global_output =
      make_tensor(make_gmem_ptr(output + output_base), make_shape(Int<64>{}, Int<8>{}),
                  make_stride(Int<1>{}, Int<512>{}));
  Tensor fragment_output = thread_mma.partition_C(global_output);
  Tensor accumulators = thread_mma.make_fragment_C(fragment_output);
  clear(accumulators);

  for (int tile_start = 0; tile_start < context_length; tile_start += tile_tokens) {
    for (int index = static_cast<int>(threadIdx.x); index < 64 * tile_tokens;
         index += static_cast<int>(blockDim.x)) {
      const int component = index / tile_tokens;
      const int token = index - component * tile_tokens;
      const std::uint64_t cache_row = tile_start + token;
      const float scale = value_scales[head_scale_base + cache_row * blocks_per_row + scale_block];
      const auto quantized =
          values[head_value_base + cache_row * dimension + component_start + component];
      shared_values(component, token) = Element(static_cast<float>(quantized) * scale);
    }
    for (int index = static_cast<int>(threadIdx.x); index < query_group * tile_tokens;
         index += static_cast<int>(blockDim.x)) {
      const int query = index / tile_tokens;
      const int token = index - query * tile_tokens;
      shared_probabilities(query, token) =
          probabilities[head_probability_base + static_cast<std::uint64_t>(query) * context_length +
                        tile_start + token];
    }
    __syncthreads();
    warpgroup_fence_operand(accumulators);
    warpgroup_arrive();
    cute::gemm(mma, fragment_values, fragment_probabilities, accumulators);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accumulators);
    __syncthreads();
  }
  copy(accumulators, fragment_output);
}

__global__ __launch_bounds__(128, 2) void global_pv_wgmma_int8_rs_segment_kernel(
    const Int8Element *values, const Int8Element *probabilities, std::int32_t *partial_outputs,
    int context_length, int cache_capacity, int segments) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  constexpr int dimension_tiles = dimension / 64;
  const int head_segment = static_cast<int>(blockIdx.x);
  const int segment = head_segment % segments;
  const int head = head_segment / segments;
  const int context_tiles = context_length / tile_tokens;
  const int tiles_per_segment = context_tiles / segments;
  const int first_tile = segment * tiles_per_segment;
  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<Int8GlobalPvRsSharedStorage *>(storage_bytes);
  Int8QkMma mma;
  ThrMMA thread_mma = mma.get_slice(static_cast<int>(threadIdx.x));
  Tensor output_template =
      make_tensor(make_gmem_ptr(partial_outputs), make_shape(Int<64>{}, Int<8>{}),
                  make_stride(Int<1>{}, Int<512>{}));
  Tensor fragment_output = thread_mma.partition_C(output_template);
  using Accumulator = decltype(thread_mma.make_fragment_C(fragment_output));
  Accumulator accumulators[dimension_tiles];
#pragma unroll
  for (int dimension_tile = 0; dimension_tile < dimension_tiles; ++dimension_tile) {
    clear(accumulators[dimension_tile]);
  }

  for (int tile_offset = 0; tile_offset < tiles_per_segment; ++tile_offset) {
    const int tile_start = (first_tile + tile_offset) * tile_tokens;
    for (int value_block = 0; value_block < 4; ++value_block) {
      Tensor shared_probabilities = make_tensor(
          make_smem_ptr(storage.probabilities[value_block].begin()), Int8PvProbabilityLayout{});
      const std::uint64_t probability_base =
          (static_cast<std::uint64_t>(value_block) * gridDim.x / segments + head) * query_group *
          context_length;
      for (int index = static_cast<int>(threadIdx.x); index < query_group * tile_tokens;
           index += static_cast<int>(blockDim.x)) {
        const int query = index / tile_tokens;
        const int token = index - query * tile_tokens;
        shared_probabilities(query, token) =
            probabilities[probability_base + static_cast<std::uint64_t>(query) * context_length +
                          tile_start + token];
      }
    }
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < dimension_tiles; ++dimension_tile) {
      constexpr int vectors_per_tile = cosize_v<Int8PvValueLayout> / 16;
      const std::uint64_t packed_tile =
          (static_cast<std::uint64_t>(head) * (cache_capacity / tile_tokens) + first_tile +
           tile_offset) *
              dimension_tiles +
          dimension_tile;
      const auto *source = values + packed_tile * cosize_v<Int8PvValueLayout>;
      auto *destination = storage.values[dimension_tile].begin();
      for (int vector_index = static_cast<int>(threadIdx.x); vector_index < vectors_per_tile;
           vector_index += static_cast<int>(blockDim.x)) {
        reinterpret_cast<uint4 *>(destination)[vector_index] =
            reinterpret_cast<const uint4 *>(source)[vector_index];
      }
    }
    __syncthreads();
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < dimension_tiles; ++dimension_tile) {
      warpgroup_fence_operand(accumulators[dimension_tile]);
    }
    warpgroup_arrive();
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < dimension_tiles; ++dimension_tile) {
      Tensor shared_values =
          make_tensor(make_smem_ptr(storage.values[dimension_tile].begin()), Int8PvValueLayout{});
      Tensor fragment_values = thread_mma.partition_A(shared_values);
      Tensor shared_probabilities =
          make_tensor(make_smem_ptr(storage.probabilities[dimension_tile / 2].begin()),
                      Int8PvProbabilityLayout{});
      Tensor fragment_probabilities = thread_mma.partition_B(shared_probabilities);
      cute::gemm(mma, fragment_values, fragment_probabilities, accumulators[dimension_tile]);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < dimension_tiles; ++dimension_tile) {
      warpgroup_fence_operand(accumulators[dimension_tile]);
    }
    __syncthreads();
  }

  const std::uint64_t partial_base =
      (static_cast<std::uint64_t>(head) * segments + segment) * query_group * dimension;
#pragma unroll
  for (int dimension_tile = 0; dimension_tile < dimension_tiles; ++dimension_tile) {
    Tensor partial_output =
        make_tensor(make_gmem_ptr(partial_outputs + partial_base + dimension_tile * 64),
                    make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
    Tensor partial_fragment = thread_mma.partition_C(partial_output);
    copy(accumulators[dimension_tile], partial_fragment);
  }
}

__global__ void reduce_global_pv_int8_segments_kernel(const std::int32_t *partial_outputs,
                                                      const float *scales, Element *output,
                                                      int head_batches, int segments) {
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t elements = static_cast<std::uint64_t>(head_batches) * query_group * dimension;
  if (index >= elements)
    return;
  const int component = static_cast<int>(index % dimension);
  const int query = static_cast<int>((index / dimension) % query_group);
  const int head = static_cast<int>(index / (dimension * query_group));
  std::int32_t total = 0;
  for (int segment = 0; segment < segments; ++segment) {
    const std::uint64_t partial =
        ((static_cast<std::uint64_t>(head) * segments + segment) * query_group + query) *
            dimension +
        component;
    total += partial_outputs[partial];
  }
  const int value_block = component / 128;
  output[index] =
      Element(static_cast<float>(total) *
              scales[(static_cast<std::uint64_t>(value_block) * head_batches + head) * query_group +
                     query]);
}

__global__ __launch_bounds__(256) void global_attention_wgmma_tile_kernel(const Element *queries,
                                                                          const Element *keys,
                                                                          const Element *values,
                                                                          Element *output,
                                                                          int context_length) {
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  const int head = static_cast<int>(blockIdx.x);
  const std::uint64_t key_value_base =
      static_cast<std::uint64_t>(head) * context_length * dimension;
  const std::uint64_t query_output_base =
      static_cast<std::uint64_t>(head) * query_group * dimension;
  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<GlobalAttentionTileSharedStorage *>(storage_bytes);
  Tensor shared_keys = make_tensor(make_smem_ptr(storage.keys.begin()), KeyLayout{});
  Tensor shared_queries = make_tensor(make_smem_ptr(storage.queries.begin()), QueryLayout{});
  Tensor shared_probabilities =
      make_tensor(make_smem_ptr(storage.probabilities.begin()), ProbabilityLayout{});
  Tensor shared_scores = make_tensor(make_smem_ptr(storage.scores.begin()), ScoreLayout{});

  for (int index = static_cast<int>(threadIdx.x); index < query_group * dimension;
       index += static_cast<int>(blockDim.x)) {
    const int query = index / dimension;
    const int component = index - query * dimension;
    shared_queries(query, component) = queries[query_output_base + index];
  }
  if (threadIdx.x < 128) {
    GlobalKeyCopy key_copy;
    ThrCopy thread_copy = key_copy.get_slice(threadIdx.x);
    Tensor global_keys =
        make_tensor(make_gmem_ptr(keys + key_value_base), make_shape(Int<64>{}, Int<512>{}),
                    make_stride(Int<512>{}, Int<1>{}));
    Tensor independent_keys = as_position_independent_swizzle_tensor(shared_keys);
    copy(key_copy, thread_copy.partition_S(global_keys), thread_copy.partition_D(independent_keys));
    cp_async_fence();
    cp_async_wait<0>();
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    GlobalQkMma qk_mma;
    ThrMMA thread_mma = qk_mma.get_slice(threadIdx.x);
    Tensor fragment_keys = thread_mma.partition_A(shared_keys);
    Tensor fragment_queries = thread_mma.partition_B(shared_queries);
    Tensor fragment_scores = thread_mma.partition_C(shared_scores);
    Tensor accumulators = thread_mma.make_fragment_C(fragment_scores);
    clear(accumulators);
    warpgroup_fence_operand(accumulators);
    warpgroup_arrive();
    cute::gemm(qk_mma, fragment_keys, fragment_queries, accumulators);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accumulators);
    copy(accumulators, fragment_scores);
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (warp < query_group) {
    const float score0 = static_cast<float>(shared_scores(lane, warp));
    const float score1 = static_cast<float>(shared_scores(lane + 32, warp));
    const float maximum = warp_maximum(max(score0, score1));
    const float exponential0 = expf(score0 - maximum);
    const float exponential1 = expf(score1 - maximum);
    const float inverse_sum = 1.0F / warp_total(exponential0 + exponential1);
    shared_probabilities(warp, lane) = Element(exponential0 * inverse_sum);
    shared_probabilities(warp, lane + 32) = Element(exponential1 * inverse_sum);
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
      Tensor shared_values =
          make_tensor(make_smem_ptr(storage.values[dimension_tile].begin()), ValueTileLayout{});
      constexpr int vectors_per_token = 8;
      for (int vector_index = static_cast<int>(threadIdx.x); vector_index < 64 * vectors_per_token;
           vector_index += 128) {
        const int token = vector_index / vectors_per_token;
        const int component_group = vector_index - token * vectors_per_token;
        const auto source = key_value_base + static_cast<std::uint64_t>(token) * 512 +
                            dimension_tile * 64 + component_group * 8;
        const uint4 packed = *reinterpret_cast<const uint4 *>(values + source);
        const auto *lanes = reinterpret_cast<const Element *>(&packed);
#pragma unroll
        for (int lane_index = 0; lane_index < 8; ++lane_index) {
          shared_values(component_group * 8 + lane_index, token) = lanes[lane_index];
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    GlobalPvMma pv_mma;
    ThrMMA thread_mma = pv_mma.get_slice(threadIdx.x);
    Tensor fragment_probabilities = thread_mma.partition_B(shared_probabilities);
    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
      Tensor shared_values =
          make_tensor(make_smem_ptr(storage.values[dimension_tile].begin()), ValueTileLayout{});
      Tensor global_output =
          make_tensor(make_gmem_ptr(output + query_output_base + dimension_tile * 64),
                      make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
      Tensor fragment_values = thread_mma.partition_A(shared_values);
      Tensor fragment_output = thread_mma.partition_C(global_output);
      Tensor accumulators = thread_mma.make_fragment_C(fragment_output);
      clear(accumulators);
      warpgroup_fence_operand(accumulators);
      warpgroup_arrive();
      cute::gemm(pv_mma, fragment_values, fragment_probabilities, accumulators);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(accumulators);
      copy(accumulators, fragment_output);
    }
  }
}

__global__ __launch_bounds__(128, 2) void global_pv_wgmma_fp8_rs_segment_kernel(
    const Fp8Element *values, const Fp8Element *probabilities, float *partial_outputs,
    Element *output, int context_length, int cache_capacity, int segments) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  constexpr int dimension_tiles = dimension / 64;
  const int head_dimension_segment = static_cast<int>(blockIdx.x);
  const int segment = head_dimension_segment % segments;
  const int head_dimension_tile = head_dimension_segment / segments;
  const int head = head_dimension_tile / dimension_tiles;
  const int dimension_tile = head_dimension_tile - head * dimension_tiles;
  const int context_tiles = context_length / tile_tokens;
  const int tiles_per_segment = context_tiles / segments;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<Fp8PvSharedStorage *>(storage_bytes);
  Tensor shared_values = make_tensor(make_smem_ptr(storage.values.begin()), Fp8ValueLayout{});
  Tensor shared_probabilities =
      make_tensor(make_smem_ptr(storage.probabilities.begin()), Fp8ProbabilityLayout{});
  Fp8PvRsMma mma;
  ThrMMA thread_mma = mma.get_slice(static_cast<int>(threadIdx.x));
  Tensor value_half = local_tile(shared_values, make_shape(Int<64>{}, Int<32>{}), make_coord(0, 0));
  Tensor value_fragment = thread_mma.partition_fragment_A(value_half);
  Tensor value_coordinates =
      thread_mma.partition_A(make_identity_tensor(make_shape(Int<64>{}, Int<32>{})));
  Tensor partitioned_probabilities = thread_mma.partition_B(shared_probabilities);
  Tensor fragment_probabilities = thread_mma.make_fragment_B(partitioned_probabilities);
  const std::uint64_t partial_base =
      (static_cast<std::uint64_t>(head_dimension_tile) * segments + segment) * 64 * query_group;
  Tensor partial_output =
      make_tensor(make_gmem_ptr(partial_outputs + partial_base), make_shape(Int<64>{}, Int<8>{}),
                  make_stride(Int<1>{}, Int<64>{}));
  Tensor fragment_output = thread_mma.partition_C(partial_output);
  Tensor accumulators = thread_mma.make_fragment_C(fragment_output);
  clear(accumulators);

  const int first_context_tile = segment * tiles_per_segment;
  for (int tile_offset = 0; tile_offset < tiles_per_segment; ++tile_offset) {
    const int token_tile = first_context_tile + tile_offset;
    const std::uint64_t value_tile =
        (static_cast<std::uint64_t>(head) * (cache_capacity / tile_tokens) + token_tile) *
            dimension_tiles +
        dimension_tile;
    const std::uint64_t probability_tile =
        static_cast<std::uint64_t>(head) * context_tiles + token_tile;
    if (threadIdx.x < cosize_v<Fp8ProbabilityLayout> / static_cast<int>(sizeof(uint4))) {
      reinterpret_cast<uint4 *>(storage.probabilities.begin())[threadIdx.x] =
          reinterpret_cast<const uint4 *>(
              probabilities + probability_tile * cosize_v<Fp8ProbabilityLayout>)[threadIdx.x];
    }
    __syncthreads();
    const Fp8Element *global_values = values + value_tile * cosize_v<Fp8ValueLayout>;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      CUTE_UNROLL
      for (int index = 0; index < size(value_fragment); ++index) {
        const auto coordinate = value_coordinates(index);
        value_fragment(index) =
            global_values[Fp8ValueLayout{}(static_cast<int>(get<0>(coordinate)),
                                           half * 32 + static_cast<int>(get<1>(coordinate)))];
      }
      warpgroup_fence_operand(value_fragment);
      warpgroup_fence_operand(accumulators);
      warpgroup_arrive();
      cute::gemm(mma, value_fragment(_, _, 0), fragment_probabilities(_, _, half), accumulators);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(value_fragment);
      warpgroup_fence_operand(accumulators);
    }
    __syncthreads();
  }
  if (segments == 1) {
    CUTE_UNROLL
    for (int index = 0; index < size(accumulators); ++index) {
      accumulators(index) *= 1.0F / (64.0F * 448.0F);
    }
    const std::uint64_t output_head = static_cast<std::uint64_t>(head) * query_group * dimension;
    Tensor global_output =
        make_tensor(make_gmem_ptr(output + output_head + dimension_tile * 64),
                    make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
    copy(accumulators, thread_mma.partition_C(global_output));
  } else {
    copy(accumulators, fragment_output);
  }
}

__global__ void reduce_global_pv_wgmma_fp8_rs_segments_kernel(const float *partial_outputs,
                                                              Element *output,
                                                              std::uint64_t output_elements,
                                                              int segments) {
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_elements)
    return;
  const std::uint64_t head = index / (query_group * dimension);
  const int head_index = static_cast<int>(index - head * query_group * dimension);
  const int query = head_index / dimension;
  const int component = head_index - query * dimension;
  const int dimension_tile = component / 64;
  const int component_in_tile = component - dimension_tile * 64;
  const std::uint64_t head_dimension_tile = head * 8 + dimension_tile;
  float total = 0.0F;
  for (int segment = 0; segment < segments; ++segment) {
    const std::uint64_t partial_base =
        (head_dimension_tile * segments + segment) * 64 * query_group;
    total += partial_outputs[partial_base + query * 64 + component_in_tile];
  }
  output[index] = Element(total * (1.0F / (64.0F * 448.0F)));
}

__global__ __launch_bounds__(128) void global_attention_wgmma_raw_rs_tile_kernel(
    const Element *queries, const Element *raw_cache, const float *inverse_rms_cache,
    const Element *key_norm_weight, const Element *cosine, const Element *sine, Element *output,
    int context_length) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int half_dimension = dimension / 2;
  constexpr int query_group = 8;
  const int head = static_cast<int>(blockIdx.x);
  const std::uint64_t raw_base = static_cast<std::uint64_t>(head) * context_length * dimension;
  const std::uint64_t inverse_rms_base = static_cast<std::uint64_t>(head) * context_length;
  const std::uint64_t query_output_base =
      static_cast<std::uint64_t>(head) * query_group * dimension;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<RsRawGlobalAttentionTileSharedStorage *>(storage_bytes);
  Tensor shared_raw = make_tensor(make_smem_ptr(storage.raw.begin()), KeyLayout{});
  Tensor shared_queries = make_tensor(make_smem_ptr(storage.queries.begin()), QueryLayout{});
  Tensor shared_probabilities =
      make_tensor(make_smem_ptr(storage.probabilities.begin()), ProbabilityLayout{});
  Tensor shared_scores = make_tensor(make_smem_ptr(storage.scores.begin()), ScoreLayout{});

  for (int index = static_cast<int>(threadIdx.x); index < query_group * dimension;
       index += static_cast<int>(blockDim.x)) {
    const int query = index / dimension;
    const int component = index - query * dimension;
    shared_queries(query, component) = queries[query_output_base + index];
  }
  if (threadIdx.x < tile_tokens) {
    storage.inverse_rms[threadIdx.x] = inverse_rms_cache[inverse_rms_base + threadIdx.x];
  }
  if (threadIdx.x < 128) {
    GlobalKeyCopy raw_copy;
    ThrCopy thread_copy = raw_copy.get_slice(threadIdx.x);
    Tensor global_raw =
        make_tensor(make_gmem_ptr(raw_cache + raw_base), make_shape(Int<64>{}, Int<512>{}),
                    make_stride(Int<512>{}, Int<1>{}));
    Tensor independent_raw = as_position_independent_swizzle_tensor(shared_raw);
    copy(raw_copy, thread_copy.partition_S(global_raw), thread_copy.partition_D(independent_raw));
    cp_async_fence();
    cp_async_wait<0>();
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    GlobalRsMma rs_mma;
    ThrMMA thread_mma = rs_mma.get_slice(threadIdx.x);
    Tensor raw_tile = local_tile(shared_raw, make_shape(Int<64>{}, Int<16>{}), make_coord(0, 0));
    Tensor key_fragment = thread_mma.partition_fragment_A(raw_tile);
    using KeyFragment = decltype(key_fragment);
    constexpr int qk_group = 4;
    KeyFragment key_fragments[qk_group] = {key_fragment, key_fragment, key_fragment, key_fragment};
    Tensor coordinates =
        thread_mma.partition_A(make_identity_tensor(make_shape(Int<64>{}, Int<16>{})));
    Tensor partitioned_queries = thread_mma.partition_B(shared_queries);
    Tensor fragment_queries = thread_mma.make_fragment_B(partitioned_queries);
    Tensor fragment_scores = thread_mma.partition_C(shared_scores);
    Tensor accumulators = thread_mma.make_fragment_C(fragment_scores);
    clear(accumulators);

#pragma unroll 1
    for (int k_group = 0; k_group < dimension / 16 / qk_group; ++k_group) {
#pragma unroll
      for (int fragment_index = 0; fragment_index < qk_group; ++fragment_index) {
        const int k_tile = k_group * qk_group + fragment_index;
        CUTE_UNROLL
        for (int index = 0; index < size(key_fragment); ++index) {
          const auto coordinate = coordinates(index);
          const int token = static_cast<int>(get<0>(coordinate));
          const int component = k_tile * 16 + static_cast<int>(get<1>(coordinate));
          const int partner =
              component < half_dimension ? component + half_dimension : component - half_dimension;
          const float inverse_rms = storage.inverse_rms[token];
          const Element normalized =
              Element(static_cast<float>(shared_raw(token, component)) * inverse_rms *
                      static_cast<float>(key_norm_weight[component]));
          const Element partner_normalized =
              Element(static_cast<float>(shared_raw(token, partner)) * inverse_rms *
                      static_cast<float>(key_norm_weight[partner]));
          const float rotated = component < half_dimension ? -static_cast<float>(partner_normalized)
                                                           : static_cast<float>(partner_normalized);
          const std::uint64_t rotary = static_cast<std::uint64_t>(token) * dimension + component;
          key_fragments[fragment_index](index) =
              Element(static_cast<float>(normalized) * static_cast<float>(cosine[rotary]) +
                      rotated * static_cast<float>(sine[rotary]));
        }
      }
#pragma unroll
      for (int fragment_index = 0; fragment_index < qk_group; ++fragment_index) {
        warpgroup_fence_operand(key_fragments[fragment_index]);
      }
      warpgroup_fence_operand(accumulators);
      warpgroup_arrive();
#pragma unroll
      for (int fragment_index = 0; fragment_index < qk_group; ++fragment_index) {
        const int k_tile = k_group * qk_group + fragment_index;
        cute::gemm(rs_mma, key_fragments[fragment_index](_, _, 0), fragment_queries(_, _, k_tile),
                   accumulators);
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
#pragma unroll
      for (int fragment_index = 0; fragment_index < qk_group; ++fragment_index) {
        warpgroup_fence_operand(key_fragments[fragment_index]);
      }
      warpgroup_fence_operand(accumulators);
    }
    copy(accumulators, fragment_scores);
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (warp < 4) {
#pragma unroll
    for (int query = warp; query < query_group; query += 4) {
      const float score0 = static_cast<float>(shared_scores(lane, query));
      const float score1 = static_cast<float>(shared_scores(lane + 32, query));
      const float maximum = warp_maximum(max(score0, score1));
      const float exponential0 = expf(score0 - maximum);
      const float exponential1 = expf(score1 - maximum);
      const float inverse_sum = 1.0F / warp_total(exponential0 + exponential1);
      shared_probabilities(query, lane) = Element(exponential0 * inverse_sum);
      shared_probabilities(query, lane + 32) = Element(exponential1 * inverse_sum);
    }
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    GlobalRsMma rs_mma;
    ThrMMA thread_mma = rs_mma.get_slice(threadIdx.x);
    Tensor raw_tile = local_tile(shared_raw, make_shape(Int<64>{}, Int<16>{}), make_coord(0, 0));
    Tensor value_fragment = thread_mma.partition_fragment_A(raw_tile);
    using ValueFragment = decltype(value_fragment);
    constexpr int pv_group = 4;
    ValueFragment value_fragments[pv_group] = {value_fragment, value_fragment, value_fragment,
                                               value_fragment};
    Tensor coordinates =
        thread_mma.partition_A(make_identity_tensor(make_shape(Int<64>{}, Int<16>{})));
    Tensor partitioned_probabilities = thread_mma.partition_B(shared_probabilities);
    Tensor fragment_probabilities = thread_mma.make_fragment_B(partitioned_probabilities);
    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
      Tensor global_output =
          make_tensor(make_gmem_ptr(output + query_output_base + dimension_tile * 64),
                      make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
      Tensor fragment_output = thread_mma.partition_C(global_output);
      Tensor accumulators = thread_mma.make_fragment_C(fragment_output);
      clear(accumulators);
#pragma unroll
      for (int token_group = 0; token_group < tile_tokens / 16 / pv_group; ++token_group) {
#pragma unroll
        for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
          const int token_tile = token_group * pv_group + fragment_index;
          CUTE_UNROLL
          for (int index = 0; index < size(value_fragment); ++index) {
            const auto coordinate = coordinates(index);
            const int component = dimension_tile * 64 + static_cast<int>(get<0>(coordinate));
            const int token = token_tile * 16 + static_cast<int>(get<1>(coordinate));
            value_fragments[fragment_index](index) = Element(
                static_cast<float>(shared_raw(token, component)) * storage.inverse_rms[token]);
          }
        }
#pragma unroll
        for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
          warpgroup_fence_operand(value_fragments[fragment_index]);
        }
        warpgroup_fence_operand(accumulators);
        warpgroup_arrive();
#pragma unroll
        for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
          const int token_tile = token_group * pv_group + fragment_index;
          cute::gemm(rs_mma, value_fragments[fragment_index](_, _, 0),
                     fragment_probabilities(_, _, token_tile), accumulators);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
#pragma unroll
        for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
          warpgroup_fence_operand(value_fragments[fragment_index]);
        }
        warpgroup_fence_operand(accumulators);
      }
      copy(accumulators, fragment_output);
    }
  }
}

__global__ __launch_bounds__(128, 2) void global_attention_wgmma_raw_rs_segment_kernel(
    const Element *queries, const Element *raw_cache, const float *inverse_rms_cache,
    const Element *key_norm_weight, const Element *cosine, const Element *sine,
    float *partial_outputs, float *partial_maxima, float *partial_sums, int context_length,
    int tiles_per_segment) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int half_dimension = dimension / 2;
  constexpr int query_group = 8;
  constexpr int pv_group = 4;
  const int tiles_per_head = context_length / tile_tokens;
  const int segments_per_head = tiles_per_head / tiles_per_segment;
  const int head_segment = static_cast<int>(blockIdx.x);
  const int head = head_segment / segments_per_head;
  const int segment = head_segment - head * segments_per_head;
  const int first_tile = segment * tiles_per_segment;
  const std::uint64_t raw_head = static_cast<std::uint64_t>(head) * context_length * dimension;
  const std::uint64_t inverse_rms_head = static_cast<std::uint64_t>(head) * context_length;
  const std::uint64_t query_head = static_cast<std::uint64_t>(head) * query_group * dimension;
  const std::uint64_t partial_head_segment =
      static_cast<std::uint64_t>(head_segment) * query_group * dimension;
  const std::uint64_t statistic_head_segment =
      static_cast<std::uint64_t>(head_segment) * query_group;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<RsRawGlobalAttentionSegmentSharedStorage *>(storage_bytes);
  Tensor shared_raw = make_tensor(make_smem_ptr(storage.raw.begin()), KeyLayout{});
  Tensor shared_queries = make_tensor(make_smem_ptr(storage.queries.begin()), QueryLayout{});
  Tensor shared_probabilities =
      make_tensor(make_smem_ptr(storage.probabilities.begin()), ProbabilityLayout{});
  Tensor shared_scores = make_tensor(make_smem_ptr(storage.scores.begin()), ScoreLayout{});
  for (int index = static_cast<int>(threadIdx.x); index < query_group * dimension;
       index += static_cast<int>(blockDim.x)) {
    const int query = index / dimension;
    const int component = index - query * dimension;
    shared_queries(query, component) = queries[query_head + index];
  }
  if (threadIdx.x < query_group) {
    storage.running_maximum[threadIdx.x] = -CUDART_INF_F;
    storage.running_sum[threadIdx.x] = 0.0F;
    storage.old_scale[threadIdx.x] = 0.0F;
  }
  __syncthreads();

  const int warpgroup_thread = static_cast<int>(threadIdx.x) & 127;
  GlobalRsMma rs_mma;
  ThrMMA thread_mma = rs_mma.get_slice(warpgroup_thread);
  Tensor output_coordinates =
      thread_mma.partition_C(make_identity_tensor(make_shape(Int<64>{}, Int<8>{})));
  Tensor first_global_output =
      make_tensor(make_gmem_ptr(partial_outputs + partial_head_segment),
                  make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
  Tensor first_fragment_output = thread_mma.partition_C(first_global_output);
  Tensor output_fragment_template = thread_mma.make_fragment_C(first_fragment_output);
  using OutputFragment = decltype(output_fragment_template);
  OutputFragment output_accumulators[8] = {output_fragment_template, output_fragment_template,
                                           output_fragment_template, output_fragment_template,
                                           output_fragment_template, output_fragment_template,
                                           output_fragment_template, output_fragment_template};
#pragma unroll
  for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
    clear(output_accumulators[dimension_tile]);
  }

  GlobalKeyCopy raw_copy;
  ThrCopy raw_thread_copy = raw_copy.get_slice(warpgroup_thread);
  Tensor independent_raw = as_position_independent_swizzle_tensor(shared_raw);
  Tensor thread_shared_raw = raw_thread_copy.partition_D(independent_raw);
  Tensor raw_fragment_source =
      local_tile(shared_raw, make_shape(Int<64>{}, Int<16>{}), make_coord(0, 0));
  Tensor fragment_template = thread_mma.partition_fragment_A(raw_fragment_source);
  using RsFragment = decltype(fragment_template);
  Tensor fragment_coordinates =
      thread_mma.partition_A(make_identity_tensor(make_shape(Int<64>{}, Int<16>{})));
  Tensor partitioned_queries = thread_mma.partition_B(shared_queries);
  Tensor fragment_queries = thread_mma.make_fragment_B(partitioned_queries);
  Tensor partitioned_probabilities = thread_mma.partition_B(shared_probabilities);
  Tensor fragment_probabilities = thread_mma.make_fragment_B(partitioned_probabilities);
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  for (int tile_offset = 0; tile_offset < tiles_per_segment; ++tile_offset) {
    const int tile = first_tile + tile_offset;
    const int first_token = tile * tile_tokens;
    const std::uint64_t tile_base = raw_head + static_cast<std::uint64_t>(first_token) * dimension;
    if (threadIdx.x < tile_tokens) {
      storage.inverse_rms[threadIdx.x] =
          inverse_rms_cache[inverse_rms_head + first_token + threadIdx.x];
    }
    if (threadIdx.x < 128) {
      Tensor global_raw =
          make_tensor(make_gmem_ptr(raw_cache + tile_base), make_shape(Int<64>{}, Int<512>{}),
                      make_stride(Int<512>{}, Int<1>{}));
      copy(raw_copy, raw_thread_copy.partition_S(global_raw), thread_shared_raw);
      cp_async_fence();
      cp_async_wait<0>();
    }
    __syncthreads();

    if (threadIdx.x < 128) {
      Tensor fragment_scores = thread_mma.partition_C(shared_scores);
      Tensor score_accumulators = thread_mma.make_fragment_C(fragment_scores);
      // RoPE couples component c with c + D/2. Process those two 16-column tiles together so the
      // exact BF16 pre-RoPE normalization is computed once per component rather than once while
      // visiting each half of the reduction. Keep these fragments scoped to QK so their registers
      // can be reused by the four PV fragments below.
      constexpr int qk_pair_group = 2;
      RsFragment qk_fragments[2 * qk_pair_group] = {fragment_template, fragment_template,
                                                    fragment_template, fragment_template};
      clear(score_accumulators);
#pragma unroll 1
      for (int pair_group = 0; pair_group < half_dimension / 16 / qk_pair_group; ++pair_group) {
#pragma unroll
        for (int pair_index = 0; pair_index < qk_pair_group; ++pair_index) {
          const int pair_tile = pair_group * qk_pair_group + pair_index;
          CUTE_UNROLL
          for (int index = 0; index < size(fragment_template); ++index) {
            const auto coordinate = fragment_coordinates(index);
            const int token = static_cast<int>(get<0>(coordinate));
            const int low_component = pair_tile * 16 + static_cast<int>(get<1>(coordinate));
            const int high_component = low_component + half_dimension;
            const float inverse_rms = storage.inverse_rms[token];
            const float low_normalized_fp32 = static_cast<float>(shared_raw(token, low_component)) *
                                              inverse_rms *
                                              static_cast<float>(key_norm_weight[low_component]);
            const float high_normalized_fp32 =
                static_cast<float>(shared_raw(token, high_component)) * inverse_rms *
                static_cast<float>(key_norm_weight[high_component]);
            const __nv_bfloat162 normalized_pair =
                __float22bfloat162_rn(make_float2(low_normalized_fp32, high_normalized_fp32));
            const __nv_bfloat16 low_normalized = __low2bfloat16(normalized_pair);
            const __nv_bfloat16 high_normalized = __high2bfloat16(normalized_pair);
            if (pair_tile < dimension / 8 / 16) {
              const std::uint64_t rotary =
                  static_cast<std::uint64_t>(low_component) * context_length + first_token + token;
              const float cosine_value = static_cast<float>(cosine[rotary]);
              const float sine_value = static_cast<float>(sine[rotary]);
              const float low_rotated = __bfloat162float(low_normalized) * cosine_value -
                                        __bfloat162float(high_normalized) * sine_value;
              const float high_rotated = __bfloat162float(high_normalized) * cosine_value +
                                         __bfloat162float(low_normalized) * sine_value;
              const __nv_bfloat162 rotated_pair =
                  __float22bfloat162_rn(make_float2(low_rotated, high_rotated));
              qk_fragments[2 * pair_index](index) = Element(__low2bfloat16(rotated_pair));
              qk_fragments[2 * pair_index + 1](index) = Element(__high2bfloat16(rotated_pair));
            } else {
              qk_fragments[2 * pair_index](index) = Element(low_normalized);
              qk_fragments[2 * pair_index + 1](index) = Element(high_normalized);
            }
          }
        }
#pragma unroll
        for (int fragment_index = 0; fragment_index < 2 * qk_pair_group; ++fragment_index) {
          warpgroup_fence_operand(qk_fragments[fragment_index]);
        }
        warpgroup_fence_operand(score_accumulators);
        warpgroup_arrive();
#pragma unroll
        for (int pair_index = 0; pair_index < qk_pair_group; ++pair_index) {
          const int pair_tile = pair_group * qk_pair_group + pair_index;
          cute::gemm(rs_mma, qk_fragments[2 * pair_index](_, _, 0),
                     fragment_queries(_, _, pair_tile), score_accumulators);
          cute::gemm(rs_mma, qk_fragments[2 * pair_index + 1](_, _, 0),
                     fragment_queries(_, _, pair_tile + half_dimension / 16), score_accumulators);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
#pragma unroll
        for (int fragment_index = 0; fragment_index < 2 * qk_pair_group; ++fragment_index) {
          warpgroup_fence_operand(qk_fragments[fragment_index]);
        }
        warpgroup_fence_operand(score_accumulators);
      }
      copy(score_accumulators, fragment_scores);
    }
    __syncthreads();

    if (warp < 4) {
#pragma unroll
      for (int query = warp; query < query_group; query += 4) {
        const float score0 = static_cast<float>(shared_scores(lane, query));
        const float score1 = static_cast<float>(shared_scores(lane + 32, query));
        const float tile_maximum = warp_maximum(max(score0, score1));
        const float next_maximum = max(storage.running_maximum[query], tile_maximum);
        const float old_scale = expf(storage.running_maximum[query] - next_maximum);
        const float exponential0 = expf(score0 - next_maximum);
        const float exponential1 = expf(score1 - next_maximum);
        const float tile_sum = warp_total(exponential0 + exponential1);
        shared_probabilities(query, lane) = Element(exponential0);
        shared_probabilities(query, lane + 32) = Element(exponential1);
        if (lane == 0) {
          storage.old_scale[query] = old_scale;
          storage.running_sum[query] = storage.running_sum[query] * old_scale + tile_sum;
          storage.running_maximum[query] = next_maximum;
        }
      }
    }
    __syncthreads();

    if (threadIdx.x < 128) {
      RsFragment pv_fragments[4] = {fragment_template, fragment_template, fragment_template,
                                    fragment_template};
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
        auto &output_accumulator = output_accumulators[dimension_tile];
        CUTE_UNROLL
        for (int index = 0; index < size(output_accumulator); ++index) {
          const int query = static_cast<int>(get<1>(output_coordinates(index)));
          output_accumulator(index) *= storage.old_scale[query];
        }
#pragma unroll
        for (int token_group = 0; token_group < tile_tokens / 16 / pv_group; ++token_group) {
#pragma unroll
          for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
            const int token_tile = token_group * pv_group + fragment_index;
            static_assert(size(fragment_template) % 2 == 0);
            CUTE_UNROLL
            for (int index = 0; index < size(fragment_template); index += 2) {
              const auto coordinate0 = fragment_coordinates(index);
              const auto coordinate1 = fragment_coordinates(index + 1);
              const int component0 = dimension_tile * 64 + static_cast<int>(get<0>(coordinate0));
              const int component1 = dimension_tile * 64 + static_cast<int>(get<0>(coordinate1));
              const int token0 = token_tile * 16 + static_cast<int>(get<1>(coordinate0));
              const int token1 = token_tile * 16 + static_cast<int>(get<1>(coordinate1));
              const float value0 =
                  static_cast<float>(shared_raw(token0, component0)) * storage.inverse_rms[token0];
              const float value1 =
                  static_cast<float>(shared_raw(token1, component1)) * storage.inverse_rms[token1];
              const __nv_bfloat162 value_pair = __float22bfloat162_rn(make_float2(value0, value1));
              pv_fragments[fragment_index](index) = Element(__low2bfloat16(value_pair));
              pv_fragments[fragment_index](index + 1) = Element(__high2bfloat16(value_pair));
            }
          }
#pragma unroll
          for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
            warpgroup_fence_operand(pv_fragments[fragment_index]);
          }
          warpgroup_fence_operand(output_accumulator);
          warpgroup_arrive();
#pragma unroll
          for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
            const int token_tile = token_group * pv_group + fragment_index;
            cute::gemm(rs_mma, pv_fragments[fragment_index](_, _, 0),
                       fragment_probabilities(_, _, token_tile), output_accumulator);
          }
          warpgroup_commit_batch();
          warpgroup_wait<0>();
#pragma unroll
          for (int fragment_index = 0; fragment_index < pv_group; ++fragment_index) {
            warpgroup_fence_operand(pv_fragments[fragment_index]);
          }
          warpgroup_fence_operand(output_accumulator);
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < 128) {
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
      Tensor global_output =
          make_tensor(make_gmem_ptr(partial_outputs + partial_head_segment + dimension_tile * 64),
                      make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
      Tensor fragment_output = thread_mma.partition_C(global_output);
      copy(output_accumulators[dimension_tile], fragment_output);
    }
  }
  if (threadIdx.x < query_group) {
    partial_maxima[statistic_head_segment + threadIdx.x] = storage.running_maximum[threadIdx.x];
    partial_sums[statistic_head_segment + threadIdx.x] = storage.running_sum[threadIdx.x];
  }
}

__global__ __launch_bounds__(256) void global_attention_wgmma_segment_kernel(
    const Element *queries, const Element *keys, const Element *values, float *partial_outputs,
    float *partial_maxima, float *partial_sums, int context_length, int tiles_per_segment) {
  constexpr int tile_tokens = 64;
  constexpr int dimension = 512;
  constexpr int query_group = 8;
  const int tiles_per_head = context_length / tile_tokens;
  const int segments_per_head = tiles_per_head / tiles_per_segment;
  const int head_segment = static_cast<int>(blockIdx.x);
  const int head = head_segment / segments_per_head;
  const int segment = head_segment - head * segments_per_head;
  const int first_tile = segment * tiles_per_segment;
  const std::uint64_t key_value_head =
      static_cast<std::uint64_t>(head) * context_length * dimension;
  const std::uint64_t query_head = static_cast<std::uint64_t>(head) * query_group * dimension;
  const std::uint64_t partial_head_segment =
      static_cast<std::uint64_t>(head_segment) * query_group * dimension;
  const std::uint64_t statistic_head_segment =
      static_cast<std::uint64_t>(head_segment) * query_group;

  extern __shared__ __align__(128) unsigned char storage_bytes[];
  auto &storage = *reinterpret_cast<GlobalAttentionTileSharedStorage *>(storage_bytes);
  Tensor shared_keys = make_tensor(make_smem_ptr(storage.keys.begin()), KeyLayout{});
  Tensor shared_queries = make_tensor(make_smem_ptr(storage.queries.begin()), QueryLayout{});
  Tensor shared_probabilities =
      make_tensor(make_smem_ptr(storage.probabilities.begin()), ProbabilityLayout{});
  Tensor shared_scores = make_tensor(make_smem_ptr(storage.scores.begin()), ScoreLayout{});
  for (int index = static_cast<int>(threadIdx.x); index < query_group * dimension;
       index += static_cast<int>(blockDim.x)) {
    const int query = index / dimension;
    const int component = index - query * dimension;
    shared_queries(query, component) = queries[query_head + index];
  }
  if (threadIdx.x < query_group) {
    storage.running_maximum[threadIdx.x] = -CUDART_INF_F;
    storage.running_sum[threadIdx.x] = 0.0F;
    storage.old_scale[threadIdx.x] = 0.0F;
  }
  __syncthreads();

  const int warpgroup_thread = static_cast<int>(threadIdx.x) & 127;
  GlobalPvMma pv_mma;
  ThrMMA pv_thread_mma = pv_mma.get_slice(warpgroup_thread);
  Tensor partial_template =
      make_tensor(make_gmem_ptr(partial_outputs + partial_head_segment),
                  make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
  Tensor partial_fragment = pv_thread_mma.partition_C(partial_template);
  using OutputAccumulator = decltype(pv_thread_mma.make_fragment_C(partial_fragment));
  OutputAccumulator output_accumulators[8];
  Tensor output_coordinates =
      pv_thread_mma.partition_C(make_identity_tensor(make_shape(Int<64>{}, Int<8>{})));
  if (threadIdx.x < 128) {
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
      clear(output_accumulators[dimension_tile]);
    }
  }

  GlobalKeyCopy key_copy;
  ThrCopy key_thread_copy = key_copy.get_slice(warpgroup_thread);
  Tensor independent_keys = as_position_independent_swizzle_tensor(shared_keys);
  Tensor thread_shared_keys = key_thread_copy.partition_D(independent_keys);
  GlobalQkMma qk_mma;
  ThrMMA qk_thread_mma = qk_mma.get_slice(warpgroup_thread);
  Tensor fragment_keys = qk_thread_mma.partition_A(shared_keys);
  Tensor fragment_queries = qk_thread_mma.partition_B(shared_queries);
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;

  for (int tile_offset = 0; tile_offset < tiles_per_segment; ++tile_offset) {
    const int tile = first_tile + tile_offset;
    const std::uint64_t tile_base =
        key_value_head + static_cast<std::uint64_t>(tile) * tile_tokens * dimension;
    if (threadIdx.x < 128) {
      Tensor global_keys =
          make_tensor(make_gmem_ptr(keys + tile_base), make_shape(Int<64>{}, Int<512>{}),
                      make_stride(Int<512>{}, Int<1>{}));
      copy(key_copy, key_thread_copy.partition_S(global_keys), thread_shared_keys);
      cp_async_fence();
      cp_async_wait<0>();
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      Tensor fragment_scores = qk_thread_mma.partition_C(shared_scores);
      Tensor score_accumulators = qk_thread_mma.make_fragment_C(fragment_scores);
      clear(score_accumulators);
      warpgroup_fence_operand(score_accumulators);
      warpgroup_arrive();
      cute::gemm(qk_mma, fragment_keys, fragment_queries, score_accumulators);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(score_accumulators);
      copy(score_accumulators, fragment_scores);
    }
    __syncthreads();

    if (warp < query_group) {
      const float score0 = static_cast<float>(shared_scores(lane, warp));
      const float score1 = static_cast<float>(shared_scores(lane + 32, warp));
      const float tile_maximum = warp_maximum(max(score0, score1));
      const float next_maximum = max(storage.running_maximum[warp], tile_maximum);
      const float old_scale = expf(storage.running_maximum[warp] - next_maximum);
      const float exponential0 = expf(score0 - next_maximum);
      const float exponential1 = expf(score1 - next_maximum);
      const float tile_sum = warp_total(exponential0 + exponential1);
      shared_probabilities(warp, lane) = Element(exponential0);
      shared_probabilities(warp, lane + 32) = Element(exponential1);
      if (lane == 0) {
        storage.old_scale[warp] = old_scale;
        storage.running_sum[warp] = storage.running_sum[warp] * old_scale + tile_sum;
        storage.running_maximum[warp] = next_maximum;
      }
    }
    __syncthreads();

    if (threadIdx.x < 128) {
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
        CUTE_UNROLL
        for (int index = 0; index < size(output_accumulators[dimension_tile]); ++index) {
          const int query = static_cast<int>(get<1>(output_coordinates(index)));
          output_accumulators[dimension_tile](index) *= storage.old_scale[query];
        }
      }
      for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
        Tensor shared_values =
            make_tensor(make_smem_ptr(storage.values[dimension_tile].begin()), ValueTileLayout{});
        constexpr int vectors_per_token = 8;
        for (int vector_index = static_cast<int>(threadIdx.x);
             vector_index < 64 * vectors_per_token; vector_index += 128) {
          const int token = vector_index / vectors_per_token;
          const int component_group = vector_index - token * vectors_per_token;
          const auto source = tile_base + static_cast<std::uint64_t>(token) * 512 +
                              dimension_tile * 64 + component_group * 8;
          const uint4 packed = *reinterpret_cast<const uint4 *>(values + source);
          const auto *lanes = reinterpret_cast<const Element *>(&packed);
#pragma unroll
          for (int lane_index = 0; lane_index < 8; ++lane_index) {
            shared_values(component_group * 8 + lane_index, token) = lanes[lane_index];
          }
        }
      }
    }
    __syncthreads();

    if (threadIdx.x < 128) {
      Tensor fragment_probabilities = pv_thread_mma.partition_B(shared_probabilities);
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
        warpgroup_fence_operand(output_accumulators[dimension_tile]);
      }
      warpgroup_arrive();
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
        Tensor shared_values =
            make_tensor(make_smem_ptr(storage.values[dimension_tile].begin()), ValueTileLayout{});
        Tensor fragment_values = pv_thread_mma.partition_A(shared_values);
        cute::gemm(pv_mma, fragment_values, fragment_probabilities,
                   output_accumulators[dimension_tile]);
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
#pragma unroll
      for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
        warpgroup_fence_operand(output_accumulators[dimension_tile]);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < 128) {
#pragma unroll
    for (int dimension_tile = 0; dimension_tile < 8; ++dimension_tile) {
      Tensor partial_output =
          make_tensor(make_gmem_ptr(partial_outputs + partial_head_segment + dimension_tile * 64),
                      make_shape(Int<64>{}, Int<8>{}), make_stride(Int<1>{}, Int<512>{}));
      Tensor fragment_output = pv_thread_mma.partition_C(partial_output);
      copy(output_accumulators[dimension_tile], fragment_output);
    }
  }
  if (threadIdx.x < query_group) {
    partial_maxima[statistic_head_segment + threadIdx.x] = storage.running_maximum[threadIdx.x];
    partial_sums[statistic_head_segment + threadIdx.x] = storage.running_sum[threadIdx.x];
  }
}

__global__ void reduce_global_attention_segments_kernel(const float *partial_outputs,
                                                        const float *partial_maxima,
                                                        const float *partial_sums, Element *output,
                                                        int segments_per_head) {
  constexpr int query_group = 8;
  constexpr int dimension = 512;
  const int head_query = static_cast<int>(blockIdx.x);
  const int head = head_query / query_group;
  const int query = head_query - head * query_group;
  float maximum = -CUDART_INF_F;
  for (int segment = 0; segment < segments_per_head; ++segment) {
    const std::uint64_t statistic =
        (static_cast<std::uint64_t>(head) * segments_per_head + segment) * query_group + query;
    maximum = max(maximum, partial_maxima[statistic]);
  }
  float denominator = 0.0F;
  for (int segment = 0; segment < segments_per_head; ++segment) {
    const std::uint64_t statistic =
        (static_cast<std::uint64_t>(head) * segments_per_head + segment) * query_group + query;
    denominator += partial_sums[statistic] * expf(partial_maxima[statistic] - maximum);
  }
  for (int component = static_cast<int>(threadIdx.x); component < dimension;
       component += static_cast<int>(blockDim.x)) {
    float numerator = 0.0F;
    for (int segment = 0; segment < segments_per_head; ++segment) {
      const std::uint64_t head_segment =
          static_cast<std::uint64_t>(head) * segments_per_head + segment;
      const std::uint64_t statistic = head_segment * query_group + query;
      const std::uint64_t partial = (head_segment * query_group + query) * dimension + component;
      numerator += partial_outputs[partial] * expf(partial_maxima[statistic] - maximum);
    }
    output[(static_cast<std::uint64_t>(head) * query_group + query) * dimension + component] =
        Element(numerator / denominator);
  }
}

} // namespace

void grouped_global_qk_wgmma_bf16(const void *query, const void *key_cache, void *scores, int batch,
                                  int kv_heads, int query_group, int context_length,
                                  int head_dimension, cudaStream_t stream, int tiles_per_block) {
  if (query == nullptr || key_cache == nullptr || scores == nullptr || batch <= 0 ||
      kv_heads <= 0 || query_group != 8 || context_length <= 0 || context_length % 64 != 0 ||
      head_dimension != 512 || tiles_per_block <= 0 ||
      (context_length / 64) % tiles_per_block != 0) {
    throw std::runtime_error("invalid Hopper global QK shape");
  }
  constexpr int threads = 128;
  constexpr std::size_t shared_bytes = sizeof(GlobalQkSharedStorage);
  check(cudaFuncSetAttribute(global_qk_wgmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set Hopper global QK shared memory");
  const int blocks = batch * kv_heads * (context_length / 64 / tiles_per_block);
  global_qk_wgmma_kernel<<<blocks, threads, shared_bytes, stream>>>(
      static_cast<const Element *>(query), static_cast<const Element *>(key_cache),
      static_cast<Element *>(scores), kv_heads, context_length, tiles_per_block);
  check(cudaPeekAtLastError(), "launch Hopper global QK");
}

void grouped_global_qk_wgmma_int8_block128(const void *query, const float *query_scales,
                                           const void *key_cache, const float *key_scales,
                                           void *scores, int batch, int kv_heads, int query_group,
                                           int context_length, int head_dimension,
                                           int cache_capacity, cudaStream_t stream,
                                           int tiles_per_block) {
  if (query == nullptr || query_scales == nullptr || key_cache == nullptr ||
      key_scales == nullptr || scores == nullptr || batch <= 0 || kv_heads <= 0 ||
      query_group != 8 || context_length <= 0 || context_length % 64 != 0 ||
      head_dimension != 512 || cache_capacity < context_length || tiles_per_block <= 0 ||
      tiles_per_block > 32 || (context_length / 64) % tiles_per_block != 0) {
    throw std::runtime_error("invalid INT8 block-128 Hopper QK shape");
  }
  constexpr int threads = 128;
  constexpr std::size_t shared_bytes = sizeof(Int8GlobalQkTileOuterSharedStorage);
  const int grid = batch * kv_heads * (context_length / 64 / tiles_per_block);
  const auto *typed_query = static_cast<const Int8Element *>(query);
  const auto *typed_keys = static_cast<const Int8Element *>(key_cache);
  auto *typed_scores = static_cast<Element *>(scores);
  if (tiles_per_block == 1) {
    global_qk_wgmma_int8_block128_tile_outer_kernel<1><<<grid, threads, shared_bytes, stream>>>(
        typed_query, query_scales, typed_keys, key_scales, typed_scores, kv_heads, context_length,
        cache_capacity);
  } else if (tiles_per_block == 2) {
    global_qk_wgmma_int8_block128_tile_outer_kernel<2><<<grid, threads, shared_bytes, stream>>>(
        typed_query, query_scales, typed_keys, key_scales, typed_scores, kv_heads, context_length,
        cache_capacity);
  } else if (tiles_per_block == 4) {
    global_qk_wgmma_int8_block128_tile_outer_kernel<4><<<grid, threads, shared_bytes, stream>>>(
        typed_query, query_scales, typed_keys, key_scales, typed_scores, kv_heads, context_length,
        cache_capacity);
  } else if (tiles_per_block == 8) {
    global_qk_wgmma_int8_block128_tile_outer_kernel<8><<<grid, threads, shared_bytes, stream>>>(
        typed_query, query_scales, typed_keys, key_scales, typed_scores, kv_heads, context_length,
        cache_capacity);
  } else if (tiles_per_block == 16) {
    global_qk_wgmma_int8_block128_tile_outer_kernel<16><<<grid, threads, shared_bytes, stream>>>(
        typed_query, query_scales, typed_keys, key_scales, typed_scores, kv_heads, context_length,
        cache_capacity);
  } else {
    global_qk_wgmma_int8_block128_tile_outer_kernel<32><<<grid, threads, shared_bytes, stream>>>(
        typed_query, query_scales, typed_keys, key_scales, typed_scores, kv_heads, context_length,
        cache_capacity);
  }
  check(cudaPeekAtLastError(), "launch INT8 block-128 Hopper QK");
}

void grouped_global_attention_wgmma_tile_bf16(const void *query, const void *key_cache,
                                              const void *value_cache, void *output, int batch,
                                              int kv_heads, int query_group, int context_length,
                                              int head_dimension, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
      batch <= 0 || kv_heads <= 0 || query_group != 8 || context_length != 64 ||
      head_dimension != 512) {
    throw std::runtime_error("invalid Hopper global-attention tile shape");
  }
  constexpr int threads = 256;
  constexpr std::size_t shared_bytes = sizeof(GlobalAttentionTileSharedStorage);
  check(cudaFuncSetAttribute(global_attention_wgmma_tile_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set Hopper global-attention tile shared memory");
  global_attention_wgmma_tile_kernel<<<batch * kv_heads, threads, shared_bytes, stream>>>(
      static_cast<const Element *>(query), static_cast<const Element *>(key_cache),
      static_cast<const Element *>(value_cache), static_cast<Element *>(output), context_length);
  check(cudaPeekAtLastError(), "launch Hopper global-attention tile");
}

void grouped_global_pv_wgmma_int8_block128(const void *value_cache, const float *value_scales,
                                           const void *probabilities, void *output, int batch,
                                           int kv_heads, int query_group, int context_length,
                                           int head_dimension, int cache_capacity,
                                           cudaStream_t stream) {
  if (value_cache == nullptr || value_scales == nullptr || probabilities == nullptr ||
      output == nullptr || batch <= 0 || kv_heads <= 0 || query_group != 8 || context_length <= 0 ||
      context_length % 64 != 0 || head_dimension != 512 || cache_capacity < context_length) {
    throw std::runtime_error("invalid INT8 block-128 Hopper PV shape");
  }
  constexpr int threads = 128;
  constexpr std::size_t shared_bytes = sizeof(Int8GlobalPvSharedStorage);
  constexpr int dimension_tiles = 8;
  global_pv_wgmma_int8_block128_kernel<<<batch * kv_heads * dimension_tiles, threads, shared_bytes,
                                         stream>>>(
      static_cast<const Int8Element *>(value_cache), value_scales,
      static_cast<const Element *>(probabilities), static_cast<Element *>(output), context_length,
      cache_capacity);
  check(cudaPeekAtLastError(), "launch INT8 block-128 Hopper PV");
}

void grouped_global_pv_wgmma_int8_rs(const void *value_cache, const void *probabilities,
                                     const float *probability_scales, void *partial_outputs,
                                     void *output, int batch, int kv_heads, int query_group,
                                     int context_length, int head_dimension, int cache_capacity,
                                     int segments, cudaStream_t stream) {
  if (value_cache == nullptr || probabilities == nullptr || probability_scales == nullptr ||
      partial_outputs == nullptr || output == nullptr || batch <= 0 || kv_heads <= 0 ||
      query_group != 8 || context_length <= 0 || context_length % 64 != 0 ||
      head_dimension != 512 || cache_capacity < context_length || segments <= 0 || segments > 32 ||
      (context_length / 64) % segments != 0) {
    throw std::runtime_error("invalid INT8 Hopper RS-PV shape");
  }
  constexpr int threads = 128;
  constexpr std::size_t shared_bytes = sizeof(Int8GlobalPvRsSharedStorage);
  check(cudaFuncSetAttribute(global_pv_wgmma_int8_rs_segment_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set INT8 Hopper RS-PV shared memory");
  const int head_batches = batch * kv_heads;
  global_pv_wgmma_int8_rs_segment_kernel<<<head_batches * segments, threads, shared_bytes,
                                           stream>>>(
      static_cast<const Int8Element *>(value_cache),
      static_cast<const Int8Element *>(probabilities), static_cast<std::int32_t *>(partial_outputs),
      context_length, cache_capacity, segments);
  check(cudaPeekAtLastError(), "launch segmented INT8 Hopper RS-PV");
  const std::uint64_t output_elements =
      static_cast<std::uint64_t>(head_batches) * query_group * head_dimension;
  constexpr int reduction_threads = 256;
  reduce_global_pv_int8_segments_kernel<<<(output_elements + reduction_threads - 1) /
                                              reduction_threads,
                                          reduction_threads, 0, stream>>>(
      static_cast<const std::int32_t *>(partial_outputs), probability_scales,
      static_cast<Element *>(output), head_batches, segments);
  check(cudaPeekAtLastError(), "launch segmented INT8 Hopper RS-PV reduction");
}

void grouped_global_pv_wgmma_fp8_rs(const void *value_cache, const void *probabilities,
                                    void *partial_outputs, void *output, int batch, int kv_heads,
                                    int query_group, int context_length, int head_dimension,
                                    int cache_capacity, int segments, cudaStream_t stream) {
  if (value_cache == nullptr || probabilities == nullptr || partial_outputs == nullptr ||
      output == nullptr || batch <= 0 || kv_heads <= 0 || query_group != 8 || context_length <= 0 ||
      context_length % 64 != 0 || head_dimension != 512 || segments <= 0 || segments > 32 ||
      (context_length / 64) % segments != 0 || cache_capacity < context_length ||
      cache_capacity % 64 != 0) {
    throw std::runtime_error("invalid FP8 Hopper RS-PV shape");
  }
  constexpr int threads = 128;
  constexpr int dimension_tiles = 8;
  constexpr std::size_t shared_bytes = sizeof(Fp8PvSharedStorage);
  global_pv_wgmma_fp8_rs_segment_kernel<<<batch * kv_heads * dimension_tiles * segments, threads,
                                          shared_bytes, stream>>>(
      static_cast<const Fp8Element *>(value_cache), static_cast<const Fp8Element *>(probabilities),
      static_cast<float *>(partial_outputs), static_cast<Element *>(output), context_length,
      cache_capacity, segments);
  check(cudaPeekAtLastError(), "launch segmented FP8 Hopper RS-PV");
  if (segments == 1)
    return;
  const std::uint64_t output_elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * head_dimension;
  constexpr int reduction_threads = 256;
  reduce_global_pv_wgmma_fp8_rs_segments_kernel<<<(output_elements + reduction_threads - 1) /
                                                      reduction_threads,
                                                  reduction_threads, 0, stream>>>(
      static_cast<const float *>(partial_outputs), static_cast<Element *>(output), output_elements,
      segments);
  check(cudaPeekAtLastError(), "launch segmented FP8 Hopper RS-PV reduction");
}

void quantize_grouped_global_values_wgmma_fp8(const void *values, void *packed_values, int batch,
                                              int kv_heads, int context_length, int head_dimension,
                                              cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || batch <= 0 || kv_heads <= 0 ||
      context_length <= 0 || context_length % 64 != 0 || head_dimension != 512) {
    throw std::runtime_error("invalid FP8 Hopper value-pack shape");
  }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context_length * head_dimension;
  constexpr int threads = 256;
  quantize_global_values_wgmma_fp8_kernel<<<(elements + threads - 1) / threads, threads, 0,
                                            stream>>>(static_cast<const Element *>(values),
                                                      static_cast<Fp8Element *>(packed_values),
                                                      elements, context_length);
  check(cudaPeekAtLastError(), "launch FP8 Hopper value pack");
}

void pack_grouped_global_values_wgmma_int8(const void *values, void *packed_values, int batch,
                                           int kv_heads, int context_length, int head_dimension,
                                           cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || batch <= 0 || kv_heads <= 0 ||
      context_length <= 0 || context_length % 64 != 0 || head_dimension != 512) {
    throw std::runtime_error("invalid INT8 global V pack shape");
  }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(batch) * kv_heads * context_length * head_dimension;
  constexpr int threads = 256;
  pack_global_values_wgmma_int8_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const Int8Element *>(values), static_cast<Int8Element *>(packed_values), elements,
      context_length);
  check(cudaPeekAtLastError(), "launch INT8 global V pack");
}

void quantize_grouped_global_values_wgmma_int8(const void *values, void *packed_values,
                                               float *scales, int batch, int kv_heads,
                                               int context_length, int head_dimension,
                                               cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || scales == nullptr || batch <= 0 ||
      kv_heads <= 0 || context_length <= 0 || context_length % 64 != 0 || head_dimension != 512) {
    throw std::runtime_error("invalid INT8 global V quantization shape");
  }
  constexpr int threads = 256;
  const int rows = batch * kv_heads * context_length;
  quantize_global_values_wgmma_int8_kernel<<<rows, threads, 0, stream>>>(
      static_cast<const Element *>(values), static_cast<Int8Element *>(packed_values), scales, rows,
      context_length);
  check(cudaPeekAtLastError(), "launch INT8 global V quantization");
}

void quantize_grouped_global_value_span_wgmma_int8(const void *values, void *packed_values,
                                                   float *scales, int kv_heads, int tokens,
                                                   int context_length, int head_dimension,
                                                   int position_start, cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || scales == nullptr || kv_heads <= 0 ||
      tokens <= 0 || context_length <= 0 || context_length % 64 != 0 || head_dimension != 512 ||
      position_start < 0 || position_start + tokens > context_length) {
    throw std::runtime_error("invalid INT8 global V span quantization shape");
  }
  constexpr int threads = 256;
  const int rows = kv_heads * tokens;
  quantize_global_value_span_wgmma_int8_kernel<<<rows, threads, 0, stream>>>(
      static_cast<const Element *>(values), static_cast<Int8Element *>(packed_values), scales,
      kv_heads, tokens, context_length, position_start);
  check(cudaPeekAtLastError(), "launch INT8 global V span quantization");
}

void quantize_grouped_global_decode_values_wgmma_int8(const void *values, void *packed_values,
                                                      float *scales, const int *positions,
                                                      const int *slots, int batch,
                                                      int maximum_slots, int kv_heads,
                                                      int context_length, int head_dimension,
                                                      cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || scales == nullptr || positions == nullptr ||
      slots == nullptr || batch <= 0 || maximum_slots < batch || kv_heads <= 0 ||
      context_length <= 0 || context_length % 64 != 0 || head_dimension != 512) {
    throw std::runtime_error("invalid INT8 global ragged V quantization shape");
  }
  constexpr int threads = 256;
  quantize_global_decode_values_wgmma_int8_kernel<<<batch * kv_heads, threads, 0, stream>>>(
      static_cast<const Element *>(values), static_cast<Int8Element *>(packed_values), scales,
      positions, slots, batch, maximum_slots, kv_heads, context_length);
  check(cudaPeekAtLastError(), "launch INT8 global ragged V quantization");
}

void quantize_grouped_global_value_span_wgmma_fp8(const void *values, void *packed_values,
                                                  int kv_heads, int tokens, int context_length,
                                                  int head_dimension, int position_start,
                                                  cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || kv_heads <= 0 || tokens <= 0 ||
      context_length <= 0 || context_length % 64 != 0 || head_dimension != 512 ||
      position_start < 0 || position_start + tokens > context_length) {
    throw std::runtime_error("invalid FP8 Hopper value-span shape");
  }
  const int elements = kv_heads * tokens * head_dimension;
  constexpr int threads = 256;
  quantize_global_value_span_wgmma_fp8_kernel<<<(elements + threads - 1) / threads, threads, 0,
                                                stream>>>(
      static_cast<const Element *>(values), static_cast<Fp8Element *>(packed_values), kv_heads,
      tokens, context_length, position_start);
  check(cudaPeekAtLastError(), "launch FP8 Hopper value-span pack");
}

void quantize_grouped_global_decode_values_wgmma_fp8(const void *values, void *packed_values,
                                                     const int *positions, const int *slots,
                                                     int batch, int maximum_slots, int kv_heads,
                                                     int context_length, int head_dimension,
                                                     cudaStream_t stream) {
  if (values == nullptr || packed_values == nullptr || positions == nullptr || slots == nullptr ||
      batch <= 0 || maximum_slots < batch || kv_heads <= 0 || context_length <= 0 ||
      context_length % 64 != 0 || head_dimension != 512) {
    throw std::runtime_error("invalid FP8 Hopper ragged value-pack shape");
  }
  const int elements = batch * kv_heads * head_dimension;
  constexpr int threads = 256;
  quantize_global_decode_values_wgmma_fp8_kernel<<<(elements + threads - 1) / threads, threads, 0,
                                                   stream>>>(
      static_cast<const Element *>(values), static_cast<Fp8Element *>(packed_values), positions,
      slots, batch, maximum_slots, kv_heads, context_length);
  check(cudaPeekAtLastError(), "launch FP8 Hopper ragged value pack");
}

void quantize_grouped_global_probabilities_wgmma_fp8(const void *probabilities,
                                                     void *packed_probabilities, int batch,
                                                     int kv_heads, int query_group,
                                                     int context_length, cudaStream_t stream) {
  if (probabilities == nullptr || packed_probabilities == nullptr || batch <= 0 || kv_heads <= 0 ||
      query_group != 8 || context_length <= 0 || context_length % 64 != 0) {
    throw std::runtime_error("invalid FP8 Hopper probability-pack shape");
  }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(batch) * kv_heads * query_group * context_length;
  constexpr int threads = 256;
  quantize_global_probabilities_wgmma_fp8_kernel<<<(elements + threads - 1) / threads, threads, 0,
                                                   stream>>>(
      static_cast<const Element *>(probabilities), static_cast<Fp8Element *>(packed_probabilities),
      elements, query_group, context_length);
  check(cudaPeekAtLastError(), "launch FP8 Hopper probability pack");
}

void grouped_global_attention_wgmma_raw_rs_tile_bf16(
    const void *query, const void *raw_cache, const float *inverse_rms_cache,
    const void *key_norm_weight, const void *cosine, const void *sine, void *output, int batch,
    int kv_heads, int query_group, int context_length, int head_dimension, cudaStream_t stream) {
  if (query == nullptr || raw_cache == nullptr || inverse_rms_cache == nullptr ||
      key_norm_weight == nullptr || cosine == nullptr || sine == nullptr || output == nullptr ||
      batch <= 0 || kv_heads <= 0 || query_group != 8 || context_length != 64 ||
      head_dimension != 512) {
    throw std::runtime_error("invalid RS-raw Hopper global-attention tile shape");
  }
  constexpr int threads = 128;
  constexpr std::size_t shared_bytes = sizeof(RsRawGlobalAttentionTileSharedStorage);
  check(cudaFuncSetAttribute(global_attention_wgmma_raw_rs_tile_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set RS-raw Hopper global-attention tile shared memory");
  global_attention_wgmma_raw_rs_tile_kernel<<<batch * kv_heads, threads, shared_bytes, stream>>>(
      static_cast<const Element *>(query), static_cast<const Element *>(raw_cache),
      inverse_rms_cache, static_cast<const Element *>(key_norm_weight),
      static_cast<const Element *>(cosine), static_cast<const Element *>(sine),
      static_cast<Element *>(output), context_length);
  check(cudaPeekAtLastError(), "launch RS-raw Hopper global-attention tile");
}

void grouped_global_attention_wgmma_raw_rs_split_bf16(
    const void *query, const void *raw_cache, const float *inverse_rms_cache,
    const void *key_norm_weight, const void *cosine, const void *sine, void *output,
    float *partial_outputs, float *partial_maxima, float *partial_sums, int batch, int kv_heads,
    int query_group, int context_length, int head_dimension, int tiles_per_segment,
    cudaStream_t stream) {
  if (query == nullptr || raw_cache == nullptr || inverse_rms_cache == nullptr ||
      key_norm_weight == nullptr || cosine == nullptr || sine == nullptr || output == nullptr ||
      partial_outputs == nullptr || partial_maxima == nullptr || partial_sums == nullptr ||
      batch <= 0 || kv_heads <= 0 || query_group != 8 || context_length <= 0 ||
      context_length % 64 != 0 || head_dimension != 512 || tiles_per_segment <= 0 ||
      (context_length / 64) % tiles_per_segment != 0) {
    throw std::runtime_error("invalid split RS-raw Hopper global-attention shape");
  }
  constexpr int threads = 128;
  constexpr std::size_t shared_bytes = sizeof(RsRawGlobalAttentionSegmentSharedStorage);
  check(cudaFuncSetAttribute(global_attention_wgmma_raw_rs_segment_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set split RS-raw Hopper global-attention shared memory");
  const int segments_per_head = context_length / 64 / tiles_per_segment;
  const int heads = batch * kv_heads;
  global_attention_wgmma_raw_rs_segment_kernel<<<heads * segments_per_head, threads, shared_bytes,
                                                 stream>>>(
      static_cast<const Element *>(query), static_cast<const Element *>(raw_cache),
      inverse_rms_cache, static_cast<const Element *>(key_norm_weight),
      static_cast<const Element *>(cosine), static_cast<const Element *>(sine), partial_outputs,
      partial_maxima, partial_sums, context_length, tiles_per_segment);
  check(cudaPeekAtLastError(), "launch split RS-raw Hopper global-attention segments");
  reduce_global_attention_segments_kernel<<<heads * query_group, 256, 0, stream>>>(
      partial_outputs, partial_maxima, partial_sums, static_cast<Element *>(output),
      segments_per_head);
  check(cudaPeekAtLastError(), "launch split RS-raw Hopper global-attention reduction");
}

void grouped_global_attention_wgmma_split_bf16(const void *query, const void *key_cache,
                                               const void *value_cache, void *output,
                                               float *partial_outputs, float *partial_maxima,
                                               float *partial_sums, int batch, int kv_heads,
                                               int query_group, int context_length,
                                               int head_dimension, int tiles_per_segment,
                                               cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr ||
      partial_outputs == nullptr || partial_maxima == nullptr || partial_sums == nullptr ||
      batch <= 0 || kv_heads <= 0 || query_group != 8 || context_length <= 0 ||
      context_length % 64 != 0 || head_dimension != 512 || tiles_per_segment <= 0 ||
      (context_length / 64) % tiles_per_segment != 0) {
    throw std::runtime_error("invalid split Hopper global-attention shape");
  }
  constexpr int threads = 256;
  constexpr std::size_t shared_bytes = sizeof(GlobalAttentionTileSharedStorage);
  check(cudaFuncSetAttribute(global_attention_wgmma_segment_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "set split Hopper global-attention shared memory");
  const int segments_per_head = context_length / 64 / tiles_per_segment;
  const int heads = batch * kv_heads;
  global_attention_wgmma_segment_kernel<<<heads * segments_per_head, threads, shared_bytes,
                                          stream>>>(
      static_cast<const Element *>(query), static_cast<const Element *>(key_cache),
      static_cast<const Element *>(value_cache), partial_outputs, partial_maxima, partial_sums,
      context_length, tiles_per_segment);
  check(cudaPeekAtLastError(), "launch split Hopper global-attention segments");
  reduce_global_attention_segments_kernel<<<heads * query_group, 256, 0, stream>>>(
      partial_outputs, partial_maxima, partial_sums, static_cast<Element *>(output),
      segments_per_head);
  check(cudaPeekAtLastError(), "launch split Hopper global-attention reduction");
}

} // namespace carat
