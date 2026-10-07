#include "carat/prefill_gemm_attention.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <map>
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

__global__ void causal_softmax_kernel(__nv_bfloat16 *scores, int query_tokens, int kv_tokens,
                                      int query_position_start) {
  __shared__ float warp_values[32];
  const int query_position = static_cast<int>(blockIdx.x) % query_tokens;
  const int maximum_key_position = min(kv_tokens - 1, query_position_start + query_position);
  __nv_bfloat16 *row = scores + static_cast<std::uint64_t>(blockIdx.x) * kv_tokens;
  float maximum = -CUDART_INF_F;
  for (int key_position = threadIdx.x; key_position <= maximum_key_position;
       key_position += blockDim.x) {
    maximum = max(maximum, __bfloat162float(row[key_position]));
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
  for (int key_position = threadIdx.x; key_position <= maximum_key_position;
       key_position += blockDim.x) {
    sum += expf(__bfloat162float(row[key_position]) - maximum);
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
  for (int key_position = threadIdx.x; key_position < kv_tokens; key_position += blockDim.x) {
    const float probability =
        key_position <= maximum_key_position
            ? expf(__bfloat162float(row[key_position]) - maximum) * inverse_sum
            : 0.0F;
    row[key_position] = __float2bfloat16_rn(probability);
  }
}

__global__ void uniform_batch_pointer_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *const *key_caches,
    const __nv_bfloat16 *const *value_caches, __nv_bfloat16 *scores, __nv_bfloat16 *output,
    int requests, int query_heads, int kv_heads, int query_tokens, int maximum_kv_tokens,
    int kv_capacity, int head_dimension, const void **key_pointers, const void **query_pointers,
    const void **value_pointers, void **score_pointers, void **output_pointers) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int batches = requests * kv_heads;
  if (index >= batches)
    return;
  const int request = index / kv_heads;
  const int kv_head = index % kv_heads;
  const int query_group = query_heads / kv_heads;
  const std::uint64_t cache_head_elements =
      static_cast<std::uint64_t>(kv_capacity) * head_dimension;
  const std::uint64_t query_head_elements =
      static_cast<std::uint64_t>(query_tokens) * head_dimension;
  const std::uint64_t score_head_elements =
      static_cast<std::uint64_t>(query_tokens) * maximum_kv_tokens;
  key_pointers[index] = key_caches[request] + kv_head * cache_head_elements;
  value_pointers[index] = value_caches[request] + kv_head * cache_head_elements;
  query_pointers[index] =
      query + (static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group) *
                  query_head_elements;
  score_pointers[index] =
      scores + (static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group) *
                   score_head_elements;
  output_pointers[index] =
      output + (static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group) *
                   query_head_elements;
}

__global__ void uniform_ring_pointer_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *const *key_caches,
    const __nv_bfloat16 *const *value_caches, __nv_bfloat16 *scores, float *fp32_output,
    int requests, int query_heads, int kv_heads, int query_tokens, int ring_capacity,
    int head_dimension, const void **key_pointers, const void **query_pointers,
    const void **value_pointers, void **score_pointers, void **output_pointers) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int batches = requests * kv_heads;
  if (index >= batches)
    return;
  const int request = index / kv_heads;
  const int kv_head = index % kv_heads;
  const int query_group = query_heads / kv_heads;
  const std::uint64_t cache_head_elements =
      static_cast<std::uint64_t>(ring_capacity) * head_dimension;
  const std::uint64_t query_head_elements =
      static_cast<std::uint64_t>(query_tokens) * head_dimension;
  const std::uint64_t score_head_elements =
      static_cast<std::uint64_t>(query_tokens) * ring_capacity;
  key_pointers[index] = key_caches[request] + kv_head * cache_head_elements;
  value_pointers[index] = value_caches[request] + kv_head * cache_head_elements;
  query_pointers[index] =
      query + (static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group) *
                  query_head_elements;
  score_pointers[index] =
      scores + (static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group) *
                   score_head_elements;
  output_pointers[index] =
      fp32_output + (static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group) *
                        query_head_elements;
}

__device__ float ring_warp_sum(float value) {
  for (unsigned offset = 16; offset > 0; offset >>= 1U) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

// A full ring containing q proposals differs from the causal virtual ring for query t only in
// proposal slots u>t. Replace those few QK scores with the pre-write K rows.
__global__ void repair_ring_scores_kernel(const __nv_bfloat16 *query,
                                          const __nv_bfloat16 *backup_keys, __nv_bfloat16 *scores,
                                          const int *position_starts, int requests, int query_heads,
                                          int kv_heads, int query_tokens, int ring_capacity,
                                          int head_dimension) {
  const int request_head = static_cast<int>(blockIdx.x);
  const int request = request_head / kv_heads;
  const int kv_head = request_head % kv_heads;
  if (request >= requests)
    return;
  const int query_group = query_heads / kv_heads;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int query_rows = query_group * query_tokens;
  if (warp >= query_rows)
    return;
  const int local_head = warp / query_tokens;
  const int query_token = warp % query_tokens;
  const std::uint64_t query_index =
      ((static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group + local_head) *
           query_tokens +
       query_token) *
      head_dimension;
  const std::uint64_t backup_head =
      (static_cast<std::uint64_t>(request) * kv_heads + kv_head) * query_tokens * head_dimension;
  const std::uint64_t score_row =
      ((static_cast<std::uint64_t>(request) * query_heads + kv_head * query_group + local_head) *
           query_tokens +
       query_token) *
      ring_capacity;
  for (int future = query_token + 1; future < query_tokens; ++future) {
    float dot = 0.0F;
    for (int component = lane; component < head_dimension; component += 32) {
      dot += __bfloat162float(query[query_index + component]) *
             __bfloat162float(
                 backup_keys[backup_head + static_cast<std::uint64_t>(future) * head_dimension +
                             component]);
    }
    dot = ring_warp_sum(dot);
    if (lane == 0) {
      const int physical = (position_starts[request] + future) % ring_capacity;
      scores[score_row + physical] = __float2bfloat16_rn(dot);
    }
  }
}

__global__ void uniform_batch_softmax_kernel(__nv_bfloat16 *scores, int rows, int width) {
  __shared__ float warp_values[32];
  const int row_index = static_cast<int>(blockIdx.x);
  if (row_index >= rows)
    return;
  __nv_bfloat16 *row = scores + static_cast<std::uint64_t>(row_index) * width;
  float maximum = -CUDART_INF_F;
  for (int key = threadIdx.x; key < width; key += blockDim.x) {
    maximum = max(maximum, __bfloat162float(row[key]));
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
  for (int key = threadIdx.x; key < width; key += blockDim.x) {
    sum += expf(__bfloat162float(row[key]) - maximum);
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
  for (int key = threadIdx.x; key < width; key += blockDim.x) {
    row[key] = __float2bfloat16_rn(expf(__bfloat162float(row[key]) - maximum) * inverse_sum);
  }
}

// PV used the proposal V in every physical slot. For query t, substitute the backup V in slots
// u>t and perform the final FP32-to-BF16 rounding only after the correction.
__global__ void repair_ring_output_kernel(const __nv_bfloat16 *const *value_caches,
                                          const __nv_bfloat16 *backup_values,
                                          const __nv_bfloat16 *scores, const float *fp32_output,
                                          __nv_bfloat16 *output, const int *position_starts,
                                          int requests, int query_heads, int kv_heads,
                                          int query_tokens, int ring_capacity, int head_dimension) {
  const int request_head = static_cast<int>(blockIdx.x);
  const int request = request_head / kv_heads;
  const int kv_head = request_head % kv_heads;
  const int component = static_cast<int>(threadIdx.x);
  if (request >= requests || component >= head_dimension)
    return;
  const int query_group = query_heads / kv_heads;
  const __nv_bfloat16 *value_cache = value_caches[request];
  const std::uint64_t cache_head =
      static_cast<std::uint64_t>(kv_head) * ring_capacity * head_dimension;
  const std::uint64_t backup_head =
      (static_cast<std::uint64_t>(request) * kv_heads + kv_head) * query_tokens * head_dimension;
  for (int local_head = 0; local_head < query_group; ++local_head) {
    for (int query_token = 0; query_token < query_tokens; ++query_token) {
      const std::uint64_t output_index = ((static_cast<std::uint64_t>(request) * query_heads +
                                           kv_head * query_group + local_head) *
                                              query_tokens +
                                          query_token) *
                                             head_dimension +
                                         component;
      const std::uint64_t score_row = output_index / head_dimension * ring_capacity;
      float value = fp32_output[output_index];
      for (int future = query_token + 1; future < query_tokens; ++future) {
        const int physical = (position_starts[request] + future) % ring_capacity;
        const float probability = __bfloat162float(scores[score_row + physical]);
        const float backup = __bfloat162float(
            backup_values[backup_head + static_cast<std::uint64_t>(future) * head_dimension +
                          component]);
        const float proposal = __bfloat162float(
            value_cache[cache_head + (static_cast<std::uint64_t>(physical) * head_dimension) +
                        component]);
        value += probability * (backup - proposal);
      }
      output[output_index] = __float2bfloat16_rn(value);
    }
  }
}

__global__ void uniform_batch_causal_softmax_kernel(__nv_bfloat16 *scores,
                                                    const int *position_starts, int requests,
                                                    int query_heads, int query_tokens,
                                                    int maximum_kv_tokens) {
  __shared__ float warp_values[32];
  const int row_index = static_cast<int>(blockIdx.x);
  const int request = row_index / (query_heads * query_tokens);
  if (request >= requests)
    return;
  const int query_position = row_index % query_tokens;
  const int maximum_key_position =
      min(maximum_kv_tokens - 1, position_starts[request] + query_position);
  __nv_bfloat16 *row = scores + static_cast<std::uint64_t>(row_index) * maximum_kv_tokens;
  float maximum = -CUDART_INF_F;
  for (int key = threadIdx.x; key <= maximum_key_position; key += blockDim.x) {
    maximum = max(maximum, __bfloat162float(row[key]));
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
  for (int key = threadIdx.x; key <= maximum_key_position; key += blockDim.x) {
    sum += expf(__bfloat162float(row[key]) - maximum);
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
  for (int key = threadIdx.x; key < maximum_kv_tokens; key += blockDim.x) {
    row[key] = key <= maximum_key_position
                   ? __float2bfloat16_rn(expf(__bfloat162float(row[key]) - maximum) * inverse_sum)
                   : __float2bfloat16_rn(0.0F);
  }
}

} // namespace

struct GemmCausalPrefillAttention::Implementation {
  struct ContiguousQkPlan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t key_layout{};
    cublasLtMatrixLayout_t query_layout{};
    cublasLtMatrixLayout_t score_layout{};
    cublasLtMatmulAlgo_t algorithm{};

    ~ContiguousQkPlan() {
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
  void *pointers{nullptr};
  void *ring_output{nullptr};
  void *lt_workspace{nullptr};
  std::uint64_t maximum_score_elements;
  int maximum_pointer_batches;
  static constexpr std::size_t lt_workspace_bytes = 32ULL * 1024 * 1024;
  std::map<std::tuple<int, int, int, int, int, int>, std::unique_ptr<ContiguousQkPlan>>
      contiguous_qk_plans;

  Implementation(int maximum_tokens, int maximum_query_heads, int maximum_requests)
      : maximum_score_elements(static_cast<std::uint64_t>(maximum_query_heads) * maximum_tokens *
                               maximum_tokens),
        maximum_pointer_batches(maximum_requests * maximum_query_heads) {
    if (maximum_tokens <= 0 || maximum_query_heads <= 0 || maximum_requests <= 0) {
      throw std::runtime_error("invalid global prefill workspace shape");
    }
    cublas_check(cublasCreate(&handle), "create global prefill cuBLAS handle");
    cublas_check(cublasLtCreate(&lt_handle), "create global prefill cuBLASLt handle");
    cuda_check(cudaMalloc(&scores, static_cast<std::size_t>(2ULL * maximum_score_elements)),
               "allocate global prefill scores");
    cuda_check(cudaMalloc(&pointers, static_cast<std::size_t>(5ULL * maximum_pointer_batches *
                                                              sizeof(void *))),
               "allocate global prefill pointer arrays");
    cuda_check(cudaMalloc(&ring_output,
                          static_cast<std::size_t>(4ULL * maximum_requests * maximum_query_heads *
                                                   512ULL * sizeof(float))),
               "allocate ring attention FP32 output");
    cuda_check(cudaMalloc(&lt_workspace, lt_workspace_bytes), "allocate contiguous QK workspace");
  }

  ~Implementation() {
    contiguous_qk_plans.clear();
    if (lt_workspace != nullptr)
      cudaFree(lt_workspace);
    if (ring_output != nullptr)
      cudaFree(ring_output);
    if (pointers != nullptr)
      cudaFree(pointers);
    if (scores != nullptr)
      cudaFree(scores);
    if (lt_handle != nullptr)
      cublasLtDestroy(lt_handle);
    if (handle != nullptr)
      cublasDestroy(handle);
  }

  ContiguousQkPlan &contiguous_qk_plan(int requests, int kv_heads, int grouped_queries,
                                       int maximum_kv_tokens, int kv_capacity, int head_dimension) {
    const auto key = std::make_tuple(requests, kv_heads, grouped_queries, maximum_kv_tokens,
                                     kv_capacity, head_dimension);
    const auto existing = contiguous_qk_plans.find(key);
    if (existing != contiguous_qk_plans.end())
      return *existing->second;
    auto plan = std::make_unique<ContiguousQkPlan>();
    cublas_check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                 "create contiguous QK descriptor");
    constexpr cublasOperation_t transpose = CUBLAS_OP_T;
    cublas_check(cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transpose, sizeof(transpose)),
                 "transpose contiguous QK keys");
    cublas_check(cublasLtMatrixLayoutCreate(&plan->key_layout, CUDA_R_16BF, head_dimension,
                                            maximum_kv_tokens, head_dimension),
                 "create contiguous QK key layout");
    cublas_check(cublasLtMatrixLayoutCreate(&plan->query_layout, CUDA_R_16BF, head_dimension,
                                            grouped_queries, head_dimension),
                 "create contiguous QK query layout");
    cublas_check(cublasLtMatrixLayoutCreate(&plan->score_layout, CUDA_R_16BF, maximum_kv_tokens,
                                            grouped_queries, maximum_kv_tokens),
                 "create contiguous QK score layout");
    const std::int32_t batches = requests * kv_heads;
    const std::int64_t key_stride = static_cast<std::int64_t>(kv_capacity) * head_dimension;
    const std::int64_t query_stride = static_cast<std::int64_t>(grouped_queries) * head_dimension;
    const std::int64_t score_stride =
        static_cast<std::int64_t>(grouped_queries) * maximum_kv_tokens;
    for (const auto layout : {plan->key_layout, plan->query_layout, plan->score_layout}) {
      cublas_check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                    &batches, sizeof(batches)),
                   "set contiguous QK batch count");
    }
    cublas_check(cublasLtMatrixLayoutSetAttribute(plan->key_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &key_stride, sizeof(key_stride)),
                 "set contiguous QK key stride");
    cublas_check(cublasLtMatrixLayoutSetAttribute(plan->query_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &query_stride, sizeof(query_stride)),
                 "set contiguous QK query stride");
    cublas_check(cublasLtMatrixLayoutSetAttribute(plan->score_layout,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &score_stride, sizeof(score_stride)),
                 "set contiguous QK score stride");
    cublasLtMatmulPreference_t preference{};
    cublas_check(cublasLtMatmulPreferenceCreate(&preference), "create contiguous QK preference");
    cublas_check(
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &lt_workspace_bytes, sizeof(lt_workspace_bytes)),
        "set contiguous QK workspace limit");
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, plan->operation, plan->key_layout, plan->query_layout, plan->score_layout,
        plan->score_layout, preference, 1, &heuristic, &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    cublas_check(status, "select contiguous QK algorithm");
    if (returned == 0)
      throw std::runtime_error("no contiguous QK algorithm");
    plan->algorithm = heuristic.algo;
    ContiguousQkPlan &result = *plan;
    contiguous_qk_plans.emplace(key, std::move(plan));
    return result;
  }
};

GemmCausalPrefillAttention::GemmCausalPrefillAttention(int maximum_tokens, int maximum_query_heads,
                                                       int maximum_requests)
    : implementation_(std::make_unique<Implementation>(maximum_tokens, maximum_query_heads,
                                                       maximum_requests)) {}
GemmCausalPrefillAttention::~GemmCausalPrefillAttention() = default;

void GemmCausalPrefillAttention::run(const void *query, const void *key, const void *value,
                                     void *output, int query_heads, int kv_heads, int tokens,
                                     int kv_tokens, int kv_capacity, int query_position_start,
                                     int head_dimension, cudaStream_t stream) {
  const int query_tokens = tokens;
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(query_heads) * query_tokens * kv_tokens;
  if (query == nullptr || key == nullptr || value == nullptr || output == nullptr ||
      query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 || query_tokens <= 0 ||
      kv_tokens < query_tokens || kv_capacity < kv_tokens || query_position_start < 0 ||
      query_position_start + query_tokens > kv_tokens || head_dimension <= 0 ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("invalid global prefill attention shape");
  }
  auto &state = *implementation_;
  cublas_check(cublasSetStream(state.handle, stream), "set global prefill stream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int query_group = query_heads / kv_heads;
  const std::int64_t query_head_elements = static_cast<std::int64_t>(query_tokens) * head_dimension;
  const std::int64_t cache_head_elements = static_cast<std::int64_t>(kv_capacity) * head_dimension;
  const std::int64_t score_head_elements = static_cast<std::int64_t>(query_tokens) * kv_tokens;
  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const auto *key_head =
        static_cast<const unsigned char *>(key) + 2ULL * kv_head * cache_head_elements;
    const auto *query_group_begin = static_cast<const unsigned char *>(query) +
                                    2ULL * kv_head * query_group * query_head_elements;
    auto *score_group = static_cast<unsigned char *>(state.scores) +
                        2ULL * kv_head * query_group * score_head_elements;
    cublas_check(cublasGemmStridedBatchedEx(
                     state.handle, CUBLAS_OP_T, CUBLAS_OP_N, kv_tokens, query_tokens,
                     head_dimension, &alpha, key_head, CUDA_R_16BF, head_dimension, 0,
                     query_group_begin, CUDA_R_16BF, head_dimension, query_head_elements, &beta,
                     score_group, CUDA_R_16BF, kv_tokens, score_head_elements, query_group,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "global prefill QK GEMM");
  }
  causal_softmax_kernel<<<query_heads * query_tokens, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), query_tokens, kv_tokens, query_position_start);
  cuda_check(cudaPeekAtLastError(), "launch global prefill causal softmax");
  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const auto *value_head =
        static_cast<const unsigned char *>(value) + 2ULL * kv_head * cache_head_elements;
    const auto *score_group = static_cast<const unsigned char *>(state.scores) +
                              2ULL * kv_head * query_group * score_head_elements;
    auto *output_group =
        static_cast<unsigned char *>(output) + 2ULL * kv_head * query_group * query_head_elements;
    cublas_check(cublasGemmStridedBatchedEx(
                     state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, query_tokens,
                     kv_tokens, &alpha, value_head, CUDA_R_16BF, head_dimension, 0, score_group,
                     CUDA_R_16BF, kv_tokens, score_head_elements, &beta, output_group, CUDA_R_16BF,
                     head_dimension, query_head_elements, query_group, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "global prefill PV GEMM");
  }
}

void GemmCausalPrefillAttention::run_uniform_batch(
    const void *query, const void *const *device_key_caches, const void *const *device_value_caches,
    void *output, const int *device_position_starts, int requests, int query_heads, int kv_heads,
    int query_tokens, int maximum_kv_tokens, int kv_capacity, int head_dimension,
    cudaStream_t stream) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(requests) * query_heads * query_tokens * maximum_kv_tokens;
  if (query == nullptr || device_key_caches == nullptr || device_value_caches == nullptr ||
      output == nullptr || device_position_starts == nullptr || requests <= 0 || query_heads <= 0 ||
      kv_heads <= 0 || query_heads % kv_heads != 0 || query_tokens <= 0 ||
      maximum_kv_tokens < query_tokens || maximum_kv_tokens > kv_capacity || head_dimension <= 0 ||
      requests * kv_heads > implementation_->maximum_pointer_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("invalid uniform global prefill attention shape");
  }
  auto &state = *implementation_;
  cublas_check(cublasSetStream(state.handle, stream), "set uniform global prefill stream");
  const int batches = requests * kv_heads;
  auto **pointer_base = static_cast<void **>(state.pointers);
  auto **key_pointers = pointer_base;
  auto **query_pointers = pointer_base + state.maximum_pointer_batches;
  auto **value_pointers = pointer_base + 2 * state.maximum_pointer_batches;
  auto **score_pointers = pointer_base + 3 * state.maximum_pointer_batches;
  auto **output_pointers = pointer_base + 4 * state.maximum_pointer_batches;
  uniform_batch_pointer_kernel<<<(batches + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(query),
      reinterpret_cast<const __nv_bfloat16 *const *>(device_key_caches),
      reinterpret_cast<const __nv_bfloat16 *const *>(device_value_caches),
      static_cast<__nv_bfloat16 *>(state.scores), static_cast<__nv_bfloat16 *>(output), requests,
      query_heads, kv_heads, query_tokens, maximum_kv_tokens, kv_capacity, head_dimension,
      const_cast<const void **>(key_pointers), const_cast<const void **>(query_pointers),
      const_cast<const void **>(value_pointers), score_pointers, output_pointers);
  cuda_check(cudaPeekAtLastError(), "build uniform global prefill pointers");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int query_group = query_heads / kv_heads;
  const int grouped_queries = query_tokens * query_group;
  cublas_check(cublasGemmBatchedEx(
                   state.handle, CUBLAS_OP_T, CUBLAS_OP_N, maximum_kv_tokens, grouped_queries,
                   head_dimension, &alpha, reinterpret_cast<const void *const *>(key_pointers),
                   CUDA_R_16BF, head_dimension,
                   reinterpret_cast<const void *const *>(query_pointers), CUDA_R_16BF,
                   head_dimension, &beta, score_pointers, CUDA_R_16BF, maximum_kv_tokens, batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "uniform global prefill QK GEMM");
  uniform_batch_causal_softmax_kernel<<<requests * query_heads * query_tokens, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), device_position_starts, requests, query_heads,
      query_tokens, maximum_kv_tokens);
  cuda_check(cudaPeekAtLastError(), "launch uniform global prefill softmax");
  cublas_check(cublasGemmBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, grouped_queries,
                   maximum_kv_tokens, &alpha, reinterpret_cast<const void *const *>(value_pointers),
                   CUDA_R_16BF, head_dimension,
                   reinterpret_cast<const void *const *>(score_pointers), CUDA_R_16BF,
                   maximum_kv_tokens, &beta, output_pointers, CUDA_R_16BF, head_dimension, batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "uniform global prefill PV GEMM");
}

void GemmCausalPrefillAttention::run_uniform_contiguous_batch(
    const void *query, const void *key_caches, const void *value_caches, void *output,
    const int *device_position_starts, int requests, int query_heads, int kv_heads,
    int query_tokens, int maximum_kv_tokens, int kv_capacity, int head_dimension,
    cudaStream_t stream) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(requests) * query_heads * query_tokens * maximum_kv_tokens;
  if (query == nullptr || key_caches == nullptr || value_caches == nullptr || output == nullptr ||
      device_position_starts == nullptr || requests <= 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 || query_tokens <= 0 || maximum_kv_tokens < query_tokens ||
      maximum_kv_tokens > kv_capacity || head_dimension <= 0 ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("invalid contiguous uniform prefill attention shape");
  }
  auto &state = *implementation_;
  cublas_check(cublasSetStream(state.handle, stream), "set contiguous uniform prefill stream");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int query_group = query_heads / kv_heads;
  const int grouped_queries = query_tokens * query_group;
  const int batches = requests * kv_heads;
  const long long cache_stride = static_cast<long long>(kv_capacity) * head_dimension;
  const long long query_stride = static_cast<long long>(grouped_queries) * head_dimension;
  const long long score_stride = static_cast<long long>(grouped_queries) * maximum_kv_tokens;
  auto &qk_plan = state.contiguous_qk_plan(requests, kv_heads, grouped_queries, maximum_kv_tokens,
                                           kv_capacity, head_dimension);
  cublas_check(cublasLtMatmul(state.lt_handle, qk_plan.operation, &alpha, key_caches,
                              qk_plan.key_layout, query, qk_plan.query_layout, &beta, state.scores,
                              qk_plan.score_layout, state.scores, qk_plan.score_layout,
                              &qk_plan.algorithm, state.lt_workspace, state.lt_workspace_bytes,
                              stream),
               "contiguous uniform prefill QK matmul");
  uniform_batch_causal_softmax_kernel<<<requests * query_heads * query_tokens, 256, 0, stream>>>(
      static_cast<__nv_bfloat16 *>(state.scores), device_position_starts, requests, query_heads,
      query_tokens, maximum_kv_tokens);
  cuda_check(cudaPeekAtLastError(), "launch contiguous uniform prefill softmax");
  cublas_check(cublasGemmStridedBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, grouped_queries,
                   maximum_kv_tokens, &alpha, value_caches, CUDA_R_16BF, head_dimension,
                   cache_stride, state.scores, CUDA_R_16BF, maximum_kv_tokens, score_stride, &beta,
                   output, CUDA_R_16BF, head_dimension, query_stride, batches, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "contiguous uniform prefill PV GEMM");
}

void GemmCausalPrefillAttention::run_uniform_ring_batch(
    const void *query, const void *const *device_key_caches, const void *const *device_value_caches,
    const void *backup_keys, const void *backup_values, void *output,
    const int *device_position_starts, int requests, int query_heads, int kv_heads,
    int query_tokens, int ring_capacity, int head_dimension, cudaStream_t stream) {
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(requests) * query_heads * query_tokens * ring_capacity;
  if (query == nullptr || device_key_caches == nullptr || device_value_caches == nullptr ||
      backup_keys == nullptr || backup_values == nullptr || output == nullptr ||
      device_position_starts == nullptr || requests <= 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 || query_tokens <= 0 || query_tokens > 8 ||
      ring_capacity <= query_tokens || head_dimension <= 0 ||
      requests * kv_heads > implementation_->maximum_pointer_batches ||
      score_elements > implementation_->maximum_score_elements) {
    throw std::runtime_error("invalid uniform ring attention shape");
  }
  auto &state = *implementation_;
  const int query_group = query_heads / kv_heads;
  cublas_check(cublasSetStream(state.handle, stream), "set uniform ring attention stream");
  const int batches = requests * kv_heads;
  auto **pointer_base = static_cast<void **>(state.pointers);
  auto **key_pointers = pointer_base;
  auto **query_pointers = pointer_base + state.maximum_pointer_batches;
  auto **value_pointers = pointer_base + 2 * state.maximum_pointer_batches;
  auto **score_pointers = pointer_base + 3 * state.maximum_pointer_batches;
  auto **output_pointers = pointer_base + 4 * state.maximum_pointer_batches;
  uniform_ring_pointer_kernel<<<(batches + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(query),
      reinterpret_cast<const __nv_bfloat16 *const *>(device_key_caches),
      reinterpret_cast<const __nv_bfloat16 *const *>(device_value_caches),
      static_cast<__nv_bfloat16 *>(state.scores), static_cast<float *>(state.ring_output), requests,
      query_heads, kv_heads, query_tokens, ring_capacity, head_dimension,
      const_cast<const void **>(key_pointers), const_cast<const void **>(query_pointers),
      const_cast<const void **>(value_pointers), score_pointers, output_pointers);
  cuda_check(cudaPeekAtLastError(), "build uniform ring attention pointers");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const int grouped_queries = query_tokens * query_group;
  cublas_check(cublasGemmBatchedEx(
                   state.handle, CUBLAS_OP_T, CUBLAS_OP_N, ring_capacity, grouped_queries,
                   head_dimension, &alpha, reinterpret_cast<const void *const *>(key_pointers),
                   CUDA_R_16BF, head_dimension,
                   reinterpret_cast<const void *const *>(query_pointers), CUDA_R_16BF,
                   head_dimension, &beta, score_pointers, CUDA_R_16BF, ring_capacity, batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "uniform ring QK GEMM");
  const int repair_score_threads = std::min(1024, query_group * query_tokens * 32);
  repair_ring_scores_kernel<<<batches, repair_score_threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(query), static_cast<const __nv_bfloat16 *>(backup_keys),
      static_cast<__nv_bfloat16 *>(state.scores), device_position_starts, requests, query_heads,
      kv_heads, query_tokens, ring_capacity, head_dimension);
  cuda_check(cudaPeekAtLastError(), "repair uniform ring scores");
  const int rows = requests * query_heads * query_tokens;
  uniform_batch_softmax_kernel<<<rows, 256, 0, stream>>>(static_cast<__nv_bfloat16 *>(state.scores),
                                                         rows, ring_capacity);
  cuda_check(cudaPeekAtLastError(), "launch uniform ring softmax");
  cublas_check(cublasGemmBatchedEx(
                   state.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dimension, grouped_queries,
                   ring_capacity, &alpha, reinterpret_cast<const void *const *>(value_pointers),
                   CUDA_R_16BF, head_dimension,
                   reinterpret_cast<const void *const *>(score_pointers), CUDA_R_16BF,
                   ring_capacity, &beta, output_pointers, CUDA_R_32F, head_dimension, batches,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "uniform ring PV GEMM");
  repair_ring_output_kernel<<<batches, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *const *>(device_value_caches),
      static_cast<const __nv_bfloat16 *>(backup_values),
      static_cast<const __nv_bfloat16 *>(state.scores),
      static_cast<const float *>(state.ring_output), static_cast<__nv_bfloat16 *>(output),
      device_position_starts, requests, query_heads, kv_heads, query_tokens, ring_capacity,
      head_dimension);
  cuda_check(cudaPeekAtLastError(), "repair uniform ring output");
}

} // namespace carat
