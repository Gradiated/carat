#pragma once

#include <cstddef>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace carat {

class GemmGroupedDecodeAttention {
public:
  GemmGroupedDecodeAttention(int maximum_batch, int maximum_kv_heads, int maximum_query_group,
                             int maximum_context);
  ~GemmGroupedDecodeAttention();
  GemmGroupedDecodeAttention(const GemmGroupedDecodeAttention &) = delete;
  GemmGroupedDecodeAttention &operator=(const GemmGroupedDecodeAttention &) = delete;

  void run(const void *query, const void *key_cache, const void *value_cache, void *output,
           int batch, int kv_heads, int query_group, int context_length, int head_dimension,
           cudaStream_t stream, int cache_capacity = 0);
  // Strided QK benchmark. Scores are BF16 in
  // [batch, kv_head, query, context] order and are not normalized.
  void run_qk(const void *query, const void *key_cache, void *scores, int batch, int kv_heads,
              int query_group, int context_length, int head_dimension, cudaStream_t stream,
              int cache_capacity = 0);
  // Compact query/output rows attend to stable cache slots with independent absolute positions.
  // positions and slots are device arrays. Invalid tail scores are zeroed before PV.
  void run_ragged(const void *query, const void *key_cache, const void *value_cache, void *output,
                  const int *positions, const int *slots, int batch, int maximum_slots,
                  int kv_heads, int query_group, int maximum_context_length, int head_dimension,
                  cudaStream_t stream, int cache_capacity);
  // Same independent-position arithmetic as run_ragged, specialized for one consecutive cache
  // interval [first_slot, first_slot + batch). This retains strided tensor-core GEMMs instead of
  // paying the pointer-array GEMM penalty.
  void run_ragged_contiguous(const void *query, const void *key_cache, const void *value_cache,
                             void *output, const int *positions, int batch, int kv_heads,
                             int query_group, int maximum_context_length, int head_dimension,
                             int first_slot, cudaStream_t stream, int cache_capacity);
  // Global-attention Q and K use Gemma-specific fixed-scale E4M3 (Q / 64, K / 1024)
  // while softmax, V and output remain BF16.
  // This path is valid only for a consecutive cache interval so cuBLASLt can retain one
  // strided-batch tensor-core launch.
  void run_ragged_contiguous_fp8_keys(const void *fp8_query, const void *fp8_key_cache,
                                      const void *value_cache, void *output, const int *positions,
                                      int batch, int kv_heads, int query_group,
                                      int maximum_context_length, int head_dimension,
                                      int first_slot, cudaStream_t stream, int cache_capacity);
  // Fixed-scale E4M3 cache. Values are stored as
  // V * 64; normalized probabilities are quantized as P * 448 before FP8 tensor-core PV.
  void run_ragged_contiguous_fp8_kv(const void *fp8_query, const void *fp8_key_cache,
                                    const void *fp8_value_cache, void *output, const int *positions,
                                    int batch, int kv_heads, int query_group,
                                    int maximum_context_length, int head_dimension, int first_slot,
                                    cudaStream_t stream, int cache_capacity);
  // Materialized signed-INT8 K baseline with one dynamic scale per token/head. Queries are
  // quantized per query/head; S8 tensor-core QK accumulates in S32, then scales before softmax.
  // V remains BF16 so this isolates the attainable K-cache byte reduction.
  void run_ragged_contiguous_int8_keys(const void *query, const void *int8_key_cache,
                                       const float *key_scales, const void *value_cache,
                                       void *output, const int *positions, int batch, int kv_heads,
                                       int query_group, int maximum_context_length,
                                       int head_dimension, int first_slot, cudaStream_t stream,
                                       int cache_capacity);
  void run_ragged_contiguous_int8_block128_keys(const void *query, const void *int8_key_cache,
                                                const float *key_scales, const void *value_cache,
                                                void *output, const int *positions, int batch,
                                                int kv_heads, int query_group,
                                                int maximum_context_length, int head_dimension,
                                                int first_slot, cudaStream_t stream,
                                                int cache_capacity, int tiles_per_block = 1);
  void run_ragged_contiguous_int8_block128_kv(
      const void *query, const void *int8_key_cache, const float *key_scales,
      const void *int8_value_cache, const float *value_scales, void *output, const int *positions,
      int batch, int kv_heads, int query_group, int maximum_context_length, int head_dimension,
      int first_slot, cudaStream_t stream, int cache_capacity, int tiles_per_block = 0);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
