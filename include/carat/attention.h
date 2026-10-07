#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace carat {

// Q/output: [batch, kv_heads, query_group, head_dimension].
// K/V: [batch, kv_heads, context_length, head_dimension].
// The kernel loads each physical K/V element once per KV-head block and applies it to every
// grouped query, avoiding the 2x/8x HBM amplification of per-query-head implementations.
void grouped_decode_attention_bf16(const void *query, const void *key_cache,
                                   const void *value_cache, void *output, int batch, int kv_heads,
                                   int query_group, int context_length, int head_dimension,
                                   cudaStream_t stream);

// WMMA prototype for global attention (D=512, GQA group=8); not used by the model runner.
void grouped_decode_attention_wmma_bf16(const void *query, const void *key_cache,
                                        const void *value_cache, void *output, int batch,
                                        int kv_heads, int query_group, int context_length,
                                        int head_dimension, cudaStream_t stream);

// Benchmark/prototyping boundary for the Hopper-native global QK tile. Produces BF16 scores in
// [batch, kv_head, query, context] order without softmax. Context must be a multiple of 64.
void grouped_global_qk_wgmma_bf16(const void *query, const void *key_cache, void *scores, int batch,
                                  int kv_heads, int query_group, int context_length,
                                  int head_dimension, cudaStream_t stream, int tiles_per_block = 1);

// Block-128 signed-INT8 QK. K/Q are row-major signed bytes and scales are
// [row, 4]. Four independently scaled S32 WGMMA partials are accumulated into BF16 scores.
void grouped_global_qk_wgmma_int8_block128(const void *query, const float *query_scales,
                                           const void *key_cache, const float *key_scales,
                                           void *scores, int batch, int kv_heads, int query_group,
                                           int context_length, int head_dimension,
                                           int cache_capacity, cudaStream_t stream,
                                           int tiles_per_block = 1);

// One 64-token Hopper global-attention tile. This keeps QK scores and
// probabilities in shared memory and performs both QK and PV with WGMMA.
void grouped_global_attention_wgmma_tile_bf16(const void *query, const void *key_cache,
                                              const void *value_cache, void *output, int batch,
                                              int kv_heads, int query_group, int context_length,
                                              int head_dimension, cudaStream_t stream);

// Fixed-scale E4M3 PV. Values are V*64 and probabilities are P*448;
// the kernel applies the inverse product scale to its FP32 WGMMA accumulator. Context is split
// into enough independent CTAs to occupy H200 at small batches, then reduced in FP32.
void grouped_global_pv_wgmma_fp8_rs(const void *value_cache, const void *probabilities,
                                    void *partial_outputs, void *output, int batch, int kv_heads,
                                    int query_group, int context_length, int head_dimension,
                                    int cache_capacity, int segments, cudaStream_t stream);

// Block-128 signed-INT8 V. V is reconstructed with per-token/block FP32 scales in
// shared memory, then BF16 WGMMA accumulates the already-normalized probabilities in FP32.
void grouped_global_pv_wgmma_int8_block128(const void *value_cache, const float *value_scales,
                                           const void *probabilities, void *output, int batch,
                                           int kv_heads, int query_group, int context_length,
                                           int head_dimension, int cache_capacity,
                                           cudaStream_t stream);
void grouped_global_pv_wgmma_int8_rs(const void *value_cache, const void *probabilities,
                                     const float *probability_scales, void *partial_outputs,
                                     void *output, int batch, int kv_heads, int query_group,
                                     int context_length, int head_dimension, int cache_capacity,
                                     int segments, cudaStream_t stream);

// Cache-write helpers for FP8 PV. Values are quantized once into the
// native K-major WGMMA tile layout; probabilities are quantized into the same per-tile contract.
void quantize_grouped_global_values_wgmma_fp8(const void *values, void *packed_values, int batch,
                                              int kv_heads, int context_length, int head_dimension,
                                              cudaStream_t stream);
void pack_grouped_global_values_wgmma_int8(const void *values, void *packed_values, int batch,
                                           int kv_heads, int context_length, int head_dimension,
                                           cudaStream_t stream);
// Block-128-scaled signed-INT8 V stored directly in the native SW64 WGMMA tile layout. The scale
// contract is four FP32 values per [physical head, token]. Span and decode variants update an
// existing cache without materializing a row-major INT8 intermediate.
void quantize_grouped_global_values_wgmma_int8(const void *values, void *packed_values,
                                               float *scales, int batch, int kv_heads,
                                               int context_length, int head_dimension,
                                               cudaStream_t stream);
void quantize_grouped_global_value_span_wgmma_int8(const void *values, void *packed_values,
                                                   float *scales, int kv_heads, int tokens,
                                                   int context_length, int head_dimension,
                                                   int position_start, cudaStream_t stream);
void quantize_grouped_global_decode_values_wgmma_int8(const void *values, void *packed_values,
                                                      float *scales, const int *positions,
                                                      const int *slots, int batch,
                                                      int maximum_slots, int kv_heads,
                                                      int context_length, int head_dimension,
                                                      cudaStream_t stream);
void quantize_grouped_global_value_span_wgmma_fp8(const void *values, void *packed_values,
                                                  int kv_heads, int tokens, int context_length,
                                                  int head_dimension, int position_start,
                                                  cudaStream_t stream);
void quantize_grouped_global_decode_values_wgmma_fp8(const void *values, void *packed_values,
                                                     const int *positions, const int *slots,
                                                     int batch, int maximum_slots, int kv_heads,
                                                     int context_length, int head_dimension,
                                                     cudaStream_t stream);
void quantize_grouped_global_probabilities_wgmma_fp8(const void *probabilities,
                                                     void *packed_probabilities, int batch,
                                                     int kv_heads, int query_group,
                                                     int context_length, cudaStream_t stream);

// One-buffer RS-WGMMA. Raw KV remains in shared memory while exact K and V fragments
// are constructed directly in the tensor-core warpgroup's registers.
void grouped_global_attention_wgmma_raw_rs_tile_bf16(
    const void *query, const void *raw_cache, const float *inverse_rms_cache,
    const void *key_norm_weight, const void *cosine, const void *sine, void *output, int batch,
    int kv_heads, int query_group, int context_length, int head_dimension, cudaStream_t stream);

// Full-context counterpart to the one-buffer RS tile. Workspace has the same shape as the
// materialized split prototype, but the persistent source is exact raw shared-projection KV.
// Cosine and sine contain Gemma's 64 active global-RoPE frequencies in transposed
// [frequency, position] order; low/high head halves share the same coefficient.
void grouped_global_attention_wgmma_raw_rs_split_bf16(
    const void *query, const void *raw_cache, const float *inverse_rms_cache,
    const void *key_norm_weight, const void *cosine, const void *sine, void *output,
    float *partial_outputs, float *partial_maxima, float *partial_sums, int batch, int kv_heads,
    int query_group, int context_length, int head_dimension, int tiles_per_segment,
    cudaStream_t stream);

// Full-context attention. Workspace contains FP32 partial outputs
// [head, segment, query, dimension] and FP32 max/sum pairs [head, segment, query].
void grouped_global_attention_wgmma_split_bf16(const void *query, const void *key_cache,
                                               const void *value_cache, void *output,
                                               float *partial_outputs, float *partial_maxima,
                                               float *partial_sums, int batch, int kv_heads,
                                               int query_group, int context_length,
                                               int head_dimension, int tiles_per_segment,
                                               cudaStream_t stream);

} // namespace carat
