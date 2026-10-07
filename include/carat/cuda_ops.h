#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace carat {

void rms_norm_bf16(const void *input, const void *weight, void *output, std::size_t rows,
                   std::size_t width, float epsilon, cudaStream_t stream);
// The row maxima are measured from the rounded BF16 output, so a later tensor-scale E4M3
// conversion can skip its separate amax pass without changing quantized bytes.
void rms_norm_bf16_with_row_amax(const void *input, const void *weight, void *output,
                                 float *row_amax, std::size_t rows, std::size_t width,
                                 float epsilon, cudaStream_t stream);
void rms_norm_add_bf16(const void *input, const void *weight, const void *residual, void *output,
                       std::size_t rows, std::size_t width, float epsilon, cudaStream_t stream);
void rms_norm_add_and_norm_bf16(const void *input, const void *weight, const void *residual,
                                const void *next_weight, void *residual_output,
                                void *normalized_output, std::size_t rows, std::size_t width,
                                float epsilon, cudaStream_t stream);
void rms_norm_add_and_norm_bf16_with_row_amax(const void *input, const void *weight,
                                              const void *residual, const void *next_weight,
                                              void *residual_output, void *normalized_output,
                                              float *normalized_row_amax, std::size_t rows,
                                              std::size_t width, float epsilon,
                                              cudaStream_t stream);
void rms_norm_add_scale_bf16(const void *input, const void *weight, const void *residual,
                             const void *scalar, void *output, std::size_t rows, std::size_t width,
                             float epsilon, cudaStream_t stream);
void scale_bf16(void *values, const void *scalar, std::size_t elements, cudaStream_t stream);
void gelu_tanh_gate_bf16(const void *gate, const void *up, void *output, std::size_t elements,
                         cudaStream_t stream);
void gelu_tanh_gate_rows_bf16(const void *packed_gate_up, void *output, int rows, int width,
                              cudaStream_t stream);
// Emits one maximum per producer CTA from the rounded BF16 GELU product. This lets a following
// tensor-scaled FP8 conversion avoid rereading the activation solely to find its maximum.
void gelu_tanh_gate_rows_bf16_with_block_amax(const void *packed_gate_up, void *output,
                                              float *block_amax, int rows, int width,
                                              cudaStream_t stream);
void reduce_block_amax_to_rows(const float *block_amax, float *row_amax, int rows,
                               int blocks_per_row, cudaStream_t stream);
void embedding_bf16(const void *embedding_weight, int token_id, void *output, int vocabulary_size,
                    int hidden_size, float scale, cudaStream_t stream);
void embedding_batch_bf16(const void *embedding_weight, const int *device_token_ids, void *output,
                          int batch, int vocabulary_size, int hidden_size, float scale,
                          cudaStream_t stream);
void gather_rows_bf16(const void *input, const int *device_row_indices, void *output, int rows,
                      int width, cudaStream_t stream);
void scatter_rows_bf16(const void *input, const int *device_row_indices, void *output, int rows,
                       int width, cudaStream_t stream);
void concatenate_rows_bf16(const void *left, int left_width, const void *right, int right_width,
                           void *output, int rows, cudaStream_t stream);
void token_to_head_bf16(const void *input, void *output, int tokens, int heads, int head_dimension,
                        cudaStream_t stream);
void head_to_token_bf16(const void *input, void *output, int tokens, int heads, int head_dimension,
                        cudaStream_t stream);
void token_to_head_batch_bf16(const void *input, void *output, int requests, int tokens, int heads,
                              int head_dimension, cudaStream_t stream);
void head_to_token_batch_bf16(const void *input, void *output, int requests, int tokens, int heads,
                              int head_dimension, cudaStream_t stream);
// The block maxima are measured from the unchanged BF16 transpose output. Blocks are
// row-local and ordered [request * tokens + token, block], with 256 values per block.
void head_to_token_batch_bf16_with_block_amax(const void *input, void *output, float *block_amax,
                                              int requests, int tokens, int heads,
                                              int head_dimension, cudaStream_t stream);
void copy_kv_to_cache_bf16(const void *source, void *destination, int heads, int tokens,
                           int head_dimension, int destination_capacity, int position_start,
                           cudaStream_t stream);
// Gemma 4 global K has an architecture-fixed RMS scale near 1/16. Its E4M3 mirror therefore
// uses K * 1024, while normalized decode queries use Q * 64.
void quantize_global_k_cache_span_fp8(const void *source, void *destination, int heads, int tokens,
                                      int head_dimension, int cache_capacity, int position_start,
                                      cudaStream_t stream);
void quantize_global_decode_qk_fp8(const void *queries, const void *key_cache, void *fp8_queries,
                                   void *fp8_key_cache, const int *positions, const int *slots,
                                   int batch, int maximum_slots, int query_heads, int kv_heads,
                                   int head_dimension, int cache_capacity, cudaStream_t stream);
// Numerical oracle: quantizes each token/head vector to symmetric INT8 and dequantizes it back
// into the BF16 cache. It measures representation quality only and gives no speedup.
void roundtrip_global_kv_int8_per_token_bf16(void *key_cache, void *value_cache, int heads,
                                             int tokens, int head_dimension, int cache_capacity,
                                             int position_start, cudaStream_t stream,
                                             bool quantize_keys = true, bool quantize_values = true,
                                             int block_width = 512);
void roundtrip_global_decode_kv_int8_per_token_bf16(
    void *key_cache, void *value_cache, const int *positions, const int *slots, int batch,
    int maximum_slots, int heads, int head_dimension, int cache_capacity, cudaStream_t stream,
    bool quantize_keys = true, bool quantize_values = true, int block_width = 512);
// Materializes contiguous BF16 rows as signed INT8 plus one FP32 dequantization scale per row.
// The stored contract is real_value ~= quantized_value * row_scale.
void quantize_bf16_rows_symmetric_int8(const void *source, void *destination, float *row_scales,
                                       int rows, int width, cudaStream_t stream);
// Block-scaled counterpart. Scales are [row, width / block_width] in row-major order.
void quantize_bf16_blocks_symmetric_int8(const void *source, void *destination, float *block_scales,
                                         int rows, int width, int block_width, cudaStream_t stream);
// Materializes block-128 signed-INT8 global K rows in the same head/slot/token order as the
// BF16 cache. Scales are [physical_head, token, 4] and reconstruct K as int8 * scale.
void quantize_global_k_cache_span_int8_block128(const void *source, void *destination,
                                                float *scales, int heads, int tokens,
                                                int head_dimension, int cache_capacity,
                                                int position_start, cudaStream_t stream);
void quantize_global_decode_k_int8_block128(const void *source, void *destination, float *scales,
                                            const int *positions, const int *slots, int batch,
                                            int maximum_slots, int heads, int head_dimension,
                                            int cache_capacity, cudaStream_t stream);
void gather_kv_ring_bf16(const void *source, void *destination, int heads, int retained_tokens,
                         int head_dimension, int source_capacity, int destination_capacity,
                         int logical_position_start, cudaStream_t stream);
void copy_kv_span_to_ring_bf16(const void *source, void *destination, int heads, int tokens,
                               int head_dimension, int source_capacity, int source_position_start,
                               int destination_capacity, int logical_position_start,
                               cudaStream_t stream);
// Uniform multi-request variants used to collapse cached suffix prefill into one launch.
void gather_kv_rings_bf16(const void *const *device_key_sources,
                          const void *const *device_value_sources, void *staged_keys,
                          void *staged_values, const int *device_position_starts, int requests,
                          int heads, int retained_tokens, int head_dimension, int source_capacity,
                          int staging_capacity, cudaStream_t stream);
void copy_kv_spans_to_rings_bf16(const void *staged_keys, const void *staged_values,
                                 void *const *device_key_destinations,
                                 void *const *device_value_destinations,
                                 const int *device_position_starts, int requests, int heads,
                                 int tokens, int head_dimension, int staging_capacity,
                                 int staging_position_start, int destination_capacity,
                                 cudaStream_t stream);
// Saves/restores the ring locations that speculative candidates may overwrite.  A rejected
// tail must be restored because those physical locations still contain live sliding-window KV.
void backup_kv_ring_spans_bf16(const void *const *device_key_sources,
                               const void *const *device_value_sources, void *backup_keys,
                               void *backup_values, const int *device_position_starts, int requests,
                               int heads, int tokens, int head_dimension, int source_capacity,
                               cudaStream_t stream);
void restore_kv_ring_suffixes_bf16(const void *backup_keys, const void *backup_values,
                                   void *const *device_key_destinations,
                                   void *const *device_value_destinations,
                                   const int *device_position_starts,
                                   const int *device_accepted_tokens, int requests, int heads,
                                   int tokens, int head_dimension, int destination_capacity,
                                   cudaStream_t stream);
// Copies a logical prefix between head-major cache allocations. A flat memcpy is only valid
// when every head's full capacity is resident; shorter prefixes contain per-head gaps.
void clone_kv_cache_prefix_bf16(const void *source_keys, const void *source_values,
                                void *destination_keys, void *destination_values, int heads,
                                int resident_tokens, int head_dimension, int cache_capacity,
                                cudaStream_t stream);
std::size_t argmax_bf16_workspace_bytes(int elements);
void argmax_bf16(const void *values, int elements, int *device_output, void *workspace,
                 cudaStream_t stream);
void argmax_bf16_rows(const void *values, int rows, int elements, int *device_output,
                      void *workspace, cudaStream_t stream);

} // namespace carat
