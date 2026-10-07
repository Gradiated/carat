#pragma once

#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace carat {

class Gemma4QkvPostprocessor {
public:
  explicit Gemma4QkvPostprocessor(int maximum_positions);
  ~Gemma4QkvPostprocessor();
  Gemma4QkvPostprocessor(const Gemma4QkvPostprocessor &) = delete;
  Gemma4QkvPostprocessor &operator=(const Gemma4QkvPostprocessor &) = delete;

  void run(const void *packed_qkv, const void *query_norm_weight, const void *key_norm_weight,
           void *queries, void *key_cache, void *value_cache, int tokens, int query_heads,
           int kv_heads, int head_dimension, bool key_equals_value, int position_start,
           int cache_capacity, float epsilon, cudaStream_t stream);
  void run_positioned(const void *packed_qkv, const void *query_norm_weight,
                      const void *key_norm_weight, void *queries, void *key_cache,
                      void *value_cache, int tokens, int query_heads, int kv_heads,
                      int head_dimension, bool key_equals_value, int position_start,
                      int cache_position_start, int cache_capacity, float epsilon,
                      cudaStream_t stream);
  void run_decode_batch(const void *packed_qkv, const void *query_norm_weight,
                        const void *key_norm_weight, void *queries, void *key_cache,
                        void *value_cache, int batch, int query_heads, int kv_heads,
                        int head_dimension, bool key_equals_value, int position, int cache_capacity,
                        float epsilon, cudaStream_t stream);
  // Compact input rows write into stable KV slots at independent absolute positions.
  // positions and slots are device arrays with `batch` entries.
  void run_decode_ragged(const void *packed_qkv, const void *query_norm_weight,
                         const void *key_norm_weight, void *queries, void *key_cache,
                         void *value_cache, const int *positions, const int *slots, int batch,
                         int maximum_slots, int query_heads, int kv_heads, int head_dimension,
                         bool key_equals_value, int cache_capacity, void *int8_key_cache,
                         float *int8_key_scales, float epsilon, cudaStream_t stream);
  // Processes equal-length request suffixes in one launch pair. Queries remain request-major
  // token rows; K/V are written into request-major staged attention buffers.
  void run_prefill_uniform(const void *packed_qkv, const void *query_norm_weight,
                           const void *key_norm_weight, void *queries, void *staged_keys,
                           void *staged_values, const int *device_position_starts, int requests,
                           int tokens_per_request, int query_heads, int kv_heads,
                           int head_dimension, bool key_equals_value, int staged_position_start,
                           int staged_capacity, float epsilon, cudaStream_t stream);
  // As above, but writes K/V directly into independent request caches. Cache pointers and
  // absolute positions live on device so an entire speculative batch still costs one query
  // launch and one K/V launch.
  void run_prefill_uniform_direct(const void *packed_qkv, const void *query_norm_weight,
                                  const void *key_norm_weight, void *queries,
                                  void *const *device_key_caches, void *const *device_value_caches,
                                  const int *device_position_starts, int requests,
                                  int tokens_per_request, int query_heads, int kv_heads,
                                  int head_dimension, bool key_equals_value, int cache_capacity,
                                  float epsilon, cudaStream_t stream);
  void run_query_ragged(const void *projected_queries, const void *query_norm_weight, void *queries,
                        const int *device_positions, int batch, int query_heads, int head_dimension,
                        bool global_attention, float epsilon, cudaStream_t stream);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
