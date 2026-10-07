#pragma once

#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace carat {

// Fused causal/sliding prefill attention. Q and output are token-major [B, S, Hq, D];
// K and V are head-major [B, Hkv, S, D]. The graph is built once per exact shape and reused.
class CudnnPrefillAttention {
public:
  CudnnPrefillAttention();
  ~CudnnPrefillAttention();
  CudnnPrefillAttention(const CudnnPrefillAttention &) = delete;
  CudnnPrefillAttention &operator=(const CudnnPrefillAttention &) = delete;

  void run(const void *query, const void *key, const void *value, void *output, int batch,
           int query_heads, int kv_heads, int query_tokens, int kv_tokens, int kv_capacity,
           int head_dimension, int left_window, cudaStream_t stream);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace carat
