#include "carat/prefill_attention.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace carat {
namespace fe = cudnn_frontend;
namespace {

constexpr std::int64_t q_uid = 1;
constexpr std::int64_t k_uid = 2;
constexpr std::int64_t v_uid = 3;
constexpr std::int64_t o_uid = 4;

void cuda_check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void cudnn_check(cudnnStatus_t result, const char *operation) {
  if (result != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(result));
  }
}

void graph_check(fe::error_t result, const char *operation) {
  if (!result.is_good()) {
    throw std::runtime_error(std::string(operation) + ": " + result.get_message());
  }
}

struct Plan {
  std::shared_ptr<fe::graph::Graph> graph;
  void *workspace{nullptr};

  ~Plan() {
    if (workspace != nullptr)
      cudaFree(workspace);
  }
};

} // namespace

struct CudnnPrefillAttention::Implementation {
  using Key = std::tuple<int, int, int, int, int, int, int, int>;

  cudnnHandle_t handle{};
  std::map<Key, std::unique_ptr<Plan>> plans;

  Implementation() {
    cudnn_check(cudnnCreate(&handle), "create cuDNN handle");
  }

  ~Implementation() {
    plans.clear();
    if (handle != nullptr)
      cudnnDestroy(handle);
  }

  Plan &plan(int batch, int query_heads, int kv_heads, int query_tokens, int kv_tokens,
             int kv_capacity, int head_dimension, int left_window) {
    const Key key{batch,     query_heads, kv_heads,       query_tokens,
                  kv_tokens, kv_capacity, head_dimension, left_window};
    const auto existing = plans.find(key);
    if (existing != plans.end())
      return *existing->second;

    auto result = std::make_unique<Plan>();
    result->graph = std::make_shared<fe::graph::Graph>();
    result->graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const std::int64_t b = batch;
    const std::int64_t hq = query_heads;
    const std::int64_t hk = kv_heads;
    const std::int64_t sq = query_tokens;
    const std::int64_t skv = kv_tokens;
    const std::int64_t capacity = kv_capacity;
    const std::int64_t d = head_dimension;
    auto query = result->graph->tensor(fe::graph::Tensor_attributes()
                                           .set_name("query")
                                           .set_uid(q_uid)
                                           .set_dim({b, hq, sq, d})
                                           .set_stride({sq * hq * d, d, hq * d, 1}));
    auto key_tensor =
        result->graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("key")
                                  .set_uid(k_uid)
                                  .set_dim({b, hk, skv, d})
                                  .set_stride({hk * capacity * d, capacity * d, d, 1}));
    auto value_tensor =
        result->graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("value")
                                  .set_uid(v_uid)
                                  .set_dim({b, hk, skv, d})
                                  .set_stride({hk * capacity * d, capacity * d, d, 1}));

    auto attributes = fe::graph::SDPA_attributes()
                          .set_name("gemma_prefill_attention")
                          .set_generate_stats(false)
                          .set_attn_scale(1.0F)
                          .set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT)
                          .set_diagonal_band_right_bound(0);
    if (left_window > 0)
      attributes.set_diagonal_band_left_bound(left_window);
    auto [output, stats] = result->graph->sdpa(query, key_tensor, value_tensor, attributes);
    (void)stats;
    output->set_output(true)
        .set_uid(o_uid)
        .set_dim({b, hq, sq, d})
        .set_stride({sq * hq * d, d, hq * d, 1});

    graph_check(result->graph->build(handle, {fe::HeurMode_t::A}), "build cuDNN SDPA graph");
    std::int64_t workspace_bytes = 0;
    graph_check(result->graph->get_workspace_size(workspace_bytes), "query cuDNN SDPA workspace");
    if (workspace_bytes > 0) {
      cuda_check(cudaMalloc(&result->workspace, static_cast<std::size_t>(workspace_bytes)),
                 "allocate cuDNN SDPA workspace");
    }
    Plan &reference = *result;
    plans.emplace(key, std::move(result));
    return reference;
  }
};

CudnnPrefillAttention::CudnnPrefillAttention()
    : implementation_(std::make_unique<Implementation>()) {}
CudnnPrefillAttention::~CudnnPrefillAttention() = default;

void CudnnPrefillAttention::run(const void *query, const void *key, const void *value, void *output,
                                int batch, int query_heads, int kv_heads, int query_tokens,
                                int kv_tokens, int kv_capacity, int head_dimension, int left_window,
                                cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || output == nullptr || batch <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 || query_tokens <= 0 ||
      kv_tokens < query_tokens || kv_capacity < kv_tokens || head_dimension <= 0 ||
      left_window < 0) {
    throw std::runtime_error("invalid prefill attention shape");
  }
  auto &state = *implementation_;
  auto &selected = state.plan(batch, query_heads, kv_heads, query_tokens, kv_tokens, kv_capacity,
                              head_dimension, left_window);
  cudnn_check(cudnnSetStream(state.handle, stream), "set cuDNN SDPA stream");
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> pointers = {
      {q_uid, const_cast<void *>(query)},
      {k_uid, const_cast<void *>(key)},
      {v_uid, const_cast<void *>(value)},
      {o_uid, output}};
  graph_check(selected.graph->execute(state.handle, pointers, selected.workspace),
              "execute cuDNN SDPA graph");
}

} // namespace carat
