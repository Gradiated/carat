#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void check(cublasStatus_t result, const char *operation) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(result));
  }
}

class Allocation {
public:
  explicit Allocation(std::uint64_t bytes) {
    check(cudaMalloc(&pointer_, static_cast<std::size_t>(bytes)), "allocate benchmark buffer");
    check(cudaMemset(pointer_, 0, static_cast<std::size_t>(bytes)), "initialize benchmark buffer");
  }
  ~Allocation() {
    cudaFree(pointer_);
  }
  void *get() const {
    return pointer_;
  }

private:
  void *pointer_{nullptr};
};

float benchmark(cudaDataType_t input_type, int input_bytes, int query_group, int context) {
  constexpr int batch = 16;
  constexpr int kv_heads = 4;
  constexpr int dimension = 512;
  constexpr int head_batches = batch * kv_heads;
  const std::uint64_t key_elements = static_cast<std::uint64_t>(head_batches) * context * dimension;
  const std::uint64_t query_elements =
      static_cast<std::uint64_t>(head_batches) * query_group * dimension;
  const std::uint64_t score_elements =
      static_cast<std::uint64_t>(head_batches) * query_group * context;
  constexpr std::size_t workspace_bytes = 32ULL * 1024 * 1024;
  Allocation key(key_elements * input_bytes);
  Allocation query(query_elements * input_bytes);
  Allocation scores(score_elements * 2ULL);
  Allocation workspace(workspace_bytes);
  Allocation scale(sizeof(float));
  constexpr float one = 1.0F;
  check(cudaMemcpy(scale.get(), &one, sizeof(one), cudaMemcpyHostToDevice), "initialize FP8 scale");

  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t key_layout{}, query_layout{}, score_layout{};
  cublasLtMatmulPreference_t preference{};
  check(cublasLtCreate(&handle), "create cuBLASLt handle");
  check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "create QK descriptor");
  constexpr cublasOperation_t transpose = CUBLAS_OP_T;
  check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                       sizeof(transpose)),
        "transpose keys");
  if (input_type == CUDA_R_8F_E4M3) {
    constexpr cublasLtMatmulMatrixScale_t scalar_scale = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    const void *scale_pointer = scale.get();
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                         &scalar_scale, sizeof(scalar_scale)),
          "set key scale mode");
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                         &scalar_scale, sizeof(scalar_scale)),
          "set query scale mode");
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                         &scale_pointer, sizeof(scale_pointer)),
          "set key scale");
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                         &scale_pointer, sizeof(scale_pointer)),
          "set query scale");
  }
  check(cublasLtMatrixLayoutCreate(&key_layout, input_type, dimension, context, dimension),
        "create key layout");
  check(cublasLtMatrixLayoutCreate(&query_layout, input_type, dimension, query_group, dimension),
        "create query layout");
  check(cublasLtMatrixLayoutCreate(&score_layout, CUDA_R_16BF, context, query_group, context),
        "create score layout");
  constexpr std::int32_t batch_count = head_batches;
  const std::int64_t key_stride = static_cast<std::int64_t>(context) * dimension;
  const std::int64_t query_stride = static_cast<std::int64_t>(query_group) * dimension;
  const std::int64_t score_stride = static_cast<std::int64_t>(query_group) * context;
  for (auto layout : {key_layout, query_layout, score_layout}) {
    check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
                                           sizeof(batch_count)),
          "set QK batch count");
  }
  check(cublasLtMatrixLayoutSetAttribute(key_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                         &key_stride, sizeof(key_stride)),
        "set key stride");
  check(cublasLtMatrixLayoutSetAttribute(query_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                         &query_stride, sizeof(query_stride)),
        "set query stride");
  check(cublasLtMatrixLayoutSetAttribute(score_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                         &score_stride, sizeof(score_stride)),
        "set score stride");
  check(cublasLtMatmulPreferenceCreate(&preference), "create QK preference");
  check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_bytes, sizeof(workspace_bytes)),
        "set QK workspace limit");
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  check(cublasLtMatmulAlgoGetHeuristic(handle, operation, key_layout, query_layout, score_layout,
                                       score_layout, preference, 1, &heuristic, &returned),
        "select QK algorithm");
  if (returned == 0)
    throw std::runtime_error("no cuBLASLt QK algorithm");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  const auto run = [&] {
    check(cublasLtMatmul(handle, operation, &alpha, key.get(), key_layout, query.get(),
                         query_layout, &beta, scores.get(), score_layout, scores.get(),
                         score_layout, &heuristic.algo, workspace.get(), workspace_bytes, nullptr),
          "run QK matmul");
  };
  run();
  run();
  check(cudaDeviceSynchronize(), "warm up QK");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create begin event");
  check(cudaEventCreate(&end), "create end event");
  constexpr int iterations = 20;
  check(cudaEventRecord(begin), "record begin event");
  for (int iteration = 0; iteration < iterations; ++iteration)
    run();
  check(cudaEventRecord(end), "record end event");
  check(cudaEventSynchronize(end), "synchronize benchmark");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure QK");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(score_layout);
  cublasLtMatrixLayoutDestroy(query_layout);
  cublasLtMatrixLayoutDestroy(key_layout);
  cublasLtMatmulDescDestroy(operation);
  cublasLtDestroy(handle);
  return milliseconds / iterations;
}

} // namespace

int main() {
  try {
    const float bf16_ms = benchmark(CUDA_R_16BF, 2, 8, 8192);
    const float fp8_ms = benchmark(CUDA_R_8F_E4M3, 1, 8, 8192);
    const float verify_bf16_ms = benchmark(CUDA_R_16BF, 2, 32, 6016);
    constexpr double bf16_key_bytes = 2.0 * 16 * 4 * 8192 * 512;
    constexpr double fp8_key_bytes = bf16_key_bytes / 2.0;
    std::cout << std::fixed << std::setprecision(4) << "bf16_qk_ms=" << bf16_ms << '\n'
              << "bf16_key_gb_s=" << bf16_key_bytes / (bf16_ms / 1000.0) / 1e9 << '\n'
              << "fp8_qk_ms=" << fp8_ms << '\n'
              << "fp8_key_gb_s=" << fp8_key_bytes / (fp8_ms / 1000.0) / 1e9 << '\n'
              << "qk_speedup=" << bf16_ms / fp8_ms << '\n'
              << "verify_q4_bf16_qk_ms=" << verify_bf16_ms << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FP8 attention benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
