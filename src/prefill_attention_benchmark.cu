#include "carat/prefill_attention.h"
#include "carat/prefill_gemm_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class Allocation {
public:
  explicit Allocation(std::size_t bytes) {
    check(cudaMalloc(&pointer_, bytes), "allocate benchmark tensor");
  }
  ~Allocation() {
    cudaFree(pointer_);
  }
  void *get() {
    return pointer_;
  }

private:
  void *pointer_{nullptr};
};

__global__ void initialize_value(__nv_bfloat16 *values, int heads, int tokens, int dimension) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int elements = heads * tokens * dimension;
  if (index < elements) {
    const int token = (index / dimension) % tokens;
    values[index] = __float2bfloat16_rn(static_cast<float>(token % 31) / 31.0F);
  }
}

float benchmark_sliding(int tokens) {
  constexpr int query_heads = 32;
  constexpr int kv_heads = 16;
  constexpr int dimension = 256;
  constexpr int window = 1024;
  const std::size_t q_bytes = 2ULL * tokens * query_heads * dimension;
  const std::size_t kv_bytes = 2ULL * tokens * kv_heads * dimension;
  Allocation query(q_bytes), key(kv_bytes), value(kv_bytes), output(q_bytes);
  check(cudaMemset(query.get(), 0, q_bytes), "clear queries");
  check(cudaMemset(key.get(), 0, kv_bytes), "clear keys");
  initialize_value<<<(kv_heads * tokens * dimension + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16 *>(value.get()), kv_heads, tokens, dimension);
  check(cudaGetLastError(), "initialize values");

  carat::CudnnPrefillAttention attention;
  attention.run(query.get(), key.get(), value.get(), output.get(), 1, query_heads, kv_heads, tokens,
                tokens, tokens, dimension, window, nullptr);
  check(cudaDeviceSynchronize(), "warm up prefill attention");
  constexpr int iterations = 10;
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create begin event");
  check(cudaEventCreate(&end), "create end event");
  check(cudaEventRecord(begin), "record begin event");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    attention.run(query.get(), key.get(), value.get(), output.get(), 1, query_heads, kv_heads,
                  tokens, tokens, tokens, dimension, window, nullptr);
  }
  check(cudaEventRecord(end), "record end event");
  check(cudaEventSynchronize(end), "synchronize end event");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure prefill attention");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

float benchmark_global(int tokens) {
  constexpr int query_heads = 32;
  constexpr int kv_heads = 4;
  constexpr int dimension = 512;
  const std::size_t q_bytes = 2ULL * tokens * query_heads * dimension;
  const std::size_t kv_bytes = 2ULL * tokens * kv_heads * dimension;
  Allocation query(q_bytes), key(kv_bytes), value(kv_bytes), output(q_bytes);
  check(cudaMemset(query.get(), 0, q_bytes), "clear global queries");
  check(cudaMemset(key.get(), 0, kv_bytes), "clear global keys");
  initialize_value<<<(kv_heads * tokens * dimension + 255) / 256, 256>>>(
      static_cast<__nv_bfloat16 *>(value.get()), kv_heads, tokens, dimension);
  check(cudaGetLastError(), "initialize global values");

  carat::GemmCausalPrefillAttention attention(tokens, query_heads);
  attention.run(query.get(), key.get(), value.get(), output.get(), query_heads, kv_heads, tokens,
                tokens, tokens, 0, dimension, nullptr);
  check(cudaDeviceSynchronize(), "warm up global prefill attention");
  constexpr int iterations = 10;
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create global begin event");
  check(cudaEventCreate(&end), "create global end event");
  check(cudaEventRecord(begin), "record global begin event");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    attention.run(query.get(), key.get(), value.get(), output.get(), query_heads, kv_heads, tokens,
                  tokens, tokens, 0, dimension, nullptr);
  }
  check(cudaEventRecord(end), "record global end event");
  check(cudaEventSynchronize(end), "synchronize global end event");
  float milliseconds = 0.0F;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "measure global prefill attention");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

} // namespace

int main() {
  try {
    std::cout << "kind,tokens,milliseconds,tokens_per_second\n";
    for (const int tokens : std::vector<int>{128, 512, 1024, 2048, 4096, 8192}) {
      const float milliseconds = benchmark_sliding(tokens);
      std::cout << "sliding," << tokens << ',' << milliseconds << ','
                << tokens / (milliseconds / 1000.0F) << '\n';
    }
    for (const int tokens : std::vector<int>{128, 512, 1024, 2048, 4096, 8192}) {
      const float milliseconds = benchmark_global(tokens);
      std::cout << "global," << tokens << ',' << milliseconds << ','
                << tokens / (milliseconds / 1000.0F) << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "prefill attention benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
