#include "carat/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Shape {
  const char *name;
  int input_width;
  int output_width;
  int multiplicity;
};

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class Allocation {
public:
  explicit Allocation(std::uint64_t bytes) : bytes_(bytes) {
    check(cudaMalloc(&pointer_, static_cast<std::size_t>(bytes)), "cudaMalloc benchmark buffer");
    check(cudaMemset(pointer_, 0, static_cast<std::size_t>(bytes)), "cudaMemset benchmark buffer");
  }
  ~Allocation() {
    cudaFree(pointer_);
  }
  void *get() {
    return pointer_;
  }

private:
  void *pointer_{nullptr};
  std::uint64_t bytes_;
};

float benchmark(carat::Bf16Linear &linear, const Shape &shape, int rows) {
  const std::uint64_t input_bytes = 2ULL * rows * shape.input_width;
  const std::uint64_t weight_bytes = 2ULL * shape.input_width * shape.output_width;
  const std::uint64_t output_bytes = 2ULL * rows * shape.output_width;
  Allocation input(input_bytes), weight(weight_bytes), output(output_bytes);
  for (int iteration = 0; iteration < 3; ++iteration) {
    linear.run(input.get(), weight.get(), output.get(), rows, shape.input_width, shape.output_width,
               nullptr);
  }
  check(cudaDeviceSynchronize(), "warmup synchronization");
  cudaEvent_t begin{}, end{};
  check(cudaEventCreate(&begin), "create begin event");
  check(cudaEventCreate(&end), "create end event");
  constexpr int iterations = 8;
  check(cudaEventRecord(begin), "record begin event");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    linear.run(input.get(), weight.get(), output.get(), rows, shape.input_width, shape.output_width,
               nullptr);
  }
  check(cudaEventRecord(end), "record end event");
  check(cudaEventSynchronize(end), "benchmark synchronization");
  float milliseconds = 0;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed time");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return milliseconds / iterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2) {
      std::cerr << "usage: carat-linear-benchmark [ROWS]\n";
      return 64;
    }
    const std::vector<Shape> shapes{
        {"sliding_qkv", 5376, 16384, 50}, {"sliding_o", 8192, 5376, 50},
        {"global_qk", 5376, 18432, 10},   {"global_o", 16384, 5376, 10},
        {"gate_up", 5376, 43008, 60},     {"down", 21504, 5376, 60},
        {"lm_head", 5376, 262144, 1},
    };
    const std::vector<int> row_counts = argc == 2 ? std::vector<int>{std::stoi(argv[1])}
                                                  : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128};
    carat::Bf16Linear linear;
    std::map<int, double> layer_milliseconds;
    std::cout << "component,rows,milliseconds,effective_weight_gb_s,tflop_s\n";
    for (const auto &shape : shapes) {
      for (const int rows : row_counts) {
        const float milliseconds = benchmark(linear, shape, rows);
        const double weight_bytes = 2.0 * shape.input_width * shape.output_width;
        const double flops = weight_bytes * rows;
        const double seconds = milliseconds / 1000.0;
        layer_milliseconds[rows] += milliseconds * shape.multiplicity;
        std::cout << shape.name << ',' << rows << ',' << std::fixed << std::setprecision(4)
                  << milliseconds << ',' << std::setprecision(2) << weight_bytes / seconds / 1e9
                  << ',' << flops / seconds / 1e12 << '\n';
      }
    }
    std::cout << "summary_rows,linear_iteration_ms,iterations_s,aggregate_tokens_s\n";
    for (const int rows : row_counts) {
      const double milliseconds = layer_milliseconds.at(rows);
      std::cout << rows << ',' << std::fixed << std::setprecision(4) << milliseconds << ','
                << 1000.0 / milliseconds << ',' << rows * 1000.0 / milliseconds << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "linear benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
