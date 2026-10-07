#pragma once

#include <cstddef>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st *;

namespace carat {

class Bf16Linear {
public:
  explicit Bf16Linear(std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL);
  ~Bf16Linear();
  Bf16Linear(const Bf16Linear &) = delete;
  Bf16Linear &operator=(const Bf16Linear &) = delete;

  void run(const void *input, const void *row_major_weight, void *output, int rows, int input_width,
           int output_width, cudaStream_t stream);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Hopper TN FP8 tensor-core matmul with BF16 output. The row-major public contract remains
// output[M,N] = input[M,K] * weight[N,K]^T.
enum class Fp8Scaling { tensor, channel, block_128 };

class Fp8Linear {
public:
  explicit Fp8Linear(Fp8Scaling scaling = Fp8Scaling::tensor,
                     std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL);
  ~Fp8Linear();
  Fp8Linear(const Fp8Linear &) = delete;
  Fp8Linear &operator=(const Fp8Linear &) = delete;

  void run(const void *fp8_input, const float *input_scale, const void *fp8_row_major_weight,
           const float *weight_scale, void *bf16_output, int rows, int input_width,
           int output_width, cudaStream_t stream);

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Hopper W4A8 mixed-input linear for Gemma decode projections. Weights are quantized in
// K-local groups, encoded as signed INT4, and reordered once for register-source WGMMA.
// Output is intentionally left without the positive tensor-wide FP8 input/weight scale product.
// Greedy argmax is invariant to that factor for the LM head; intermediate projections must fuse
// or otherwise apply it before this primitive can be selected by the model runner.
class Int4Fp8Linear {
public:
  Int4Fp8Linear(const void *fp8_row_major_weight, int input_width, int output_width,
                int group_width = 256, int tile_columns = 256, int scale_refinement_iterations = 0);
  ~Int4Fp8Linear();
  Int4Fp8Linear(const Int4Fp8Linear &) = delete;
  Int4Fp8Linear &operator=(const Int4Fp8Linear &) = delete;

  [[nodiscard]] bool supports_rows(int rows) const;
  void run(const void *fp8_input, void *bf16_output, int rows, cudaStream_t stream,
           int output_row_stride = 0);
  [[nodiscard]] std::size_t allocated_bytes() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Hopper W4A16 mixed-input linear. Unlike the FP8-input variant, weights are quantized directly
// from BF16 and group scales remain BF16, avoiding the E4M3 lookup precision/range boundary. This
// is intended for the bandwidth-bound Gemma LM head, where activation bytes are negligible next
// to the tied embedding stream.
class Int4Bf16Linear {
public:
  Int4Bf16Linear(const void *row_major_weight, int input_width, int output_width,
                 int group_width = 256, bool source_is_fp8 = false);
  ~Int4Bf16Linear();
  Int4Bf16Linear(const Int4Bf16Linear &) = delete;
  Int4Bf16Linear &operator=(const Int4Bf16Linear &) = delete;

  [[nodiscard]] bool supports_rows(int rows) const;
  void run(const void *bf16_input, void *bf16_output, int rows, cudaStream_t stream,
           int output_row_stride = 0);
  [[nodiscard]] std::size_t allocated_bytes() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

// Computes a tensor amax, stores scale=amax/448, and emits E4M3 bytes. Scale remains device-side.
void quantize_bf16_to_fp8_e4m3(const void *input, void *output, float *scale, std::size_t elements,
                               cudaStream_t stream, float scale_multiplier = 1.0F);
// Converts a row-major BF16 tensor using maxima emitted by its producer. Every CTA derives the
// same tensor scale from the small row-amax vector, avoiding a reread pass and a grid barrier.
void quantize_bf16_to_fp8_e4m3_from_row_amax(const void *input, void *output, float *scale,
                                             const float *row_amax, int rows, int width,
                                             cudaStream_t stream, float scale_multiplier = 1.0F);
// General producer-metadata form. The values may be one maximum per row or multiple
// row-local block maxima; every conversion CTA derives the same tensor scale from them.
void quantize_bf16_to_fp8_e4m3_from_amax_values(const void *input, void *output, float *scale,
                                                const float *amax_values, int amax_count, int rows,
                                                int width, cudaStream_t stream,
                                                float scale_multiplier = 1.0F);
void quantize_bf16_rows_to_fp8_e4m3(const void *input, void *output, float *scales, int rows,
                                    int width, cudaStream_t stream);

[[nodiscard]] std::size_t fp8_weight_block_scale_elements(int output_width, int input_width);
[[nodiscard]] std::size_t fp8_activation_block_scale_elements(int rows, int input_width);
// Weights use one scale per 128x128 tile; activations use one per row and 128 K values. These
// layouts are the native Hopper cuBLASLt A=BLK128x128/B=VEC128 contract.
void quantize_bf16_weight_to_fp8_block_128(const void *input, void *output, float *scales,
                                           int output_width, int input_width, cudaStream_t stream);
void quantize_bf16_activation_to_fp8_block_128(const void *input, void *output, float *scales,
                                               int rows, int input_width, cudaStream_t stream);

// Research oracle for a packed signed-INT4 weight representation. Each independent K block is
// reduced to the signed INT4 [-8, 7] codebook and reconstructed in place as E4M3. The existing FP8
// GEMM then isolates model-quality impact without claiming the reconstruction is a fast path.
void roundtrip_fp8_weight_via_int4_blocks(void *fp8_weight, int output_width, int input_width,
                                          int block_width, cudaStream_t stream,
                                          int scale_refinement_iterations = 0);

} // namespace carat
