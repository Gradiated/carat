#include "carat/linear.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/mixed_dtype_utils.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carat {
namespace {

using Bf16ElementInput = cutlass::bfloat16_t;
using Bf16ElementWeight = cutlass::int4b_t;
using Bf16ElementScale = cutlass::bfloat16_t;
using Bf16ElementOutput = cutlass::bfloat16_t;
using Bf16LayoutInput = cutlass::layout::RowMajor;
using Bf16LayoutWeight = cutlass::layout::ColumnMajor;
using Bf16LayoutOutput = cutlass::layout::RowMajor;
using Bf16StrideInput = cutlass::detail::TagToStrideA_t<Bf16LayoutInput>;
using Bf16StrideWeight = cutlass::detail::TagToStrideB_t<Bf16LayoutWeight>;
// Establish correctness with the unpermuted MMA value order first. CUTLASS's optional BF16
// ValueShuffle is a later tuning step and requires an explicit roundtrip test for this transposed
// output-major geometry.
using Bf16LayoutAtomWeight = decltype(cutlass::compute_memory_reordering_atom<Bf16ElementInput>());
using Bf16ReorderedWeightLayout = decltype(cute::tile_to_shape(
    Bf16LayoutAtomWeight{}, cute::Layout<cute::Shape<int, int, int>, Bf16StrideWeight>{}));

constexpr int bf16_alignment_input = 128 / cutlass::sizeof_bits<Bf16ElementInput>::value;
constexpr int bf16_alignment_weight = 128 / cutlass::sizeof_bits<Bf16ElementWeight>::value;
constexpr int bf16_alignment_output = 128 / cutlass::sizeof_bits<Bf16ElementOutput>::value;

void bf16_int4_cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void bf16_int4_cutlass_check(cutlass::Status status, const char *operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
  }
}

__device__ float bf16_int4_reduce_max(float value, float *warp_maxima) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  if (lane == 0)
    warp_maxima[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < static_cast<int>(blockDim.x) / 32 ? warp_maxima[lane] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    if (lane == 0)
      warp_maxima[0] = value;
  }
  __syncthreads();
  return warp_maxima[0];
}

__device__ std::uint8_t bf16_int4_code(int value) {
  return static_cast<std::uint8_t>(value & 0x0f);
}

template <typename Source> __device__ float bf16_int4_source_value(const Source &value) {
  if constexpr (std::is_same_v<Source, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename Source>
__global__ void quantize_bf16_int4_kernel(const Source *input, std::uint8_t *row_major_output,
                                          Bf16ElementScale *scales, int output_width,
                                          int input_width, int group_width, int groups) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= output_width)
    return;
  __shared__ float warp_maxima[8];
  const std::size_t row_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t packed_row_base = row_base / 2;
  for (int group = 0; group < groups; ++group) {
    const int begin = group * group_width;
    const int column = begin + static_cast<int>(threadIdx.x);
    float value = 0.0F;
    if (threadIdx.x < group_width && column < input_width) {
      value = bf16_int4_source_value(input[row_base + column]);
    }
    const float maximum = bf16_int4_reduce_max(fabsf(value), warp_maxima);
    const Bf16ElementScale encoded_scale(maximum > 0.0F ? maximum / 7.0F : 1.0F);
    const float scale = static_cast<float>(encoded_scale);
    if (threadIdx.x == 0) {
      // Hopper mixed-input StrideScale is group-major after swapping/transposing operands.
      scales[static_cast<std::size_t>(group) * output_width + row] = encoded_scale;
    }
    if (threadIdx.x < static_cast<unsigned>(group_width / 2)) {
      const int first_column = begin + static_cast<int>(threadIdx.x) * 2;
      const int second_column = first_column + 1;
      const float first = first_column < input_width
                              ? bf16_int4_source_value(input[row_base + first_column])
                              : 0.0F;
      const float second = second_column < input_width
                               ? bf16_int4_source_value(input[row_base + second_column])
                               : 0.0F;
      const int first_quantized = max(-7, min(7, __float2int_rn(first / scale)));
      const int second_quantized = max(-7, min(7, __float2int_rn(second / scale)));
      row_major_output[packed_row_base + begin / 2 + threadIdx.x] =
          bf16_int4_code(first_quantized) |
          static_cast<std::uint8_t>(bf16_int4_code(second_quantized) << 4U);
    }
    __syncthreads();
  }
}

template <int TileRows> struct Bf16Int4KernelConfiguration {
  using TileShape = cute::Shape<cute::_256, cute::Int<TileRows>, cute::_64>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, Bf16ElementOutput,
      typename cutlass::layout::LayoutTranspose<Bf16LayoutOutput>::type, bf16_alignment_output,
      Bf16ElementOutput, typename cutlass::layout::LayoutTranspose<Bf16LayoutOutput>::type,
      bf16_alignment_output, cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cute::tuple<Bf16ElementWeight, Bf16ElementScale>, Bf16ReorderedWeightLayout,
      bf16_alignment_weight, Bf16ElementInput,
      typename cutlass::layout::LayoutTranspose<Bf16LayoutInput>::type, bf16_alignment_input, float,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

} // namespace

struct Int4Bf16Linear::Implementation {
  struct PlanBase {
    virtual ~PlanBase() = default;
    virtual void run(cudaStream_t stream) = 0;
    const void *input{};
    void *output{};
    int output_row_stride{};
  };

  template <int TileRows> struct Plan final : PlanBase {
    using Configuration = Bf16Int4KernelConfiguration<TileRows>;
    using Gemm = typename Configuration::Gemm;
    Gemm gemm;
    void *workspace{};

    Plan(Implementation &state, const void *input, void *output, int rows, int output_row_stride) {
      this->input = input;
      this->output = output;
      this->output_row_stride = output_row_stride;
      const auto stride_input = cutlass::make_cute_packed_stride(
          Bf16StrideInput{}, cute::make_shape(rows, state.input_width, 1));
      using StrideOutput = typename Configuration::Kernel::StrideC;
      const StrideOutput stride_output{cute::Int<1>{}, static_cast<std::int64_t>(output_row_stride),
                                       static_cast<std::int64_t>(output_row_stride) * rows};
      typename Gemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {state.output_width, rows, state.input_width, 1},
          {static_cast<const Bf16ElementWeight *>(state.packed_weight), state.reordered_layout,
           static_cast<const Bf16ElementInput *>(input), stride_input,
           static_cast<const Bf16ElementScale *>(state.scales), state.scale_stride,
           state.group_width},
          {{1.0F, 0.0F},
           static_cast<const Bf16ElementOutput *>(output),
           stride_output,
           static_cast<Bf16ElementOutput *>(output),
           stride_output}};
      const std::size_t workspace_bytes = Gemm::get_workspace_size(arguments);
      if (workspace_bytes != 0) {
        bf16_int4_cuda_check(cudaMalloc(&workspace, workspace_bytes),
                             "allocate BF16 INT4 linear workspace");
      }
      bf16_int4_cutlass_check(gemm.can_implement(arguments), "validate BF16 INT4 linear plan");
      bf16_int4_cutlass_check(gemm.initialize(arguments, workspace),
                              "initialize BF16 INT4 linear plan");
    }

    ~Plan() override {
      if (workspace != nullptr)
        cudaFree(workspace);
    }

    void run(cudaStream_t stream) override {
      bf16_int4_cutlass_check(gemm.run(stream), "run BF16 INT4 linear");
    }
  };

  int input_width;
  int output_width;
  int group_width;
  int groups;
  std::size_t weight_bytes;
  std::size_t scale_bytes;
  void *packed_weight{};
  void *scales{};
  Bf16ReorderedWeightLayout reordered_layout;
  using ScaleStride = typename Bf16Int4KernelConfiguration<16>::Mainloop::StrideScale;
  ScaleStride scale_stride;
  std::map<int, std::unique_ptr<PlanBase>> plans;

  Implementation(const void *source_weight, int requested_input_width, int requested_output_width,
                 int requested_group_width, bool source_is_fp8)
      : input_width(requested_input_width), output_width(requested_output_width),
        group_width(requested_group_width), groups((input_width + group_width - 1) / group_width),
        weight_bytes(static_cast<std::size_t>(output_width) * input_width / 2),
        scale_bytes(static_cast<std::size_t>(output_width) * groups * sizeof(Bf16ElementScale)),
        reordered_layout(cute::tile_to_shape(Bf16LayoutAtomWeight{},
                                             cute::make_shape(output_width, input_width, 1))),
        scale_stride(cutlass::make_cute_packed_stride(ScaleStride{},
                                                      cute::make_shape(output_width, groups, 1))) {
    if (source_weight == nullptr || input_width <= 0 || output_width <= 0 || group_width != 256 ||
        input_width % group_width != 0 || input_width % 2 != 0 || output_width % 256 != 0) {
      throw std::runtime_error("invalid BF16 INT4 linear weight shape");
    }
    void *row_major_weight = nullptr;
    void *reordered_staging = nullptr;
    bf16_int4_cuda_check(cudaMalloc(&row_major_weight, weight_bytes),
                         "allocate row-major BF16 INT4 staging weight");
    try {
      bf16_int4_cuda_check(cudaMalloc(&reordered_staging, weight_bytes * 2),
                           "allocate reordered BF16 INT4 weight");
      bf16_int4_cuda_check(cudaMalloc(&scales, scale_bytes), "allocate BF16 INT4 scales");
      if (source_is_fp8) {
        quantize_bf16_int4_kernel<<<output_width, 256>>>(
            static_cast<const __nv_fp8_e4m3 *>(source_weight),
            static_cast<std::uint8_t *>(row_major_weight), static_cast<Bf16ElementScale *>(scales),
            output_width, input_width, group_width, groups);
      } else {
        quantize_bf16_int4_kernel<<<output_width, 256>>>(
            static_cast<const __nv_bfloat16 *>(source_weight),
            static_cast<std::uint8_t *>(row_major_weight), static_cast<Bf16ElementScale *>(scales),
            output_width, input_width, group_width, groups);
      }
      bf16_int4_cuda_check(cudaPeekAtLastError(), "launch BF16 INT4 weight quantization");
      bf16_int4_cuda_check(cudaDeviceSynchronize(), "finish BF16 INT4 weight quantization");
      const auto source_stride = cutlass::make_cute_packed_stride(
          Bf16StrideWeight{}, cute::make_shape(output_width, input_width, 1));
      const auto source_layout =
          cute::make_layout(cute::make_shape(output_width, input_width, 1), source_stride);
      cutlass::reorder_tensor(static_cast<const Bf16ElementWeight *>(row_major_weight),
                              source_layout, static_cast<Bf16ElementWeight *>(reordered_staging),
                              reordered_layout);
      packed_weight = reordered_staging;
      reordered_staging = nullptr;
    } catch (...) {
      cudaFree(row_major_weight);
      if (reordered_staging != nullptr)
        cudaFree(reordered_staging);
      if (scales != nullptr)
        cudaFree(scales);
      if (packed_weight != nullptr)
        cudaFree(packed_weight);
      scales = nullptr;
      packed_weight = nullptr;
      throw;
    }
    cudaFree(row_major_weight);
  }

  ~Implementation() {
    plans.clear();
    if (scales != nullptr)
      cudaFree(scales);
    if (packed_weight != nullptr)
      cudaFree(packed_weight);
  }

  PlanBase &plan(const void *input, void *output, int rows, int output_row_stride) {
    const auto found = plans.find(rows);
    if (found != plans.end()) {
      if (found->second->input != input || found->second->output != output ||
          found->second->output_row_stride != output_row_stride) {
        throw std::runtime_error("BF16 INT4 linear buffers changed for a cached row shape");
      }
      return *found->second;
    }
    std::unique_ptr<PlanBase> created;
    if (rows <= 16) {
      created = std::make_unique<Plan<16>>(*this, input, output, rows, output_row_stride);
    } else if (rows <= 32) {
      created = std::make_unique<Plan<32>>(*this, input, output, rows, output_row_stride);
    } else if (rows <= 64) {
      created = std::make_unique<Plan<64>>(*this, input, output, rows, output_row_stride);
    } else {
      created = std::make_unique<Plan<128>>(*this, input, output, rows, output_row_stride);
    }
    PlanBase &reference = *created;
    plans.emplace(rows, std::move(created));
    return reference;
  }
};

Int4Bf16Linear::Int4Bf16Linear(const void *row_major_weight, int input_width, int output_width,
                               int group_width, bool source_is_fp8)
    : implementation_(std::make_unique<Implementation>(row_major_weight, input_width, output_width,
                                                       group_width, source_is_fp8)) {}
Int4Bf16Linear::~Int4Bf16Linear() = default;

bool Int4Bf16Linear::supports_rows(int rows) const {
  return rows > 0 && rows <= 128;
}

void Int4Bf16Linear::run(const void *bf16_input, void *bf16_output, int rows, cudaStream_t stream,
                         int output_row_stride) {
  if (bf16_input == nullptr || bf16_output == nullptr || !supports_rows(rows)) {
    throw std::runtime_error("invalid BF16 INT4 linear activation shape");
  }
  const int selected_stride =
      output_row_stride == 0 ? implementation_->output_width : output_row_stride;
  if (selected_stride < implementation_->output_width) {
    throw std::runtime_error("BF16 INT4 linear output stride is too small");
  }
  implementation_->plan(bf16_input, bf16_output, rows, selected_stride).run(stream);
}

std::size_t Int4Bf16Linear::allocated_bytes() const {
  return implementation_->weight_bytes * 2 + implementation_->scale_bytes;
}

} // namespace carat
