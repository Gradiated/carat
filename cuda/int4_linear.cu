#include "carat/linear.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
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

using ElementInput = cutlass::float_e4m3_t;
using ElementWeight = cutlass::int4b_t;
using ElementScale = cutlass::float_e4m3_t;
using PackedScale = cutlass::Array<ElementScale, 8>;
using ElementOutput = cutlass::bfloat16_t;
using LayoutInput = cutlass::layout::RowMajor;
using LayoutWeight = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using StrideInput = cutlass::detail::TagToStrideA_t<LayoutInput>;
using StrideWeight = cutlass::detail::TagToStrideB_t<LayoutWeight>;
using LayoutAtomWeight = decltype(cutlass::compute_memory_reordering_atom<ElementInput>());
using ReorderedWeightLayout = decltype(cute::tile_to_shape(
    LayoutAtomWeight{}, cute::Layout<cute::Shape<int, int, int>, StrideWeight>{}));

constexpr int alignment_input = 128 / cutlass::sizeof_bits<ElementInput>::value;
constexpr int alignment_weight = 128 / cutlass::sizeof_bits<ElementWeight>::value;
constexpr int alignment_output = 128 / cutlass::sizeof_bits<ElementOutput>::value;

void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void cutlass_check(cutlass::Status status, const char *operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
  }
}

__device__ float reduce_max(float value, float *warp_maxima) {
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

__device__ float reduce_sum(float value, float *warp_sums) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  if (lane == 0)
    warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < static_cast<int>(blockDim.x) / 32 ? warp_sums[lane] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0)
      warp_sums[0] = value;
  }
  __syncthreads();
  value = warp_sums[0];
  __syncthreads();
  return value;
}

__device__ std::uint8_t unified_int4_code(int value) {
  return static_cast<std::uint8_t>(value > 0 ? 8 - value : value & 0x0f);
}

__global__ void quantize_fp8_int4_lut_kernel(const __nv_fp8_e4m3 *input,
                                             std::uint8_t *row_major_output,
                                             PackedScale *packed_scales, int output_width,
                                             int input_width, int group_width, int groups,
                                             int scale_refinement_iterations) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= output_width)
    return;
  __shared__ float warp_maxima[8];
  __shared__ float optimized_scale;
  const std::size_t row_base = static_cast<std::size_t>(row) * input_width;
  const std::size_t packed_row_base = row_base / 2;
  for (int group = 0; group < groups; ++group) {
    const int begin = group * group_width;
    const int column = begin + static_cast<int>(threadIdx.x);
    float value = 0.0F;
    if (threadIdx.x < group_width && column < input_width) {
      value = static_cast<float>(input[row_base + column]);
    }
    const float maximum = reduce_max(fabsf(value), warp_maxima);
    // The unified signed-INT4 conversion always materializes an internal -8 lookup entry. E4M3
    // tops out at 448, so scale=max/7 is invalid for blocks whose FP8 maximum exceeds 392: the
    // otherwise-unused lookup entry overflows and poisons the tensor-core conversion. Cap the
    // scale at 448/8 and use the available -8 code for negative outliers.
    if (threadIdx.x == 0) {
      optimized_scale = maximum > 0.0F ? fminf(maximum / 7.0F, 56.0F) : 1.0F;
    }
    __syncthreads();
    for (int iteration = 0; iteration < scale_refinement_iterations; ++iteration) {
      const int quantized = max(-8, min(7, __float2int_rn(value / optimized_scale)));
      const float numerator = reduce_sum(value * static_cast<float>(quantized), warp_maxima);
      const float denominator = reduce_sum(static_cast<float>(quantized * quantized), warp_maxima);
      if (threadIdx.x == 0 && denominator > 0.0F) {
        optimized_scale = fminf(numerator / denominator, 56.0F);
      }
      __syncthreads();
    }
    const float scale = optimized_scale;
    if (threadIdx.x == 0) {
      PackedScale lookup;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        lookup[index] = ElementScale(scale * static_cast<float>(index - 8));
      }
      // StrideScale is [1, output_width, output_width * groups] after the mixed-input
      // operand swap: rows are contiguous inside each K-group plane.
      packed_scales[static_cast<std::size_t>(group) * output_width + row] = lookup;
    }
    if (threadIdx.x < static_cast<unsigned>(group_width / 2)) {
      const int first_column = begin + static_cast<int>(threadIdx.x) * 2;
      const int second_column = first_column + 1;
      const float first =
          first_column < input_width ? static_cast<float>(input[row_base + first_column]) : 0.0F;
      const float second =
          second_column < input_width ? static_cast<float>(input[row_base + second_column]) : 0.0F;
      const int first_quantized = max(-8, min(7, __float2int_rn(first / scale)));
      const int second_quantized = max(-8, min(7, __float2int_rn(second / scale)));
      row_major_output[packed_row_base + begin / 2 + threadIdx.x] =
          unified_int4_code(first_quantized) |
          static_cast<std::uint8_t>(unified_int4_code(second_quantized) << 4U);
    }
    __syncthreads();
  }
}

template <int TileColumns, int TileRows> struct KernelConfiguration {
  using TileShape = cute::Shape<cute::Int<TileColumns>, cute::Int<TileRows>, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  // CUTLASS requires at least 128 M columns for the cooperative two-warpgroup schedule. A
  // 64-column tile is still useful for Gemma's 5,376-wide projections, where the cooperative
  // 256-column tile exposes only 21 CTAs on 132 SMs, so use the single-warpgroup schedule there.
  using KernelSchedule =
      std::conditional_t<TileColumns == 64, cutlass::gemm::KernelTmaWarpSpecialized,
                         cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementOutput,
      typename cutlass::layout::LayoutTranspose<LayoutOutput>::type, alignment_output,
      ElementOutput, typename cutlass::layout::LayoutTranspose<LayoutOutput>::type,
      alignment_output, cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cute::tuple<ElementWeight, PackedScale>,
      ReorderedWeightLayout, alignment_weight, ElementInput,
      typename cutlass::layout::LayoutTranspose<LayoutInput>::type, alignment_input, float,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

} // namespace

struct Int4Fp8Linear::Implementation {
  struct PlanBase {
    virtual ~PlanBase() = default;
    virtual void run(cudaStream_t stream) = 0;
    const void *input{};
    void *output{};
    int output_row_stride{};
  };

  template <int TileColumns, int TileRows> struct Plan final : PlanBase {
    using Configuration = KernelConfiguration<TileColumns, TileRows>;
    using Gemm = typename Configuration::Gemm;
    Gemm gemm;
    void *workspace{};
    std::size_t workspace_bytes{};

    Plan(Implementation &state, const void *input, void *output, int rows, int output_row_stride) {
      this->input = input;
      this->output = output;
      this->output_row_stride = output_row_stride;
      const auto stride_input = cutlass::make_cute_packed_stride(
          StrideInput{}, cute::make_shape(rows, state.input_width, 1));
      using StrideOutput = typename Configuration::Kernel::StrideC;
      const StrideOutput stride_output{cute::Int<1>{}, static_cast<std::int64_t>(output_row_stride),
                                       static_cast<std::int64_t>(output_row_stride) * rows};
      typename Gemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {state.output_width, rows, state.input_width, 1},
          {static_cast<const ElementWeight *>(state.packed_weight), state.reordered_layout,
           static_cast<const ElementInput *>(input), stride_input,
           static_cast<const PackedScale *>(state.packed_scales), state.scale_stride,
           state.group_width},
          {{1.0F, 0.0F},
           static_cast<const ElementOutput *>(output),
           stride_output,
           static_cast<ElementOutput *>(output),
           stride_output},
          {0, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0)},
          {}};
      workspace_bytes = Gemm::get_workspace_size(arguments);
      if (workspace_bytes != 0) {
        cuda_check(cudaMalloc(&workspace, workspace_bytes), "allocate INT4 linear workspace");
      }
      cutlass_check(gemm.can_implement(arguments), "validate INT4 linear plan");
      cutlass_check(gemm.initialize(arguments, workspace), "initialize INT4 linear plan");
    }

    ~Plan() override {
      if (workspace != nullptr)
        cudaFree(workspace);
    }

    void run(cudaStream_t stream) override {
      cutlass_check(gemm.run(stream), "run INT4 linear");
    }
  };

  int input_width;
  int output_width;
  int group_width;
  int groups;
  int tile_columns;
  int scale_refinement_iterations;
  std::size_t weight_bytes;
  std::size_t scale_bytes;
  void *packed_weight{};
  void *packed_scales{};
  ReorderedWeightLayout reordered_layout;
  using ScaleStride = typename KernelConfiguration<256, 16>::Mainloop::StrideScale;
  ScaleStride scale_stride;
  std::map<int, std::unique_ptr<PlanBase>> plans;

  Implementation(const void *fp8_weight, int requested_input_width, int requested_output_width,
                 int requested_group_width, int requested_tile_columns,
                 int requested_scale_refinement_iterations)
      : input_width(requested_input_width), output_width(requested_output_width),
        group_width(requested_group_width), groups((input_width + group_width - 1) / group_width),
        tile_columns(requested_tile_columns),
        scale_refinement_iterations(requested_scale_refinement_iterations),
        weight_bytes(static_cast<std::size_t>(output_width) * input_width / 2),
        scale_bytes(static_cast<std::size_t>(output_width) * groups * sizeof(PackedScale)),
        reordered_layout(cute::tile_to_shape(LayoutAtomWeight{},
                                             cute::make_shape(output_width, input_width, 1))),
        scale_stride(cutlass::make_cute_packed_stride(ScaleStride{},
                                                      cute::make_shape(output_width, groups, 1))) {
    if (fp8_weight == nullptr || input_width <= 0 || output_width <= 0 ||
        (group_width != 128 && group_width != 256) || input_width % group_width != 0 ||
        input_width % 2 != 0 || output_width % 256 != 0 ||
        (tile_columns != 64 && tile_columns != 128 && tile_columns != 256) ||
        scale_refinement_iterations < 0 || scale_refinement_iterations > 8) {
      throw std::runtime_error("invalid INT4 FP8 linear weight shape");
    }
    void *row_major_weight = nullptr;
    void *reordered_staging = nullptr;
    cuda_check(cudaMalloc(&row_major_weight, weight_bytes),
               "allocate row-major INT4 staging weight");
    try {
      // CUTLASS 4.6's Hopper mixed-input shuffle uses a byte-addressed sparse layout for logical
      // INT4 elements. Keep its complete one-byte-per-element extent; compacting the address span
      // corrupts K>256 shapes even though small diagonal probes can appear correct.
      cuda_check(cudaMalloc(&reordered_staging, weight_bytes * 2),
                 "allocate reordered INT4 staging weight");
      cuda_check(cudaMalloc(&packed_scales, scale_bytes), "allocate INT4 lookup scales");
      quantize_fp8_int4_lut_kernel<<<output_width, 256>>>(
          static_cast<const __nv_fp8_e4m3 *>(fp8_weight),
          static_cast<std::uint8_t *>(row_major_weight), static_cast<PackedScale *>(packed_scales),
          output_width, input_width, group_width, groups, scale_refinement_iterations);
      cuda_check(cudaPeekAtLastError(), "launch INT4 LM-head quantization");
      cuda_check(cudaDeviceSynchronize(), "finish INT4 LM-head quantization");
      const auto source_stride = cutlass::make_cute_packed_stride(
          StrideWeight{}, cute::make_shape(output_width, input_width, 1));
      const auto source_layout =
          cute::make_layout(cute::make_shape(output_width, input_width, 1), source_stride);
      cutlass::reorder_tensor(static_cast<const ElementWeight *>(row_major_weight), source_layout,
                              static_cast<ElementWeight *>(reordered_staging), reordered_layout);
      packed_weight = reordered_staging;
      reordered_staging = nullptr;
    } catch (...) {
      cudaFree(row_major_weight);
      if (reordered_staging != nullptr)
        cudaFree(reordered_staging);
      if (packed_scales != nullptr)
        cudaFree(packed_scales);
      if (packed_weight != nullptr)
        cudaFree(packed_weight);
      packed_scales = nullptr;
      packed_weight = nullptr;
      throw;
    }
    cudaFree(row_major_weight);
    cudaFree(reordered_staging);
  }

  ~Implementation() {
    plans.clear();
    if (packed_scales != nullptr)
      cudaFree(packed_scales);
    if (packed_weight != nullptr)
      cudaFree(packed_weight);
  }

  PlanBase &plan(const void *input, void *output, int rows, int output_row_stride) {
    const auto found = plans.find(rows);
    if (found != plans.end()) {
      if (found->second->input != input || found->second->output != output ||
          found->second->output_row_stride != output_row_stride) {
        throw std::runtime_error("INT4 linear buffers changed for a cached row shape");
      }
      return *found->second;
    }
    const auto create = [&]<int TileRows>() -> std::unique_ptr<PlanBase> {
      if (tile_columns == 64) {
        return std::make_unique<Plan<64, TileRows>>(*this, input, output, rows, output_row_stride);
      }
      if (tile_columns == 128) {
        return std::make_unique<Plan<128, TileRows>>(*this, input, output, rows, output_row_stride);
      }
      return std::make_unique<Plan<256, TileRows>>(*this, input, output, rows, output_row_stride);
    };
    std::unique_ptr<PlanBase> created;
    if (rows <= 16) {
      created = create.template operator()<16>();
    } else if (rows <= 32) {
      created = create.template operator()<32>();
    } else if (rows <= 64) {
      created = create.template operator()<64>();
    } else {
      created = create.template operator()<128>();
    }
    PlanBase &reference = *created;
    plans.emplace(rows, std::move(created));
    return reference;
  }
};

Int4Fp8Linear::Int4Fp8Linear(const void *fp8_row_major_weight, int input_width, int output_width,
                             int group_width, int tile_columns, int scale_refinement_iterations)
    : implementation_(std::make_unique<Implementation>(fp8_row_major_weight, input_width,
                                                       output_width, group_width, tile_columns,
                                                       scale_refinement_iterations)) {}
Int4Fp8Linear::~Int4Fp8Linear() = default;

bool Int4Fp8Linear::supports_rows(int rows) const {
  return rows > 0 && rows <= 128;
}

void Int4Fp8Linear::run(const void *fp8_input, void *bf16_output, int rows, cudaStream_t stream,
                        int output_row_stride) {
  if (fp8_input == nullptr || bf16_output == nullptr || !supports_rows(rows)) {
    throw std::runtime_error("invalid INT4 FP8 linear activation shape");
  }
  const int selected_stride =
      output_row_stride == 0 ? implementation_->output_width : output_row_stride;
  if (selected_stride < implementation_->output_width) {
    throw std::runtime_error("INT4 FP8 linear output stride is too small");
  }
  implementation_->plan(fp8_input, bf16_output, rows, selected_stride).run(stream);
}

std::size_t Int4Fp8Linear::allocated_bytes() const {
  return implementation_->weight_bytes * 2 + implementation_->scale_bytes;
}

} // namespace carat
