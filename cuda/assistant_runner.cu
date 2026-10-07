#include "carat/assistant_runner.h"

#include "carat/cuda_ops.h"
#include "carat/device_weights.h"
#include "carat/gemm_attention.h"
#include "carat/linear.h"
#include "carat/qkv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace carat {
namespace {

constexpr int target_hidden = 5376;
constexpr int assistant_hidden = 1024;
constexpr int intermediate = 8192;
constexpr int vocabulary = 262144;
constexpr int maximum_draft_depth = 8;
constexpr float epsilon = 1.0e-6F;

enum class AssistantFp8Mode { off, lm_head, all };

AssistantFp8Mode assistant_fp8_mode(const Fp8WeightArena *weights) {
  const char *configured = std::getenv("CARAT_ASSISTANT_FP8_MODE");
  const std::string mode = configured == nullptr ? "off" : configured;
  if (mode == "off")
    return AssistantFp8Mode::off;
  if (weights == nullptr) {
    throw std::runtime_error("assistant FP8 mode requires quantized assistant weights");
  }
  if (mode == "lm_head")
    return AssistantFp8Mode::lm_head;
  if (mode == "all")
    return AssistantFp8Mode::all;
  throw std::runtime_error("CARAT_ASSISTANT_FP8_MODE must be off, lm_head or all");
}

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

__global__ void offset_positions_kernel(const int *context_positions, int *query_positions,
                                        int active, int offset) {
  const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < active)
    query_positions[row] = context_positions[row] + offset;
}

class Allocation {
public:
  explicit Allocation(std::uint64_t bytes) {
    check(cudaMalloc(&pointer_, static_cast<std::size_t>(bytes)), "allocate assistant workspace");
  }
  Allocation(const Allocation &) = delete;
  Allocation &operator=(const Allocation &) = delete;

  ~Allocation() {
    cudaFree(pointer_);
  }
  void *get() {
    return pointer_;
  }

private:
  void *pointer_{nullptr};
};

} // namespace

struct Gemma4AssistantRunner::Implementation {
  const DeviceWeightArena &weights;
  const Fp8WeightArena *fp8_weights;
  AssistantFp8Mode fp8_mode;
  int maximum_batch;
  int maximum_context;
  Bf16Linear linear;
  std::unique_ptr<Fp8Linear> fp8_linear;
  Gemma4QkvPostprocessor qkv;
  GemmGroupedDecodeAttention attention;
  Allocation target_embeddings;
  Allocation previous_hidden;
  Allocation concatenated;
  Allocation hidden_a;
  Allocation hidden_b;
  Allocation normalized;
  Allocation projected_queries;
  Allocation queries;
  Allocation attention_output;
  Allocation projected_attention;
  Allocation first_residual;
  Allocation pre_mlp;
  Allocation gate_up;
  Allocation activated;
  Allocation down;
  Allocation final_hidden;
  Allocation logits;
  Allocation device_tokens;
  Allocation device_outputs;
  Allocation device_forced_tokens;
  Allocation device_proposals;
  Allocation device_context_positions;
  Allocation device_query_positions;
  Allocation device_slots;
  Allocation argmax_workspace;
  Allocation fp8_input;
  Allocation fp8_scale;

  Implementation(const DeviceWeightArena &assistant_weights,
                 const Fp8WeightArena *assistant_fp8_weights, int batch, int context)
      : weights(assistant_weights), fp8_weights(assistant_fp8_weights),
        fp8_mode(assistant_fp8_mode(assistant_fp8_weights)), maximum_batch(batch),
        maximum_context(context), linear(),
        fp8_linear(fp8_mode == AssistantFp8Mode::off
                       ? nullptr
                       : std::make_unique<Fp8Linear>(Fp8Scaling::tensor)),
        qkv(context), attention(batch, 16, 8, context),
        target_embeddings(2ULL * batch * target_hidden),
        previous_hidden(2ULL * batch * target_hidden), concatenated(4ULL * batch * target_hidden),
        hidden_a(2ULL * batch * assistant_hidden), hidden_b(2ULL * batch * assistant_hidden),
        normalized(2ULL * batch * assistant_hidden), projected_queries(2ULL * batch * 16384),
        queries(2ULL * batch * 16384), attention_output(2ULL * batch * 16384),
        projected_attention(2ULL * batch * assistant_hidden),
        first_residual(2ULL * batch * assistant_hidden), pre_mlp(2ULL * batch * assistant_hidden),
        gate_up(4ULL * batch * intermediate), activated(2ULL * batch * intermediate),
        down(2ULL * batch * assistant_hidden), final_hidden(2ULL * batch * assistant_hidden),
        logits(2ULL * batch * vocabulary), device_tokens(sizeof(int) * batch),
        device_outputs(sizeof(int) * batch), device_forced_tokens(sizeof(int) * batch),
        device_proposals(sizeof(int) * batch * maximum_draft_depth),
        device_context_positions(sizeof(int) * batch), device_query_positions(sizeof(int) * batch),
        device_slots(sizeof(int) * batch),
        argmax_workspace(argmax_bf16_workspace_bytes(vocabulary) * batch),
        fp8_input(static_cast<std::uint64_t>(batch) * 16384), fp8_scale(sizeof(float)) {
    if (batch <= 0 || context <= 0)
      throw std::runtime_error("invalid assistant shape");
  }

  void run_linear(const void *input, const std::string &name, void *output, int rows,
                  int input_width, int output_width) {
    const bool use_fp8 = fp8_mode == AssistantFp8Mode::all ||
                         (fp8_mode == AssistantFp8Mode::lm_head && name == "token_embedding");
    if (!use_fp8) {
      linear.run(input, weights.at(name), output, rows, input_width, output_width, nullptr);
      return;
    }
    quantize_bf16_to_fp8_e4m3(input, fp8_input.get(), static_cast<float *>(fp8_scale.get()),
                              static_cast<std::size_t>(rows) * input_width, nullptr);
    fp8_linear->run(fp8_input.get(), static_cast<const float *>(fp8_scale.get()),
                    fp8_weights->at(name), fp8_weights->scale(name), output, rows, input_width,
                    output_width, nullptr);
  }

  void run_layer(int layer, const void *input, void *output, int active,
                 const int *context_positions, const int *query_positions, const int *slots,
                 int maximum_slots, const void *keys, const void *values, int cache_capacity,
                 int maximum_context_length, int global_split, int first_global_context_length) {
    const bool global = layer == 3;
    const int head_dimension = global ? 512 : 256;
    const int kv_heads = global ? 4 : 16;
    const int query_group = 32 / kv_heads;
    const int query_width = 32 * head_dimension;
    const std::string prefix = "layer." + std::to_string(layer) + ".";
    rms_norm_bf16(input, weights.at(prefix + "input_norm"), normalized.get(), active,
                  assistant_hidden, epsilon, nullptr);
    run_linear(normalized.get(), prefix + "q", projected_queries.get(), active, assistant_hidden,
               query_width);
    qkv.run_query_ragged(projected_queries.get(), weights.at(prefix + "q_norm"), queries.get(),
                         query_positions, active, 32, head_dimension, global, epsilon, nullptr);
    if (global && global_split > 0 && global_split < active) {
      attention.run_ragged(queries.get(), keys, values, attention_output.get(), context_positions,
                           slots, global_split, maximum_slots, kv_heads, query_group,
                           first_global_context_length, head_dimension, nullptr, cache_capacity);
      const std::size_t row_bytes = 2ULL * query_width;
      attention.run_ragged(
          static_cast<const unsigned char *>(queries.get()) + row_bytes * global_split, keys,
          values, static_cast<unsigned char *>(attention_output.get()) + row_bytes * global_split,
          context_positions + global_split, slots + global_split, active - global_split,
          maximum_slots, kv_heads, query_group, maximum_context_length, head_dimension, nullptr,
          cache_capacity);
    } else {
      attention.run_ragged(queries.get(), keys, values, attention_output.get(), context_positions,
                           slots, active, maximum_slots, kv_heads, query_group,
                           maximum_context_length, head_dimension, nullptr, cache_capacity);
    }
    run_linear(attention_output.get(), prefix + "o", projected_attention.get(), active, query_width,
               assistant_hidden);
    rms_norm_add_and_norm_bf16(projected_attention.get(),
                               weights.at(prefix + "post_attention_norm"), input,
                               weights.at(prefix + "pre_mlp_norm"), first_residual.get(),
                               pre_mlp.get(), active, assistant_hidden, epsilon, nullptr);
    run_linear(pre_mlp.get(), prefix + "gate_up", gate_up.get(), active, assistant_hidden,
               2 * intermediate);
    gelu_tanh_gate_rows_bf16(gate_up.get(), activated.get(), active, intermediate, nullptr);
    run_linear(activated.get(), prefix + "down", down.get(), active, intermediate,
               assistant_hidden);
    rms_norm_add_scale_bf16(down.get(), weights.at(prefix + "post_mlp_norm"), first_residual.get(),
                            weights.at(prefix + "scalar"), output, active, assistant_hidden,
                            epsilon, nullptr);
  }
};

Gemma4AssistantRunner::Gemma4AssistantRunner(const DeviceWeightArena &weights,
                                             const Fp8WeightArena *fp8_weights, int maximum_batch,
                                             int maximum_context)
    : implementation_(
          std::make_unique<Implementation>(weights, fp8_weights, maximum_batch, maximum_context)) {}
Gemma4AssistantRunner::~Gemma4AssistantRunner() = default;

std::vector<std::vector<int>> Gemma4AssistantRunner::draft_four(
    const void *target_embedding_weight, const void *target_hidden_by_slot,
    const std::vector<int> &last_tokens, const std::vector<int> &context_positions,
    const std::vector<int> &slots, const std::vector<int> &forced_first_tokens, int maximum_slots,
    int depth, const void *sliding_keys, const void *sliding_values, int sliding_capacity,
    const void *global_keys, const void *global_values, int global_capacity) {
  auto &state = *implementation_;
  const int active = static_cast<int>(last_tokens.size());
  if (active <= 0 || active > state.maximum_batch ||
      context_positions.size() != last_tokens.size() || slots.size() != last_tokens.size() ||
      maximum_slots < active || target_embedding_weight == nullptr ||
      target_hidden_by_slot == nullptr || sliding_keys == nullptr || sliding_values == nullptr ||
      global_keys == nullptr || global_values == nullptr || depth <= 0 ||
      depth > maximum_draft_depth) {
    throw std::runtime_error("invalid assistant draft batch");
  }
  if (!forced_first_tokens.empty() && forced_first_tokens.size() != last_tokens.size()) {
    throw std::runtime_error("forced assistant first tokens do not match draft batch");
  }
  int maximum_global_context = 0;
  for (int row = 0; row < active; ++row) {
    if (last_tokens[static_cast<std::size_t>(row)] < 0 ||
        last_tokens[static_cast<std::size_t>(row)] >= vocabulary ||
        context_positions[static_cast<std::size_t>(row)] < 0 ||
        context_positions[static_cast<std::size_t>(row)] >= global_capacity ||
        slots[static_cast<std::size_t>(row)] < 0 ||
        slots[static_cast<std::size_t>(row)] >= maximum_slots) {
      throw std::runtime_error("invalid assistant row metadata");
    }
    maximum_global_context =
        std::max(maximum_global_context, context_positions[static_cast<std::size_t>(row)] + 1);
  }
  int global_split = 0;
  int first_global_context = maximum_global_context;
  if (active >= 4 && std::is_sorted(context_positions.begin(), context_positions.end()) &&
      context_positions.back() - context_positions.front() > 512) {
    global_split = (active + 1) / 2;
    first_global_context = context_positions[static_cast<std::size_t>(global_split - 1)] + 1;
  }
  check(cudaMemcpy(state.device_tokens.get(), last_tokens.data(), active * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy assistant input tokens");
  check(cudaMemcpy(state.device_context_positions.get(), context_positions.data(),
                   active * sizeof(int), cudaMemcpyHostToDevice),
        "copy assistant context positions");
  check(cudaMemcpy(state.device_slots.get(), slots.data(), active * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy assistant slots");
  if (!forced_first_tokens.empty()) {
    check(cudaMemcpy(state.device_forced_tokens.get(), forced_first_tokens.data(),
                     active * sizeof(int), cudaMemcpyHostToDevice),
          "copy forced assistant first tokens");
  }
  gather_rows_bf16(target_hidden_by_slot, static_cast<const int *>(state.device_slots.get()),
                   state.previous_hidden.get(), active, target_hidden, nullptr);

  std::vector<std::vector<int>> proposals(static_cast<std::size_t>(active),
                                          std::vector<int>(static_cast<std::size_t>(depth)));
  for (int draft_index = 0; draft_index < depth; ++draft_index) {
    const void *input_tokens =
        draft_index == 0 ? state.device_tokens.get() : state.device_outputs.get();
    embedding_batch_bf16(target_embedding_weight, static_cast<const int *>(input_tokens),
                         state.target_embeddings.get(), active, vocabulary, target_hidden,
                         std::sqrt(static_cast<float>(target_hidden)), nullptr);
    concatenate_rows_bf16(state.target_embeddings.get(), target_hidden, state.previous_hidden.get(),
                          target_hidden, state.concatenated.get(), active, nullptr);
    state.run_linear(state.concatenated.get(), "pre_projection", state.hidden_a.get(), active,
                     2 * target_hidden, assistant_hidden);
    offset_positions_kernel<<<(active + 255) / 256, 256>>>(
        static_cast<const int *>(state.device_context_positions.get()),
        static_cast<int *>(state.device_query_positions.get()), active, draft_index);
    check(cudaPeekAtLastError(), "compute assistant query positions");
    void *input = state.hidden_a.get();
    void *output = state.hidden_b.get();
    for (int layer = 0; layer < 4; ++layer) {
      const bool global = layer == 3;
      state.run_layer(layer, input, output, active,
                      static_cast<const int *>(state.device_context_positions.get()),
                      static_cast<const int *>(state.device_query_positions.get()),
                      static_cast<const int *>(state.device_slots.get()), maximum_slots,
                      global ? global_keys : sliding_keys, global ? global_values : sliding_values,
                      global ? global_capacity : sliding_capacity,
                      global ? maximum_global_context : sliding_capacity, global_split,
                      first_global_context);
      std::swap(input, output);
    }
    rms_norm_bf16(input, state.weights.at("final_norm"), state.final_hidden.get(), active,
                  assistant_hidden, epsilon, nullptr);
    const bool forced_target_token = draft_index == 0 && !forced_first_tokens.empty();
    if (forced_target_token) {
      // The target has already produced this token. It seeds the recurrent second proposal,
      // so evaluating and then overwriting the assistant's 262k-way LM head is dead work.
      check(cudaMemcpyAsync(state.device_outputs.get(), state.device_forced_tokens.get(),
                            active * sizeof(int), cudaMemcpyDeviceToDevice),
            "select forced assistant first tokens");
    } else {
      state.run_linear(state.final_hidden.get(), "token_embedding", state.logits.get(), active,
                       assistant_hidden, vocabulary);
      argmax_bf16_rows(state.logits.get(), active, vocabulary,
                       static_cast<int *>(state.device_outputs.get()), state.argmax_workspace.get(),
                       nullptr);
    }
    state.run_linear(state.final_hidden.get(), "post_projection", state.previous_hidden.get(),
                     active, assistant_hidden, target_hidden);
    check(cudaMemcpyAsync(static_cast<int *>(state.device_proposals.get()) + draft_index * active,
                          state.device_outputs.get(), active * sizeof(int),
                          cudaMemcpyDeviceToDevice),
          "store device assistant proposals");
  }
  std::vector<int> depth_major(static_cast<std::size_t>(active * depth));
  check(cudaMemcpy(depth_major.data(), state.device_proposals.get(),
                   depth_major.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "copy assistant proposals");
  for (int draft_index = 0; draft_index < depth; ++draft_index) {
    for (int row = 0; row < active; ++row) {
      proposals[static_cast<std::size_t>(row)][static_cast<std::size_t>(draft_index)] =
          depth_major[static_cast<std::size_t>(draft_index * active + row)];
    }
  }
  return proposals;
}

} // namespace carat
