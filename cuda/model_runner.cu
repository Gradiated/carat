#include "carat/model_runner.h"

#include "carat/cuda_ops.h"
#include "carat/device_weights.h"
#include "carat/gemma4_config.h"
#include "carat/layer_runner.h"
#include "carat/linear.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace carat {
namespace {

constexpr std::uint64_t alignment = 256;

std::uint64_t align_up(std::uint64_t value) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class Allocation {
public:
  explicit Allocation(std::uint64_t bytes) {
    check(cudaMalloc(&pointer_, bytes), "allocate model workspace");
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

struct CacheLocation {
  std::uint64_t offset;
  int capacity;
};

} // namespace

struct Gemma4ModelRunner::Implementation {
  const Gemma4Config &config;
  const DeviceWeightArena &weights;
  int maximum_context;
  int current_position{0};
  int hidden;
  int vocabulary;
  std::vector<CacheLocation> caches;
  std::uint64_t cache_payload_bytes{0};
  Gemma4LayerRunner layers;
  Bf16Linear lm_head;
  Allocation cache_arena;
  Allocation hidden_a;
  Allocation hidden_b;
  Allocation final_hidden;
  Allocation logits;
  Allocation device_token;
  Allocation argmax_workspace;

  static std::uint64_t cache_payload(const Gemma4Config &config, int maximum_context,
                                     std::vector<CacheLocation> *locations) {
    std::uint64_t bytes = 0;
    for (const auto &layer : config.layers) {
      bytes = align_up(bytes);
      const int capacity = layer.attention == AttentionKind::sliding
                               ? std::min(maximum_context, static_cast<int>(config.sliding_window))
                               : maximum_context;
      locations->push_back({bytes, capacity});
      bytes += 2ULL * capacity * layer.kv_width();
    }
    return align_up(bytes);
  }

  Implementation(const Gemma4Config &model_config, const DeviceWeightArena &device_weights,
                 int max_context)
      : config(model_config), weights(device_weights), maximum_context(max_context),
        hidden(static_cast<int>(model_config.hidden_size)),
        vocabulary(static_cast<int>(model_config.vocabulary_size)), caches(),
        cache_payload_bytes(cache_payload(model_config, max_context, &caches)),
        layers(model_config, device_weights, max_context, max_context), lm_head(),
        cache_arena(2ULL * cache_payload_bytes), hidden_a(2ULL * max_context * hidden),
        hidden_b(2ULL * max_context * hidden), final_hidden(2ULL * hidden),
        logits(2ULL * vocabulary), device_token(sizeof(int) * max_context),
        argmax_workspace(argmax_bf16_workspace_bytes(vocabulary)) {
    if (max_context <= 0)
      throw std::runtime_error("invalid model context");
  }

  void *key_cache(std::size_t layer) {
    return static_cast<unsigned char *>(cache_arena.get()) + caches.at(layer).offset;
  }
  void *value_cache(std::size_t layer) {
    return static_cast<unsigned char *>(cache_arena.get()) + cache_payload_bytes +
           caches.at(layer).offset;
  }
};

Gemma4ModelRunner::Gemma4ModelRunner(const Gemma4Config &config, const DeviceWeightArena &weights,
                                     int maximum_context)
    : implementation_(std::make_unique<Implementation>(config, weights, maximum_context)) {}
Gemma4ModelRunner::~Gemma4ModelRunner() = default;

int Gemma4ModelRunner::append(int token_id) {
  auto &state = *implementation_;
  if (state.current_position >= state.maximum_context)
    throw std::runtime_error("model context is full");
  embedding_bf16(state.weights.at("token_embedding"), token_id, state.hidden_a.get(),
                 state.vocabulary, state.hidden, std::sqrt(static_cast<float>(state.hidden)),
                 nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    const auto &cache = state.caches[layer];
    state.layers.run_decode(static_cast<int>(layer), input, output, state.current_position,
                            state.key_cache(layer), state.value_cache(layer), cache.capacity,
                            nullptr);
    std::swap(input, output);
  }
  rms_norm_bf16(input, state.weights.at("final_norm"), state.final_hidden.get(), 1, state.hidden,
                static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                    state.logits.get(), 1, state.hidden, state.vocabulary, nullptr);
  argmax_bf16(state.logits.get(), state.vocabulary, static_cast<int *>(state.device_token.get()),
              state.argmax_workspace.get(), nullptr);
  int next_token = 0;
  check(
      cudaMemcpy(&next_token, state.device_token.get(), sizeof(next_token), cudaMemcpyDeviceToHost),
      "copy greedy token");
  ++state.current_position;
  return next_token;
}

int Gemma4ModelRunner::prefill(const std::vector<int> &token_ids) {
  auto &state = *implementation_;
  if (state.current_position != 0)
    throw std::runtime_error("prefill requires an empty model runner");
  if (token_ids.empty() || token_ids.size() > static_cast<std::size_t>(state.maximum_context)) {
    throw std::runtime_error("prompt does not fit the model context");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0 || token_id >= state.vocabulary) {
      throw std::runtime_error("prompt contains an invalid token id");
    }
  }
  check(cudaMemcpy(state.device_token.get(), token_ids.data(), token_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice),
        "copy prompt token ids");
  const int tokens = static_cast<int>(token_ids.size());
  embedding_batch_bf16(state.weights.at("token_embedding"),
                       static_cast<const int *>(state.device_token.get()), state.hidden_a.get(),
                       tokens, state.vocabulary, state.hidden,
                       std::sqrt(static_cast<float>(state.hidden)), nullptr);
  void *input = state.hidden_a.get();
  void *output = state.hidden_b.get();
  for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
    const auto &cache = state.caches[layer];
    state.layers.run_prefill(static_cast<int>(layer), input, output, tokens, state.key_cache(layer),
                             state.value_cache(layer), cache.capacity, nullptr);
    std::swap(input, output);
  }
  const auto *last_hidden =
      static_cast<const unsigned char *>(input) + 2ULL * (tokens - 1) * state.hidden;
  rms_norm_bf16(last_hidden, state.weights.at("final_norm"), state.final_hidden.get(), 1,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                    state.logits.get(), 1, state.hidden, state.vocabulary, nullptr);
  argmax_bf16(state.logits.get(), state.vocabulary, static_cast<int *>(state.device_token.get()),
              state.argmax_workspace.get(), nullptr);
  int next_token = 0;
  check(
      cudaMemcpy(&next_token, state.device_token.get(), sizeof(next_token), cudaMemcpyDeviceToHost),
      "copy prefill greedy token");
  state.current_position = tokens;
  return next_token;
}

int Gemma4ModelRunner::prefill_chunked(const std::vector<int> &token_ids, int chunk_tokens) {
  auto &state = *implementation_;
  if (state.current_position != 0)
    throw std::runtime_error("chunked prefill requires an empty model runner");
  if (token_ids.empty() || token_ids.size() > static_cast<std::size_t>(state.maximum_context) ||
      chunk_tokens <= 0 || chunk_tokens > state.maximum_context) {
    throw std::runtime_error("invalid chunked prompt shape");
  }
  for (const int token_id : token_ids) {
    if (token_id < 0 || token_id >= state.vocabulary) {
      throw std::runtime_error("prompt contains an invalid token id");
    }
  }
  void *final_input = nullptr;
  int final_chunk_tokens = 0;
  for (int position = 0; position < static_cast<int>(token_ids.size()); position += chunk_tokens) {
    const int tokens = std::min(chunk_tokens, static_cast<int>(token_ids.size()) - position);
    check(cudaMemcpy(state.device_token.get(), token_ids.data() + position, tokens * sizeof(int),
                     cudaMemcpyHostToDevice),
          "copy prompt chunk token ids");
    embedding_batch_bf16(state.weights.at("token_embedding"),
                         static_cast<const int *>(state.device_token.get()), state.hidden_a.get(),
                         tokens, state.vocabulary, state.hidden,
                         std::sqrt(static_cast<float>(state.hidden)), nullptr);
    void *input = state.hidden_a.get();
    void *output = state.hidden_b.get();
    for (std::size_t layer = 0; layer < state.config.layers.size(); ++layer) {
      const auto &cache = state.caches[layer];
      state.layers.run_prefill_chunk(static_cast<int>(layer), input, output, tokens, position,
                                     state.key_cache(layer), state.value_cache(layer),
                                     cache.capacity, nullptr);
      std::swap(input, output);
    }
    final_input = input;
    final_chunk_tokens = tokens;
  }
  const auto *last_hidden = static_cast<const unsigned char *>(final_input) +
                            2ULL * (final_chunk_tokens - 1) * state.hidden;
  rms_norm_bf16(last_hidden, state.weights.at("final_norm"), state.final_hidden.get(), 1,
                state.hidden, static_cast<float>(state.config.rms_norm_epsilon), nullptr);
  state.lm_head.run(state.final_hidden.get(), state.weights.at("token_embedding"),
                    state.logits.get(), 1, state.hidden, state.vocabulary, nullptr);
  argmax_bf16(state.logits.get(), state.vocabulary, static_cast<int *>(state.device_token.get()),
              state.argmax_workspace.get(), nullptr);
  int next_token = 0;
  check(
      cudaMemcpy(&next_token, state.device_token.get(), sizeof(next_token), cudaMemcpyDeviceToHost),
      "copy chunked prefill greedy token");
  state.current_position = static_cast<int>(token_ids.size());
  return next_token;
}

void Gemma4ModelRunner::reset_sequence() {
  implementation_->current_position = 0;
}

int Gemma4ModelRunner::position() const {
  return implementation_->current_position;
}

} // namespace carat
