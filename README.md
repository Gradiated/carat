# Carat

Carat is a C++20/CUDA inference engine for the text decoder of Gemma 4 31B on
NVIDIA Hopper GPUs, developed on an H200. It uses sliding-window KV ring buffers,
model-specific attention kernels, continuous batching, prefix-cache reuse, and
optional FP8 execution and assistant-model speculative decoding.

Read the [blog post](https://www.gradiated.com/library/gemma-inference-engine/)
for the design and performance results.

## Build

On Linux, install CMake 3.25 or newer and a C++20 compiler. The default build
provides the model inspector and CPU tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/carat-inspect /path/to/gemma-4-31B-it
```

To build the GPU runtime, also install a CUDA toolkit supporting `sm_90a` and
cuDNN 9 headers and libraries. CMake fetches pinned CUTLASS and cuDNN frontend
sources. The GPU backend targets Hopper (`sm_90a`).

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCARAT_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

## Run

Download `google/gemma-4-31B-it` separately under its own access terms. Pass the
local model directory containing `config.json` and the safetensors weights:

```sh
CARAT_RUNTIME_HOST=127.0.0.1 ./build-cuda/carat-runtime /path/to/gemma-4-31B-it
```

The server accepts token IDs. Use the model's tokenizer to encode your prompt and
decode the returned `output_ids`. The runtime uses greedy decoding and supports
streaming, cancellation, and prefix-cache reuse, with an 8,192-token context.

```sh
curl http://127.0.0.1:30000/health
curl http://127.0.0.1:30000/v1/token-completions \
  -H 'Content-Type: application/json' \
  -d '{"input_ids":[2,100],"max_tokens":32}'
```

Add `"stream":true` for server-sent events. Optional `stop_token_ids` supplement
the model's EOS tokens. `GET /metrics` exposes Prometheus metrics.

| Setting | Default | Purpose |
| --- | --- | --- |
| `CARAT_RUNTIME_HOST` | `0.0.0.0` | IPv4 bind address |
| `CARAT_RUNTIME_PORT` | `30000` | HTTP port |
| `CARAT_RUNTIME_API_KEY` | unset | Require `Authorization: Bearer <key>` when set |
| `CARAT_FP8_DECODE` | `off` | Set `tensor` to enable FP8 decode |
| `CARAT_ASSISTANT_MODEL_PATH` | unset | Assistant checkpoint; requires FP8 decode |
| `CARAT_ASSISTANT_FP8_MODE` | `off` | Assistant FP8 mode: `off`, `lm_head`, or `all` |

Parity and benchmark programs live in `src/`; fixture and smoke tools live in
`tools/`. Python model tools require PyTorch, safetensors, and a Transformers
release with Gemma 4 support. The HTTP smoke tool uses the Python standard library.
