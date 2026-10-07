# Carat

Carat runs Gemma 4 31B text inference on a single NVIDIA H200. It is written in C++ and CUDA,
with ring buffers for the model's sliding-window attention and an HTTP server
for token completions. It supports continuous batching, prefix-cache reuse,
FP8 decode, and speculative decoding with a Gemma assistant model.

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

To build the GPU runtime, install the CUDA 12.9 toolkit or newer, including
cuRAND headers, and cuDNN 9.10 or newer headers and libraries. Put `nvcc` on your `PATH`.
CMake fetches pinned CUTLASS and cuDNN frontend sources. The GPU backend targets
Hopper (`sm_90a`). The runtime reserves 24 request slots with an 8,192-token context each, so an H200 is needed for the full model
and KV cache.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCARAT_ENABLE_CUDA=ON
cmake --build build-cuda -j 4
ctest --test-dir build-cuda --output-on-failure
```

Tested on an H200 with Ubuntu 24.04, GCC 13.3, CUDA 12.9, and cuDNN 9.27.
The test suites, BF16 and FP8 HTTP generation, streaming, batching, and cache
reuse were checked with the checkpoint below. BF16 greedy output matched
Transformers 5.19.0 on three short prompts.

## Run

Download `google/gemma-4-31B-it` under its model license. The checkpoint used
here is revision `b9ea41a2887d8607f594846523f94c6cc75ac8a4`. Pass the local
directory containing `config.json` and the safetensors weights:

```sh
CARAT_RUNTIME_HOST=127.0.0.1 ./build-cuda/carat-runtime /path/to/gemma-4-31B-it
```

For text prompts, install Transformers and use the client:

```sh
python3 -m venv .venv
.venv/bin/pip install transformers==5.19.0
.venv/bin/python tools/chat.py /path/to/gemma-4-31B-it "What is a ring buffer?"
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

Set `-DCARAT_BUILD_BENCHMARKS=ON` to build the parity and benchmark programs in
`src/`. Fixture and smoke tools live in `tools/`. Python model tools require
PyTorch, safetensors, and a Transformers release with Gemma 4 support. The HTTP smoke tool uses the Python standard library.
