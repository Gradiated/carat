cmake_minimum_required(VERSION 3.25)

project(carat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CARAT_ENABLE_CUDA "Build the CUDA backend and HTTP runtime" OFF)
option(CARAT_BUILD_BENCHMARKS "Build GPU benchmarks and model parity tools" OFF)

include(CTest)
find_package(Threads REQUIRED)

add_library(carat_support support/src/http.cpp support/src/json.cpp)
target_include_directories(carat_support PUBLIC support/include)
target_link_libraries(carat_support PUBLIC Threads::Threads)
target_compile_options(carat_support PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_library(carat_model
  src/json.cpp
  src/gemma4_config.cpp
  src/gemma4_model.cpp
  src/safetensors.cpp
  src/weight_plan.cpp
)
target_include_directories(carat_model PUBLIC include)
target_compile_options(carat_model PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_library(carat_runtime
  src/runtime/completion_request.cpp
  src/runtime/completion_response.cpp
  src/runtime/metrics.cpp
  src/runtime/settings.cpp
  src/runtime/slot_policy.cpp
  src/runtime/speculation.cpp
)
target_include_directories(carat_runtime PUBLIC src)
target_link_libraries(carat_runtime PRIVATE carat_model)
target_compile_options(carat_runtime PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(carat-inspect src/inspect_main.cpp)
target_link_libraries(carat-inspect PRIVATE carat_model)
target_compile_options(carat-inspect PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(CARAT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12.9 REQUIRED)
  find_path(CARAT_CURAND_INCLUDE_DIR curand_kernel.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn_version.h REQUIRED)
  foreach(part MAJOR MINOR PATCHLEVEL)
    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" version_line
      REGEX "^#define CUDNN_${part} [0-9]+")
    string(REGEX MATCH "[0-9]+" CUDNN_${part} "${version_line}")
  endforeach()
  set(CUDNN_VERSION "${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}")
  if(CUDNN_VERSION VERSION_LESS 9.10)
    message(FATAL_ERROR "Carat requires cuDNN 9.10 or newer; found ${CUDNN_VERSION}")
  endif()
  include(FetchContent)
  set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    cudnn_frontend
    URL https://codeload.github.com/NVIDIA/cudnn-frontend/tar.gz/b372d39879d44c91a8d5b342022e74802b6a8da2
    URL_HASH SHA256=8fbdd859e484924f1ccbafa3a6f8c4bdeb970f4cbbcf20ddf54a9ceaa3bf91ac
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(cudnn_frontend)
  FetchContent_Declare(
    cutlass
    URL https://codeload.github.com/NVIDIA/cutlass/tar.gz/6c65a175668952f09bcbf66cb97a8de1b734b4a0
    URL_HASH SHA256=6e2f4b08bd4b54ee795a18de4b7c0a65a65b62fcb83c2785e872e18c1fae28bc
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _carat_header_only
  )
  FetchContent_MakeAvailable(cutlass)
  find_library(CUDNN_LIBRARY NAMES cudnn REQUIRED)
  add_library(carat_cuda
    cuda/attention.cu
    cuda/assistant_runner.cu
    cuda/batch_model_runner.cu
    cuda/gemm_attention.cu
    cuda/hopper_attention.cu
    cuda/prefill_gemm_attention.cu
    cuda/ops.cu
    cuda/prefill_attention.cu
    cuda/device_weights.cu
    cuda/int4_bf16_linear.cu
    cuda/int4_linear.cu
    cuda/linear.cu
    cuda/layer_runner.cu
    cuda/model_runner.cu
    cuda/qkv.cu
  )
  target_include_directories(carat_cuda PUBLIC include PRIVATE
    ${CARAT_CURAND_INCLUDE_DIR}
    ${CUDNN_INCLUDE_DIR}
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/tools/util/include)
  target_compile_options(carat_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(carat_cuda PUBLIC carat_model CUDA::cudart CUDA::cublas
    CUDA::cublasLt cudnn_frontend ${CUDNN_LIBRARY})
  set_target_properties(carat_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 90a
  )
  if(BUILD_TESTING)
    add_executable(carat_cuda_tests tests/cuda_ops_tests.cu)
    target_link_libraries(carat_cuda_tests PRIVATE carat_cuda CUDA::cudart)
    set_target_properties(carat_cuda_tests PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES 90
    )
    add_test(NAME carat_cuda_tests COMMAND carat_cuda_tests)
  endif()
  function(carat_cuda_tool name source)
    add_executable(carat-${name} ${source})
    target_link_libraries(carat-${name} PRIVATE carat_cuda)
    set_target_properties(carat-${name} PROPERTIES
      CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 90)
  endfunction()

  add_executable(carat-load src/load_main.cpp)
  target_link_libraries(carat-load PRIVATE carat_cuda)

  if(CARAT_BUILD_BENCHMARKS)
    foreach(tool IN ITEMS linear-benchmark fp8-linear-benchmark fp8-model-parity fp8-quality-eval
        attention-benchmark fp8-attention-benchmark prefill-attention-benchmark layer-parity
        model-parity prefill-model-parity prefill-benchmark batch-benchmark ragged-batch-parity)
      string(REPLACE "-" "_" source_name "${tool}")
      carat_cuda_tool(${tool} src/${source_name}.cu)
    endforeach()
  endif()

  add_executable(carat-runtime src/native_runtime_main.cpp)
  target_link_libraries(carat-runtime PRIVATE carat_cuda carat_runtime carat_support)
  target_compile_options(carat-runtime PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  install(TARGETS carat-runtime RUNTIME DESTINATION bin)
endif()

if(BUILD_TESTING)
  add_executable(carat_support_tests
    support/tests/http_server_tests.cpp
    support/tests/support/test_support.cpp)
  target_include_directories(carat_support_tests PRIVATE support/tests/support)
  target_link_libraries(carat_support_tests PRIVATE carat_support)
  add_test(NAME carat_support_tests COMMAND carat_support_tests)
  set_tests_properties(carat_support_tests PROPERTIES TIMEOUT 60)

  set(CARAT_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

  add_library(carat_test_support STATIC tests/support/test_support.cpp)
  target_include_directories(carat_test_support PUBLIC tests)
  target_compile_options(carat_test_support PRIVATE ${CARAT_WARNINGS})

  add_library(carat_model_fixtures STATIC tests/support/model_fixtures.cpp)
  target_include_directories(carat_model_fixtures PUBLIC tests)
  target_compile_options(carat_model_fixtures PRIVATE ${CARAT_WARNINGS})

  add_executable(carat_model_tests
    tests/json_tests.cpp
    tests/safetensors_tests.cpp
    tests/gemma4_config_tests.cpp
    tests/weight_plan_tests.cpp
  )
  target_link_libraries(carat_model_tests PRIVATE carat_model
    carat_model_fixtures carat_test_support)
  target_compile_options(carat_model_tests PRIVATE ${CARAT_WARNINGS})
  add_test(NAME carat_model_tests COMMAND carat_model_tests)

  add_executable(carat_runtime_tests
    tests/completion_request_tests.cpp
    tests/completion_response_tests.cpp
    tests/runtime_metrics_tests.cpp
    tests/runtime_settings_tests.cpp
    tests/slot_policy_tests.cpp
    tests/speculation_tests.cpp
  )
  target_link_libraries(carat_runtime_tests PRIVATE carat_runtime
    carat_test_support)
  target_compile_options(carat_runtime_tests PRIVATE ${CARAT_WARNINGS})
  add_test(NAME carat_runtime_tests COMMAND carat_runtime_tests)

  add_executable(carat_model_fixtures_writer tests/support/write_model_fixtures_main.cpp)
  target_link_libraries(carat_model_fixtures_writer PRIVATE carat_model_fixtures)
  target_compile_options(carat_model_fixtures_writer PRIVATE ${CARAT_WARNINGS})
  add_test(NAME carat_inspect_cli COMMAND ${CMAKE_COMMAND}
    -DFIXTURES=$<TARGET_FILE:carat_model_fixtures_writer>
    -DINSPECT=$<TARGET_FILE:carat-inspect>
    -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/engine-inspect-cli
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/inspect_cli_test.cmake)

  if(NOT CARAT_ENABLE_CUDA)
    # The CUDA build links the runtime; host builds still compile it against the CUDA-free headers.
    add_library(carat_runtime_host_compile OBJECT src/native_runtime_main.cpp)
    target_link_libraries(carat_runtime_host_compile PRIVATE carat_model
      carat_runtime carat_support)
    target_compile_options(carat_runtime_host_compile PRIVATE ${CARAT_WARNINGS})
  endif()
endif()

install(TARGETS carat-inspect RUNTIME DESTINATION bin)
