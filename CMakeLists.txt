cmake_minimum_required(VERSION 3.18)
project(HP_GEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────────
option(ENABLE_CUDA   "Build CUDA Layer 4 kernels"   OFF)
option(ENABLE_OPENMP "Build with OpenMP threading"  ON)

# ─── OpenMP ───────────────────────────────────────────────────────────────────
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# ─── CUDA (Optional) ──────────────────────────────────────────────────────────
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    # sm_75=Turing(RTX20xx), sm_86=Ampere(RTX30xx), sm_89=Ada(RTX40xx)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# ─── Include Directories ──────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/benchmark)

# ─── Layer 1: Cache-friendly GEMM ─────────────────────────────────────────────
add_library(gemm_layer1 STATIC
    src/layer1/gemm_naive.cpp
    src/layer1/gemm_ikj.cpp
    src/layer1/gemm_blocked.cpp
)
target_compile_options(gemm_layer1 PRIVATE -O3 -march=native -mavx2 -mfma)

# ─── Layer 2: SIMD Vectorization ──────────────────────────────────────────────
add_library(gemm_layer2 STATIC
    src/layer2/gemm_avx2.cpp
    src/layer2/gemm_simd_blocked.cpp
)
target_compile_options(gemm_layer2 PRIVATE -O3 -march=native -mavx2 -mfma -ffast-math)

# ─── Layer 3: OpenMP Multi-threading ──────────────────────────────────────────
add_library(gemm_layer3 STATIC
    src/layer3/gemm_openmp.cpp
)
target_compile_options(gemm_layer3 PRIVATE -O3 -march=native -mavx2 -mfma -ffast-math)
if(ENABLE_OPENMP)
    target_link_libraries(gemm_layer3 PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─── Layer 4: CUDA GPU (optional) ─────────────────────────────────────────────
if(ENABLE_CUDA)
    add_library(gemm_layer4 STATIC
        src/layer4/gemm_cuda_naive.cu
        src/layer4/gemm_cuda_shared.cu
    )
    target_compile_definitions(gemm_layer4 PUBLIC ENABLE_CUDA)
    target_compile_options(gemm_layer4 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
    )
    target_link_libraries(gemm_layer4 PUBLIC CUDA::cudart)
    set_target_properties(gemm_layer4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ─── Benchmark Executable ─────────────────────────────────────────────────────
set(BENCH_SOURCES
    benchmark/bench_main.cpp
    benchmark/bench_layer1.cpp
    benchmark/bench_layer2.cpp
    benchmark/bench_layer3.cpp
)
if(ENABLE_CUDA)
    list(APPEND BENCH_SOURCES benchmark/bench_layer4.cpp)
endif()

add_executable(hp_gemm_bench ${BENCH_SOURCES})
target_link_libraries(hp_gemm_bench PRIVATE gemm_layer1 gemm_layer2 gemm_layer3)
if(ENABLE_CUDA)
    target_link_libraries(hp_gemm_bench PRIVATE gemm_layer4)
    target_compile_definitions(hp_gemm_bench PRIVATE ENABLE_CUDA)
endif()

# ─── Test Executable ──────────────────────────────────────────────────────────
set(TEST_SOURCES
    tests/test_main.cpp
    tests/test_layer1.cpp
    tests/test_layer2.cpp
    tests/test_layer3.cpp
)
if(ENABLE_CUDA)
    list(APPEND TEST_SOURCES tests/test_layer4.cpp)
endif()

add_executable(hp_gemm_tests ${TEST_SOURCES})
target_link_libraries(hp_gemm_tests PRIVATE gemm_layer1 gemm_layer2 gemm_layer3)
if(ENABLE_CUDA)
    target_link_libraries(hp_gemm_tests PRIVATE gemm_layer4)
    target_compile_definitions(hp_gemm_tests PRIVATE ENABLE_CUDA)
endif()

enable_testing()
add_test(NAME correctness COMMAND hp_gemm_tests)
