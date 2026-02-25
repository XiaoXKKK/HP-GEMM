#ifdef ENABLE_CUDA

#include "timer.h"
#include "gemm_common.h"
#include "gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>

void bench_layer4(int N, double naive_gflops) {
    // Host matrices
    Matrix A(N, N), B(N, N), C_host(N, N);
    A.fill_random(1); B.fill_random(2);

    // Device matrices
    CudaMatrix d_A(N, N), d_B(N, N), d_C(N, N);
    d_A.copy_from_host(A.ptr());
    d_B.copy_from_host(B.ptr());

    // Benchmark naive CUDA kernel
    double cuda_naive_gflops = BenchmarkTimer::measure_gflops(
        [&]{
            // Zero device C before each run
            cudaMemset(d_C.d_ptr, 0, (size_t)N * N * sizeof(float));
            gemm_cuda_naive(d_A.d_ptr, d_B.d_ptr, d_C.d_ptr, N, N, N);
        },
        N, N, N);
    BenchmarkTimer::print_result("cuda naive", N, cuda_naive_gflops, naive_gflops);

    // Benchmark shared memory tiled kernel
    double cuda_shared_gflops = BenchmarkTimer::measure_gflops(
        [&]{
            cudaMemset(d_C.d_ptr, 0, (size_t)N * N * sizeof(float));
            gemm_cuda_shared(d_A.d_ptr, d_B.d_ptr, d_C.d_ptr, N, N, N);
        },
        N, N, N);
    BenchmarkTimer::print_result("cuda shared mem", N, cuda_shared_gflops, naive_gflops);
}

#endif // ENABLE_CUDA
