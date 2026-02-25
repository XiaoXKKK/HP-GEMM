#include "timer.h"
#include "gemm_common.h"
#include "gemm_simd.h"

void bench_layer2(int N, double naive_gflops) {
    Matrix A(N, N), B(N, N), C(N, N);
    A.fill_random(1); B.fill_random(2);

    double avx2_gflops = BenchmarkTimer::measure_gflops(
        [&]{ C.zero(); gemm_avx2(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
        N, N, N);
    BenchmarkTimer::print_result("avx2 1x8", N, avx2_gflops, naive_gflops);

    double simd4x8_gflops = BenchmarkTimer::measure_gflops(
        [&]{ C.zero(); gemm_simd_blocked(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
        N, N, N);
    BenchmarkTimer::print_result("avx2 4x8 (reg block)", N, simd4x8_gflops, naive_gflops);
}
