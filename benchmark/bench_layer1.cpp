#include "timer.h"
#include "gemm_common.h"
#include "gemm_naive.h"

void bench_layer1(int N, double naive_gflops) {
    Matrix A(N, N), B(N, N), C(N, N);
    A.fill_random(1); B.fill_random(2);

    // ikj (loop reorder)
    double ikj_gflops = BenchmarkTimer::measure_gflops(
        [&]{ C.zero(); gemm_ikj(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
        N, N, N);
    BenchmarkTimer::print_result("ikj (loop reorder)", N, ikj_gflops, naive_gflops);

    // blocked (2-level tiling)
    double blocked_gflops = BenchmarkTimer::measure_gflops(
        [&]{ C.zero(); gemm_blocked(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
        N, N, N);
    BenchmarkTimer::print_result("blocked (L2+reg tile)", N, blocked_gflops, naive_gflops);
}
