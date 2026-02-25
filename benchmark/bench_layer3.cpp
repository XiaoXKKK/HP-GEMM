#include "timer.h"
#include "gemm_common.h"
#include "gemm_openmp.h"
#include <cstdio>
#ifdef _OPENMP
#include <omp.h>
#endif

void bench_layer3(int N, double naive_gflops) {
    Matrix A(N, N), B(N, N), C(N, N);
    A.fill_random(1); B.fill_random(2);

    // Benchmark across different thread counts
    int thread_counts[] = {1, 2, 4, 8, 16};
    for (int nt : thread_counts) {
#ifdef _OPENMP
        omp_set_num_threads(nt);
        int actual = omp_get_max_threads();
        if (actual != nt) continue;  // skip if fewer physical cores
#else
        if (nt > 1) continue;
#endif
        char label[64];
        snprintf(label, sizeof(label), "openmp (%d threads)", nt);

        double gflops = BenchmarkTimer::measure_gflops(
            [&]{ C.zero(); gemm_openmp(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
            N, N, N);
        BenchmarkTimer::print_result(label, N, gflops, naive_gflops);
    }
}
