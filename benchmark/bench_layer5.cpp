#ifdef ENABLE_OPENBLAS

#include "timer.h"
#include "gemm_common.h"
#include "gemm_blas.h"
#include <cblas.h>
#include <cstdio>

void bench_layer5(int N, double naive_gflops) {
    Matrix A(N, N), B(N, N), C(N, N);
    A.fill_random(1); B.fill_random(2);

    // ── Single-threaded OpenBLAS ──────────────────────────────────────────────
    // Force 1 thread to get a fair per-core comparison against our Layer 2/3
    openblas_set_num_threads(1);

    double blas1_gflops = BenchmarkTimer::measure_gflops(
        [&]{ C.zero(); gemm_openblas(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
        N, N, N);
    BenchmarkTimer::print_result("OpenBLAS (1 thread)", N, blas1_gflops, naive_gflops);

    // ── Multi-threaded OpenBLAS (default: uses all cores) ─────────────────────
    // OpenBLAS defaults to OMP_NUM_THREADS or the number of physical cores.
    // Reset to default (0 = auto-detect)
    openblas_set_num_threads(0);

    double blasN_gflops = BenchmarkTimer::measure_gflops(
        [&]{ C.zero(); gemm_openblas(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
        N, N, N);

    int num_threads = openblas_get_num_threads();
    char label[64];
    snprintf(label, sizeof(label), "OpenBLAS (%d threads)", num_threads);
    BenchmarkTimer::print_result(label, N, blasN_gflops, naive_gflops);
}

#endif // ENABLE_OPENBLAS
