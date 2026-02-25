#include "timer.h"
#include "gemm_common.h"
#include "gemm_naive.h"
#include "gemm_simd.h"
#include "gemm_openmp.h"
#ifdef ENABLE_CUDA
#include "gemm_cuda.h"
#endif
#ifdef ENABLE_OPENBLAS
#include "gemm_blas.h"
#endif

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Forward declarations for per-layer bench runners
void bench_layer1(int N, double naive_gflops);
void bench_layer2(int N, double naive_gflops);
void bench_layer3(int N, double naive_gflops);
#ifdef ENABLE_CUDA
void bench_layer4(int N, double naive_gflops);
#endif
#ifdef ENABLE_OPENBLAS
void bench_layer5(int N, double naive_gflops);
#endif

static void print_usage(const char* prog) {
    printf("Usage: %s [--layer 1|2|3|4|5|all] [--size 256|512|1024|2048|4096|all] [--runs N]\n",
           prog);
}

int main(int argc, char* argv[]) {
    // Defaults
    int layer_filter = 0;   // 0 = all
    int size_filter  = 0;   // 0 = all
    int bench_runs   = 5;
    int warmup_runs  = 2;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--layer" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v != "all") layer_filter = std::stoi(v);
        } else if (arg == "--size" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v != "all") size_filter = std::stoi(v);
        } else if (arg == "--runs" && i + 1 < argc) {
            bench_runs = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::vector<int> sizes;
    if (size_filter == 0)
        sizes = {256, 512, 1024, 2048, 4096};
    else
        sizes = {size_filter};

    printf("\n=== HP-GEMM Benchmark ===\n");
    printf("  Runs: %d (warmup: %d)\n\n", bench_runs, warmup_runs);

    for (int N : sizes) {
        printf("──── Matrix size: %d × %d ────\n", N, N);

        // Always compute naive baseline for speedup reference
        double naive_gflops = 0.0;
        if (layer_filter == 0 || layer_filter == 1) {
            // Compute naive baseline
            Matrix A(N, N), B(N, N), C(N, N);
            A.fill_random(1); B.fill_random(2);
            naive_gflops = BenchmarkTimer::measure_gflops(
                [&]{ C.zero(); gemm_naive(A.ptr(), B.ptr(), C.ptr(), N, N, N); },
                N, N, N, warmup_runs, bench_runs);
            BenchmarkTimer::print_result("naive (ijk)", N, naive_gflops, 0.0);
        }

        if (layer_filter == 0 || layer_filter == 1)
            bench_layer1(N, naive_gflops);
        if (layer_filter == 0 || layer_filter == 2)
            bench_layer2(N, naive_gflops);
        if (layer_filter == 0 || layer_filter == 3)
            bench_layer3(N, naive_gflops);
#ifdef ENABLE_CUDA
        if (layer_filter == 0 || layer_filter == 4)
            bench_layer4(N, naive_gflops);
#endif
#ifdef ENABLE_OPENBLAS
        if (layer_filter == 0 || layer_filter == 5)
            bench_layer5(N, naive_gflops);
#endif
        printf("\n");
    }

    return 0;
}
