#pragma once

#include <chrono>
#include <functional>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <string>

// ─── Benchmark Timer ──────────────────────────────────────────────────────────
//
// Usage:
//   double gflops = BenchmarkTimer::measure_gflops(
//       [&]{ gemm_blocked(A, B, C, M, N, K); },
//       M, N, K, /*warmup=*/2, /*runs=*/5);
//

class BenchmarkTimer {
public:
    // Run func once and return elapsed seconds
    static double time_once(const std::function<void()>& func) {
        auto t0 = std::chrono::high_resolution_clock::now();
        func();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }

    // Run warmup_runs times (discarded), then bench_runs times.
    // Returns MEDIAN elapsed time in seconds.
    static double median_time(const std::function<void()>& func,
                               int warmup_runs = 2,
                               int bench_runs  = 5)
    {
        // Warmup: let CPU reach steady-state frequency (turbo boost stabilisation)
        for (int i = 0; i < warmup_runs; ++i)
            func();

        std::vector<double> times(bench_runs);
        for (int i = 0; i < bench_runs; ++i)
            times[i] = time_once(func);

        std::sort(times.begin(), times.end());
        return times[bench_runs / 2];  // median
    }

    // Convenience: returns GFLOPS given a GEMM kernel
    // C is expected to be pre-zeroed EACH call; caller must handle this.
    static double measure_gflops(const std::function<void()>& func,
                                  int M, int N, int K,
                                  int warmup = 2, int runs = 5)
    {
        double t = median_time(func, warmup, runs);
        double flops = 2.0 * M * N * K;
        return (flops / t) * 1e-9;
    }

    // Print a formatted benchmark result row to stdout
    //   name    : kernel label (e.g., "blocked")
    //   N       : matrix dimension (square assumed for display)
    //   gflops  : measured GFLOPS
    //   baseline: GFLOPS of reference (for speedup calc), 0 = skip speedup
    static void print_result(const std::string& name, int N,
                              double gflops, double baseline = 0.0)
    {
        if (baseline > 0.0)
            std::printf("  %-20s | N=%5d | %6.2f GFLOPS | %5.2fx vs naive\n",
                        name.c_str(), N, gflops, gflops / baseline);
        else
            std::printf("  %-20s | N=%5d | %6.2f GFLOPS\n",
                        name.c_str(), N, gflops);
    }

    static void print_header() {
        std::printf("  %-20s | %-7s | %-15s | %s\n",
                    "Kernel", "N", "GFLOPS", "Speedup");
        std::printf("  %s\n", std::string(60, '-').c_str());
    }
};
