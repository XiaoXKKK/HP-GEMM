#pragma once

#include <vector>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

// ─── Row-major Matrix ─────────────────────────────────────────────────────────
//
// Storage layout: C = A * B
//   element (i, j) is at data[i * cols + j]
//
// Memory is 32-byte aligned (required for AVX2 _mm256_load_ps).
//
struct Matrix {
    int rows;
    int cols;
    float* data;  // raw aligned pointer (not owned by std::vector to allow alignment)

    Matrix(int r, int c)
        : rows(r), cols(c)
    {
        // Allocate 32-byte aligned memory for AVX2 compatibility
        std::size_t bytes = static_cast<std::size_t>(r) * c * sizeof(float);
        // aligned_alloc requires size to be multiple of alignment
        std::size_t aligned_bytes = (bytes + 31) & ~static_cast<std::size_t>(31);
#if defined(_WIN32)
        data = static_cast<float*>(_aligned_malloc(aligned_bytes, 32));
#else
        data = static_cast<float*>(std::aligned_alloc(32, aligned_bytes));
#endif
        std::memset(data, 0, aligned_bytes);
    }

    ~Matrix() {
#if defined(_WIN32)
        _aligned_free(data);
#else
        std::free(data);
#endif
    }

    // Disable copy to avoid double-free; use explicit clone() if needed
    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Move support
    Matrix(Matrix&& o) noexcept
        : rows(o.rows), cols(o.cols), data(o.data)
    {
        o.data = nullptr;
        o.rows = o.cols = 0;
    }

    float& operator()(int i, int j)       { return data[i * cols + j]; }
    float  operator()(int i, int j) const { return data[i * cols + j]; }

    float*       ptr()       { return data; }
    const float* ptr() const { return data; }

    // Zero out all elements
    void zero() { std::memset(data, 0, static_cast<std::size_t>(rows) * cols * sizeof(float)); }

    // Fill with uniform random values in [-1, 1]
    void fill_random(unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        int n = rows * cols;
        for (int i = 0; i < n; ++i)
            data[i] = dist(rng);
    }
};

// ─── GFLOPS Calculation ───────────────────────────────────────────────────────
//
// GEMM performs 2*M*N*K floating-point operations:
//   - M*N*K multiplications
//   - M*N*K additions (or fused into FMA = 2 FLOPs per FMA)
//
inline double compute_gflops(int M, int N, int K, double elapsed_seconds) {
    double flops = 2.0 * M * N * K;
    return (flops / elapsed_seconds) * 1e-9;
}

// ─── Correctness Check ────────────────────────────────────────────────────────
//
// Returns max |a[i][j] - b[i][j]| over all elements.
// Use this to verify optimized kernels produce the same result as the reference.
//
inline float max_abs_error(const Matrix& a, const Matrix& b) {
    float max_err = 0.0f;
    int n = a.rows * a.cols;
    for (int i = 0; i < n; ++i) {
        float err = std::fabs(a.data[i] - b.data[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ─── Function Pointer Type ────────────────────────────────────────────────────
// Unified signature for all GEMM kernels: C = A * B  (C must be pre-zeroed)
using GemmFunc = void (*)(const float* A, const float* B, float* C,
                           int M, int N, int K);
