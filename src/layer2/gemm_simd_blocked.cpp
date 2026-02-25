#include "gemm_simd.h"
#include <immintrin.h>
#include <algorithm>
#include <cstring>

// ─── AVX2 GEMM: 4×8 Register-Blocked Micro-Kernel ───────────────────────────
//
// WHY 4×8 AND NOT 1×8?
//
// The 1×8 kernel (gemm_avx2) processes one output row at a time.
// For each of the K inner iterations:
//   - It loads 1 scalar from A → 1 broadcast
//   - It loads 8 floats from B → 1 YMM load
//   - It performs 1 FMA
//
// FMA throughput: modern cores can execute 2 FMAs per cycle.
// With only 1 accumulator (1 YMM), the FMA pipeline is barely utilized.
//
// SOLUTION: process 4 rows simultaneously with 4 YMM accumulators.
// In each inner k-iteration:
//   - Load 1 B vector (8 floats): shared across all 4 rows → amortized cost
//   - Broadcast 4 A scalars (one per row)
//   - Execute 4 FMAs in parallel (the CPU can pipeline these!)
//
// B:   load 1 YMM register  (8 floats)
// A:   4 scalar broadcasts   (4 floats, 4 separate registers)
// FMA: 4 operations → 4 × 2 FLOPs = 32 FLOPs per k-iteration
//
// The ratio "FLOPs per byte loaded from A+B" doubles compared to 1×8.
// This is called "arithmetic intensity" — higher is better for utilization.
//
// ── Internal helper: 4×8 micro-kernel for a tile ─────────────────────────────
//
// Processes rows [i_start, i_start+4) × columns [j_start, j_start+8)
// Assumes the caller has verified bounds (or handles partial blocks separately).
//
static inline void kernel_4x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int i_start, int j_start,
    int M, int N, int K)
{
    // Load 4 × 8 = 32 C accumulators from memory into 4 YMM registers
    __m256 c0 = _mm256_loadu_ps(&C[(i_start + 0) * N + j_start]);
    __m256 c1 = _mm256_loadu_ps(&C[(i_start + 1) * N + j_start]);
    __m256 c2 = _mm256_loadu_ps(&C[(i_start + 2) * N + j_start]);
    __m256 c3 = _mm256_loadu_ps(&C[(i_start + 3) * N + j_start]);

    // Accumulate over K dimension
    for (int k = 0; k < K; ++k) {
        // Load 8 consecutive B values: B[k][j_start .. j_start+7]
        __m256 b = _mm256_loadu_ps(&B[k * N + j_start]);

        // Broadcast A[i_start + r][k] for each of the 4 rows
        __m256 a0 = _mm256_set1_ps(A[(i_start + 0) * K + k]);
        __m256 a1 = _mm256_set1_ps(A[(i_start + 1) * K + k]);
        __m256 a2 = _mm256_set1_ps(A[(i_start + 2) * K + k]);
        __m256 a3 = _mm256_set1_ps(A[(i_start + 3) * K + k]);

        // 4 FMAs: each YMM accumulator updated independently
        // The CPU can pipeline these (out-of-order execution + 2 FMA ports)
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }

    // Write back 4 × 8 = 32 output values
    _mm256_storeu_ps(&C[(i_start + 0) * N + j_start], c0);
    _mm256_storeu_ps(&C[(i_start + 1) * N + j_start], c1);
    _mm256_storeu_ps(&C[(i_start + 2) * N + j_start], c2);
    _mm256_storeu_ps(&C[(i_start + 3) * N + j_start], c3);
}

// ── Partial row handler: when i_start+4 > M (fewer than 4 remaining rows) ────
static inline void kernel_partial_rows(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int i_start, int j_start,
    int M, int N, int K)
{
    int rows = std::min(4, M - i_start);
    for (int ri = 0; ri < rows; ++ri) {
        int j = j_start;
        for (; j <= N - 8; j += 8) {
            __m256 c = _mm256_loadu_ps(&C[(i_start + ri) * N + j]);
            for (int k = 0; k < K; ++k) {
                __m256 a = _mm256_set1_ps(A[(i_start + ri) * K + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                c = _mm256_fmadd_ps(a, b, c);
            }
            _mm256_storeu_ps(&C[(i_start + ri) * N + j], c);
        }
        // scalar tail for j
        for (; j < N; ++j) {
            float sum = C[(i_start + ri) * N + j];
            for (int k = 0; k < K; ++k)
                sum += A[(i_start + ri) * K + k] * B[k * N + j];
            C[(i_start + ri) * N + j] = sum;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────
void gemm_simd_blocked(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float*       __restrict__ C,
                       int M, int N, int K)
{
    constexpr int ROW_BLOCK = 4;   // register block: 4 rows per micro-kernel
    constexpr int COL_BLOCK = 8;   // register block: 8 cols per YMM register

    // Iterate over row blocks of 4
    for (int i = 0; i < M; i += ROW_BLOCK) {
        if (i + ROW_BLOCK <= M) {
            // Full 4×8 micro-kernel path
            int j = 0;
            for (; j <= N - COL_BLOCK; j += COL_BLOCK)
                kernel_4x8(A, B, C, i, j, M, N, K);
            // Handle remaining columns with partial-row handler (rows still 4)
            if (j < N)
                kernel_partial_rows(A, B, C, i, j, M, N, K);
        } else {
            // Fewer than 4 rows remaining
            kernel_partial_rows(A, B, C, i, 0, M, N, K);
        }
    }
}

// ── Helper exposed for Layer 3 (OpenMP calls this per row-block) ──────────────
//
// Processes rows [i_start, i_end) using the 4×8 SIMD micro-kernel.
// This function is declared extern "C" for clean linkage from gemm_openmp.cpp.
//
void gemm_simd_block_rows(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int i_start, int i_end,
                           int M, int N, int K)
{
    constexpr int ROW_BLOCK = 4;
    constexpr int COL_BLOCK = 8;

    for (int i = i_start; i < i_end; i += ROW_BLOCK) {
        if (i + ROW_BLOCK <= i_end) {
            int j = 0;
            for (; j <= N - COL_BLOCK; j += COL_BLOCK)
                kernel_4x8(A, B, C, i, j, M, N, K);
            if (j < N)
                kernel_partial_rows(A, B, C, i, j, M, N, K);
        } else {
            kernel_partial_rows(A, B, C, i, 0, M, N, K);
        }
    }
}
