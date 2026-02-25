#include "gemm_openmp.h"
#include "gemm_simd.h"
#include <algorithm>

// Forward declaration of the helper exposed by gemm_simd_blocked.cpp
void gemm_simd_block_rows(const float* A, const float* B, float* C,
                           int i_start, int i_end, int M, int N, int K);

// ─── OpenMP Parallel GEMM ─────────────────────────────────────────────────────
//
// STRATEGY: parallelize the outermost i-dimension of the 4×8 SIMD kernel.
//
// WHY IS THIS SAFE (no race conditions)?
//   Each output row C[i][*] is written by exactly ONE thread.
//   Threads process disjoint row ranges → no two threads write the same memory.
//   A and B are read-only → no write-write or write-read conflicts.
//
// WHY NOT PARALLELIZE j OR k?
//   j-loop: fine for OpenMP but shorter trips → higher scheduling overhead.
//   k-loop: DANGEROUS — multiple threads would accumulate into the SAME C[i][j].
//            This is a data race. Would require atomic ops or reduction → slow.
//
// SCHEDULE(STATIC):
//   Divides the M rows evenly across OMP_NUM_THREADS threads.
//   Each thread gets M/nthreads rows — roughly equal work (all same size).
//   Static is better than dynamic here because work per row is uniform.
//
// FALSE SHARING AVOIDANCE:
//   Each thread writes a contiguous block of rows. Within a row block,
//   C elements are stored contiguously (row-major). Cache lines (64 bytes = 16 floats)
//   are NOT shared across thread boundaries (rows are 4-aligned, >> 16 floats wide).
//
// THREAD COUNT:
//   Set by OMP_NUM_THREADS env var before running:
//     OMP_NUM_THREADS=4 ./hp_gemm_bench --layer 3
//   Or call omp_set_num_threads() at runtime.
//
void gemm_openmp(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float*       __restrict__ C,
                 int M, int N, int K)
{
    constexpr int ROW_BLOCK = 4;  // must match the micro-kernel row block size

    // Each OpenMP thread processes a contiguous range of row blocks.
    // The loop variable i steps by ROW_BLOCK = 4 (the SIMD micro-kernel size).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i += ROW_BLOCK) {
        int i_end = std::min(i + ROW_BLOCK, M);
        // Delegate to the SIMD micro-kernel for this row block
        gemm_simd_block_rows(A, B, C, i, i_end, M, N, K);
    }
}
