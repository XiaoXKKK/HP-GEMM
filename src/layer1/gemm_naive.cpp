#include "gemm_naive.h"

// ─── Naive GEMM: ijk loop order ───────────────────────────────────────────────
//
// This is the textbook O(M*N*K) matrix multiplication.
// The inner loop accesses B[k][j], which moves down a column in row-major storage.
// Column access means each B element is in a DIFFERENT cache line → cache miss on every access.
//
// Memory access pattern (for square N×N matrices):
//   A[i][k]: sequential along row i → GOOD (stride 1)
//   B[k][j]: sequential down column j → BAD (stride N floats = 4*N bytes per step)
//   C[i][j]: written once per (i,j) pair → OK
//
// At N=1024: each column step = 4096 bytes, far beyond L1 cache line (64 bytes).
// Result: nearly every B access is a cold cache miss.
//
void gemm_naive(const float* __restrict__ A,
                const float* __restrict__ B,
                float*       __restrict__ C,
                int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];  // B: column access (bad!)
            C[i * N + j] += sum;
        }
}
