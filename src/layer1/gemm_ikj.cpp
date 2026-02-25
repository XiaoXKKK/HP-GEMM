#include "gemm_naive.h"

// ─── Cache-friendly GEMM: ikj loop order ─────────────────────────────────────
//
// KEY INSIGHT: swap the j and k loops compared to naive.
// The innermost loop now advances j (not k), which means:
//
//   A[i*K+k]: loaded ONCE before the inner loop → held in a register (broadcast)
//   B[k*N+j]: j increments by 1 → sequential memory access (STRIDE 1 = cache-friendly)
//   C[i*N+j]: j increments by 1 → sequential memory access (STRIDE 1 = cache-friendly)
//
// Why does this matter?
//   A cache line holds 64 bytes = 16 floats.
//   In the inner loop we read 16 consecutive B values and 16 consecutive C values
//   from each cache line fetched, instead of wasting a full cache line per scalar.
//
// Typical speedup vs naive: 3-5x for large N.
// No special hardware features needed — pure loop ordering.
//
void gemm_ikj(const float* __restrict__ A,
              const float* __restrict__ B,
              float*       __restrict__ C,
              int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];  // scalar: kept in register for inner loop
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a_ik * B[k * N + j];  // both B and C: stride-1 access
        }
}
