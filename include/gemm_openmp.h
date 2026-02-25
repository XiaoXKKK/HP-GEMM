#pragma once

// ─── Layer 3 kernel declarations ──────────────────────────────────────────────

// OpenMP parallel GEMM: parallelizes outer i-loop over the 4x8 SIMD micro-kernel.
// Thread count controlled by OMP_NUM_THREADS environment variable.
// Each thread works on disjoint rows of C (no false sharing).
void gemm_openmp(const float* A, const float* B, float* C,
                 int M, int N, int K);
