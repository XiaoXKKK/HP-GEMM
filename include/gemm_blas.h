#pragma once

// ─── Layer 5: OpenBLAS GEMM wrapper ──────────────────────────────────────────
// cblas_sgemm signature:
//   C = alpha * A * B + beta * C
// We always use alpha=1.0, beta=0.0 (C = A * B, C pre-zeroed)
#ifdef ENABLE_OPENBLAS

void gemm_openblas(const float* A, const float* B, float* C,
                   int M, int N, int K);

#endif // ENABLE_OPENBLAS
