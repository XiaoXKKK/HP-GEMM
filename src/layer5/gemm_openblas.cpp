#ifdef ENABLE_OPENBLAS

#include "gemm_blas.h"
#include <cblas.h>

// ─── OpenBLAS GEMM Wrapper ────────────────────────────────────────────────────
//
// cblas_sgemm computes: C = alpha*A*B + beta*C
//
// Parameters:
//   CblasRowMajor : row-major storage (matches our Matrix layout)
//   CblasNoTrans  : A and B are not transposed
//   alpha = 1.0f  : no scaling of the product
//   beta  = 0.0f  : overwrite C (not accumulate), equivalent to C being pre-zeroed
//
// Under the hood, OpenBLAS uses:
//   - SIMD auto-vectorization (AVX2/AVX-512 depending on CPU)
//   - Multi-level cache blocking tuned for the target architecture
//   - Optimized micro-kernels written in hand-tuned assembly
//   - Optionally multi-threaded (controlled by OPENBLAS_NUM_THREADS env var)
//
// This is the "industry reference" level — what production HPC code uses.
//
void gemm_openblas(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    cblas_sgemm(
        CblasRowMajor,  // storage order
        CblasNoTrans,   // A is not transposed
        CblasNoTrans,   // B is not transposed
        M, N, K,        // dimensions
        1.0f,           // alpha
        A, K,           // A matrix and its leading dimension (cols = K)
        B, N,           // B matrix and its leading dimension (cols = N)
        0.0f,           // beta (overwrite C)
        C, N            // C matrix and its leading dimension (cols = N)
    );
}

#endif // ENABLE_OPENBLAS
