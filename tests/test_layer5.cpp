#ifdef ENABLE_OPENBLAS

#include "gemm_common.h"
#include "gemm_naive.h"
#include "gemm_blas.h"
#include <cblas.h>
#include <cstdio>

static const float BLAS_TOL = 1e-3f;

static int test_blas_kernel(const char* name,
    void (*func)(const float*, const float*, float*, int, int, int),
    int M, int N, int K)
{
    Matrix A(M, K), B(K, N), C_ref(M, N), C_test(M, N);
    A.fill_random(42);
    B.fill_random(99);

    gemm_naive(A.ptr(), B.ptr(), C_ref.ptr(), M, N, K);
    func(A.ptr(), B.ptr(), C_test.ptr(), M, N, K);

    float err = max_abs_error(C_ref, C_test);
    if (err > BLAS_TOL) {
        printf("  FAIL: %-20s M=%d N=%d K=%d  error=%.2e  (tol=%.2e)\n",
               name, M, N, K, (double)err, (double)BLAS_TOL);
        return 1;
    }
    printf("  PASS: %-20s M=%d N=%d K=%d  error=%.2e\n",
           name, M, N, K, (double)err);
    return 0;
}

int run_layer5_tests() {
    int failures = 0;
    failures += test_blas_kernel("openblas", gemm_openblas,  64,  64,  64);
    failures += test_blas_kernel("openblas", gemm_openblas, 128, 128, 128);
    failures += test_blas_kernel("openblas", gemm_openblas, 257, 257, 257);
    failures += test_blas_kernel("openblas", gemm_openblas, 100, 200, 150);
    return failures;
}

#endif // ENABLE_OPENBLAS
