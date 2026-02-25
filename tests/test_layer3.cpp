#include "gemm_common.h"
#include "gemm_naive.h"
#include "gemm_openmp.h"
#include <cstdio>

static const float OMP_TOL = 1e-2f;

static int test_kernel(const char* name, GemmFunc func, int M, int N, int K) {
    Matrix A(M, K), B(K, N), C_ref(M, N), C_test(M, N);
    A.fill_random(42);
    B.fill_random(99);

    gemm_naive(A.ptr(), B.ptr(), C_ref.ptr(), M, N, K);
    func(A.ptr(), B.ptr(), C_test.ptr(), M, N, K);

    float err = max_abs_error(C_ref, C_test);
    if (err > OMP_TOL) {
        printf("  FAIL: %-20s M=%d N=%d K=%d  error=%.2e  (tol=%.2e)\n",
               name, M, N, K, (double)err, (double)OMP_TOL);
        return 1;
    }
    printf("  PASS: %-20s M=%d N=%d K=%d  error=%.2e\n",
           name, M, N, K, (double)err);
    return 0;
}

int run_layer3_tests() {
    int failures = 0;

    failures += test_kernel("openmp", gemm_openmp,  64,  64,  64);
    failures += test_kernel("openmp", gemm_openmp,  128, 128, 128);
    failures += test_kernel("openmp", gemm_openmp,  257, 257, 257);
    failures += test_kernel("openmp", gemm_openmp,  100, 200, 150);
    failures += test_kernel("openmp", gemm_openmp,  512, 512, 512);

    return failures;
}
