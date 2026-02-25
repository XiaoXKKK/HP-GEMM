#include "gemm_common.h"
#include "gemm_naive.h"
#include "gemm_simd.h"
#include <cstdio>

static const float SIMD_TOL = 1e-2f;  // FMA reordering can introduce tiny differences

static int test_kernel(const char* name, GemmFunc func, int M, int N, int K) {
    Matrix A(M, K), B(K, N), C_ref(M, N), C_test(M, N);
    A.fill_random(42);
    B.fill_random(99);

    gemm_naive(A.ptr(), B.ptr(), C_ref.ptr(), M, N, K);
    func(A.ptr(), B.ptr(), C_test.ptr(), M, N, K);

    float err = max_abs_error(C_ref, C_test);
    if (err > SIMD_TOL) {
        printf("  FAIL: %-24s M=%d N=%d K=%d  error=%.2e  (tol=%.2e)\n",
               name, M, N, K, (double)err, (double)SIMD_TOL);
        return 1;
    }
    printf("  PASS: %-24s M=%d N=%d K=%d  error=%.2e\n",
           name, M, N, K, (double)err);
    return 0;
}

int run_layer2_tests() {
    int failures = 0;

    // Power-of-2 sizes (aligned paths)
    failures += test_kernel("avx2_1x8",       gemm_avx2,          8,   8,   8);
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  8,   8,   8);
    failures += test_kernel("avx2_1x8",       gemm_avx2,          64,  64,  64);
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  64,  64,  64);
    failures += test_kernel("avx2_1x8",       gemm_avx2,          128, 128, 128);
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  128, 128, 128);

    // Non-power-of-2: tests tail handling (j%8 != 0, rows%4 != 0)
    failures += test_kernel("avx2_1x8",       gemm_avx2,          100, 100, 100);
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  100, 100, 100);
    failures += test_kernel("avx2_1x8",       gemm_avx2,          257, 257, 257);
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  257, 257, 257);

    // Non-square
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  100, 200, 150);
    failures += test_kernel("avx2_4x8",       gemm_simd_blocked,  513, 256, 128);

    return failures;
}
