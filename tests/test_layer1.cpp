#include "gemm_common.h"
#include "gemm_naive.h"
#include <cstdio>
#include <cstring>

// ─── Test helpers ─────────────────────────────────────────────────────────────

static const float CPU_TOL = 1e-3f;

static int test_kernel(const char* name, GemmFunc func, int M, int N, int K) {
    Matrix A(M, K), B(K, N), C_ref(M, N), C_test(M, N);
    A.fill_random(42);
    B.fill_random(99);

    // Reference: naive ijk
    gemm_naive(A.ptr(), B.ptr(), C_ref.ptr(), M, N, K);

    // Under test
    func(A.ptr(), B.ptr(), C_test.ptr(), M, N, K);

    float err = max_abs_error(C_ref, C_test);
    if (err > CPU_TOL) {
        printf("  FAIL: %-20s M=%d N=%d K=%d  error=%.2e  (tol=%.2e)\n",
               name, M, N, K, (double)err, (double)CPU_TOL);
        return 1;
    }
    printf("  PASS: %-20s M=%d N=%d K=%d  error=%.2e\n",
           name, M, N, K, (double)err);
    return 0;
}

// ─── Layer 1 tests ────────────────────────────────────────────────────────────

int run_layer1_tests() {
    int failures = 0;

    // Edge cases: very small matrices
    failures += test_kernel("naive",   gemm_naive,   4,   4,   4);
    failures += test_kernel("ikj",     gemm_ikj,     4,   4,   4);
    failures += test_kernel("blocked", gemm_blocked, 4,   4,   4);

    // Small power-of-2
    failures += test_kernel("naive",   gemm_naive,   8,   8,   8);
    failures += test_kernel("ikj",     gemm_ikj,     8,   8,   8);
    failures += test_kernel("blocked", gemm_blocked, 8,   8,   8);

    // Medium
    failures += test_kernel("ikj",     gemm_ikj,     64,  64,  64);
    failures += test_kernel("blocked", gemm_blocked, 64,  64,  64);
    failures += test_kernel("ikj",     gemm_ikj,     128, 128, 128);
    failures += test_kernel("blocked", gemm_blocked, 128, 128, 128);

    // Non-power-of-2: critical for boundary correctness in blocking
    failures += test_kernel("ikj",     gemm_ikj,     257, 257, 257);
    failures += test_kernel("blocked", gemm_blocked, 257, 257, 257);
    failures += test_kernel("ikj",     gemm_ikj,     100, 200, 150);
    failures += test_kernel("blocked", gemm_blocked, 100, 200, 150);

    // Non-square
    failures += test_kernel("blocked", gemm_blocked, 513, 256, 128);

    return failures;
}
