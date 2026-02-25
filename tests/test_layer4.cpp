#ifdef ENABLE_CUDA

#include "gemm_common.h"
#include "gemm_naive.h"
#include "gemm_cuda.h"
#include <cstdio>

static const float CUDA_TOL = 1e-2f;

static int test_cuda_kernel(const char* name,
    void (*func)(const float*, const float*, float*, int, int, int),
    int M, int N, int K)
{
    // Host matrices
    Matrix A(M, K), B(K, N), C_ref(M, N), C_host(M, N);
    A.fill_random(42);
    B.fill_random(99);
    gemm_naive(A.ptr(), B.ptr(), C_ref.ptr(), M, N, K);

    // Device matrices
    CudaMatrix d_A(M, K), d_B(K, N), d_C(M, N);
    d_A.copy_from_host(A.ptr());
    d_B.copy_from_host(B.ptr());

    // Run CUDA kernel
    func(d_A.d_ptr, d_B.d_ptr, d_C.d_ptr, M, N, K);

    // Copy result back
    d_C.copy_to_host(C_host.ptr());

    float err = max_abs_error(C_ref, C_host);
    if (err > CUDA_TOL) {
        printf("  FAIL: %-24s M=%d N=%d K=%d  error=%.2e  (tol=%.2e)\n",
               name, M, N, K, (double)err, (double)CUDA_TOL);
        return 1;
    }
    printf("  PASS: %-24s M=%d N=%d K=%d  error=%.2e\n",
           name, M, N, K, (double)err);
    return 0;
}

int run_layer4_tests() {
    int failures = 0;

    failures += test_cuda_kernel("cuda_naive",  gemm_cuda_naive,  64,  64,  64);
    failures += test_cuda_kernel("cuda_shared", gemm_cuda_shared, 64,  64,  64);
    failures += test_cuda_kernel("cuda_naive",  gemm_cuda_naive,  128, 128, 128);
    failures += test_cuda_kernel("cuda_shared", gemm_cuda_shared, 128, 128, 128);
    failures += test_cuda_kernel("cuda_shared", gemm_cuda_shared, 257, 257, 257);  // non-power-of-2
    failures += test_cuda_kernel("cuda_shared", gemm_cuda_shared, 100, 200, 150);  // non-square

    return failures;
}

#endif // ENABLE_CUDA
