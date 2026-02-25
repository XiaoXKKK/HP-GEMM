#include <cstdio>
#include <cstdlib>

// test_main.cpp: test runner entry point

extern int run_layer1_tests();
extern int run_layer2_tests();
extern int run_layer3_tests();
#ifdef ENABLE_CUDA
extern int run_layer4_tests();
#endif
#ifdef ENABLE_OPENBLAS
extern int run_layer5_tests();
#endif

int main() {
    int failures = 0;

    printf("\n=== HP-GEMM Correctness Tests ===\n\n");

    printf("--- Layer 1: Cache-friendly GEMM ---\n");
    failures += run_layer1_tests();

    printf("\n--- Layer 2: AVX2 SIMD GEMM ---\n");
    failures += run_layer2_tests();

    printf("\n--- Layer 3: OpenMP GEMM ---\n");
    failures += run_layer3_tests();

#ifdef ENABLE_CUDA
    printf("\n--- Layer 4: CUDA GEMM ---\n");
    failures += run_layer4_tests();
#endif

#ifdef ENABLE_OPENBLAS
    printf("\n--- Layer 5: OpenBLAS GEMM ---\n");
    failures += run_layer5_tests();
#endif

    printf("\n=== Summary: %d test(s) failed ===\n\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
