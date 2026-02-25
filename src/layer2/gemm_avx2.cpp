#include "gemm_simd.h"
#include <immintrin.h>  // AVX2 + FMA intrinsics
#include <algorithm>

// ─── AVX2 GEMM: 1×8 register block ───────────────────────────────────────────
//
// SIMD (Single Instruction, Multiple Data) allows processing 8 float32 values
// in parallel using a single CPU instruction.
//
// AVX2 adds 256-bit "YMM" registers. Each YMM register holds 8 × float32.
// The CPU has 16 YMM registers (ymm0 – ymm15).
//
// KEY INTRINSICS USED:
//   _mm256_set1_ps(x)        — broadcast scalar x to all 8 lanes of a YMM
//   _mm256_loadu_ps(ptr)     — load 8 floats from unaligned memory address
//   _mm256_fmadd_ps(a, b, c) — fused multiply-add: c = a*b + c  (1 instruction = 2 FLOPs)
//   _mm256_storeu_ps(ptr, v) — store 8 floats to unaligned memory address
//
// MICRO-KERNEL (1×8):
//   For a fixed row i and columns j..j+7:
//     broadcast A[i][k] to all 8 lanes
//     load B[k][j..j+7] into one YMM
//     FMA: c_accum += A_broadcast * B_vec
//   After K iterations, store 8 C values at once.
//
// This processes 8 output elements per inner-loop iteration vs 1 in scalar code.
//
void gemm_avx2(const float* __restrict__ A,
               const float* __restrict__ B,
               float*       __restrict__ C,
               int M, int N, int K)
{
    constexpr int SIMD_WIDTH = 8;  // 256-bit / 32-bit = 8 floats per YMM

    for (int i = 0; i < M; ++i) {
        // Process 8 columns of C at a time (one YMM register)
        int j = 0;
        for (; j <= N - SIMD_WIDTH; j += SIMD_WIDTH) {
            // Initialize accumulator from existing C values
            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);

            // Inner loop: accumulate over K dimension
            for (int k = 0; k < K; ++k) {
                // Broadcast A[i][k] to all 8 SIMD lanes
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);

                // Load 8 consecutive B values: B[k][j..j+7]
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);

                // FMA: c_vec += a_broadcast * b_vec  (2 FLOPs per lane = 16 FLOPs total)
                c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
            }

            // Store 8 results back to C
            _mm256_storeu_ps(&C[i * N + j], c_vec);
        }

        // Handle remaining columns (tail: j to N-1) with scalar code
        for (; j < N; ++j) {
            float sum = C[i * N + j];
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}
