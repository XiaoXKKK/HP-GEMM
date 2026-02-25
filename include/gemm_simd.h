#pragma once

// ─── Layer 2 kernel declarations ──────────────────────────────────────────────

// Basic AVX2 kernel: 1x8 register block using _mm256_fmadd_ps
// Processes 8 output columns at a time per row
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K);

// Advanced AVX2 kernel: 4x8 register block (4 rows × 8 cols per micro-kernel)
// Maximizes FMA throughput by keeping 4 YMM accumulators in registers
void gemm_simd_blocked(const float* A, const float* B, float* C,
                       int M, int N, int K);
