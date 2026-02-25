#pragma once

// ─── Layer 1 kernel declarations ──────────────────────────────────────────────

// Naive ijk loop: inner loop accesses B column-wise (stride-N, cache-unfriendly)
void gemm_naive(const float* A, const float* B, float* C,
                int M, int N, int K);

// Loop reordered to ikj: A[i][k] broadcast, B and C accessed row-wise (stride-1)
void gemm_ikj(const float* A, const float* B, float* C,
              int M, int N, int K);

// Two-level cache blocking: L2 tile (128x128) + register micro-kernel (4x4)
void gemm_blocked(const float* A, const float* B, float* C,
                  int M, int N, int K);
