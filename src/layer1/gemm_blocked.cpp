#include "gemm_naive.h"
#include <algorithm>  // std::min

// ─── Two-Level Cache-Blocked GEMM ────────────────────────────────────────────
//
// PROBLEM WITH ikj: even with stride-1 access, at large N the working set
// of the inner loops exceeds L2 cache. We repeatedly evict and reload data.
//
// SOLUTION: divide the computation into tiles that FIT in L2 cache.
//
// ── Level 1: L2 Blocking ──────────────────────────────────────────────────────
//
// Choose L2_BLOCK so that 3 tiles (A_tile, B_tile, C_tile) fit in L2:
//   tile_bytes = L2_BLOCK * L2_BLOCK * 4 bytes (float)
//   3 * L2_BLOCK^2 * 4 <= L2_size (e.g. 256 KB)
//   L2_BLOCK <= sqrt(256*1024 / 12) ≈ 148 → round down to 128
//
// The outer 3 loops (i0, k0, j0) iterate over L2-sized tiles.
// Within each (i0, k0, j0) tile, the data fits in L2 → no evictions.
//
// ── Level 2: Register Blocking (Micro-Kernel) ─────────────────────────────────
//
// Within each L2 tile, we further block into REG_M × REG_N sub-tiles.
// The REG_M * REG_N float accumulators are kept in CPU registers across
// the inner k-loop. This avoids repeated load/store to C in memory.
//
// REG_M=4, REG_N=4 → 16 float registers. Modern CPUs have 16 XMM registers
// (or 16 YMM for AVX2). We stay within the register file.
//
// Combined effect: near-optimal use of the memory hierarchy:
//   Registers: C accumulators (hot, no memory traffic)
//   L1 cache:  A and B micro-tiles during inner k-loop
//   L2 cache:  full A, B, C tiles for the L2 block
//
static constexpr int L2_BLOCK = 128;  // L2 tile size (tune for your L2 size)
static constexpr int REG_M    = 4;    // rows in register micro-kernel
static constexpr int REG_N    = 4;    // cols in register micro-kernel

void gemm_blocked(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float*       __restrict__ C,
                  int M, int N, int K)
{
    // ── Outer L2 tiles ────────────────────────────────────────────────────────
    for (int i0 = 0; i0 < M; i0 += L2_BLOCK) {
        int i_end = std::min(i0 + L2_BLOCK, M);

        for (int k0 = 0; k0 < K; k0 += L2_BLOCK) {
            int k_end = std::min(k0 + L2_BLOCK, K);

            for (int j0 = 0; j0 < N; j0 += L2_BLOCK) {
                int j_end = std::min(j0 + L2_BLOCK, N);

                // ── Inner register micro-kernel ───────────────────────────────
                // Process REG_M rows of A × REG_N cols of B at a time.
                // Accumulators c[ri][rj] stay in registers across the k-loop.
                for (int i = i0; i < i_end; i += REG_M) {
                    for (int j = j0; j < j_end; j += REG_N) {

                        // Load C accumulators into registers
                        float c[REG_M][REG_N] = {};
                        int ri_max = std::min(REG_M, i_end - i);
                        int rj_max = std::min(REG_N, j_end - j);

                        for (int ri = 0; ri < ri_max; ++ri)
                            for (int rj = 0; rj < rj_max; ++rj)
                                c[ri][rj] = C[(i + ri) * N + (j + rj)];

                        // Accumulate over k-dimension
                        for (int k = k0; k < k_end; ++k) {
                            for (int ri = 0; ri < ri_max; ++ri) {
                                float a_val = A[(i + ri) * K + k];
                                for (int rj = 0; rj < rj_max; ++rj)
                                    c[ri][rj] += a_val * B[k * N + (j + rj)];
                            }
                        }

                        // Write back accumulators to C
                        for (int ri = 0; ri < ri_max; ++ri)
                            for (int rj = 0; rj < rj_max; ++rj)
                                C[(i + ri) * N + (j + rj)] = c[ri][rj];
                    }
                }

            }  // j0
        }  // k0
    }  // i0
}
