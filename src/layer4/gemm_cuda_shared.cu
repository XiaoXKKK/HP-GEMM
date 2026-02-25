#ifdef ENABLE_CUDA

#include "gemm_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Shared Memory Tiled GEMM Kernel ─────────────────────────────────────────
//
// KEY IDEA: instead of each thread loading its own A/B values from slow
// global memory (600+ cycle latency), threads in a block cooperate to load
// a "tile" of A and B into fast shared memory (4 cycle latency).
// Then they compute from shared memory — much faster.
//
// TILE SIZE = 32:
//   Each block: 32×32 = 1024 threads (maximum on modern GPUs)
//   Each block computes a 32×32 tile of the output C matrix
//   Shared memory used: 2 × 32×32×4 bytes = 8 KB (well within 48KB/SM limit)
//
// ALGORITHM (two-phase loop over K):
//
//   For each K-tile t:
//
//   Phase 1 — LOAD (cooperative):
//     thread (ty, tx) loads:
//       As[ty][tx] = A[row][t*TILE + tx]    (one element from A's K-tile)
//       Bs[ty][tx] = B[t*TILE + ty][col]    (one element from B's K-tile)
//     __syncthreads(): CRITICAL BARRIER — no thread can proceed until ALL
//                      threads have finished loading. Without this, some threads
//                      would use uninitialized/stale shared memory values.
//
//   Phase 2 — COMPUTE:
//     Each thread accumulates: sum += As[ty][k] * Bs[k][tx]  for k in [0, TILE)
//     This is pure shared memory reads (fast!) — no global memory traffic.
//     __syncthreads(): CRITICAL BARRIER — no thread can overwrite the shared
//                      memory tile until ALL threads have finished computing.
//
// MEMORY COALESCING:
//   Loading As[ty][tx] = A[row*K + t*TILE + tx]:
//     tx = threadIdx.x, threads 0..31 have tx=0..31 → consecutive addresses → COALESCED
//   Loading Bs[ty][tx] = B[(t*TILE+ty)*N + col]:
//     col varies by threadIdx.x → consecutive addresses → COALESCED
//
// GLOBAL MEMORY TRAFFIC REDUCTION:
//   Each element of A/B is loaded from global memory once per tile.
//   It is reused TILE_SIZE times within the tile.
//   Reduction factor = TILE_SIZE = 32x fewer global memory loads vs naive.
//
#define TILE_SIZE 32

__global__ static void shared_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float*       __restrict__ C,
                                      int M, int N, int K)
{
    // Shared memory tiles — 32×32 arrays allocated in on-chip SMEM
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;  // output row this thread computes
    int col = blockIdx.x * TILE_SIZE + tx;  // output col this thread computes

    float sum = 0.0f;

    // Iterate over K in tiles of TILE_SIZE
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {

        // ── Phase 1: Cooperative load of A and B tiles into shared memory ────
        int a_col = t * TILE_SIZE + tx;  // column index in A
        int b_row = t * TILE_SIZE + ty;  // row index in B

        // Bounds check: handle cases where M, N, K are not multiples of TILE_SIZE
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // SYNCHRONIZE: all threads must finish loading before any thread computes
        __syncthreads();

        // ── Phase 2: Compute from shared memory ───────────────────────────────
        // Each thread accumulates TILE_SIZE multiply-add operations using SMEM
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        // SYNCHRONIZE: all threads must finish computing before the next tile
        // overwrites the shared memory arrays
        __syncthreads();
    }

    // Write result (bounds check)
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void gemm_cuda_shared(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K)
{
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    shared_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif // ENABLE_CUDA
