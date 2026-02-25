#include "gemm_cuda.h"
#ifdef ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ─── Helper: check CUDA errors ────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Naive CUDA Kernel ────────────────────────────────────────────────────────
//
// EXECUTION MODEL:
//   Grid of (ceil(N/16) × ceil(M/16)) blocks
//   Each block: 16×16 = 256 threads
//   Each thread computes ONE output element C[row][col]
//
// MEMORY ACCESS PATTERN:
//   Thread (tx, ty) in block (bx, by):
//     row = by * 16 + ty
//     col = bx * 16 + tx
//
//   For the inner K-loop:
//     A[row][k] = A[row*K + k]  → all threads in same row access same k → NOT coalesced
//     B[k][col] = B[k*N + col]  → threads in same warp have adjacent col → COALESCED
//
//   Problem: A access is stride-N across warp threads (warp = 32 consecutive threadIdx.x).
//   This causes "uncoalesced" global memory reads → bandwidth waste.
//   Layer 4 shared memory kernel fixes this.
//
__global__ static void naive_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float*       __restrict__ C,
                                     int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void gemm_cuda_naive(const float* d_A, const float* d_B, float* d_C,
                     int M, int N, int K)
{
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    naive_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── CudaMatrix RAII ─────────────────────────────────────────────────────────
CudaMatrix::CudaMatrix(int r, int c) : rows(r), cols(c) {
    CUDA_CHECK(cudaMalloc(&d_ptr, static_cast<size_t>(r) * c * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, static_cast<size_t>(r) * c * sizeof(float)));
}

CudaMatrix::~CudaMatrix() {
    if (d_ptr) cudaFree(d_ptr);
}

void CudaMatrix::copy_from_host(const float* h_ptr) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr,
                          static_cast<size_t>(rows) * cols * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void CudaMatrix::copy_to_host(float* h_ptr) const {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr,
                          static_cast<size_t>(rows) * cols * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

#endif // ENABLE_CUDA
