#pragma once
#ifdef ENABLE_CUDA

// ─── Layer 4 CUDA kernel host-side declarations ───────────────────────────────

// Naive CUDA kernel: one thread per output element C[row][col]
// Launch: dim3(16,16) threads, grid covers full M×N output
void gemm_cuda_naive(const float* d_A, const float* d_B, float* d_C,
                     int M, int N, int K);

// Shared memory tiling: TILE_SIZE=32, cooperative tile loading
// Reduces global memory traffic by TILE_SIZE factor vs naive
void gemm_cuda_shared(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K);

// ─── RAII device memory wrapper ───────────────────────────────────────────────
struct CudaMatrix {
    float* d_ptr = nullptr;
    int rows, cols;

    CudaMatrix(int r, int c);
    ~CudaMatrix();

    // Copy from host to device (h_ptr must have rows*cols floats)
    void copy_from_host(const float* h_ptr);
    // Copy from device to host
    void copy_to_host(float* h_ptr) const;

    // Prevent accidental copies
    CudaMatrix(const CudaMatrix&)            = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;
};

#endif // ENABLE_CUDA
