# Layer 4: CUDA GPU 实现

## 动机：为什么需要 GPU？

CPU 的 SIMD + 多核方案已经能达到数百 GFLOPS，但 GPU 的算力体量完全不同：

| 硬件 | 核心/单元数 | 理论 GFLOPS（FP32） |
|------|-----------|-------------------|
| Intel i7-12700 (16核) | 16 P-cores | ~400 GFLOPS |
| NVIDIA RTX 3080 | 8704 CUDA cores | ~29700 GFLOPS |
| NVIDIA RTX 4090 | 16384 CUDA cores | ~82600 GFLOPS |

GPU 通过**大规模并行**弥补单核性能较低的劣势：数千个小型处理器同时工作。

**适用场景**：深度学习训练、科学计算、金融蒙特卡洛模拟——任何可以分解成大量独立小任务的问题。

---

## 关键概念

### CUDA 执行模型

```
Grid（网格）
  └── Block（线程块，若干个）
        └── Thread（线程，若干个）

GPU 以 Warp（32 个线程）为最小调度单元
所有同一 Warp 的线程执行相同指令（SIMT 模型）
```

对于矩阵乘法 C(M×N) = A(M×K) × B(K×N)：
```
每个 Thread  → 计算 1 个输出元素 C[row][col]
每个 Block   → 计算一个 TILE_SIZE×TILE_SIZE 的 C 子矩阵
Grid         → 覆盖整个 M×N 输出矩阵
```

### GPU 内存层次

| 内存类型 | 容量 | 延迟 | 带宽 | 作用域 |
|---------|------|------|------|--------|
| 全局内存（Global） | 8-24 GB | 200+ 周期 | ~1 TB/s | 所有线程可见 |
| 共享内存（Shared） | 48-96 KB/SM | 4 周期 | ~20 TB/s | 同一 Block 线程 |
| 寄存器（Register） | ~256 KB/SM | 0 周期 | 无限 | 单个线程 |
| L1/Texture Cache | 32-128 KB/SM | 30-50 周期 | ~4 TB/s | 自动管理 |

**关键**：Shared Memory 比 Global Memory 快约 **100×**，是 CUDA 优化的核心工具。

### 内存合并访问（Memory Coalescing）

一个 Warp（32 个线程）同时访问内存时，如果 32 个线程的地址是连续的，
GPU 可以将 32 次独立访问合并为 **1 次内存事务**（128 字节，32×4 字节）。

```
合并访问（好）：
  thread 0 → addr 0, thread 1 → addr 4, ..., thread 31 → addr 124
  → 1 次内存事务，效率 100%

非合并访问（坏）：
  thread 0 → addr 0, thread 1 → addr 512, ..., thread 31 → addr 15872
  → 32 次独立内存事务，效率 3%
```

---

## 实现分析

### Step 1：朴素 CUDA Kernel

**文件**：`src/layer4/gemm_cuda_naive.cu`

```cuda
__global__ void naive_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 输出行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 输出列

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];  // A 访问 row 固定，k 变 → 同一 Warp 非合并
        C[row * N + col] = sum;
    }
}
```

**内存访问分析**：
- `B[k*N + col]`：col = blockIdx.x * 16 + threadIdx.x → 同一 Warp 的 threadIdx.x=0..15 对应连续地址 → **合并访问（好）**
- `A[row*K + k]`：row = blockIdx.y * 16 + threadIdx.y → 同一 Warp（相同 threadIdx.y，不同 threadIdx.x）对应 **相同地址** → 广播（OK，但浪费带宽）

朴素 kernel 虽然简单，但每个 k 迭代都访问全局内存（无缓存重用）。

**启动配置**：
```cuda
dim3 threads(16, 16);   // 256 threads per block
dim3 blocks((N+15)/16, (M+15)/16);
```

---

### Step 2：Shared Memory Tiling（核心优化）

**文件**：`src/layer4/gemm_cuda_shared.cu`

**核心思想**：每个 Block 计算一个 32×32 的 C 子矩阵。
在计算前，32×32 的线程**协作**从全局内存加载 A 和 B 的小 Tile 到共享内存，
然后从（快速的）共享内存计算。

```
全局内存访问次数（naive）  = M × N × K × 2 次（每次 k 迭代加载 A 和 B 各 1 次）
全局内存访问次数（tiling） = M × N × K × 2 / TILE_SIZE 次
```

当 TILE_SIZE=32 时，全局内存访问减少 **32×**。

```cuda
#define TILE_SIZE 32

__global__ void shared_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    // 共享内存：每个 Block 独有，访问延迟 4 周期
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历 K 维的 Tile
    for (int t = 0; t < (K + TILE_SIZE-1) / TILE_SIZE; ++t) {

        // ── 阶段 1：协作加载 ──────────────────────────────────────────────
        // 1024 个线程（32×32 Block）协作加载两个 32×32 Tile
        // 每个线程加载 1 个 A 元素 + 1 个 B 元素

        int a_col = t * TILE_SIZE + tx;       // A 中的列索引
        int b_row = t * TILE_SIZE + ty;       // B 中的行索引

        // 边界保护：Tile 超出矩阵边界时填 0
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // ── 关键同步 1 ───────────────────────────────────────────────────
        // 必须等所有线程完成加载后，才能开始计算
        // 否则某些线程会读取到未初始化的 As/Bs 内容
        __syncthreads();

        // ── 阶段 2：从共享内存计算（100× 比全局内存快）───────────────────
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        // ── 关键同步 2 ───────────────────────────────────────────────────
        // 必须等所有线程完成计算后，下一次循环才能覆写 As/Bs
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**`__syncthreads()` 的两处作用**：
1. 阶段 1 后：保证所有 As/Bs 元素已被加载完毕（读写安全）
2. 阶段 2 后：保证所有线程完成计算，下次循环可以安全覆写共享内存

漏掉任何一个 `__syncthreads()` 都会导致数据竞争和错误结果。

---

### RAII 设备内存管理

**文件**：`include/gemm_cuda.h` + `src/layer4/gemm_cuda_naive.cu`

```cpp
struct CudaMatrix {
    float* d_ptr;  // device pointer
    int rows, cols;

    CudaMatrix(int r, int c) {
        cudaMalloc(&d_ptr, r * c * sizeof(float));
        cudaMemset(d_ptr, 0, r * c * sizeof(float));
    }

    ~CudaMatrix() { cudaFree(d_ptr); }  // 析构时自动释放 GPU 内存

    void copy_from_host(const float* h) {
        cudaMemcpy(d_ptr, h, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    }
};
```

RAII 模式确保 GPU 内存不泄漏（类似 C++ 的 unique_ptr）。

---

## WSL2 CUDA 环境配置

```bash
# 1. 验证 GPU 可见（使用 Windows 驱动直通，不在 WSL2 内安装驱动）
nvidia-smi
# 正常输出应显示 GPU 型号、驱动版本、CUDA 版本

# 2. 在 WSL2 (Ubuntu 22.04) 内安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install cuda-toolkit-12-3

# 3. 设置环境变量（加入 ~/.bashrc）
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 4. 验证 nvcc
nvcc --version

# 5. 构建 CUDA 版本
bash scripts/build_cuda.sh
```

**常见问题**：
- `nvidia-smi` 失败 → Windows NVIDIA 驱动版本需 >= 525.x
- nvcc 找不到 → 检查 PATH 是否包含 /usr/local/cuda/bin
- 构建失败 `sm_XX not supported` → 修改 `CMakeLists.txt` 中的 `CMAKE_CUDA_ARCHITECTURES`

查看 GPU 的 Compute Capability：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## 性能结果

> 运行后填入：
> ```bash
> build_cuda/hp_gemm_bench --layer 4 --size all
> ```

| 矩阵大小 | CPU naive (GFLOPS) | CUDA naive (GFLOPS) | CUDA shared (GFLOPS) |
|---------|------------------|-------------------|---------------------|
| 256×256 | [填入] | [填入] | [填入] |
| 512×512 | [填入] | [填入] | [填入] |
| 1024×1024 | [填入] | [填入] | [填入] |
| 2048×2048 | [填入] | [填入] | [填入] |

**注意**：小矩阵（256×256）下 GPU 可能比 CPU 慢——GPU 启动 kernel 有固定开销（数微秒），
而 CPU 处理小矩阵只需几毫秒。GPU 的优势在大矩阵（N ≥ 1024）时才能体现。

---

## 简历模板

```
- 实现 CUDA GPU GEMM：朴素 kernel（每线程 1 个输出元素）到 Shared Memory Tiling（32×32 Tile），
  全局内存访问次数减少 32×
- 识别并解决 __syncthreads() 两处关键同步点（Phase1 加载完成、Phase2 计算完成），
  防止共享内存数据竞争
- 在 [GPU 型号] 上达到 [填入] GFLOPS，是 CPU 朴素实现的 [填入]×；
  与 cuBLAS 对比：相当于 [填入]% cuBLAS 吞吐量
```
