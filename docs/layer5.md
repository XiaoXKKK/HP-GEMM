# Layer 5: 与 OpenBLAS 对标 Benchmark

## 动机：我们离工业级实现还有多远？

前四层的实现是自底向上的优化探索。Layer 5 回答一个更实际的问题：
**我们写的代码，和工业界用了几十年打磨的 BLAS 库相比，差距在哪？差距有多大？**

这个对比本身就是简历的核心亮点——能准确知道自己和 SOTA 的差距，比盲目声称"高性能"更有说服力。

---

## OpenBLAS 是什么

[OpenBLAS](https://github.com/xianyi/OpenBLAS) 是 BLAS（Basic Linear Algebra Subprograms）接口的开源实现：

- **由 GotoBLAS2 演化而来**，作者 Kazushige Goto 是 BLAS 优化领域的传奇人物
- **架构特定优化**：对每种 CPU 微架构（Haswell、Skylake、Zen3 等）有单独的汇编微内核
- **自动多线程**：默认使用所有物理核心（通过 OpenMP 或 pthreads）
- **LAPACK 兼容**：被 NumPy、SciPy、MATLAB（底层）广泛使用

### `cblas_sgemm` 接口

```cpp
// BLAS 标准接口：C = alpha * A * B + beta * C
cblas_sgemm(
    CblasRowMajor,  // 存储顺序（行优先 = 我们的 Matrix 布局）
    CblasNoTrans,   // A 不转置
    CblasNoTrans,   // B 不转置
    M, N, K,        // 矩阵维度
    1.0f,           // alpha = 1（不缩放乘积）
    A, K,           // A 指针 + leading dimension（行优先时 = 列数 K）
    B, N,           // B 指针 + leading dimension（= N）
    0.0f,           // beta = 0（覆盖 C，等同于 C 预清零）
    C, N            // C 指针 + leading dimension（= N）
);
```

---

## 关键概念

### BLAS 分级

| Level | 操作类型 | 示例 | 复杂度 |
|-------|---------|------|--------|
| BLAS-1 | 向量-向量 | dot product, axpy | O(n) |
| BLAS-2 | 矩阵-向量 | gemv | O(n²) |
| **BLAS-3** | **矩阵-矩阵** | **sgemm** | **O(n³)** |

BLAS-3 操作（如 GEMM）有极高的**算术强度**（FLOPs / bytes），因此能充分利用 CPU 算力。
这就是为什么 DNN 训练用 GEMM 而不是逐元素操作。

### OpenBLAS 的优化策略（比我们的实现多了什么）

| 优化层次 | 我们的实现 | OpenBLAS |
|---------|----------|---------|
| 寄存器分块 | 4×8（手写） | 架构特定（如 Haswell: 4×12 或 6×16） |
| SIMD | AVX2 256-bit | AVX2/AVX-512 + FMA，汇编手写 |
| Cache 分块 | 固定 L2_BLOCK=128 | 运行时探测 L1/L2/L3 大小，动态调整 |
| 内存打包（Packing） | 无 | 将 A/B 重排为连续内存（减少 TLB miss） |
| 多线程 | OpenMP 外层循环 | 专用线程池，减少线程启动开销 |
| 预取（Prefetch） | 无 | 显式 `_mm_prefetch` 指令 |
| 微架构适配 | 无 | 编译时探测 CPU，选择最优微内核 |

**最关键的差异：数据打包（Packing）**

OpenBLAS 在计算前先将 A 和 B 的子块**重排**为连续内存布局：
```
原始 B 子块（列主序访问有 gap）:    Packed B（连续，无 stride）:
B[0][j], B[1][j], ...（stride=N） → b0, b1, b2, ...（stride=1）
```
这消除了 TLB（Translation Lookaside Buffer）压力，对大矩阵有显著提升。

---

## 实现分析

**文件**：`src/layer5/gemm_openblas.cpp`

```cpp
void gemm_openblas(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,   // leading dim of A (row-major) = K
                B, N,         // leading dim of B = N
                0.0f, C, N);  // beta=0: overwrite C
}
```

封装极其简单——复杂度全在 OpenBLAS 内部。

### 线程控制

```bash
# 方法 1：环境变量（影响整个进程）
OPENBLAS_NUM_THREADS=1 ./hp_gemm_bench --layer 5 --size 1024

# 方法 2：代码内动态设置（bench_layer5.cpp 使用此方法）
openblas_set_num_threads(1);  // 单线程对比 Layer 2
openblas_set_num_threads(0);  // 0 = 恢复默认（所有核心）
```

---

## 性能结果

> 运行后填入：
> ```bash
> # 安装 OpenBLAS
> sudo apt-get install libopenblas-dev
>
> # 重新构建（从 WSL2 项目目录）
> cmake -S . -B build_blas -DENABLE_OPENBLAS=ON -DENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
> cmake --build build_blas -j$(nproc)
> cd build_blas && ctest --output-on-failure
>
> # 运行 Layer 5 benchmark
> ./hp_gemm_bench --layer 5 --size all
> ```

### 单线程对比（我们的最好实现 vs OpenBLAS 单线程）

| N | avx2 4×8 (GFLOPS) | OpenBLAS 1T (GFLOPS) | 差距 |
|---|------------------|---------------------|------|
| 256 | 47.99 | [填入] | [填入]× |
| 512 | 35.24 | [填入] | [填入]× |
| 1024 | 42.18 | [填入] | [填入]× |
| 2048 | 7.25 | [填入] | [填入]× |
| 4096 | 6.27 | [填入] | [填入]× |

### 多线程对比（OpenMP Layer3 vs OpenBLAS 多线程）

| N | OpenMP 8T (GFLOPS) | OpenBLAS 8T (GFLOPS) | 差距 |
|---|-------------------|---------------------|------|
| 1024 | ~[填入] | [填入] | [填入]× |
| 2048 | 43.13 | [填入] | [填入]× |
| 4096 | [填入] | [填入] | [填入]× |

---

## 差距分析：为什么 OpenBLAS 更快

根据已有的 Layer 1-4 数据分析预期差距：

**大矩阵（N=2048+）差距大的原因**：
1. **数据打包（Packing）**：我们的 4×8 kernel 在 k-loop 中读取 B 时有 stride-N 访问，
   TLB miss 随 N 增大而增加。OpenBLAS 打包后消除此问题。
2. **更优的 Cache 分块参数**：OpenBLAS 运行时探测 CPU 缓存大小，我们固定使用 L2_BLOCK=128。
3. **更大的寄存器块**：现代 CPU 有 16 个 YMM 寄存器，OpenBLAS 的微内核往往使用 6×16 或 4×24
   的寄存器块（取决于 CPU），而我们只用了 4×8。

**小矩阵（N≤256）差距小的原因**：
数据完全在 L1/L2 Cache 中，打包收益不明显；我们的实现也能充分利用 SIMD。

---

## 理论峰值计算（填入实测后对比）

```bash
# i7-10700: 2.9 GHz base, 4.8 GHz boost（单核）, 16 线程
# 单核理论峰值（AVX2 + FMA）：
#   4.8 GHz × 2 FMA ports × 8 floats/YMM × 2 FLOPs/FMA = 153.6 GFLOPS
# 8核（不含超线程）：
#   4.0 GHz × 8 × 2 × 8 × 2 = 1024 GFLOPS（理论上限）

# 查看实际频率
grep "cpu MHz" /proc/cpuinfo | head -8
```

| 实现 | 测量值（N=1024） | 单核峰值利用率 |
|------|---------------|--------------|
| avx2 4×8 (单线程) | 42.18 GFLOPS | 42.18/153.6 = **27.5%** |
| OpenBLAS 单线程 | [填入] | [填入]% |
| OpenMP 8线程 | ~[填入] | [填入]% |
| OpenBLAS 多线程 | [填入] | [填入]% |

---

## 简历模板

```
高性能矩阵乘法多层优化 | C++17, CUDA, AVX2, OpenMP  [2025.xx]
- 从朴素 O(n³) 实现出发，通过 ikj 循环重排 + 两级 Cache 分块（L2 Tile=128×128，
  寄存器块=4×4），Cache Miss 率降低至 <5%，实测提升 [填入]× （N=2048）
- 手写 AVX2 4×8 FMA 寄存器块 micro-kernel（_mm256_fmadd_ps），
  达到 42.18 GFLOPS（单核理论峰值的 27.5%）
- OpenMP schedule(static) 并行化外层循环，8 核获得 5.88× 加速比
  （效率 73.5%，内存带宽饱和由 perf LLC-miss 计数验证）
- CUDA Shared Memory Tiling（TILE=32），全局内存流量降低 32×，
  RTX 3060 上达到 501 GFLOPS（vs CPU naive 343×）
- 与 OpenBLAS [版本] 对标：单线程达到其 [填入]%，多线程达到 [填入]%；
  差距原因：缺少数据打包（Packing）和运行时 Cache 参数自适应
```

---

## 延伸阅读

| 资源 | 说明 |
|------|------|
| [BLIS 论文](https://dl.acm.org/doi/10.1145/2764454) | 现代 GEMM 优化的学术参考实现 |
| [BLISlab 教程](https://github.com/flame/blislab) | 从零实现 GEMM 到 BLAS 级别 |
| [Goto & van de Geijn 2008](https://dl.acm.org/doi/10.1145/1356052.1356053) | GotoBLAS 算法原始论文 |
| [how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm) | 中文教程，与本项目路线高度吻合 |
