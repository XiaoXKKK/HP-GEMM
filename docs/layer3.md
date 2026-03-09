# Layer 3: OpenMP 多线程并行

## 动机：Layer 2 的瓶颈在哪？

Layer 2 的 4×8 SIMD kernel 已经充分利用了单核的向量计算能力。
但现代 CPU 有多个物理核心——每个核心都有自己的 ALU 和 FMA 单元。

**问题**：Layer 2 只用了 1 个核心，其他核心空闲。

**解决方案**：使用 OpenMP 将工作分发给多个核心，实现**数据并行**。

---

## 关键概念

### OpenMP 编程模型

OpenMP 是 C/C++ 的多线程并行编程 API，通过编译指令（pragma）来标注并行区域：

```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i)
    work(i);
// 运行时：N 次迭代被分配给 OMP_NUM_THREADS 个线程并行执行
```

编译时需要 `-fopenmp` flag，链接时需要 `-lgomp`（Linux）。

### 为什么可以并行化 i-循环？

在矩阵乘法中，不同行 i 的输出 `C[i][*]` 是**完全独立**的：
- 计算 `C[i][j]` 只需要 A 的第 i 行和 B 的整列
- 不同 i 之间没有写入相同内存位置的情况

因此，不同线程处理不同的 i 区间是**无数据竞争**的。

### 为什么不并行化 k-循环？

```cpp
// 危险！多线程同时写入 C[i][j]
#pragma omp parallel for  // ← 这是错的
for (int k = 0; k < K; ++k)
    C[i][j] += A[i][k] * B[k][j];  // 数据竞争！
```

多个 k 的累加都要写入同一个 `C[i][j]`，这是典型的**数据竞争**（write-write conflict）。需要 atomic 或 reduction 才能正确，但会极大降低性能。

### 假共享（False Sharing）

```
Cache Line = 64 字节 = 16 个 float

线程 0 写入 C[0][0]~C[0][15]   ← 同一 Cache Line
线程 1 写入 C[0][16]~C[0][31]  ← 如果两个写入在同一 Cache Line，会触发 Cache 一致性协议

本项目按行块分配工作（步长 4 行），行宽 N >> 16，
所以线程之间不共享 Cache Line → 无假共享。
```

### Amdahl 定律

```
加速比 S(n) = 1 / (f_serial + (1 - f_serial) / n)

其中 f_serial = 串行部分比例，n = 核心数

若 f_serial = 5%：
  S(4)  = 1 / (0.05 + 0.95/4)  ≈ 3.5×
  S(8)  = 1 / (0.05 + 0.95/8)  ≈ 5.9×
  S(16) = 1 / (0.05 + 0.95/16) ≈ 8.9×
```

实际中受**内存带宽**限制：所有核心共享 L3 Cache 和 DRAM 带宽。
当线程数超过内存带宽瓶颈时，增加线程反而无法线性提升。

---

## 实现分析

**文件**：`src/layer3/gemm_openmp.cpp`

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < M; i += ROW_BLOCK) {
    int i_end = std::min(i + ROW_BLOCK, M);
    gemm_simd_block_rows(A, B, C, i, i_end, M, N, K);
}
```

**设计要点**：

1. **步长 = ROW_BLOCK = 4**：与 Layer 2 的 4×8 SIMD micro-kernel 对齐
   - 每次循环分配 4 行给一个线程
   - 保持 SIMD 对齐，不破坏 Layer 2 的向量化效果

2. **`schedule(static)`**：静态调度
   - 编译时将 M/4 个工作单元平均分给 n 个线程
   - 每个线程得到连续的行块（减少 Cache Line 竞争）
   - 工作量均匀（所有线程处理相同数量的行），static 是最优选择

3. **零同步开销**：
   - 没有锁，没有 atomic
   - `#pragma omp barrier` 隐式在并行区域结束时执行一次（等待所有线程完成）

---

## 性能结果（线程扩展性）

测试环境：Intel i7-10700 @ 2.90GHz（8 物理核心 / 16 逻辑核心），N=2048

**N=2048 矩阵，线程扩展性**：

| 线程数 | GFLOPS | vs 1 线程 | 效率 |
|-------|--------|----------|------|
| 1  | 7.33  | 1.00× | 100.00% |
| 2  | 13.79 | 1.88× | 94.07%  |
| 4  | 28.19 | 3.85× | 96.15%  |
| 8  | 43.13 | 5.88× | 73.57%  |
| 16 | 42.08 | 5.74× | 35.88%  |

**效率 = 实际加速比 / 理论线程数**，反映了内存带宽饱和程度。

关键观察：
- **4 线程（4 物理核心）效率 96.15%**：接近完美线性加速，说明并行化开销极小
- **8 线程效率跌至 73.57%**：L3/DRAM 带宽成为瓶颈，多核同时读 A/B 矩阵耗尽带宽
- **16 线程效率仅 35.88%**：超线程（2 逻辑核心共享 1 套 ALU），不能带来额外算力

**查看 CPU 核心数**：
```bash
nproc           # 逻辑核心数（含超线程）
lscpu | grep Core  # 物理核心数
```

---

## 分析子线性加速的原因

使用 `perf` 观察内存带宽利用率：

```bash
# 测量内存带宽使用情况
perf stat -e LLC-load-misses,LLC-store-misses \
    build_cpu/hp_gemm_bench --layer 3 --size 2048
```

当线程数超过某个阈值后，LLC-load-misses 不再增加但 GFLOPS 也不再线性提升，
说明已达到 **DRAM 带宽上限**（典型值：DDR4 双通道 ~40-50 GB/s）。

---

## 简历模板

```
- 以 OpenMP `#pragma omp parallel for schedule(static)` 并行化 SIMD kernel 的
  最外层 i 循环，每线程处理不相交的连续行块（无假共享，无数据竞争）
- 8 物理核心下获得 5.88× 加速比（效率 73.57%），4 核心下获得 3.85× 加速比
  （效率 96.15%）；sub-linear 原因为 L3/DRAM 带宽饱和（由 perf LLC-miss 计数验证）
```

---

## 面试问答

### Q1：OpenMP 怎么用的？

**答**：

1. **编译**：加 `-fopenmp` flag（链接 `-lgomp`），CMake 中用 `find_package(OpenMP)`。

2. **核心 pragma**：在最外层 i-循环前加一行：
   ```cpp
   #pragma omp parallel for schedule(static)
   for (int i = 0; i < M; i += ROW_BLOCK) {
       gemm_simd_block_rows(A, B, C, i, std::min(i+ROW_BLOCK, M), M, N, K);
   }
   ```
   - `parallel for`：运行时 fork 出 N 个线程，各自领取不同的 i 区间执行
   - `schedule(static)`：编译期均分，线程 t 处理 `[t*chunk, (t+1)*chunk)` 行块；工作量均匀时 static 比 dynamic 开销更低

3. **线程数控制**：
   ```bash
   OMP_NUM_THREADS=4 ./hp_gemm_bench --layer 3   # 环境变量（推荐）
   ```
   或在代码里 `omp_set_num_threads(4)`。

4. **安全性**：不同 i 区间的线程写入 C 的不同行（disjoint rows）→ 无数据竞争。A/B 只读，无写冲突。

---

### Q2：为什么加速是 4 倍？

**简答**：4 个物理核心各自独立计算，工作量四等分，串行开销极小，所以接近 4×。

**展开回答**：

#### 1. 理论上限：Amdahl 定律

```
加速比 S(n) = 1 / (f_serial + (1-f_serial)/n)
```

本实现中：
- **并行部分**：GEMM 核心计算，O(M·N·K)，全部在 `#pragma omp parallel for` 覆盖范围内
- **串行部分**：OpenMP fork/join + barrier，O(1)，对大矩阵可忽略（≈ 0.5% 以内）

代入 f_serial ≈ 0.01（保守估计 1% 串行）、n = 4 核：

```
S(4) = 1 / (0.01 + 0.99/4) ≈ 3.88×
```

实测 N=2048 时 4 线程效率 **96.15%**（3.85×），与理论值高度吻合，说明串行开销确实在 1% 以内。

#### 2. 为什么 4 核效率高（96%）但 8 核效率低（73%）？

| 核心数 | 限制因素 |
|-------|---------|
| 1–4  | 计算受限（FMA 单元满载），带宽够用 |
| 5–8  | **内存带宽瓶颈**：全部核心共享 L3 Cache 和 DDR4 DRAM 带宽（~40 GB/s），同时读 A/B 矩阵导致带宽饱和 |
| 9–16 | 超线程：2 个逻辑核心共用 1 套 ALU，无额外算力，效率继续下降 |

**物理含义**：N=2048 时，三个矩阵（A、B、C）共占 3 × 2048² × 4B ≈ **48 MB**，远超 L3 Cache（16 MB）。线程越多，对 DRAM 的争用越严重，边际收益递减。

#### 3. 为什么不会超线性（> 4×）？

矩阵乘法的算术强度（Arithmetic Intensity）有限。即便计算是完美并行的，读写内存的带宽是所有核心共享的，无法随核心数线性扩展。

#### 4. 一句话总结

> **4 核下接近 4× 是因为：①计算是完全独立的（无同步开销），②工作均分（static 调度），③4 核时内存带宽尚未饱和。** 超过 4 核（8 核 73% 效率）后带宽成为瓶颈，继续加核心收益递减。
