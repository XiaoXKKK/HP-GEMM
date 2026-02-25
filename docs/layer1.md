# Layer 1: Cache-Friendly GEMM

## 动机：为什么朴素实现这么慢？

在开始优化之前，先要理解朴素实现的瓶颈在哪里。

矩阵乘法 C = A × B 的朴素实现是三重循环：

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            C[i][j] += A[i][k] * B[k][j];
```

**问题**：在行优先（row-major）存储下，`B[k][j]` 的 k 循环是按**列**访问 B 矩阵。
- 相邻两次访问的地址差：`B[k][j]` 和 `B[k+1][j]` 相差 N×4 字节
- N=1024 时：相差 4096 字节，而 Cache Line 只有 64 字节
- 每次访问 B 几乎都是 Cache Miss，需要从内存重新加载

---

## 关键概念

### CPU 存储层次结构

| 层级 | 典型大小 | 延迟 | 带宽 |
|------|---------|------|------|
| 寄存器 | ~1 KB | 0 周期 | 无限 |
| L1 D-Cache | 32 KB | 4 周期 | ~1 TB/s |
| L2 Cache | 256 KB | 12 周期 | ~400 GB/s |
| L3 Cache | 6-32 MB | 40 周期 | ~200 GB/s |
| DRAM | ∞ | 200 周期 | ~50 GB/s |

**Cache Line**：内存和缓存之间数据传输的最小单位 = **64 字节 = 16 个 float32**。

无论你只访问 1 个元素，CPU 都会加载 64 字节到缓存。如果你能在这 64 字节被驱逐前用上所有 16 个 float，Cache Line 利用率就是 100%。

### 空间局部性（Spatial Locality）

访问 `array[k]` 后立即访问 `array[k+1]`：同一 Cache Line 中，不需要额外加载。这就是**步长为 1（stride-1）访问模式**好的原因。

---

## 实现分析

### Step 1：朴素实现（ijk）

**文件**：`src/layer1/gemm_naive.cpp`

```cpp
// ijk 循环顺序 — 内循环访问 B 的列（stride-N，cache 不友好）
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[i * K + k] * B[k * N + j];  // B: 步长 N 访问
        C[i * N + j] += sum;
    }
```

内循环 k 每次递增，`B[k*N+j]` 的地址增加 N×4 字节：
- N=256：每步 1024 字节（16 个 Cache Line 跨度）
- N=1024：每步 4096 字节（64 个 Cache Line 跨度）

**结论**：每次 B 访问几乎必然 Cache Miss。

---

### Step 2：循环重排（ikj）

**文件**：`src/layer1/gemm_ikj.cpp`

```cpp
// ikj 循环顺序 — 将 j 移到最内层，使 B 和 C 都是步长 1
for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
        const float a_ik = A[i * K + k];  // 标量，保存在寄存器中
        for (int j = 0; j < N; ++j)
            C[i * N + j] += a_ik * B[k * N + j];  // B 和 C 都是步长 1
    }
```

**内循环分析**：
- `A[i*K+k]`：在进入内循环前已加载，存放在寄存器（无内存访问）
- `B[k*N+j]`：j 递增 → 步长 1 → 顺序读取 → Cache 友好
- `C[i*N+j]`：j 递增 → 步长 1 → 顺序读写 → Cache 友好

每次内循环迭代处理 1 个 Cache Line 中的 16 个 B 值，利用率从 1/N 提升到 16/16 = 100%。

**预期提升**：3-5× vs naive（大矩阵）

---

### Step 3：两级缓存分块（Blocking/Tiling）

**文件**：`src/layer1/gemm_blocked.cpp`

即使 ikj 访问模式好，当 N 很大时，整个 B 矩阵无法放入 L2 Cache，仍然需要反复从 L3/DRAM 加载数据。

**解决方案**：将计算分割成"刚好能装进 L2"的小块（Tile）。

#### L2 分块尺寸推导

```
目标：三个 Tile（A_tile, B_tile, C_tile）总大小 ≤ L2 Cache（256 KB）
tile_bytes = L2_BLOCK × L2_BLOCK × 4 bytes
3 × L2_BLOCK² × 4 ≤ 256 × 1024
L2_BLOCK ≤ sqrt(256×1024 / 12) ≈ 148 → 取 128
```

#### 寄存器分块（Register Micro-Kernel）

在 L2 Tile 内部，进一步分成 4×4 的小块：
- `c[4][4]`：16 个 float 累加器保存在 **CPU 寄存器** 中
- 整个 K 维循环期间，这 16 个值 **不写回内存**
- 避免了 REG_M × REG_N 次的 C 矩阵 load/store

```cpp
// 寄存器累加器（保持在寄存器，不经过 Cache）
float c[REG_M][REG_N] = {};

// 先从 C 加载当前值
for (int ri = 0; ri < REG_M; ++ri)
    for (int rj = 0; rj < REG_N; ++rj)
        c[ri][rj] = C[(i+ri)*N + (j+rj)];

// 内循环：A 和 B 都在 L1 Cache 中（L2 Tile 保证）
for (int k = k0; k < k_end; ++k)
    for (int ri = 0; ri < REG_M; ++ri) {
        float a_val = A[(i+ri)*K + k];
        for (int rj = 0; rj < REG_N; ++rj)
            c[ri][rj] += a_val * B[k*N + (j+rj)];
    }

// 只写回一次
for (int ri = 0; ri < REG_M; ++ri)
    for (int rj = 0; rj < REG_N; ++rj)
        C[(i+ri)*N + (j+rj)] = c[ri][rj];
```

**内存层次利用**：
```
寄存器：c[4][4] 累加器（16 个 float = 64 字节）
L1 Cache：当前 A/B 的 L1 小块
L2 Cache：整个 L2 Tile（A_tile 64KB + B_tile 64KB + C_tile 64KB ≈ 192KB）
L3/DRAM：只在跨 L2 Tile 边界时访问
```

---

## 性能结果

> 以下数字在运行后填入。在 WSL2 中执行：
> ```bash
> bash scripts/build_cpu.sh
> build_cpu/hp_gemm_bench --layer 1 --size all
> ```

| 矩阵大小 | naive (GFLOPS) | ikj (GFLOPS) | blocked (GFLOPS) | blocked vs naive |
|---------|---------------|-------------|-----------------|-----------------|
| 256×256 | [填入] | [填入] | [填入] | [填入]× |
| 512×512 | [填入] | [填入] | [填入] | [填入]× |
| 1024×1024 | [填入] | [填入] | [填入] | [填入]× |
| 2048×2048 | [填入] | [填入] | [填入] | [填入]× |
| 4096×4096 | [填入] | [填入] | [填入] | [填入]× |

**理论峰值**：在运行时查看：
```bash
# 查看 CPU 频率和核心数
lscpu | grep -E "Model name|MHz|Core"
# 单核理论峰值 = 频率(GHz) × 每周期FLOPs
# 对于 scalar (无SIMD): 频率 × 2 (1 FMA = 2 FLOPs)
```

---

## 性能分析工具

### 用 perf 测量 Cache Miss

```bash
# 测量 naive 的 Cache Miss 率
perf stat -e cache-misses,cache-references,instructions,cycles \
    build_cpu/hp_gemm_bench --layer 1 --size 1024

# 对比 blocked 的 Cache Miss 率
```

典型结果：naive 在 N=1024 时 Cache Miss 率 > 50%，blocked 降低到 < 5%。

---

## 关键调优参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `L2_BLOCK` | 128 | L2 Tile 大小（调整以匹配你的 L2 大小） |
| `REG_M` | 4 | 寄存器块行数（4 = 4个标量寄存器） |
| `REG_N` | 4 | 寄存器块列数（4 = 4个标量寄存器） |

**调优方法**：
```bash
# 修改 src/layer1/gemm_blocked.cpp 中的常量后重建
# 尝试 L2_BLOCK = 64, 96, 128, 160
```

---

## 简历模板

```
高性能矩阵乘法多层优化 | C++17, AVX2, OpenMP  [2025.xx]
- 分析朴素 O(n³) GEMM 的 Cache Miss 病因：B 矩阵列访问导致 stride-N 加载，
  Cache Line 利用率 < 2%（N=1024）
- 实现 ikj 循环重排消除 stride-N 访问，达到 [填入] GFLOPS（vs naive [填入] GFLOPS，
  [填入]× 提升）
- 实现两级 Cache 分块（L2 Tile=128×128，寄存器块=4×4），进一步提升至 [填入] GFLOPS
  （vs naive [填入]×），Cache Miss 率降低至 < 5%
```
