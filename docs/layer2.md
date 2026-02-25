# Layer 2: AVX2 SIMD 向量化

## 动机：Layer 1 的瓶颈在哪？

Layer 1 的 blocked 实现已经很好地利用了 Cache，但每次乘加运算仍然是**标量**（scalar）操作——每条指令处理 **1 个 float**。

现代 x86 CPU 配备了宽向量寄存器，可以在一条指令中处理 **8 个 float**：

```
标量 FMA: a[0] += b[0] * c[0]         — 1 条指令, 2 FLOPs
AVX2 FMA: a[0..7] += b[0..7] * c[0..7] — 1 条指令, 16 FLOPs
```

理论上，仅靠 SIMD 就能获得 **8× 的吞吐量提升**（8-wide float32）。

---

## 关键概念

### x86 SIMD 寄存器层次

| 指令集 | 寄存器名 | 宽度 | float32 个数 |
|--------|---------|------|------------|
| SSE/SSE2 | XMM | 128-bit | 4 |
| AVX/AVX2 | YMM | 256-bit | **8** |
| AVX-512 | ZMM | 512-bit | 16 |

本项目使用 **AVX2 YMM 寄存器（256-bit）**，每寄存器装 8 个 float32。
CPU 配备 **16 个 YMM 寄存器**（ymm0 - ymm15）。

### FMA 指令（Fused Multiply-Add）

```
FMA: c = a * b + c   →   1 条指令 = 2 FLOPs（1 乘 + 1 加）
```

**为什么 Fused 很重要**：
- 非融合：2 条指令（VMULPS + VADDPS），1 次中间结果写入/读出
- 融合：1 条指令（VFMADD231PS），更少的指令开销，更高精度（中间结果不截断）

需要编译器 flag：`-mavx2 -mfma`

### 关键 AVX2 Intrinsics

AVX2 通过 C 头文件 `<immintrin.h>` 提供：

```cpp
// 类型
__m256   // 256-bit 向量，存储 8 个 float32

// 加载 / 存储
__m256 _mm256_loadu_ps(const float* mem);     // 从未对齐内存加载 8 个 float
__m256 _mm256_load_ps(const float* mem);      // 从 32-byte 对齐内存加载（更快）
void   _mm256_storeu_ps(float* mem, __m256 a); // 存储 8 个 float 到未对齐内存

// 广播（把 1 个标量复制到所有 8 个 lane）
__m256 _mm256_set1_ps(float a);               // broadcast scalar → [a, a, a, a, a, a, a, a]

// 算术
__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c); // c = a*b + c (FMA)
__m256 _mm256_add_ps(__m256 a, __m256 b);              // a + b
__m256 _mm256_mul_ps(__m256 a, __m256 b);              // a * b
```

---

## 实现分析

### Step 1：1×8 AVX2 Micro-Kernel

**文件**：`src/layer2/gemm_avx2.cpp`

**思路**：对于输出矩阵 C 的每一行 i，按 8 列一组处理：

```
处理前（scalar）:           处理后（SIMD）:
C[i][j]   += A[i][k] * B[k][j]   →   C[i][j:j+8] += A[i][k] * B[k][j:j+8]
C[i][j+1] += A[i][k] * B[k][j+1]
...（8 次标量操作）
```

```cpp
for (int i = 0; i < M; ++i) {
    for (int j = 0; j <= N - 8; j += 8) {

        __m256 c_vec = _mm256_loadu_ps(&C[i*N + j]);  // 加载 8 个 C 累加器

        for (int k = 0; k < K; ++k) {
            // A[i][k]：标量 → 广播到 8 个 SIMD lane
            __m256 a = _mm256_set1_ps(A[i*K + k]);

            // B[k][j..j+7]：连续 8 个 float → 1 次 Cache Line 操作
            __m256 b = _mm256_loadu_ps(&B[k*N + j]);

            // FMA：c_vec += a * b（8 个 lane 同时计算）
            c_vec = _mm256_fmadd_ps(a, b, c_vec);
        }

        _mm256_storeu_ps(&C[i*N + j], c_vec);  // 写回 8 个结果
    }
    // 尾部处理（N % 8 != 0 的剩余列）
}
```

**计算密度**：每次内循环迭代 = 16 FLOPs（8 个 lane × 2 FLOPs/FMA），但只需 3 条指令（broadcast + load + FMA）。

---

### Step 2：4×8 寄存器分块 Micro-Kernel

**文件**：`src/layer2/gemm_simd_blocked.cpp`

**问题**：1×8 kernel 每个 k 迭代有 **1 个 FMA** 操作。现代 CPU 有 **2 个 FMA 执行单元**，每周期最多完成 2 次 FMA，意味着 1×8 只用到了 50% 的 FMA 能力。

**解决方案**：同时处理 **4 行** A，使用 4 个独立的 YMM 累加器 c0..c3：

```
每个 k 迭代：
  加载 B[k][j:j+8]    → 1 次 load（4 行共享）
  广播 A[i+0][k]      → broadcast → FMA c0  ┐
  广播 A[i+1][k]      → broadcast → FMA c1  │ 4 个独立 FMA，
  广播 A[i+2][k]      → broadcast → FMA c2  │ CPU 可以流水线并行
  广播 A[i+3][k]      → broadcast → FMA c3  ┘
```

每个 k 迭代 = 4 × 16 = **64 FLOPs**，同时 B 的加载被 4 行复用（分摊代价）。

```cpp
// 4 个独立 YMM 累加器
__m256 c0 = _mm256_loadu_ps(&C[(i+0)*N + j]);
__m256 c1 = _mm256_loadu_ps(&C[(i+1)*N + j]);
__m256 c2 = _mm256_loadu_ps(&C[(i+2)*N + j]);
__m256 c3 = _mm256_loadu_ps(&C[(i+3)*N + j]);

for (int k = 0; k < K; ++k) {
    __m256 b  = _mm256_loadu_ps(&B[k*N + j]);      // 加载一次，被 4 行共用

    __m256 a0 = _mm256_set1_ps(A[(i+0)*K + k]);
    __m256 a1 = _mm256_set1_ps(A[(i+1)*K + k]);
    __m256 a2 = _mm256_set1_ps(A[(i+2)*K + k]);
    __m256 a3 = _mm256_set1_ps(A[(i+3)*K + k]);

    // 4 个 FMA 可以流水线执行（乱序执行 CPU 会自动 overlap）
    c0 = _mm256_fmadd_ps(a0, b, c0);
    c1 = _mm256_fmadd_ps(a1, b, c1);
    c2 = _mm256_fmadd_ps(a2, b, c2);
    c3 = _mm256_fmadd_ps(a3, b, c3);
}

// 写回 4×8 = 32 个输出值
_mm256_storeu_ps(&C[(i+0)*N + j], c0);
_mm256_storeu_ps(&C[(i+1)*N + j], c1);
_mm256_storeu_ps(&C[(i+2)*N + j], c2);
_mm256_storeu_ps(&C[(i+3)*N + j], c3);
```

**算术强度（Arithmetic Intensity）**：
```
FLOPs per iteration = 4 × 8 × 2 = 64
Bytes loaded        = B(32 bytes) + 4×A(16 bytes) = 48 bytes
AI = 64 / 48 ≈ 1.33 FLOPs/byte
```

比 1×8 kernel（AI ≈ 0.67）高一倍，更接近 CPU 的 Roofline 峰值。

---

### 内存对齐（Memory Alignment）

`_mm256_load_ps` 要求地址 **32-byte 对齐**，比 `_mm256_loadu_ps`（unaligned）快约 1-5%。

`gemm_common.h` 中的 `Matrix` 结构体使用 `std::aligned_alloc(32, size)` 分配 32-byte 对齐内存，因此可以安全使用 aligned 版本（如果 N×4 是 32 的倍数）。

本实现保守地使用 `_mm256_loadu_ps` 以处理所有情况，实际性能差异很小。

---

## 性能结果

> 运行后填入：
> ```bash
> build_cpu/hp_gemm_bench --layer 2 --size all
> ```

| 矩阵大小 | naive (GFLOPS) | avx2 1×8 (GFLOPS) | avx2 4×8 (GFLOPS) | 4×8 vs naive |
|---------|---------------|------------------|------------------|-------------|
| 256×256 | [填入] | [填入] | [填入] | [填入]× |
| 512×512 | [填入] | [填入] | [填入] | [填入]× |
| 1024×1024 | [填入] | [填入] | [填入] | [填入]× |
| 2048×2048 | [填入] | [填入] | [填入] | [填入]× |
| 4096×4096 | [填入] | [填入] | [填入] | [填入]× |

**CPU 理论峰值（单核）**：
```
理论 GFLOPS = 频率(GHz) × 2(FMA ports) × 8(floats/YMM) × 2(FLOPs/FMA)
例：3.5 GHz = 3.5 × 2 × 8 × 2 = 112 GFLOPS/核
```

---

## 调试技巧：验证 SIMD 编译生效

```bash
# 检查汇编输出，确认编译器生成了 vfmadd 指令
objdump -d build_cpu/libgemm_layer2.a | grep -c "vfmadd"
# 应该看到非零数字

# 或者：
g++ -O3 -mavx2 -mfma -S src/layer2/gemm_avx2.cpp -o /tmp/gemm_avx2.s
grep -c "vfmadd" /tmp/gemm_avx2.s
```

---

## 简历模板

```
- 使用 AVX2 256-bit SIMD intrinsics 手动向量化 GEMM 内核，
  一条 FMA 指令并行处理 8 个 float32（_mm256_fmadd_ps）
- 设计 4×8 寄存器分块 micro-kernel，使 4 个独立 FMA 操作可被 CPU 流水线并行执行，
  达到 [填入] GFLOPS（[填入]% 单核理论峰值 on [CPU 型号]）
```
