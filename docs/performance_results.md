# Performance Results

## System Information

> Fill in after running benchmarks:
> ```bash
> bash scripts/run_benchmarks.sh
> ```

| Item | Value |
|------|-------|
| CPU |  Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz |
| Physical Cores | 16 |
| L2 Cache | 2 MiB |
| L3 Cache | 16 MiB |
| Memory | 8G |
| GPU |  NVIDIA GeForce RTX 3060 |
| OS | WSL2 Ubuntu |
| Compiler | 13.3.0 |
| CUDA | cuda 12.6 |

---

## Layer 1: Cache-Friendly GEMM

| N | naive GFLOPS | ikj GFLOPS | ikj speedup | blocked GFLOPS | blocked speedup |
|---|-------------|-----------|------------|---------------|----------------|
| 256 | 2.25 | 33.55 | 14.91× | 13.67 | 6.07× |
| 512 | 1.32 | 30.47 | 23.08× | 11.47 | 8.68× |
| 1024 | 1.46 | 25.89 | 17.70× | 9.79 | 6.69× |
| 2048 | 0.35 | 8.38 | 24.19× | 9.56 | 27.59× |
| 4096 | 0.24 | 6.56 | 27.04× | 9.84 | 40.56× |

---

## Layer 2: AVX2 SIMD

| N | naive GFLOPS | avx2 1×8 GFLOPS | avx2 4×8 GFLOPS | 4×8 speedup |
|---|-------------|----------------|----------------|------------|
| 256 | 2.25 | 17.96 | 47.99 | 21.32× |
| 512 | 1.32 | 10.55 | 35.24 | 26.69× |
| 1024 | 1.46 | 11.47 | 42.18 | 28.83× |
| 2048 | 0.35 | 2.09 | 7.25 | 20.93× |
| 4096 | 0.24 | 1.67 | 6.27 | 25.88× |

---

## Layer 3: OpenMP Threading (N=2048)

| Threads | GFLOPS | Speedup vs 1T | Efficiency |
|---------|--------|--------------|-----------|
| 1 | 7.33 | 1.00× | 100.00% |
| 2 | 13.79 | 1.88× | 94.07% |
| 4 | 28.19 | 3.85× | 96.15% |
| 8 | 43.13 | 5.88× | 73.57% |
| 16 | 42.08 | 5.74× | 35.88% |

---

## Layer 4: CUDA GPU

| N | CPU naive | CUDA naive | CUDA shared | GPU/CPU naive |
|---|-----------|-----------|------------|--------------|
| 256 | 2.25 | 351.31 | 294.21 | 156.14× |
| 512 | 1.32 | 620.10 | 523.48 | 469.77× |
| 1024 | 1.46 | 720.62 | 501.16 | 493.58× |
| 2048 | 0.35 | 656.91 | 562.99 | 1876.89× |

---

## Layer 5: OpenBLAS Comparison

> Run after installing OpenBLAS:
> ```bash
> sudo apt-get install libopenblas-dev
> cmake -S . -B build_blas -DENABLE_OPENBLAS=ON -DENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
> cmake --build build_blas -j$(nproc)
> build_blas/hp_gemm_bench --layer 5 --size all
> ```

### Single-thread: our avx2 4×8 vs OpenBLAS 1T

| N | avx2 4×8 (GFLOPS) | OpenBLAS 1T (GFLOPS) | OpenBLAS/ours |
|---|------------------|---------------------|--------------|
| 256 | 47.99 | [填入] | [填入]× |
| 512 | 35.24 | [填入] | [填入]× |
| 1024 | 42.18 | [填入] | [填入]× |
| 2048 | 7.25 | [填入] | [填入]× |
| 4096 | 6.27 | [填入] | [填入]× |

### Multi-thread: our OpenMP vs OpenBLAS (all cores)

| N | our OpenMP (GFLOPS) | OpenBLAS MT (GFLOPS) | OpenBLAS/ours |
|---|--------------------|--------------------|--------------|
| 1024 | [填入] | [填入] | [填入]× |
| 2048 | 43.13 | [填入] | [填入]× |
| 4096 | [填入] | [填入] | [填入]× |

---

## Summary: Peak Performance vs Naive

| Kernel | N=1024 GFLOPS | vs naive |
|--------|--------------|---------|
| naive (baseline) | 1.46 | 1.0× |
| ikj | 25.89 | 17.70× |
| blocked | 9.79 | 6.69× |
| avx2 4×8 | 42.18 | 28.83× |
| openmp (max threads) | 112.03 | 76.59× |
| cuda shared | 501.16 | 343.26× |
| OpenBLAS 1T | [填入] | [填入]× |
| OpenBLAS MT | [填入] | [填入]× |
