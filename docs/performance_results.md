# Performance Results

## System Information

> Fill in after running benchmarks:
> ```bash
> bash scripts/run_benchmarks.sh
> ```

| Item | Value |
|------|-------|
| CPU | [lscpu 结果] |
| Physical Cores | [nproc --all / 2] |
| L2 Cache | [lscpu \| grep L2] |
| L3 Cache | [lscpu \| grep L3] |
| Memory | [free -h] |
| GPU | [nvidia-smi 结果] |
| OS | WSL2 Ubuntu |
| Compiler | GCC [version] |
| CUDA | [nvcc --version] |

---

## Layer 1: Cache-Friendly GEMM

| N | naive GFLOPS | ikj GFLOPS | ikj speedup | blocked GFLOPS | blocked speedup |
|---|-------------|-----------|------------|---------------|----------------|
| 256 | | | | | |
| 512 | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |

---

## Layer 2: AVX2 SIMD

| N | naive GFLOPS | avx2 1×8 GFLOPS | avx2 4×8 GFLOPS | 4×8 speedup |
|---|-------------|----------------|----------------|------------|
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |
| 4096 | | | | |

---

## Layer 3: OpenMP Threading (N=2048)

| Threads | GFLOPS | Speedup vs 1T | Efficiency |
|---------|--------|--------------|-----------|
| 1 | | 1.0× | 100% |
| 2 | | | |
| 4 | | | |
| 8 | | | |
| 16 | | | |

---

## Layer 4: CUDA GPU

| N | CPU naive | CUDA naive | CUDA shared | GPU/CPU naive |
|---|-----------|-----------|------------|--------------|
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |

---

## Summary: Peak Performance vs Naive

| Kernel | N=1024 GFLOPS | vs naive |
|--------|--------------|---------|
| naive (baseline) | | 1.0× |
| ikj | | |
| blocked | | |
| avx2 4×8 | | |
| openmp (max threads) | | |
| cuda shared | | |
