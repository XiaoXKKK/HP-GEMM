# CPU Perf
$ ./hp_gemm_bench --layer all --size all

=== HP-GEMM Benchmark ===
  Runs: 5 (warmup: 2)

──── Matrix size: 256 × 256 ────
  naive (ijk)          | N=  256 |   2.25 GFLOPS
  ikj (loop reorder)   | N=  256 |  33.55 GFLOPS | 14.91x vs naive
  blocked (L2+reg tile) | N=  256 |  13.67 GFLOPS |  6.07x vs naive
  avx2 1x8             | N=  256 |  17.96 GFLOPS |  7.98x vs naive
  avx2 4x8 (reg block) | N=  256 |  47.99 GFLOPS | 21.32x vs naive
  openmp (1 threads)   | N=  256 |  52.46 GFLOPS | 23.31x vs naive
  openmp (2 threads)   | N=  256 | 106.17 GFLOPS | 47.17x vs naive
  openmp (4 threads)   | N=  256 | 202.40 GFLOPS | 89.93x vs naive
  openmp (8 threads)   | N=  256 | 229.70 GFLOPS | 102.05x vs naive
  openmp (16 threads)  | N=  256 | 428.00 GFLOPS | 190.16x vs naive

──── Matrix size: 512 × 512 ────
  naive (ijk)          | N=  512 |   1.32 GFLOPS
  ikj (loop reorder)   | N=  512 |  30.47 GFLOPS | 23.08x vs naive
  blocked (L2+reg tile) | N=  512 |  11.47 GFLOPS |  8.68x vs naive
  avx2 1x8             | N=  512 |  10.55 GFLOPS |  7.99x vs naive
  avx2 4x8 (reg block) | N=  512 |  35.24 GFLOPS | 26.69x vs naive
  openmp (1 threads)   | N=  512 |  35.86 GFLOPS | 27.16x vs naive
  openmp (2 threads)   | N=  512 |  58.86 GFLOPS | 44.58x vs naive
  openmp (4 threads)   | N=  512 | 111.23 GFLOPS | 84.24x vs naive
  openmp (8 threads)   | N=  512 | 158.77 GFLOPS | 120.23x vs naive
  openmp (16 threads)  | N=  512 | 314.51 GFLOPS | 238.18x vs naive

──── Matrix size: 1024 × 1024 ────
  naive (ijk)          | N= 1024 |   1.46 GFLOPS
  ikj (loop reorder)   | N= 1024 |  25.89 GFLOPS | 17.70x vs naive
  blocked (L2+reg tile) | N= 1024 |   9.79 GFLOPS |  6.69x vs naive
  avx2 1x8             | N= 1024 |  11.47 GFLOPS |  7.84x vs naive
  avx2 4x8 (reg block) | N= 1024 |  42.18 GFLOPS | 28.83x vs naive
  openmp (1 threads)   | N= 1024 |  38.78 GFLOPS | 26.51x vs naive
  openmp (2 threads)   | N= 1024 |  83.30 GFLOPS | 56.94x vs naive
  openmp (4 threads)   | N= 1024 | 112.03 GFLOPS | 76.59x vs naive
  openmp (8 threads)   | N= 1024 |  99.37 GFLOPS | 67.93x vs naive
  openmp (16 threads)  | N= 1024 |  92.86 GFLOPS | 63.49x vs naive

──── Matrix size: 2048 × 2048 ────
  naive (ijk)          | N= 2048 |   0.35 GFLOPS
  ikj (loop reorder)   | N= 2048 |   8.38 GFLOPS | 24.19x vs naive
  blocked (L2+reg tile) | N= 2048 |   9.56 GFLOPS | 27.59x vs naive
  avx2 1x8             | N= 2048 |   2.09 GFLOPS |  6.02x vs naive
  avx2 4x8 (reg block) | N= 2048 |   7.25 GFLOPS | 20.93x vs naive
  openmp (1 threads)   | N= 2048 |   7.33 GFLOPS | 21.17x vs naive
  openmp (2 threads)   | N= 2048 |  13.79 GFLOPS | 39.80x vs naive
  openmp (4 threads)   | N= 2048 |  28.19 GFLOPS | 81.39x vs naive
  openmp (8 threads)   | N= 2048 |  43.13 GFLOPS | 124.50x vs naive
  openmp (16 threads)  | N= 2048 |  42.08 GFLOPS | 121.49x vs naive

──── Matrix size: 4096 × 4096 ────
  naive (ijk)          | N= 4096 |   0.24 GFLOPS
  ikj (loop reorder)   | N= 4096 |   6.56 GFLOPS | 27.04x vs naive
  blocked (L2+reg tile) | N= 4096 |   9.84 GFLOPS | 40.56x vs naive
  avx2 1x8             | N= 4096 |   1.67 GFLOPS |  6.87x vs naive
  avx2 4x8 (reg block) | N= 4096 |   6.27 GFLOPS | 25.88x vs naive
  openmp (1 threads)   | N= 4096 |   6.27 GFLOPS | 25.87x vs naive
  openmp (2 threads)   | N= 4096 |  10.29 GFLOPS | 42.41x vs naive
  openmp (4 threads)   | N= 4096 |  11.50 GFLOPS | 47.43x vs naive
  openmp (8 threads)   | N= 4096 |  14.01 GFLOPS | 57.75x vs naive
  openmp (16 threads)  | N= 4096 |  12.73 GFLOPS | 52.51x vs naive

# GPU Perf

$ ./hp_gemm_bench --layer 4 --size all

=== HP-GEMM Benchmark ===
  Runs: 5 (warmup: 2)

──── Matrix size: 256 × 256 ────
  cuda naive           | N=  256 | 351.31 GFLOPS
  cuda shared mem      | N=  256 | 294.21 GFLOPS

──── Matrix size: 512 × 512 ────
  cuda naive           | N=  512 | 620.10 GFLOPS
  cuda shared mem      | N=  512 | 523.48 GFLOPS

──── Matrix size: 1024 × 1024 ────
  cuda naive           | N= 1024 | 720.62 GFLOPS
  cuda shared mem      | N= 1024 | 501.16 GFLOPS

──── Matrix size: 2048 × 2048 ────
  cuda naive           | N= 2048 | 656.91 GFLOPS
  cuda shared mem      | N= 2048 | 562.99 GFLOPS

──── Matrix size: 4096 × 4096 ────
  cuda naive           | N= 4096 | 694.97 GFLOPS
  cuda shared mem      | N= 4096 | 576.78 GFLOPS