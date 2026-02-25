#!/bin/bash
# build_cpu.sh - Configure and build HP-GEMM (CPU-only, no CUDA)
# Run this from the repository root inside WSL2.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build_cpu"

echo "=== HP-GEMM CPU Build ==="
echo "  Project: ${PROJECT_ROOT}"
echo "  Build:   ${BUILD_DIR}"
echo ""

cmake -S "${PROJECT_ROOT}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=OFF \
      -DENABLE_OPENMP=ON

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "  Test:      cd ${BUILD_DIR} && ctest --output-on-failure"
echo "  Benchmark: ${BUILD_DIR}/hp_gemm_bench --layer all --size all"
