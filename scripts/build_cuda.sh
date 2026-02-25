#!/bin/bash
# build_cuda.sh - Configure and build HP-GEMM with CUDA enabled
# Prerequisites: nvidia-smi works in WSL2, nvcc installed via cuda-toolkit
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build_cuda"

echo "=== HP-GEMM CUDA Build ==="
echo ""

# Verify CUDA environment
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install cuda-toolkit inside WSL2:"
    echo "  sudo apt-get install cuda-toolkit-12-x"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi failed. GPU may not be accessible in WSL2."
    echo "  Ensure Windows NVIDIA driver >= 525.x is installed."
fi

echo "  nvcc: $(nvcc --version | head -1)"
echo ""

cmake -S "${PROJECT_ROOT}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON \
      -DENABLE_OPENMP=ON

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "  Test:      cd ${BUILD_DIR} && ctest --output-on-failure"
echo "  Benchmark: ${BUILD_DIR}/hp_gemm_bench --layer 4 --size 1024"
