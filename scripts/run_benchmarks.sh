#!/bin/bash
# run_benchmarks.sh - Run all benchmarks and save results to results/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
BENCH="${PROJECT_ROOT}/build_cpu/hp_gemm_bench"

if [ ! -f "${BENCH}" ]; then
    echo "ERROR: benchmark binary not found. Run scripts/build_cpu.sh first."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${RESULTS_DIR}/bench_${TIMESTAMP}.txt"

echo "=== HP-GEMM Full Benchmark ===" | tee "${OUTPUT}"
echo "  Date: $(date)" | tee -a "${OUTPUT}"
echo "  CPU:  $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)" | tee -a "${OUTPUT}"
echo "  Cores: $(nproc)" | tee -a "${OUTPUT}"
echo "" | tee -a "${OUTPUT}"

"${BENCH}" --layer all --size all 2>&1 | tee -a "${OUTPUT}"

echo ""
echo "Results saved to: ${OUTPUT}"
