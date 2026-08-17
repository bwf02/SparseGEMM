#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCH="${CUDA_ARCH:-90a}"
mkdir -p "${BUILD_DIR}"

COMMON=(
  -std=c++17 -O3 -shared -Xcompiler=-fPIC
  "-I${ROOT}/third_party"
  "-I${CUDA_HOME}/include"
  "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
)

"${CUDA_HOME}/bin/nvcc" "${COMMON[@]}" \
  "${ROOT}/csrc/cusparselt_batch.cu" \
  -L"${CUDA_HOME}/lib64" -lcusparseLt -lcudart \
  -o "${BUILD_DIR}/libslidesparse_batch.so"

"${CUDA_HOME}/bin/nvcc" "${COMMON[@]}" -x cu \
  "${ROOT}/third_party/sputnik/spmm/cuda_spmm.cu.cc" \
  "${ROOT}/csrc/sputnik_batch_binding.cu" \
  -L"${CUDA_HOME}/lib64" -lcudart \
  -o "${BUILD_DIR}/libsputnik_batch.so"

echo "Built baselines in ${BUILD_DIR}"
