#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/tmp/sglang-venv/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/cublas-grouped-kernel}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"
{
  date -u
  git -C "$ROOT" rev-parse HEAD
  git -C "$ROOT" status --short
  nvidia-smi
  "$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'
} > "$OUTPUT_DIR/environment.txt"
"$PYTHON_BIN" "$ROOT/baselines/moe_batch/test_cublas_grouped.py" \
  2>&1 | tee "$OUTPUT_DIR/correctness.log"
for seed in ${SEEDS:-1234 1235 1236 1237 1238}; do
  if [[ -e "$OUTPUT_DIR/seed${seed}.csv" ]]; then
    echo "Refusing to overwrite $OUTPUT_DIR/seed${seed}.csv" >&2
    exit 1
  fi
  "$PYTHON_BIN" -u "$ROOT/benchmarks/bench_moe_model_shapes.py" \
    --models qwen15 deepseek_v2_lite qwen3_30b llama4_scout \
    --projections gate_up down \
    --batch-sizes 8 16 32 64 128 4096 8192 16384 32768 \
    --native-only --with-cublas \
    --warmup 25 --iterations 100 --seed "$seed" \
    --output "$OUTPUT_DIR/seed${seed}.csv" \
    2>&1 | tee "$OUTPUT_DIR/seed${seed}.log"
done
