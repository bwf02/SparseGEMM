#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/tmp/sglang-venv/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/kernel-paper-small-batch}
WARMUP=${WARMUP:-25}
ITERATIONS=${ITERATIONS:-100}
SEEDS=${SEEDS:-1234 1235 1236 1237 1238}

mkdir -p "$OUTPUT_DIR"
for seed in $SEEDS; do
  output="$OUTPUT_DIR/seed${seed}.csv"
  if [[ -s "$output" ]]; then
    echo "reuse $output"
    continue
  fi
  echo "run seed=$seed -> $output"
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    "$REPO_ROOT/benchmarks/bench_moe_model_shapes.py" \
    --models qwen15 deepseek_v2_lite qwen3_30b llama4_scout \
    --projections gate_up down \
    --batch-sizes 8 16 32 \
    --external-backends cusparselt \
    --native-layouts masked \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS" \
    --seed "$seed" \
    --output "$output"
done
