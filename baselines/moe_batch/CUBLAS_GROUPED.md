# cuBLAS Grouped GEMM Baseline

This baseline calls `cublasGemmGroupedBatchedEx`, not a Python loop over GEMMs
or the uniform batched API. It requires a cuBLAS library exposing that symbol.
`CUBLAS_LIBRARY` can select the shared library if it is not on the loader path.

## Matched Comparison

- Four existing TP1 model shapes: Qwen1.5-MoE, DeepSeek-V2-Lite,
  Qwen3-30B-A3B, and Llama 4 Scout; gate/up and down projections.
- The batch size is total input tokens, not the per-expert M. Existing balanced
  routing distributes `BS * top_k` rows across the model's experts.
- All three native backends receive the same activation tensor and the same
  weight values. LoSparse packs 25%-sparse weights; cuBLAS and DeepGEMM use
  the corresponding zero-materialized dense weights.
- cuBLAS uses one group per active expert, with one GEMM per group and actual
  expert M. Pointer offsets respect the existing padded buffer strides, but
  padded rows are not computed. Empty experts are omitted.
- BF16 inputs and output with FP32 accumulation. No shape autotuning is added.
- Timing follows the existing benchmark: CUDA events around 100 ordinary calls,
  following 25 warmups, without CUDA Graphs. Handle creation and host/device
  pointer-array preparation are outside timing. API-internal setup and gaps
  between launches can affect this measurement; it is not an isolated NCU
  kernel-duration measurement. Routing/activation preparation is excluded for
  all three native backends.
- Small-M LoSparse retains the existing prebind selection; DeepGEMM uses its
  masked grouped implementation. No production kernel code is modified.

## Run and Archive

On the designated remote H20, after synchronizing approved source changes:

```bash
CUDA_VISIBLE_DEVICES=0 OUTPUT_DIR=/tmp/cublas-grouped-kernel \
  bash benchmarks/run_cublas_grouped.sh
```

This runs independent seeds 1234--1238 over BS 8/16/32/64/128/4K/8K/16K/32K,
including the existing main grid and small-batch supplements. The correctness
test covers empty experts, uneven M, untouched padding, and a nondefault stream.
Each measured case also checks cuBLAS against DeepGEMM on valid output rows.
Retain every seed CSV, log, environment record, and this source version locally.
The CSV latency is the per-call average in microseconds, not a per-iteration
sample. A complete seed has 216 native rows (72 shapes times three backends).

Plot locally after collection:

```bash
python benchmarks/plot_cublas_grouped.py \
  --inputs /path/to/archive/seed*.csv --output-dir /path/to/archive/plots
```

Plots use fresh same-run DeepGEMM as 1.0x. They do not mix historical timings
with new timings. The manuscript and its existing figures remain unchanged.

Reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex
