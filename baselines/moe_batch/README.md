# MoE Batched Sparse Baselines

This directory provides two fixed-shape batched GEMM baselines at 25% weight
sparsity:

- **SlideSparse + cuSPARSELt:** magnitude-prunes two values per group of eight,
  expands each 6:8 group into three overlapping 2:4 groups, then invokes one
  cuSPARSELt strided-batch matmul. The physical K is `1.5 * K`.
- **Sputnik batch adaptation:** stores the same pruned 2:8 weights in CSR and
  runs Sputnik's FP16 SpMM tile implementation with one matrix batch per
  `grid.z` slice. This is a single fixed-shape batch launch.

Upstream Sputnik does not expose batched matrices. Its documentation calls the
dense RHS width `n` a "batch size", but `CudaSpmm` accepts only one CSR matrix.
The vendored Apache-2.0 source is pinned to commit
`bbf5840ba5efccf01f862855c785f71bcc6ff1f0`; the local change adds only fixed
batch strides and `grid.z` dispatch.

## Build and run

```bash
cd /path/to/SparseGEMM
CUDA_HOME=/usr/local/cuda CUDA_ARCH=90a \
  bash baselines/moe_batch/build.sh
python benchmarks/bench_moe_batch_baselines.py \
  --experts 8 --m 128 256 512 1024 2048 4096 --n 1408 --k 2048
```

Both kernels use FP16 because upstream Sputnik has FP16 and FP32 APIs but no
BF16 implementation. The benchmark uses identical sparse weights, activations,
fixed batch shapes, warmup, and timing. `slidesparse_cusparselt` reports only
the batched GEMM latency; `slide_activation` reports the online expansion
separately. `sputnik_batch` likewise excludes the layout transpose, which is
reported as `sputnik_transpose`. Weight sliding, cuSPARSELt compression, and
CSR conversion are offline preprocessing and are excluded from kernel latency.

For MoE routing, set `M` to the same padded expert capacity for every batch.
This baseline intentionally does not implement variable-M grouped GEMM.
