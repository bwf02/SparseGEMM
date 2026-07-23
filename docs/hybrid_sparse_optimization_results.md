# Hybrid Sparse Kernel Optimization Results

## Scope

This document records optimization results produced with the standard M sweep:

```text
M = 128, 256, 512, 1024, 2048, 4096
```

Do not mix results from the earlier benchmark shapes into this document. The
historical results remain in `hybrid_sparse_tensorcore_performance.md`.

## Benchmark Protocol

- GPU: NVIDIA H20 (SM90).
- Data type: BF16 input/output with FP32 accumulation.
- Accumulation: FP32.
- Weight pattern: hybrid block sparse, one 2:4 sparse block per two blocks.
- Element sparsity: 25%.
- Default weight block: `64 x 64` unless an iteration states otherwise.
- Projection shapes:
  - Gate/up: `N = 1408, K = 2048`.
  - Down: `N = 2048, K = 1408`.
- Baseline: DeepGEMM BF16 dense GEMM on the zero-filled dense weight.
- Timing: `bench_kineto`; weight conversion and packing are excluded.
- Cache policy: L2 flush disabled.
- Measurements: 100 per shape.
- Report dense, sparse, reduction, total, and DeepGEMM latency in microseconds.
- Run correctness validation against the dense Torch/DeepGEMM result before
  recording performance.
- Record whether L2 flush is enabled and keep that setting identical across
  compared versions.

## Gate/Up Projection

`N = 1408, K = 2048`

| Version | M | Dense (us) | Sparse (us) | Reduce (us) | Total (us) | DeepGEMM (us) | DG / Hybrid | Correct |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | 128 | 9.17 | 12.46 | 5.04 | 26.68 | 9.68 | 0.363x | Yes |
| Metadata prefetch | 128 | 9.20 | 9.07 | 5.06 | 23.33 | 9.68 | 0.415x | Yes |
| Baseline | 256 | 12.08 | 14.12 | 5.34 | 31.54 | 13.89 | 0.440x | Yes |
| Metadata prefetch | 256 | 12.10 | 11.25 | 5.34 | 28.69 | 13.89 | 0.484x | Yes |
| Baseline | 512 | 23.03 | 27.70 | 5.96 | 56.70 | 25.49 | 0.450x | Yes |
| Metadata prefetch | 512 | 23.05 | 20.86 | 5.97 | 49.87 | 25.49 | 0.511x | Yes |
| Baseline | 1024 | 34.39 | 42.18 | 6.55 | 83.13 | 48.98 | 0.589x | Yes |
| Metadata prefetch | 1024 | 34.38 | 31.00 | 6.62 | 71.99 | 48.98 | 0.680x | Yes |
| Baseline | 2048 | 57.30 | 74.98 | 13.44 | 145.73 | 95.68 | 0.657x | Yes |
| Metadata prefetch | 2048 | 57.50 | 53.19 | 13.36 | 124.04 | 95.68 | 0.771x | Yes |
| Baseline | 4096 | 104.16 | 133.88 | 32.02 | 270.06 | 184.10 | 0.682x | Yes |
| Metadata prefetch | 4096 | 103.84 | 94.79 | 32.05 | 230.68 | 184.10 | 0.798x | Yes |

## Down Projection

`N = 2048, K = 1408`

| Version | M | Dense (us) | Sparse (us) | Reduce (us) | Total (us) | DeepGEMM (us) | DG / Hybrid | Correct |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | 128 | 7.01 | 9.22 | 5.08 | 21.32 | 8.61 | 0.404x | Yes |
| Metadata prefetch | 128 | 7.03 | 7.27 | 5.10 | 19.39 | 8.61 | 0.444x | Yes |
| Baseline | 256 | 9.18 | 10.56 | 5.28 | 25.01 | 13.35 | 0.534x | Yes |
| Metadata prefetch | 256 | 9.22 | 8.85 | 5.29 | 23.35 | 13.35 | 0.572x | Yes |
| Baseline | 512 | 17.85 | 20.56 | 6.08 | 44.48 | 24.82 | 0.558x | Yes |
| Metadata prefetch | 512 | 17.86 | 16.38 | 6.11 | 40.34 | 24.82 | 0.615x | Yes |
| Baseline | 1024 | 33.23 | 44.01 | 7.97 | 85.22 | 47.74 | 0.560x | Yes |
| Metadata prefetch | 1024 | 32.94 | 31.44 | 7.98 | 72.36 | 47.74 | 0.660x | Yes |
| Baseline | 2048 | 58.96 | 73.51 | 18.80 | 151.27 | 93.63 | 0.619x | Yes |
| Metadata prefetch | 2048 | 59.20 | 60.90 | 18.75 | 138.85 | 93.63 | 0.674x | Yes |
| Baseline | 4096 | 105.07 | 136.45 | 44.95 | 286.47 | 183.30 | 0.640x | Yes |
| Metadata prefetch | 4096 | 104.99 | 105.58 | 44.97 | 255.54 | 183.30 | 0.717x | Yes |

## Grouped GEMM

Use total valid M matching the standard sweep where practical. Record the
expert count, token distribution, grouped mode, and capacity in the iteration
entry because total M alone does not fully describe a grouped workload.

| Version | Mode | Experts | Total valid M | Distribution | Total (us) | DeepGEMM (us) | DG / Hybrid | Correct |
|:---|:---|---:|---:|:---|---:|---:|---:|:---:|
| Baseline | - | - | 128 | - | - | - | - | - |
| Baseline | - | - | 256 | - | - | - | - | - |
| Baseline | - | - | 512 | - | - | - | - | - |
| Baseline | - | - | 1024 | - | - | - | - | - |
| Baseline | - | - | 2048 | - | - | - | - | - |
| Baseline | - | - | 4096 | - | - | - | - | - |

## Fused GEMM

These measurements use the fused dense/sparse mainloop. `Direct` writes each
BF16 output element from registers, while `STSM/TMA` stages BF16 through
swizzled shared memory and uses a TMA store.

### Gate/Up Projection

| M | Direct (us) | STSM/TMA (us) | DeepGEMM (us) | Direct / STSM | DG / STSM | Correct |
|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 18.42 | 18.27 | 10.32 | 1.008x | 0.565x | Yes |
| 256 | 26.54 | 25.92 | 14.91 | 1.024x | 0.575x | Yes |
| 512 | 71.66 | 32.66 | 27.35 | 2.195x | 0.837x | Yes |
| 1024 | 76.84 | 60.59 | 52.78 | 1.268x | 0.871x | Yes |
| 2048 | 128.37 | 122.86 | 102.93 | 1.045x | 0.838x | Yes |
| 4096 | 248.27 | 185.84 | 198.94 | 1.336x | 1.071x | Yes |

### Down Projection

| M | Direct (us) | STSM/TMA (us) | DeepGEMM (us) | Direct / STSM | DG / STSM | Correct |
|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 13.76 | 13.65 | 9.26 | 1.008x | 0.678x | Yes |
| 256 | 19.33 | 19.03 | 14.29 | 1.016x | 0.751x | Yes |
| 512 | 38.31 | 39.08 | 26.71 | 0.980x | 0.683x | Yes |
| 1024 | 72.73 | 80.61 | 51.41 | 0.902x | 0.638x | Yes |
| 2048 | 130.99 | 102.04 | 100.49 | 1.284x | 0.985x | Yes |
| 4096 | 278.75 | 191.15 | 197.71 | 1.458x | 1.034x | Yes |

## Iteration Log

Add one section per attempted optimization. Keep rejected versions in the log.

### Iteration 0: Standard-Sweep Baseline

- Status: completed.
- Kernel version: current `64 x 64` two-stage TMA implementation.
- Kernel commit: `605ac28`.
- Benchmark harness commit: `6e8d58f`.
- Date: 2026-07-22.
- L2 flush: disabled.
- Number of measurements: 100 per shape.
- Correctness: passed against the zero-filled dense DeepGEMM result.
- NCU workload: `M=512, N=1408, K=2048`, sparse kernel.
- NCU summary: `36.96 us`, SM throughput `14.62%`, memory throughput
  `11.82%`, DRAM throughput `2.64%`, achieved occupancy `15.45%`, and 94
  registers per thread. L1TEX long-scoreboard stalls represented `43.37%` of
  the average cycles between issued instructions.
- Result: baseline recorded for all 12 standard projection shapes.
- Decision: retain as the comparison baseline.

### Iteration 1: Block-Row Metadata Prefetch

- Status: completed.
- Checklist item: topology/metadata prefetching.
- Kernel version: `64 x 64` two-stage TMA with raw metadata staged in shared
  memory before the sparse mainloop.
- Kernel commit: `ecf77bb`.
- Benchmark harness commit: `6e8d58f`.
- Date: 2026-07-22.
- GPU: NVIDIA H20 (SM90).
- Weight block: `64 x 64`.
- Output tile: `64 x 64`.
- Pipeline stages: two for weight and activation; block-row metadata is
  prefetched once per CTA.
- L2 flush: disabled.
- Number of measurements: 100 per shape.
- Correctness: all six legal 2:4 metadata pairs and row-varying metadata tests
  passed against the Torch reference. The complete hybrid sparse CUDA test
  file passed all 13 tests.
- NCU workload: `M=512, N=1408, K=2048`, sparse kernel.
- NCU changes: duration `36.96 -> 24.74 us`, SM throughput
  `14.62% -> 22.14%`, memory throughput `11.82% -> 15.14%`, DRAM throughput
  `2.64% -> 3.97%`, and achieved occupancy `15.45% -> 15.66%`. Register count
  remained 94. The previous `43.37%` L1TEX long-scoreboard warning disappeared.
- Performance change: total latency improved on every standard shape by
  `6.6%` to `15.1%`. The gate/up M=2048 sparse path improved from
  `74.98 us` to `53.19 us`; gate/up M=4096 improved from `133.88 us` to
  `94.79 us`.
- Result: contiguous shared-memory staging removes fine-grained metadata
  global loads from the WGMMA.SP critical path.
- Decision: retain as the new preferred `64 x 64` kernel version.

### Iteration 2: Fused Dense/Sparse Mainloop

- Status: completed.
- Checklist item: fused dense/sparse mainloop.
- Kernel version: `64 x 64` two-stage TMA with one FP32 accumulator and direct
  BF16 global stores.
- Commit: `e9ac569`.
- Date: 2026-07-23.
- GPU: NVIDIA H20 (SM90).
- Correctness: all six legal 2:4 metadata pairs and row-varying metadata tests
  passed against the Torch reference.
- Result: removed the two partial-output buffers, reduction kernel, and three
  kernel launches from the steady-state path.
- Decision: retain as the fused-mainloop baseline.

### Iteration 3: BF16 STSM/TMA Epilogue

- Status: completed.
- Checklist item: fused STSM/TMA epilogue.
- Kernel version: fused `64 x 64` mainloop with FP32-to-BF16 conversion,
  `stmatrix.sync.aligned.x2.m8n8.shared.b16.trans`, 128-byte XOR-swizzled
  shared memory, and TMA global store.
- Commit: `9e1f379`; include fix: `708d7bd`.
- Date: 2026-07-23.
- GPU: NVIDIA H20 (SM90).
- Weight block: `64 x 64`.
- Output tile: `64 x 64`.
- Pipeline stages: two for weight and activation; one synchronous output stage.
- L2 flush: disabled.
- Number of measurements: 100 per shape.
- Correctness: all 13 hybrid sparse CUDA tests passed, including all six legal
  metadata pairs and row-varying metadata.
- NCU workload: `M=512, N=1408, K=2048`.
- NCU changes from fused direct: duration `54.37 -> 34.18 us`, registers per
  thread `94 -> 74`, theoretical occupancy `25.0% -> 37.5%`, achieved
  occupancy `14.93% -> 18.09%`, and SM throughput `30.16% -> 48.35%`.
  The direct kernel's half-utilized global-store sector warning disappeared.
- Remaining bottleneck: the grid covers only `0.75` wave per SM and barrier
  stalls account for `37.0%` of average issue latency.
- Performance change: STSM/TMA improves the gate/up projection at M=4096 from
  `248.27 us` to `185.84 us` and slightly exceeds DeepGEMM at both M=4096
  projection shapes. Small-M latency remains behind DeepGEMM.
- Result: retain the DeepGEMM-style BF16 STSM/TMA data path. Test persistent
  output-stage overlap as a separate iteration.
- Decision: retain.

## Iteration Template

```markdown
### Iteration N: Technique Name

- Status: completed / failed / pending.
- Checklist item:
- Kernel version:
- Commit:
- Date:
- GPU:
- Weight block:
- Output tile:
- Pipeline stages:
- L2 flush:
- Number of measurements:
- Correctness:
- NCU workload:
- NCU changes:
- Performance change:
- Result:
- Decision: retain / reject / investigate.
```
