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

## Current Performance Summary

Current kernel: fused dense/sparse mainloop with the BF16 STSM/TMA epilogue.
Because all contributions accumulate in one kernel, separate sparse, dense,
and reduction latency do not exist for this version and are marked `N/A`.
Speedup is always calculated as `DeepGEMM / Total`.

| Shape (M x N x K) | Sparse latency (us) | Dense latency (us) | Reduce (us) | Total (us) | DeepGEMM (us) | Speedup over DeepGEMM |
|:---|---:|---:|---:|---:|---:|---:|
| 128 x 1408 x 2048 | N/A | N/A | N/A | 18.27 | 10.32 | 0.565x |
| 128 x 2048 x 1408 | N/A | N/A | N/A | 13.65 | 9.26 | 0.678x |
| 256 x 1408 x 2048 | N/A | N/A | N/A | 25.92 | 14.91 | 0.575x |
| 256 x 2048 x 1408 | N/A | N/A | N/A | 19.03 | 14.29 | 0.751x |
| 512 x 1408 x 2048 | N/A | N/A | N/A | 32.66 | 27.35 | 0.837x |
| 512 x 2048 x 1408 | N/A | N/A | N/A | 39.08 | 26.71 | 0.683x |
| 1024 x 1408 x 2048 | N/A | N/A | N/A | 60.59 | 52.78 | 0.871x |
| 1024 x 2048 x 1408 | N/A | N/A | N/A | 80.61 | 51.41 | 0.638x |
| 2048 x 1408 x 2048 | N/A | N/A | N/A | 122.86 | 102.93 | 0.838x |
| 2048 x 2048 x 1408 | N/A | N/A | N/A | 102.04 | 100.49 | 0.985x |
| 4096 x 1408 x 2048 | N/A | N/A | N/A | 185.84 | 198.94 | 1.071x |
| 4096 x 2048 x 1408 | N/A | N/A | N/A | 191.15 | 197.71 | 1.034x |

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
