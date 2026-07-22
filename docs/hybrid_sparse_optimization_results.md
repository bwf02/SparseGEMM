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
- Report dense, sparse, reduction, total, and DeepGEMM latency in microseconds.
- Run correctness validation against the dense Torch/DeepGEMM result before
  recording performance.
- Record whether L2 flush is enabled and keep that setting identical across
  compared versions.

## Gate/Up Projection

`N = 1408, K = 2048`

| Version | M | Dense (us) | Sparse (us) | Reduce (us) | Total (us) | DeepGEMM (us) | DG / Hybrid | Correct |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | 128 | - | - | - | - | - | - | - |
| Baseline | 256 | - | - | - | - | - | - | - |
| Baseline | 512 | - | - | - | - | - | - | - |
| Baseline | 1024 | - | - | - | - | - | - | - |
| Baseline | 2048 | - | - | - | - | - | - | - |
| Baseline | 4096 | - | - | - | - | - | - | - |

## Down Projection

`N = 2048, K = 1408`

| Version | M | Dense (us) | Sparse (us) | Reduce (us) | Total (us) | DeepGEMM (us) | DG / Hybrid | Correct |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | 128 | - | - | - | - | - | - | - |
| Baseline | 256 | - | - | - | - | - | - | - |
| Baseline | 512 | - | - | - | - | - | - | - |
| Baseline | 1024 | - | - | - | - | - | - | - |
| Baseline | 2048 | - | - | - | - | - | - | - |
| Baseline | 4096 | - | - | - | - | - | - | - |

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

## Iteration Log

Add one section per attempted optimization. Keep rejected versions in the log.

### Iteration 0: Standard-Sweep Baseline

- Status: pending.
- Kernel version: current `64 x 64` two-stage TMA implementation.
- Commit: pending.
- Date: pending.
- L2 flush: pending.
- Number of measurements: pending.
- Correctness: pending.
- NCU workload: pending.
- NCU summary: pending.
- Result: pending.
- Decision: pending.

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
