# Hybrid Sparse Kernel Optimization Checklist

## Status Convention

- `[ ]`: not tested.
- `[x]`: experiment completed. A checked item does not necessarily mean the
  technique is retained; see its result and decision.
- Each performance experiment should record the kernel version, workload,
  latency, relevant NCU metrics, and final decision.
- Keep every optimized kernel in a separate implementation file so completed
  experiments remain reproducible.

## Current Baseline

- GPU: NVIDIA H20 (SM90).
- Data type: BF16 input/output with FP32 accumulation.
- Hybrid pattern: one 2:4 sparse weight block per two blocks (25% element
  sparsity).
- Preferred small-M configuration: `64 x 64` weight block and `64 x 64`
  output tile.
- Execution: separate dense and sparse kernels followed by FP32 reduction.
- Pipeline: two-stage TMA producer/consumer pipeline.
- Baseline: DeepGEMM BF16 dense GEMM on the zero-filled dense weight.
- Standard benchmark sweep: `M = 128, 256, 512, 1024, 2048, 4096`.
- For grouped GEMM, use total valid M matching the standard sweep where
  practical.
- Record new measurements and per-version decisions in
  `hybrid_sparse_optimization_results.md`; keep the earlier performance document
  unchanged as historical data.

## P0: Mainloop Latency

- [ ] **Prepack hardware-ready sparse metadata**
  - Generate lane-ready WGMMA.SP metadata during weight conversion.
  - Remove byte-code decoding and metadata nibble construction from the CUDA
    mainloop.
  - Measure sparse-kernel latency, long-scoreboard stalls, and Tensor Core
    utilization.
  - Result: pending.

- [ ] **Increase the TMA pipeline from two to four stages**
  - Keep the `64 x 64` weight block and change only the stage count.
  - Compare three, four, and five stages if four stages improves latency.
  - Measure shared-memory usage, occupancy, barrier stalls, long scoreboard,
    and total latency.
  - Result: pending.

- [ ] **Merge multiple K tiles before WGMMA commit/wait**
  - Start with `merge_k = 2` and issue two K tiles before `commit_group` and
    `wait_group`.
  - Preserve each TMA stage until all dependent WGMMA operations complete.
  - Measure wait stalls, Tensor Core utilization, and register pressure.
  - Result: pending.

- [ ] **Apply warpgroup register reallocation**
  - Deallocate producer registers and allocate more registers to the math
    warpgroup, following the DeepGEMM producer/consumer pattern.
  - Test only after the deeper pipeline and merged K loop are correct.
  - Result: pending.

## P1: Scheduling And Shape Selection

- [ ] **Add CTA tile swizzling for L2 locality**
  - Group output tiles so adjacent CTAs reuse the same activation or weight
    region through L2.
  - Compare L2 hit rate and latency with the current two-dimensional order.
  - Result: pending.

- [ ] **Add shape-aware kernel dispatch**
  - Retain `64 x 64` for small-M workloads.
  - Consider `128 x 128` for larger M where its smaller sparse grid is no
    longer the dominant limitation.
  - Select the weight block, output tile, and pipeline stages from M/N/K and
    available shared memory.
  - Result: pending.

- [ ] **Add a persistent scheduler for grouped GEMM**
  - Launch approximately one CTA per SM and let each CTA consume multiple
    expert tiles.
  - Cover uneven token counts, empty experts, and grouped-layout alignment.
  - Measure load balance and end-to-end grouped latency.
  - Result: pending.

- [ ] **Overlap epilogue/store with the next persistent tile**
  - Stage output through shared memory and overlap TMA store with the next
    tile's mainloop where dependencies permit.
  - Result: pending.

## P2: Kernel Fusion

- [ ] **Fuse dense and sparse paths into one hybrid mainloop**
  - Decode the shared block topology inside one CTA.
  - Issue dense WGMMA or WGMMA.SP for each weight block into the same FP32
    accumulators.
  - Eliminate duplicated activation loads and the two partial-output buffers.
  - Result: pending.

- [ ] **Remove the standalone FP32 reduction kernel**
  - Produce the final output in the fused hybrid kernel.
  - Measure the reduction latency, launch overhead, and partial-buffer traffic
    removed from the end-to-end path.
  - Result: pending.

- [ ] **Use STSM plus TMA for the fused epilogue**
  - Store accumulators into swizzled shared memory, then issue a TMA store to
    the final row-major output.
  - Evaluate only with the fused hybrid mainloop.
  - Result: pending.

## P3: Advanced Hopper Features

- [ ] **Evaluate TMA multicast and CTA clusters**
  - Share activation tiles across adjacent N tiles or weight tiles across M
    tiles.
  - Enable only for shapes with enough waves and measurable operand reuse.
  - Result: pending.

- [ ] **Evaluate a persistent normal-GEMM path**
  - Compare it with static scheduling only after the mainloop latency work.
  - Do not expect it to increase parallelism for very small grids by itself.
  - Result: pending.

- [ ] **Evaluate topology/metadata prefetching**
  - Prefetch upcoming block selectors and sparse metadata independently from
    the weight TMA pipeline.
  - Measure long-scoreboard stalls and instruction overhead.
  - Result: pending.

## Completed Experiments

- [x] **Replace synchronous loads with a two-stage TMA pipeline**
  - Result: large improvement over the synchronous WGMMA implementation;
    retained as the current baseline.

- [x] **Test `64 x 64`, `128 x 32`, `128 x 64`, and `128 x 128` weight blocks**
  - Result: `64 x 64` remains best at M=128; `128 x 128` is faster at M=512.
  - Decision: retain all variants and use the result when implementing
    shape-aware dispatch.

- [x] **Test two consumer warpgroups with a `128 x 128` output tile**
  - Result: occupancy increased, but sparse-kernel duration also increased
    because the grid fell from 88 to 44 CTAs.
  - Decision: retain for comparison, but do not use as the preferred kernel.

- [x] **Profile shared-memory bank conflicts**
  - Result: no bank conflicts were reported in the dense or sparse TMA
    mainloops.
  - Decision: do not prioritize TMA swizzle changes solely for bank-conflict
    removal.

- [x] **Profile memory-bandwidth utilization**
  - Result: DRAM utilization is low; current kernels are latency-bound rather
    than HBM-bandwidth-bound.
  - Decision: prioritize metadata delivery, pipeline depth, and WGMMA wait
    reduction.

## Experiment Record Template

Copy this block under the relevant checklist item after each experiment:

```text
Kernel version:
Commit:
GPU:
Workload (M/N/K, grouped layout):
Correctness:
Latency before / after:
DeepGEMM latency:
NCU changes:
Decision (retain / reject / investigate):
Notes:
```
