# Hybrid Sparse Tensor Core Performance

## Warp MMA Configuration

- GPU: NVIDIA H20
- Data type: BF16 input/output with FP32 accumulation
- Hybrid layout: `64 x 64`, one 2:4 sparse block per two blocks
- Element sparsity: 25%
- Current implementation: separate dense `mma.sync` and sparse `mma.sp.sync`
  kernels followed by FP32 partial reduction
- Pipeline: direct synchronous global-to-register loads; no TMA, `cp.async`,
  multistage pipeline, or overlap
- Baseline: DeepGEMM BF16 dense GEMM on the zero-filled weight
- Timing: `bench_kineto`, 10 measurements with L2 flush; weight packing excluded

## Warp MMA Results

| M | N | K | Current (us) | DeepGEMM (us) | Current TFLOPS | DeepGEMM TFLOPS | Relative to DeepGEMM | Latency gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1408 | 2048 | 222.8 | 8.5 | 0.03 | 0.68 | 3.82% | 26.21x |
| 1 | 2048 | 1408 | 155.7 | 7.8 | 0.04 | 0.74 | 5.01% | 19.96x |
| 8 | 1408 | 2048 | 224.5 | 8.6 | 0.21 | 5.36 | 3.83% | 26.10x |
| 8 | 2048 | 1408 | 156.8 | 7.7 | 0.29 | 5.99 | 4.91% | 20.36x |
| 32 | 1408 | 2048 | 236.0 | 8.7 | 0.78 | 21.21 | 3.69% | 27.13x |
| 32 | 2048 | 1408 | 165.0 | 7.6 | 1.12 | 24.28 | 4.61% | 21.71x |
| 128 | 1408 | 2048 | 253.0 | 11.1 | 2.92 | 66.50 | 4.39% | 22.79x |
| 128 | 2048 | 1408 | 179.4 | 10.3 | 4.11 | 71.67 | 5.74% | 17.42x |

`Relative to DeepGEMM` is `DeepGEMM latency / current latency`. `Latency gap`
is `current latency / DeepGEMM latency`; lower is better.

## Synchronous WGMMA Results

This version uses one 128-thread warpgroup per `64 x 64` output tile. Dense
blocks use `HGMMA.64x64x16`; sparse blocks use
`HGMMA.SP.64x64x32`. Both paths synchronously stage global memory in shared
memory and retain the separate FP32 reduction kernel. Timing uses 20
measurements with L2 flush.

| M | N | K | Warp MMA (us) | Sync WGMMA (us) | DeepGEMM (us) | WGMMA / Warp MMA | WGMMA / DeepGEMM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1408 | 2048 | 222.0 | 281.0 | 8.7 | 1.27x | 32.30x |
| 1 | 2048 | 1408 | 156.1 | 198.2 | 7.6 | 1.27x | 26.08x |
| 8 | 1408 | 2048 | 224.4 | 287.3 | 8.5 | 1.28x | 33.80x |
| 8 | 2048 | 1408 | 156.8 | 199.7 | 7.7 | 1.27x | 25.94x |
| 32 | 1408 | 2048 | 234.6 | 297.3 | 8.5 | 1.27x | 34.98x |
| 32 | 2048 | 1408 | 165.2 | 207.1 | 7.7 | 1.25x | 26.90x |
| 128 | 1408 | 2048 | 251.2 | 322.7 | 10.8 | 1.28x | 29.88x |
| 128 | 2048 | 1408 | 179.4 | 224.6 | 10.2 | 1.25x | 22.02x |

At `M=128, N=1408, K=2048`, NCU reports `218.9 us` for the dense
compute kernel and `111.2 us` for the sparse compute kernel. Both have about
`6.25%` active warps. The implementation validates WGMMA and WGMMA.SP
correctness, but synchronous staging and a wait after every block leave the
Tensor Cores underutilized. TMA staging and mainloop overlap are required in
the next version.

## TMA WGMMA Results

The current best version uses a two-stage TMA pipeline with one producer
warpgroup and one WGMMA consumer warpgroup. Dense and sparse blocks remain
separate kernels, followed by FP32 partial reduction. SASS contains
`UTMALDG.2D`, `HGMMA.64x64x16.F32.BF16`, and
`HGMMA.SP.64x64x32.F32.BF16`. Timing uses 20 measurements with L2 flush.

| M | N | K | TMA WGMMA (us) | DeepGEMM (us) | Effective TFLOPS | Relative to DeepGEMM | Latency gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1408 | 2048 | 29.9 | 8.5 | 0.19 | 28.5% | 3.52x |
| 1 | 2048 | 1408 | 23.1 | 7.7 | 0.25 | 33.3% | 3.00x |
| 8 | 1408 | 2048 | 30.6 | 8.6 | 1.51 | 28.3% | 3.56x |
| 8 | 2048 | 1408 | 23.5 | 7.7 | 1.97 | 32.7% | 3.05x |
| 32 | 1408 | 2048 | 32.0 | 8.7 | 5.77 | 27.2% | 3.68x |
| 32 | 2048 | 1408 | 24.7 | 7.8 | 7.46 | 31.4% | 3.17x |
| 128 | 1408 | 2048 | 33.9 | 11.1 | 21.78 | 32.7% | 3.05x |
| 128 | 2048 | 1408 | 26.5 | 10.3 | 27.81 | 38.7% | 2.57x |

At `M=128, N=1408, K=2048`, NCU reports about `12.0 us` for the
dense path and `19.49 us` for the sparse path. Active warps remain about
`8%`; sparse long-scoreboard stalls are `46%`, so sparse metadata and data
arrival latency remain the primary mainloop bottleneck.

## Tile Experiment

The preserved `128 x 64` variant reuses each weight tile across two 64-row
activation subtiles. It passes the same metadata and M-tail correctness tests,
but is slower than `64 x 64` for every measured shape from `M=1` through
`M=512`. At `M=128, N=1408, K=2048`, latency rises from `33.9 us` to
`46.9 us`; NCU reports `16.99 us` dense and `24.29 us` sparse. Halving the CTA
count and doubling accumulator pressure outweigh weight reuse.

With the current mapping, WGMMA's fixed `m64` dimension corresponds to 64
output channels. A true `128 x 32` output tile would compute and discard half
of each WGMMA result. A `128 x 128` CTA can compose two independent 64-row
weight block-rows, but their hybrid selectors differ, so it cannot generally
reuse packed weights and requires roughly twice the accumulator state. These
shapes are therefore not candidates for the default kernel under the current
`64 x 64` block format.
