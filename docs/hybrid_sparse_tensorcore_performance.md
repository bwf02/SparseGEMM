# Hybrid Sparse Tensor Core Performance

## Configuration

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

## Results

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
