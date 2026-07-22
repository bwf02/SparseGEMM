# SparseGEMM Agent Notes

## Hybrid Sparse Terminology

- `weight block size` refers to the hybrid sparse weight format block, currently
  represented by `block_h x block_w`.
- `block_h` is the weight row or output-channel dimension.
- `block_w` is the K dimension.
- `weight tile size` refers to the kernel's logical weight-operand tile
  `weight_tile_n x weight_tile_k`. Keep it equal to one sparse storage block:
  `weight_tile_n == block_h` and `weight_tile_k == block_w`.
- `output tile size` refers to the CTA scheduling tile
  `output_tile_m x output_tile_n`. It is independent of the weight block shape;
  do not require it to equal `block_h x block_w`.
- Output M/N scheduling, warpgroup decomposition, pipeline stages, and the
  number of weight tiles composed by a CTA may be tuned independently.

## Kernel Iteration

- Keep older kernel versions in separate files when adding optimized variants,
  so correctness and performance can be compared across iterations.

## JIT Kernel Organization

- Do not place full CUDA kernel bodies inside JIT-generated raw strings.
- Follow the DeepGEMM style: put readable kernel implementations under
  `deep_gemm/include/deep_gemm/impls/*.cuh`.
- Keep `csrc/jit_kernels/impls/*.hpp` focused on host-side argument checks,
  runtime setup, launch configuration, and a small generated instantiation stub.
- The generated JIT code should normally only include the target `.cuh` file and
  instantiate the selected kernel symbol.
