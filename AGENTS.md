# SparseGEMM Agent Notes

## Hybrid Sparse Terminology

- `weight block size` refers to the hybrid sparse weight format block, currently
  represented by `block_h x block_w`.
- `block_h` is the weight row or output-channel dimension.
- `block_w` is the K dimension.
- `CUDA tile size` refers to the kernel scheduling tile, not the sparse weight
  storage block.
- When tuning tile size against weight block size, keep the CUDA tile
  output-channel dimension equal to the sparse weight block row dimension.

## Kernel Iteration

- Keep older kernel versions in separate files when adding optimized variants,
  so correctness and performance can be compared across iterations.
