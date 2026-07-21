"""CUDA entry points for hybrid block sparse weights."""

from typing import Optional

import torch

from .format import HybridBlockSparseWeight


def hybrid_block_sparse_gemm_naive(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the naive three-kernel BF16 implementation on Hopper.

    The implementation uses independent dense-block and 2:4-block kernels
    that write FP32 partial outputs, followed by a BF16 reduction kernel.
    Version one is intentionally fixed to a 64x64 hybrid block layout.
    """
    if not isinstance(packed_weight, HybridBlockSparseWeight):
        raise TypeError("packed_weight must be a HybridBlockSparseWeight")
    if len(packed_weight.original_shape) != 2:
        raise ValueError("naive GEMM requires packed weight shape [N, K]")
    if packed_weight.layout.block_h != 64 or packed_weight.layout.block_w != 64:
        raise ValueError("naive GEMM currently requires block_h=block_w=64")
    if a.dim() != 2:
        raise ValueError(f"activation must have shape [M, K], got {tuple(a.shape)}")
    if a.dtype != torch.bfloat16:
        raise TypeError("activation must have dtype torch.bfloat16")
    if not a.is_cuda:
        raise ValueError("activation must be a CUDA tensor")
    if not a.is_contiguous():
        raise ValueError("activation must be contiguous")

    n, k = packed_weight.original_shape
    if a.shape[1] != k:
        raise ValueError(f"activation K ({a.shape[1]}) must match weight K ({k})")
    if packed_weight.dense_values.dtype != torch.bfloat16:
        raise TypeError("packed weight values must have dtype torch.bfloat16")
    if packed_weight.dense_values.device != a.device:
        raise ValueError("activation and packed weight must be on the same device")
    packed_tensors = (
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        packed_weight.sparse_metadata,
    )
    if any(tensor.device != a.device for tensor in packed_tensors):
        raise ValueError("all packed tensors must be on the activation device")
    if any(not tensor.is_contiguous() for tensor in packed_tensors):
        raise ValueError("all packed tensors must be contiguous")

    if out is None:
        out = torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)
    else:
        if out.shape != (a.shape[0], n):
            raise ValueError(
                f"out must have shape {(a.shape[0], n)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16 or out.device != a.device:
            raise ValueError("out must be BF16 on the same CUDA device as activation")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    import deep_gemm  # Imported lazily so format and reference APIs remain CPU-only.

    deep_gemm._C.hybrid_block_sparse_bf16_gemm_naive(
        a,
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        packed_weight.sparse_metadata,
        out,
        packed_weight.layout.block_n,
        packed_weight.layout.block_m,
    )
    return out
