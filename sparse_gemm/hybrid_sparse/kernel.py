"""CUDA entry points for hybrid block sparse weights."""

from typing import Optional

import torch

from .format import HybridBlockSparseWeight


def _validate_grouped_inputs(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    grouped_index: torch.Tensor,
    expected_a_dim: int,
) -> tuple[int, int, int]:
    if not isinstance(packed_weight, HybridBlockSparseWeight):
        raise TypeError("packed_weight must be a HybridBlockSparseWeight")
    if len(packed_weight.original_shape) != 3:
        raise ValueError("grouped GEMM requires packed weight shape [E, N, K]")
    if packed_weight.layout.block_h != 64 or packed_weight.layout.block_w != 64:
        raise ValueError("naive grouped GEMM currently requires block_h=block_w=64")
    if a.dim() != expected_a_dim:
        raise ValueError(
            f"activation must be {expected_a_dim}D, got shape {tuple(a.shape)}"
        )
    if a.dtype != torch.bfloat16:
        raise TypeError("activation must have dtype torch.bfloat16")
    if not a.is_cuda:
        raise ValueError("activation must be a CUDA tensor")
    if not a.is_contiguous():
        raise ValueError("activation must be contiguous")

    experts, n, k = packed_weight.original_shape
    if a.shape[-1] != k:
        raise ValueError(f"activation K ({a.shape[-1]}) must match weight K ({k})")
    if grouped_index.shape != (experts,):
        raise ValueError(
            f"grouped index must have shape {(experts,)}, got {tuple(grouped_index.shape)}"
        )
    if grouped_index.dtype != torch.int32:
        raise TypeError("grouped index must have dtype torch.int32")
    if grouped_index.device != a.device or not grouped_index.is_contiguous():
        raise ValueError("grouped index must be contiguous on the activation device")

    packed_tensors = (
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        packed_weight.sparse_metadata,
    )
    if packed_weight.dense_values.dtype != torch.bfloat16:
        raise TypeError("packed weight values must have dtype torch.bfloat16")
    if any(tensor.device != a.device for tensor in packed_tensors):
        raise ValueError("all packed tensors must be on the activation device")
    if any(not tensor.is_contiguous() for tensor in packed_tensors):
        raise ValueError("all packed tensors must be contiguous")
    return experts, n, k


def _prepare_grouped_out(
    shape: tuple[int, ...], a: torch.Tensor, out: Optional[torch.Tensor]
) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=a.device)
    if out.shape != shape:
        raise ValueError(f"out must have shape {shape}, got {tuple(out.shape)}")
    if out.dtype != torch.bfloat16 or out.device != a.device:
        raise ValueError("out must be BF16 on the same CUDA device as activation")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


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


def hybrid_block_sparse_gemm_tensorcore(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the synchronous two-path BF16 Tensor Core implementation on Hopper."""
    if not isinstance(packed_weight, HybridBlockSparseWeight):
        raise TypeError("packed_weight must be a HybridBlockSparseWeight")
    if len(packed_weight.original_shape) != 2:
        raise ValueError("Tensor Core GEMM requires packed weight shape [N, K]")
    if packed_weight.layout.block_h != 64 or packed_weight.layout.block_w != 64:
        raise ValueError("Tensor Core GEMM currently requires block_h=block_w=64")
    if a.dim() != 2:
        raise ValueError(f"activation must have shape [M, K], got {tuple(a.shape)}")
    if a.dtype != torch.bfloat16 or not a.is_cuda or not a.is_contiguous():
        raise ValueError("activation must be contiguous BF16 on CUDA")

    n, k = packed_weight.original_shape
    if a.shape[1] != k:
        raise ValueError(f"activation K ({a.shape[1]}) must match weight K ({k})")
    packed_tensors = (
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        packed_weight.sparse_metadata,
    )
    if packed_weight.dense_values.dtype != torch.bfloat16:
        raise TypeError("packed weight values must have dtype torch.bfloat16")
    if any(tensor.device != a.device for tensor in packed_tensors):
        raise ValueError("all packed tensors must be on the activation device")
    if any(not tensor.is_contiguous() for tensor in packed_tensors):
        raise ValueError("all packed tensors must be contiguous")

    if out is None:
        out = torch.empty((a.shape[0], n), dtype=torch.bfloat16, device=a.device)
    elif (
        out.shape != (a.shape[0], n)
        or out.dtype != torch.bfloat16
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 with shape [M, N] on CUDA")

    import deep_gemm

    deep_gemm._C.hybrid_block_sparse_bf16_gemm_tensorcore(
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


def hybrid_block_sparse_grouped_contiguous_naive(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    grouped_layout: torch.Tensor,
    m_alignment: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the naive BF16 grouped GEMM with psum contiguous layout semantics."""
    _, n, _ = _validate_grouped_inputs(a, packed_weight, grouped_layout, 2)
    if not isinstance(m_alignment, int) or isinstance(m_alignment, bool):
        raise TypeError("m_alignment must be an integer")
    if m_alignment <= 0:
        raise ValueError("m_alignment must be greater than zero")
    out = _prepare_grouped_out((a.shape[0], n), a, out)

    import deep_gemm

    deep_gemm._C.hybrid_block_sparse_bf16_grouped_contiguous_naive(
        a,
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        packed_weight.sparse_metadata,
        grouped_layout,
        out,
        m_alignment,
        packed_weight.layout.block_n,
        packed_weight.layout.block_m,
    )
    return out


def hybrid_block_sparse_grouped_masked_naive(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    masked_m: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the naive BF16 grouped GEMM with per-expert valid M counts."""
    experts, n, _ = _validate_grouped_inputs(a, packed_weight, masked_m, 3)
    if a.shape[0] != experts:
        raise ValueError(
            f"activation experts ({a.shape[0]}) must match weight experts ({experts})"
        )
    out = _prepare_grouped_out((experts, a.shape[1], n), a, out)

    import deep_gemm

    deep_gemm._C.hybrid_block_sparse_bf16_grouped_masked_naive(
        a,
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        packed_weight.sparse_metadata,
        masked_m,
        out,
        packed_weight.layout.block_n,
        packed_weight.layout.block_m,
    )
    return out
