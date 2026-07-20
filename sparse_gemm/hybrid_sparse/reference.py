"""Torch correctness references for hybrid block sparse GEMM kernels."""

from typing import Optional

import torch

from .format import HybridBlockSparseWeight


def _resolve_out_dtype(a: torch.Tensor, out_dtype: Optional[torch.dtype]) -> torch.dtype:
    dtype = a.dtype if out_dtype is None else out_dtype
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("out_dtype must be a floating-point dtype")
    return dtype


def _validate_activation(a: torch.Tensor, dimensions: int) -> None:
    if a.dim() != dimensions:
        raise ValueError(f"activation must be {dimensions}D, got shape {tuple(a.shape)}")
    if not a.is_floating_point():
        raise TypeError("activation must use a floating-point dtype")


def _validate_weight_device(a: torch.Tensor, weight: torch.Tensor) -> None:
    if a.device != weight.device:
        raise ValueError("activation and packed weight must be on the same device")


def hybrid_block_sparse_gemm_ref(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    c: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute ``A[M,K] @ W[N,K].T`` using a decoded packed weight."""
    _validate_activation(a, 2)
    weight = packed_weight.to_dense()
    if weight.dim() != 2:
        raise ValueError("ordinary GEMM requires packed weight shape [N, K]")
    _validate_weight_device(a, weight)
    if a.shape[1] != weight.shape[1]:
        raise ValueError(
            f"activation K ({a.shape[1]}) must match weight K ({weight.shape[1]})"
        )

    result = a.float() @ weight.float().t()
    if c is not None:
        if c.shape != result.shape:
            raise ValueError(
                f"c shape must be {tuple(result.shape)}, got {tuple(c.shape)}"
            )
        if c.device != a.device:
            raise ValueError("c and activation must be on the same device")
        if not c.is_floating_point():
            raise TypeError("c must use a floating-point dtype")
        result = result + c.float()
    return result.to(_resolve_out_dtype(a, out_dtype))


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _validate_index_tensor(
    tensor: torch.Tensor, name: str, expected_size: int, device: torch.device
) -> None:
    if tensor.dim() != 1 or tensor.numel() != expected_size:
        raise ValueError(f"{name} must have shape [{expected_size}]")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have dtype int32 or int64")
    if tensor.device != device:
        raise ValueError(f"{name} and activation must be on the same device")


def hybrid_block_sparse_grouped_contiguous_ref(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    grouped_layout: torch.Tensor,
    m_alignment: int,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute an M-grouped GEMM using SparseGEMM's psum layout semantics."""
    _validate_activation(a, 2)
    weight = packed_weight.to_dense()
    if weight.dim() != 3:
        raise ValueError("grouped GEMM requires packed weight shape [E, N, K]")
    _validate_weight_device(a, weight)
    experts, n, k = weight.shape
    if a.shape[1] != k:
        raise ValueError(f"activation K ({a.shape[1]}) must match weight K ({k})")
    if not isinstance(m_alignment, int) or isinstance(m_alignment, bool):
        raise TypeError("m_alignment must be an integer")
    if m_alignment <= 0:
        raise ValueError("m_alignment must be greater than zero")
    _validate_index_tensor(grouped_layout, "grouped_layout", experts, a.device)

    ends = grouped_layout.tolist()
    total_m = a.shape[0]
    output = torch.zeros((total_m, n), dtype=torch.float32, device=a.device)
    previous_end = 0
    for expert, end in enumerate(ends):
        start = 0 if expert == 0 else _align(previous_end, m_alignment)
        if end < start:
            raise ValueError(
                f"grouped_layout[{expert}] ({end}) is before group start ({start})"
            )
        if end > total_m:
            raise ValueError(
                f"grouped_layout[{expert}] ({end}) exceeds total_m ({total_m})"
            )
        if end > start:
            output[start:end] = a[start:end].float() @ weight[expert].float().t()
        previous_end = end
    if experts and _align(previous_end, m_alignment) > total_m:
        raise ValueError("activation does not include the final psum alignment region")
    return output.to(_resolve_out_dtype(a, out_dtype))


def hybrid_block_sparse_grouped_masked_ref(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    masked_m: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute a masked M-grouped GEMM with deterministic zero tail rows."""
    _validate_activation(a, 3)
    weight = packed_weight.to_dense()
    if weight.dim() != 3:
        raise ValueError("grouped GEMM requires packed weight shape [E, N, K]")
    _validate_weight_device(a, weight)
    experts, max_m, k = a.shape
    weight_experts, n, weight_k = weight.shape
    if experts != weight_experts:
        raise ValueError(
            f"activation experts ({experts}) must match weight experts ({weight_experts})"
        )
    if k != weight_k:
        raise ValueError(f"activation K ({k}) must match weight K ({weight_k})")
    _validate_index_tensor(masked_m, "masked_m", experts, a.device)

    counts = masked_m.tolist()
    output = torch.zeros(
        (experts, max_m, n), dtype=torch.float32, device=a.device
    )
    for expert, count in enumerate(counts):
        if count < 0 or count > max_m:
            raise ValueError(
                f"masked_m[{expert}] must be in [0, {max_m}], got {count}"
            )
        if count:
            output[expert, :count] = (
                a[expert, :count].float() @ weight[expert].float().t()
            )
    return output.to(_resolve_out_dtype(a, out_dtype))
