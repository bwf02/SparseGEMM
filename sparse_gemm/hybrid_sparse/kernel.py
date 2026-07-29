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


def hybrid_block_sparse_gemm_wgmma_sync(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the synchronous-load, warpgroup WGMMA BF16 implementation on Hopper."""
    if not isinstance(packed_weight, HybridBlockSparseWeight):
        raise TypeError("packed_weight must be a HybridBlockSparseWeight")
    if len(packed_weight.original_shape) != 2:
        raise ValueError("WGMMA GEMM requires packed weight shape [N, K]")
    if packed_weight.layout.block_h != 64 or packed_weight.layout.block_w != 64:
        raise ValueError("WGMMA GEMM currently requires block_h=block_w=64")
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

    deep_gemm._C.hybrid_block_sparse_bf16_gemm_wgmma_sync(
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


def _hybrid_block_sparse_gemm_wgmma_tma(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor],
    binding: str,
    block_h: int = 64,
    block_w: int = 64,
    metadata: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(packed_weight, HybridBlockSparseWeight):
        raise TypeError("packed_weight must be a HybridBlockSparseWeight")
    if len(packed_weight.original_shape) != 2:
        raise ValueError("WGMMA TMA GEMM requires packed weight shape [N, K]")
    if (
        packed_weight.layout.block_h != block_h
        or packed_weight.layout.block_w != block_w
    ):
        raise ValueError(
            f"{binding} requires block_h={block_h}, block_w={block_w}"
        )
    if a.dim() != 2:
        raise ValueError(f"activation must have shape [M, K], got {tuple(a.shape)}")
    if a.dtype != torch.bfloat16 or not a.is_cuda or not a.is_contiguous():
        raise ValueError("activation must be contiguous BF16 on CUDA")

    n, k = packed_weight.original_shape
    if a.shape[1] != k:
        raise ValueError(f"activation K ({a.shape[1]}) must match weight K ({k})")
    kernel_metadata = packed_weight.sparse_metadata if metadata is None else metadata
    packed_tensors = (
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        kernel_metadata,
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

    getattr(deep_gemm._C, binding)(
        a,
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        kernel_metadata,
        out,
        packed_weight.layout.block_n,
        packed_weight.layout.block_m,
    )
    return out


def hybrid_block_sparse_gemm_wgmma_tma(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 64x64 two-stage TMA and WGMMA BF16 implementation on Hopper."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma",
    )


def hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 64x64 TMA kernel with block-row metadata staged in shared memory."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_metadata_prefetch",
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_direct(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fused 64x64 hybrid mainloop with direct BF16 output stores."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_direct",
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fused 64x64 mainloop with a BF16 STSM/TMA epilogue."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm",
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fused STSM/TMA kernel with persistent output-tile scheduling."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent",
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the persistent fused kernel with pre-encoded WGMMA.SP metadata."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run Stage 3 with direct global lane-ready metadata loads."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Merge two logical K blocks into each lane-ready WGMMA group."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the lane-ready persistent kernel with three TMA stages."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Overlap the next tile with the direct-metadata epilogue."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prefetch lane-ready metadata while waiting for the TMA stage."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stage each sparse block's lane-ready metadata with TMA."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stage metadata with a cooperative producer-warp vector copy."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reuse each weight tile across two 64-row activation tiles."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Broadcast each pipeline stage's block kind through shared memory."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run output128 stage-kind with reusable GMMA descriptors."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run output128 descriptor reuse with a compact 320-thread CTA."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the stage-kind kernel with a 32x64 small-M output tile."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 32x64 kernel with three merged K blocks and six TMA stages."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 64x64 output-tile kernel with lane-ready metadata."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 64x64 kernel with three asynchronous WGMMA groups."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the N:M-group staged 64x64 output-tile kernel."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
    partial: Optional[torch.Tensor] = None,
    tile_counters: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run split-K2 with FP32 partials and an in-kernel final reduction."""
    if not isinstance(packed_weight, HybridBlockSparseWeight):
        raise TypeError("packed_weight must be a HybridBlockSparseWeight")
    if len(packed_weight.original_shape) != 2:
        raise ValueError("split-K2 GEMM requires packed weight shape [N, K]")
    if (
        packed_weight.layout.block_h != 64
        or packed_weight.layout.block_w != 64
        or packed_weight.layout.block_n != 1
        or packed_weight.layout.block_m != 2
    ):
        raise ValueError("split-K2 GEMM requires 64x64 blocks with N:M=1:2")
    if a.dim() != 2:
        raise ValueError(f"activation must have shape [M, K], got {tuple(a.shape)}")
    if a.dtype != torch.bfloat16 or not a.is_cuda or not a.is_contiguous():
        raise ValueError("activation must be contiguous BF16 on CUDA")

    n, k = packed_weight.original_shape
    m = a.shape[0]
    if a.shape[1] != k:
        raise ValueError(f"activation K ({a.shape[1]}) must match weight K ({k})")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError("packed_weight does not contain lane-ready hardware metadata")
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    packed_tensors = (
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        metadata,
    )
    if packed_weight.dense_values.dtype != torch.bfloat16:
        raise TypeError("packed weight values must have dtype torch.bfloat16")
    if any(tensor.device != a.device for tensor in packed_tensors):
        raise ValueError("all packed tensors must be on the activation device")
    if any(not tensor.is_contiguous() for tensor in packed_tensors):
        raise ValueError("all packed tensors must be contiguous")

    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    elif (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 with shape [M, N] on CUDA")

    partial_shape = (2, m, n)
    if partial is None:
        partial = torch.empty(partial_shape, dtype=torch.float32, device=a.device)
    elif (
        partial.shape != partial_shape
        or partial.dtype != torch.float32
        or partial.device != a.device
        or not partial.is_contiguous()
    ):
        raise ValueError(
            "partial must be contiguous FP32 with shape [2, M, N] on CUDA"
        )

    num_tiles = ((m + 63) // 64) * ((n + 63) // 64)
    if tile_counters is None:
        tile_counters = torch.zeros(num_tiles, dtype=torch.int32, device=a.device)
    elif (
        tile_counters.shape != (num_tiles,)
        or tile_counters.dtype != torch.int32
        or tile_counters.device != a.device
        or not tile_counters.is_contiguous()
    ):
        raise ValueError(
            "tile_counters must be contiguous INT32 with one entry per output tile"
        )

    import deep_gemm

    deep_gemm._C.hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce(
        a,
        packed_weight.block_selector,
        packed_weight.dense_values,
        packed_weight.sparse_values,
        metadata,
        partial,
        tile_counters,
        out,
        packed_weight.layout.block_n,
        packed_weight.layout.block_m,
    )
    return out


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the N:M-group staged 48x64 output-tile kernel."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 48x64 group-stage kernel with a compiled 1:2 fast path."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 48x64 1:2 fast path with one producer warp."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 48x64 1:2 fast path with two K groups per commit."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("merge2 requires block_n=1 and block_m=2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 48x64 1:2 fast path with reusable GMMA descriptors."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("descriptor reuse requires block_n=1 and block_m=2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 1:2 fast path with lane-ready metadata loaded by TMA."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 80x64 group-stage kernel with a compiled 1:2 fast path."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 96x64 group-stage kernel with a compiled 1:2 fast path."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the output96 1:2 fast path with reusable GMMA descriptors."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("output96 descriptor reuse requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 88-row 1:2 fast path with reusable GMMA descriptors."""
    if (packed_weight.layout.block_n, packed_weight.layout.block_m) != (1, 2):
        raise ValueError("output88 group-stage fast path requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the output80 1:2 fast path with reusable GMMA descriptors."""
    if (packed_weight.layout.block_n, packed_weight.layout.block_m) != (1, 2):
        raise ValueError("output80 group-stage fast path requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run output128 with one pipeline stage per 1:2 block group."""
    if (packed_weight.layout.block_n, packed_weight.layout.block_m) != (1, 2):
        raise ValueError("output128 group-stage fast path requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 64x64 group-stage kernel with a compiled 1:2 fast path."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 64x64 1:2 fast path with reusable GMMA descriptors."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("descriptor reuse requires block_n=1 and block_m=2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run descriptor reuse with M/N/K compiled into the JIT kernel."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("fixed-shape descriptor reuse requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run fixed-shape descriptor reuse with a seven-stage TMA pipeline."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("stage7 fixed-shape kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run stage7 with the compile-time K-group loops fully unrolled."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("stage7 unroll-k kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run six mixed sparse/dense WGMMA instructions as one async group."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("fused MMA-group kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fused MMA group with lane-uniform descriptor bases."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("uniform-descriptor fused MMA kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the fused MMA group with compile-time shared-memory descriptors."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("constexpr-descriptor fused MMA kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run constexpr descriptors with complete WGMMA groups per selector branch."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("constexpr branch-group kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one constexpr branch-group output tile per CTA."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("constexpr branch-group full-grid kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run unrolled stage7 with a 32x64 output tile."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("output32 stage7 unroll-k kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run stage7 while keeping up to three WGMMA groups in flight."""
    if packed_weight.layout.block_n != 1 or packed_weight.layout.block_m != 2:
        raise ValueError("async-group3 kernel requires N:M=1:2")
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3",
        metadata=metadata,
    )


def _select_hybrid_block_sparse_gemm_wgmma_tuned(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
):
    layout = packed_weight.layout
    shape = (a.shape[0], *packed_weight.original_shape)
    if (layout.block_n, layout.block_m) == (1, 2):
        tuned = {
            (128, 1408, 2048): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse,
            (256, 1408, 2048): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse,
            (512, 1408, 2048): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse,
            (1024, 1408, 2048): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse,
            (128, 2048, 1408): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group,
            (256, 2048, 1408): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse,
            (512, 2048, 1408): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse,
            (1024, 2048, 1408): hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath,
        }
        if shape in tuned:
            return tuned[shape]
    return hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64


def hybrid_block_sparse_gemm_wgmma_tuned(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch verified Qwen MoE shapes and fall back to generic group staging."""
    kernel = _select_hybrid_block_sparse_gemm_wgmma_tuned(a, packed_weight)
    return kernel(a, packed_weight, out=out)


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Broadcast one selector per block group through shared memory."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run Stage 3 with a 40/128 producer/math warpgroup register budget."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the lane-ready persistent kernel with four TMA stages."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the lane-ready persistent kernel with five TMA stages."""
    metadata = packed_weight.hardware_metadata
    if metadata is None:
        raise ValueError(
            "packed_weight does not contain lane-ready hardware metadata"
        )
    if metadata.dtype != torch.int32:
        raise TypeError("hardware_metadata must have dtype torch.int32")
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5",
        metadata=metadata,
    )


def hybrid_block_sparse_gemm_wgmma_tma_128x64(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 128x64 two-stage TMA and WGMMA BF16 implementation on Hopper."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_128x64",
    )


def hybrid_block_sparse_gemm_wgmma_tma_block128x32(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 128x32 weight-block TMA and WGMMA BF16 implementation."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32",
        block_h=128,
        block_w=32,
    )


def hybrid_block_sparse_gemm_wgmma_tma_block128x32_stage3(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 128x32 weight-block kernel with a three-stage TMA pipeline."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_stage3",
        block_h=128,
        block_w=32,
    )


def hybrid_block_sparse_gemm_wgmma_tma_block128x32_output128x128(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 128x32 weight tile with a 128x128 CTA output tile."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x32_output128x128",
        block_h=128,
        block_w=32,
    )


def hybrid_block_sparse_gemm_wgmma_tma_block128x64(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 128x64 weight-block TMA and WGMMA BF16 implementation."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x64",
        block_h=128,
        block_w=64,
    )


def hybrid_block_sparse_gemm_wgmma_tma_block128x128(
    a: torch.Tensor,
    packed_weight: HybridBlockSparseWeight,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the 128x128 weight-block TMA and WGMMA BF16 implementation."""
    return _hybrid_block_sparse_gemm_wgmma_tma(
        a,
        packed_weight,
        out,
        "hybrid_block_sparse_bf16_gemm_wgmma_tma_block128x128",
        block_h=128,
        block_w=128,
    )


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
