"""Hybrid dense-block and 2:4-block sparse weight utilities."""

from .format import (
    HybridBlockSparseLayout,
    HybridBlockSparseWeight,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_to_dense,
)
from .reference import (
    hybrid_block_sparse_gemm_ref,
    hybrid_block_sparse_grouped_contiguous_ref,
    hybrid_block_sparse_grouped_masked_ref,
)
from .kernel import (
    hybrid_block_sparse_gemm_naive,
    hybrid_block_sparse_grouped_contiguous_naive,
    hybrid_block_sparse_grouped_masked_naive,
)

__all__ = [
    "HybridBlockSparseLayout",
    "HybridBlockSparseWeight",
    "dense_to_hybrid_block_sparse",
    "hybrid_block_sparse_gemm_ref",
    "hybrid_block_sparse_gemm_naive",
    "hybrid_block_sparse_grouped_contiguous_naive",
    "hybrid_block_sparse_grouped_contiguous_ref",
    "hybrid_block_sparse_grouped_masked_naive",
    "hybrid_block_sparse_grouped_masked_ref",
    "hybrid_block_sparse_to_dense",
]
