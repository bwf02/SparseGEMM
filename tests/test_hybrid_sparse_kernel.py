import unittest

import torch

from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_naive,
    hybrid_block_sparse_gemm_tensorcore,
    hybrid_block_sparse_gemm_wgmma_sync,
    hybrid_block_sparse_gemm_wgmma_tma,
    hybrid_block_sparse_gemm_wgmma_tma_fused_direct,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_output_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_async_group2,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage4_async_group2,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata,
    hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch,
    hybrid_block_sparse_gemm_wgmma_tma_128x64,
    hybrid_block_sparse_gemm_wgmma_tma_block128x32,
    hybrid_block_sparse_gemm_wgmma_tma_block128x32_stage3,
    hybrid_block_sparse_gemm_wgmma_tma_block128x32_output128x128,
    hybrid_block_sparse_gemm_wgmma_tma_block128x64,
    hybrid_block_sparse_gemm_wgmma_tma_block128x128,
    hybrid_block_sparse_gemm_wgmma_tuned,
    hybrid_block_sparse_gemm_ref,
    hybrid_block_sparse_grouped_contiguous_naive,
    hybrid_block_sparse_grouped_contiguous_ref,
    hybrid_block_sparse_grouped_contiguous_wgmma_tma,
    hybrid_block_sparse_grouped_masked_naive,
    hybrid_block_sparse_grouped_masked_ref,
    hybrid_block_sparse_grouped_masked_wgmma_tma,
)


def make_mask(weight, layout, sparse_block_ids):
    mask = torch.zeros_like(weight, dtype=torch.bool)
    block_rows = weight.shape[0] // layout.block_h
    block_columns = weight.shape[1] // layout.block_w
    for block_row in range(block_rows):
        row_start = block_row * layout.block_h
        for group_start in range(0, block_columns, layout.block_m):
            for local_block in sparse_block_ids:
                column_start = (group_start + local_block) * layout.block_w
                block = mask[
                    row_start : row_start + layout.block_h,
                    column_start : column_start + layout.block_w,
                ].reshape(layout.block_h, -1, 4)
                block[..., 2:] = True
    return mask


def make_grouped_mask(weight, layout, sparse_block_ids):
    return torch.stack(
        [make_mask(expert_weight, layout, sparse_block_ids) for expert_weight in weight]
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestHybridSparseNaiveKernel(unittest.TestCase):
    def test_output96_descriptor_reuse_matches_reference(self):
        torch.manual_seed(127)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output88_descriptor_reuse_matches_reference(self):
        torch.manual_seed(125)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output80_descriptor_reuse_matches_reference(self):
        torch.manual_seed(126)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath_desc_reuse(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_group_stage_descriptor_reuse_matches_reference(self):
        torch.manual_seed(124)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_fused_mma_group_matches_reference(self):
        torch.manual_seed(134)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_fixed_shape_unroll_k_matches_reference(self):
        torch.manual_seed(135)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_stage5_output_reuse_matches_reference(self):
        torch.manual_seed(136)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_output_reuse(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_stage5_async_group2_matches_reference(self):
        torch.manual_seed(137)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_async_group2(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_stage4_async_group2_matches_reference(self):
        torch.manual_seed(138)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage4_async_group2(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_stage7_async_group3_matches_reference(self):
        torch.manual_seed(126)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_fixed_shape_stage7_matches_reference(self):
        torch.manual_seed(125)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_fixed_shape_stage7_unroll_k_matches_reference(self):
        torch.manual_seed(127)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_fixed_shape_stage7_fused_mma_group_matches_reference(self):
        torch.manual_seed(129)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_fixed_shape_stage7_fused_mma_group_uniform_desc_matches_reference(self):
        torch.manual_seed(130)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_uniform_desc(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_fixed_shape_stage7_fused_mma_group_constexpr_desc_matches_reference(self):
        torch.manual_seed(131)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_fixed_shape_stage7_constexpr_desc_branch_group_matches_reference(self):
        torch.manual_seed(132)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output64_constexpr_desc_branch_group_full_grid_matches_reference(self):
        torch.manual_seed(133)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(1280, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_fused_mma_group_constexpr_desc_branch_group_full_grid(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output32_fixed_shape_stage7_unroll_k_matches_reference(self):
        torch.manual_seed(128)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        packed = dense_to_hybrid_block_sparse(
            weight, make_mask(weight, layout, (1,)), layout
        )
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output32x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_unroll_k(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_splitk2_fused_reduce_matches_reference_and_resets_counters(self):
        torch.manual_seed(124)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 128, 129):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                out = torch.empty(
                    m, 128, device="cuda", dtype=torch.bfloat16
                )
                partial = torch.empty(
                    2, m, 128, device="cuda", dtype=torch.float32
                )
                counters = torch.zeros(
                    ((m + 63) // 64) * 2,
                    device="cuda",
                    dtype=torch.int32,
                )
                for _ in range(2):
                    actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce(
                        activation,
                        packed,
                        out=out,
                        partial=partial,
                        tile_counters=counters,
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=1e-2, atol=1e-2
                    )
                    self.assertEqual(torch.count_nonzero(counters).item(), 0)

    def test_tensorcore_matches_reference_for_all_metadata_pairs(self):
        torch.manual_seed(100)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros_like(weight, dtype=torch.bool)
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        sparse_block = mask[:, :64].reshape(64, 16, 4)
        for quartet in range(16):
            keep = pairs[quartet % len(pairs)]
            sparse_block[:, quartet, :] = True
            sparse_block[:, quartet, keep[0]] = False
            sparse_block[:, quartet, keep[1]] = False
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(73, 128, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_tensorcore(activation, packed)
        wgmma_actual = hybrid_block_sparse_gemm_wgmma_sync(activation, packed)
        tma_actual = hybrid_block_sparse_gemm_wgmma_tma(activation, packed)
        metadata_prefetch_actual = (
            hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
                activation, packed
            )
        )
        fused_direct_actual = hybrid_block_sparse_gemm_wgmma_tma_fused_direct(
            activation, packed
        )
        fused_stsm_actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm(
            activation, packed
        )
        persistent_actual = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent(
                activation, packed
            )
        )
        lane_ready_actual = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
                activation, packed
            )
        )
        tma_128x64_actual = hybrid_block_sparse_gemm_wgmma_tma_128x64(
            activation, packed
        )

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(wgmma_actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tma_actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            metadata_prefetch_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            fused_direct_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            fused_stsm_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            persistent_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            lane_ready_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(tma_128x64_actual, expected, rtol=1e-2, atol=1e-2)

    def test_tensorcore_matches_reference_for_row_varying_metadata(self):
        torch.manual_seed(102)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros_like(weight, dtype=torch.bool)
        pairs = torch.tensor(
            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
            device="cuda",
        )
        for block_row in range(2):
            for block_column in (0, 128):
                sparse_block = mask[
                    block_row * 64 : (block_row + 1) * 64,
                    block_column : block_column + 64,
                ].reshape(64, 16, 4)
                sparse_block[:] = True
                pair_ids = torch.randint(0, len(pairs), (64, 16), device="cuda")
                keep = pairs[pair_ids]
                sparse_block.scatter_(2, keep, False)
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(65, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_tensorcore(activation, packed)
        wgmma_actual = hybrid_block_sparse_gemm_wgmma_sync(activation, packed)
        tma_actual = hybrid_block_sparse_gemm_wgmma_tma(activation, packed)
        metadata_prefetch_actual = (
            hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
                activation, packed
            )
        )
        fused_direct_actual = hybrid_block_sparse_gemm_wgmma_tma_fused_direct(
            activation, packed
        )
        fused_stsm_actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm(
            activation, packed
        )
        persistent_actual = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent(
                activation, packed
            )
        )
        lane_ready_actual = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
                activation, packed
            )
        )
        tma_128x64_actual = hybrid_block_sparse_gemm_wgmma_tma_128x64(
            activation, packed
        )

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(wgmma_actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tma_actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            metadata_prefetch_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            fused_direct_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            fused_stsm_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            persistent_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            lane_ready_actual, expected, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(tma_128x64_actual, expected, rtol=1e-2, atol=1e-2)

    def test_wgmma_sync_matches_reference_for_m_tails(self):
        torch.manual_seed(103)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 8, 63, 64, 65, 73):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 128, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                sync_actual = hybrid_block_sparse_gemm_wgmma_sync(activation, packed)
                tma_actual = hybrid_block_sparse_gemm_wgmma_tma(activation, packed)
                tma_128x64_actual = hybrid_block_sparse_gemm_wgmma_tma_128x64(
                    activation, packed
                )
                torch.testing.assert_close(sync_actual, expected, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(tma_actual, expected, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(
                    tma_128x64_actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_wgmma_tma_block128x32_matches_reference(self):
        torch.manual_seed(104)
        layout = HybridBlockSparseLayout(128, 32, 1, 2)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 73, 128):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = hybrid_block_sparse_gemm_wgmma_tma_block128x32(
                    activation, packed
                )
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_lane_ready_merge_k2_matches_reference_for_even_and_odd_k_blocks(self):
        torch.manual_seed(109)
        for block_m, k, sparse_block_ids in (
            (2, 256, (0,)),
            (3, 384, (1,)),
        ):
            with self.subTest(block_m=block_m):
                layout = HybridBlockSparseLayout(64, 64, 1, block_m)
                weight = torch.randn(
                    128, k, device="cuda", dtype=torch.bfloat16
                )
                mask = make_mask(weight, layout, sparse_block_ids)
                packed = dense_to_hybrid_block_sparse(weight, mask, layout)
                activation = torch.randn(
                    129, k, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_merge_k2(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )
    def test_lane_ready_stage3_matches_reference(self):
        torch.manual_seed(110)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage3(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_direct_metadata_matches_reference(self):
        torch.manual_seed(114)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_direct_metadata_epilogue_overlap_matches_reference(self):
        torch.manual_seed(115)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_epilogue_overlap(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_direct_metadata_register_prefetch_matches_reference(self):
        torch.manual_seed(116)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_direct_metadata_register_prefetch(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_tma_metadata_matches_reference(self):
        torch.manual_seed(117)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_producer_metadata_copy_matches_reference(self):
        torch.manual_seed(118)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_producer_metadata_copy_output128x64_matches_reference(self):
        torch.manual_seed(119)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 127, 128, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_output128x64_stage_kind_matches_reference(self):
        torch.manual_seed(120)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 65, 127, 128, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_output128_stage_kind_descriptor_reuse_matches_reference(self):
        torch.manual_seed(122)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_output128_descriptor_reuse_compact_matches_reference(self):
        torch.manual_seed(123)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact(
            activation, packed
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_lane_ready_output128x64_stage_selector_matches_reference(self):
        torch.manual_seed(121)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 65, 127, 128, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_output32x64_stage_kind_matches_reference(self):
        torch.manual_seed(122)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 31, 32, 33, 47, 48, 49, 63, 64, 65, 128, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )
                merged = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    merged, expected, rtol=1e-2, atol=1e-2
                )
                wide = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    wide, expected, rtol=1e-2, atol=1e-2
                )
                async_group3 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind_async_group3(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    async_group3, expected, rtol=1e-2, atol=1e-2
                )
                group_stage = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage, expected, rtol=1e-2, atol=1e-2
                )
                group_stage_64_desc_reuse = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_64_desc_reuse,
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                )
                group_stage_64_fixed_shape = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_64_fixed_shape,
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                )
                group_stage_48 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_48, expected, rtol=1e-2, atol=1e-2
                )
                group_stage_48_nm12 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_48_nm12, expected, rtol=1e-2, atol=1e-2
                )
                group_stage_48_nm12_compact = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_48_nm12_compact,
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                )
                group_stage_48_nm12_merge2 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_48_nm12_merge2,
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                )
                group_stage_48_nm12_desc_reuse = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_desc_reuse(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_48_nm12_desc_reuse,
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                )
                group_stage_48_nm12_tma_metadata = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_48_nm12_tma_metadata,
                    expected,
                    rtol=1e-2,
                    atol=1e-2,
                )
                group_stage_80_nm12 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_80_nm12, expected, rtol=1e-2, atol=1e-2
                )
                group_stage_96_nm12 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_96_nm12, expected, rtol=1e-2, atol=1e-2
                )
                group_stage_64_nm12 = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    group_stage_64_nm12, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_reg_realloc_matches_reference(self):
        torch.manual_seed(113)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_reg_realloc(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_stage4_matches_reference(self):
        torch.manual_seed(111)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage4(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_lane_ready_stage5_matches_reference(self):
        torch.manual_seed(112)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, (1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 63, 64, 65, 129, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_stage5(
                        activation, packed
                    )
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-2, atol=1e-2
                )

    def test_wgmma_tma_block128x32_stage3_matches_reference(self):
        torch.manual_seed(108)
        layout = HybridBlockSparseLayout(128, 32, 1, 2)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 73, 128, 513, 1024):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = hybrid_block_sparse_gemm_wgmma_tma_block128x32_stage3(
                    activation, packed
                )
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_wgmma_tma_block128x64_matches_reference(self):
        torch.manual_seed(105)
        layout = HybridBlockSparseLayout(128, 64, 1, 2)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 73, 128):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = hybrid_block_sparse_gemm_wgmma_tma_block128x64(
                    activation, packed
                )
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_wgmma_tma_block128x32_output128x128_matches_reference(self):
        torch.manual_seed(107)
        layout = HybridBlockSparseLayout(128, 32, 1, 2)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 73, 128, 129, 256):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 256, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = (
                    hybrid_block_sparse_gemm_wgmma_tma_block128x32_output128x128(
                        activation, packed
                    )
                )
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_wgmma_tma_block128x128_matches_reference(self):
        torch.manual_seed(106)
        layout = HybridBlockSparseLayout(128, 128, 1, 2)
        weight = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        for m in (1, 64, 73, 128):
            with self.subTest(m=m):
                activation = torch.randn(
                    m, 512, device="cuda", dtype=torch.bfloat16
                )
                expected = hybrid_block_sparse_gemm_ref(activation, packed)
                actual = hybrid_block_sparse_gemm_wgmma_tma_block128x128(
                    activation, packed
                )
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_matches_reference_for_one_of_two_blocks(self):
        torch.manual_seed(101)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(73, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_naive(activation, packed)

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_matches_reference_for_two_of_four_blocks(self):
        torch.manual_seed(202)
        layout = HybridBlockSparseLayout(64, 64, 2, 4)
        weight = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(0, 2))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(9, 64, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        returned = hybrid_block_sparse_gemm_naive(activation, packed, out=out)

        self.assertIs(returned, out)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

        group_stage = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage, expected, rtol=1e-2, atol=1e-2
        )
        group_stage_48 = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_48, expected, rtol=1e-2, atol=1e-2
        )
        group_stage_48_nm12 = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_48_nm12, expected, rtol=1e-2, atol=1e-2
        )
        group_stage_48_nm12_compact = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_compact(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_48_nm12_compact,
            expected,
            rtol=1e-2,
            atol=1e-2,
        )
        group_stage_48_nm12_tma_metadata = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_tma_metadata(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_48_nm12_tma_metadata,
            expected,
            rtol=1e-2,
            atol=1e-2,
        )
        group_stage_80_nm12 = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output80x64_nm12_fastpath(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_80_nm12, expected, rtol=1e-2, atol=1e-2
        )
        group_stage_96_nm12 = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_96_nm12, expected, rtol=1e-2, atol=1e-2
        )
        group_stage_64_nm12 = (
            hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath(
                activation, packed
            )
        )
        torch.testing.assert_close(
            group_stage_64_nm12, expected, rtol=1e-2, atol=1e-2
        )
        tuned_fallback = hybrid_block_sparse_gemm_wgmma_tuned(
            activation, packed
        )
        torch.testing.assert_close(
            tuned_fallback, expected, rtol=1e-2, atol=1e-2
        )

    def test_rejects_non_64_block_layout(self):
        layout = HybridBlockSparseLayout(16, 16, 1, 2)
        weight = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        mask = make_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        with self.assertRaisesRegex(ValueError, "block_h=block_w=64"):
            hybrid_block_sparse_gemm_naive(
                torch.randn(1, 32, device="cuda", dtype=torch.bfloat16), packed
            )

    def test_grouped_contiguous_matches_reference_and_zeros_padding(self):
        torch.manual_seed(303)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(3, 128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_grouped_mask(weight, layout, sparse_block_ids=(1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        grouped_layout = torch.tensor([3, 64, 66], device="cuda", dtype=torch.int32)
        out = torch.empty(128, 128, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_grouped_contiguous_ref(
            activation, packed, grouped_layout, m_alignment=64
        )
        returned = hybrid_block_sparse_grouped_contiguous_naive(
            activation, packed, grouped_layout, m_alignment=64, out=out
        )

        self.assertIs(returned, out)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(torch.count_nonzero(out[3:64]).item(), 0)
        self.assertEqual(torch.count_nonzero(out[66:]).item(), 0)

        fused_out = hybrid_block_sparse_grouped_contiguous_wgmma_tma(
            activation, packed, grouped_layout, m_alignment=64
        )
        torch.testing.assert_close(
            fused_out, expected, rtol=1e-2, atol=1e-2
        )
        self.assertEqual(torch.count_nonzero(fused_out[3:64]).item(), 0)
        self.assertEqual(torch.count_nonzero(fused_out[66:]).item(), 0)

    def test_grouped_contiguous_output128_matches_reference(self):
        torch.manual_seed(304)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(
            3, 128, 256, device="cuda", dtype=torch.bfloat16
        )
        mask = make_grouped_mask(weight, layout, sparse_block_ids=(1,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(
            384, 256, device="cuda", dtype=torch.bfloat16
        )
        grouped_layout = torch.tensor(
            [3, 129, 258], device="cuda", dtype=torch.int32
        )

        expected = hybrid_block_sparse_grouped_contiguous_ref(
            activation, packed, grouped_layout, m_alignment=128
        )
        actual = hybrid_block_sparse_grouped_contiguous_wgmma_tma(
            activation, packed, grouped_layout, m_alignment=128
        )

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(torch.count_nonzero(actual[3:128]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[129:256]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[258:]).item(), 0)

    def test_grouped_masked_matches_reference_and_zeros_tail(self):
        torch.manual_seed(404)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(3, 128, 256, device="cuda", dtype=torch.bfloat16)
        mask = make_grouped_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(3, 9, 256, device="cuda", dtype=torch.bfloat16)
        masked_m = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)

        expected = hybrid_block_sparse_grouped_masked_ref(
            activation, packed, masked_m
        )
        actual = hybrid_block_sparse_grouped_masked_naive(
            activation, packed, masked_m
        )

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[1, 4:]).item(), 0)

    def test_grouped_masked_wgmma_tma_matches_reference_and_zeros_tail(self):
        torch.manual_seed(405)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(3, 128, 256, device="cuda", dtype=torch.bfloat16)
        masks = torch.stack(
            [
                make_mask(weight[0], layout, (0,)),
                make_mask(weight[1], layout, (1,)),
                make_mask(weight[2], layout, (0,)),
            ]
        )
        packed = dense_to_hybrid_block_sparse(weight, masks, layout)
        activation = torch.randn(3, 64, 256, device="cuda", dtype=torch.bfloat16)
        masked_m = torch.tensor([0, 17, 64], device="cuda", dtype=torch.int32)

        expected = hybrid_block_sparse_grouped_masked_ref(
            activation, packed, masked_m
        )
        actual = hybrid_block_sparse_grouped_masked_wgmma_tma(
            activation, packed, masked_m
        )

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[1, 17:]).item(), 0)

    def test_grouped_masked_wgmma_tma_output128_matches_reference(self):
        torch.manual_seed(406)
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(
            3, 128, 256, device="cuda", dtype=torch.bfloat16
        )
        masks = torch.stack(
            [
                make_mask(weight[0], layout, (0,)),
                make_mask(weight[1], layout, (1,)),
                make_mask(weight[2], layout, (0,)),
            ]
        )
        packed = dense_to_hybrid_block_sparse(weight, masks, layout)
        activation = torch.randn(
            3, 128, 256, device="cuda", dtype=torch.bfloat16
        )
        masked_m = torch.tensor(
            [0, 65, 128], device="cuda", dtype=torch.int32
        )

        expected = hybrid_block_sparse_grouped_masked_ref(
            activation, packed, masked_m
        )
        actual = hybrid_block_sparse_grouped_masked_wgmma_tma(
            activation, packed, masked_m
        )

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[1, 65:]).item(), 0)

    def test_grouped_wgmma_tma_requires_aligned_m_layout(self):
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        mask = make_grouped_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            hybrid_block_sparse_grouped_contiguous_wgmma_tma(
                torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
                packed,
                torch.tensor([1], device="cuda", dtype=torch.int32),
                m_alignment=32,
            )
        with self.assertRaisesRegex(ValueError, "capacity must be divisible by 64"):
            hybrid_block_sparse_grouped_masked_wgmma_tma(
                torch.randn(1, 9, 128, device="cuda", dtype=torch.bfloat16),
                packed,
                torch.tensor([1], device="cuda", dtype=torch.int32),
            )

    def test_grouped_kernel_requires_int32_index(self):
        layout = HybridBlockSparseLayout(64, 64, 1, 2)
        weight = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        mask = make_grouped_mask(weight, layout, sparse_block_ids=(0,))
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        with self.assertRaisesRegex(TypeError, "dtype torch.int32"):
            hybrid_block_sparse_grouped_masked_naive(
                torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16),
                packed,
                torch.ones(1, device="cuda", dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
