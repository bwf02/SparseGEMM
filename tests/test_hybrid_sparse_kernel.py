import unittest

import torch

from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_naive,
    hybrid_block_sparse_gemm_tensorcore,
    hybrid_block_sparse_gemm_ref,
    hybrid_block_sparse_grouped_contiguous_naive,
    hybrid_block_sparse_grouped_contiguous_ref,
    hybrid_block_sparse_grouped_masked_naive,
    hybrid_block_sparse_grouped_masked_ref,
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

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

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
            sparse_block = mask[
                block_row * 64 : (block_row + 1) * 64, :64
            ].reshape(64, 16, 4)
            sparse_block[:] = True
            pair_ids = torch.randint(0, len(pairs), (64, 16), device="cuda")
            keep = pairs[pair_ids]
            sparse_block.scatter_(2, keep, False)
        packed = dense_to_hybrid_block_sparse(weight, mask, layout)
        activation = torch.randn(65, 256, device="cuda", dtype=torch.bfloat16)

        expected = hybrid_block_sparse_gemm_ref(activation, packed)
        actual = hybrid_block_sparse_gemm_tensorcore(activation, packed)

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
