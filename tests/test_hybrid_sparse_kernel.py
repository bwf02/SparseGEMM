import unittest

import torch

from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_naive,
    hybrid_block_sparse_gemm_ref,
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestHybridSparseNaiveKernel(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
