import unittest

import torch

from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    HybridBlockSparseWeight,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_ref,
    hybrid_block_sparse_grouped_contiguous_ref,
    hybrid_block_sparse_grouped_masked_ref,
)


def make_valid_mask(weight, layout, sparse_block_ids=None):
    mask = torch.zeros_like(weight, dtype=torch.bool)
    sparse_block_ids = (
        tuple(range(layout.block_n))
        if sparse_block_ids is None
        else tuple(sparse_block_ids)
    )
    flat_mask = mask.reshape(-1, weight.shape[-2], weight.shape[-1])
    block_columns = weight.shape[-1] // layout.block_w
    for batch in range(flat_mask.shape[0]):
        for block_row in range(weight.shape[-2] // layout.block_h):
            row_start = block_row * layout.block_h
            row_end = row_start + layout.block_h
            for group_start in range(0, block_columns, layout.block_m):
                for local_block in sparse_block_ids:
                    column_start = (group_start + local_block) * layout.block_w
                    block = flat_mask[
                        batch,
                        row_start:row_end,
                        column_start : column_start + layout.block_w,
                    ].reshape(layout.block_h, -1, 4)
                    block[..., 2:] = True
    return mask


class TestHybridSparseFormat(unittest.TestCase):
    def test_round_trip_and_compact_streams(self):
        layout = HybridBlockSparseLayout(2, 4, 1, 2)
        weight = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        mask = make_valid_mask(weight, layout)

        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        self.assertEqual(packed.block_selector.tolist(), [[1]])
        self.assertEqual(tuple(packed.dense_values.shape), (1, 1, 1, 2, 4))
        self.assertEqual(tuple(packed.sparse_values.shape), (1, 1, 1, 2, 2))
        self.assertEqual(tuple(packed.sparse_metadata.shape), (1, 1, 1, 2, 1))
        self.assertTrue(torch.equal(packed.to_dense(), weight.masked_fill(mask, 0)))

    def test_round_trip_multiple_experts_and_outer_nm(self):
        layout = HybridBlockSparseLayout(2, 4, 2, 4)
        weight = torch.randn(3, 4, 32, generator=torch.Generator().manual_seed(7))
        mask = make_valid_mask(weight, layout, sparse_block_ids=(0, 2))

        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        self.assertEqual(tuple(packed.block_selector.shape), (3, 2, 2))
        self.assertTrue(torch.equal(packed.to_dense(), weight.masked_fill(mask, 0)))

    def test_all_two_of_four_metadata_codes(self):
        layout = HybridBlockSparseLayout(1, 24, 1, 1)
        weight = torch.arange(1, 25, dtype=torch.float32).reshape(1, 24)
        mask = torch.ones_like(weight, dtype=torch.bool).reshape(1, 6, 4)
        for quartet, pair in enumerate(
            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ):
            mask[0, quartet, pair[0]] = False
            mask[0, quartet, pair[1]] = False
        mask = mask.reshape_as(weight)

        packed = dense_to_hybrid_block_sparse(weight, mask, layout)

        self.assertEqual(packed.sparse_metadata.flatten().tolist(), list(range(6)))
        self.assertTrue(torch.equal(packed.to_dense(), weight.masked_fill(mask, 0)))

    def test_lane_ready_metadata_matches_wgmma_encoding(self):
        layout = HybridBlockSparseLayout(64, 64, 1, 1)
        weight = torch.randn(64, 64, generator=torch.Generator().manual_seed(8))
        mask = torch.ones_like(weight, dtype=torch.bool).reshape(64, 16, 4)
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for row in range(64):
            for quartet in range(16):
                first, second = pairs[(row + quartet) % len(pairs)]
                mask[row, quartet, first] = False
                mask[row, quartet, second] = False
        packed = dense_to_hybrid_block_sparse(weight, mask.reshape_as(weight), layout)

        self.assertIsNotNone(packed.hardware_metadata)
        self.assertEqual(tuple(packed.hardware_metadata.shape), (1, 1, 1, 2, 4, 16))
        nibbles = (0x4, 0x8, 0xC, 0x9, 0xD, 0xE)
        expected = 0
        for quartet in range(4):
            lower_code = int(packed.sparse_metadata[0, 0, 0, 0, quartet])
            upper_code = int(packed.sparse_metadata[0, 0, 0, 8, quartet])
            expected |= nibbles[lower_code] << (quartet * 4)
            expected |= nibbles[upper_code] << ((quartet + 4) * 4)
        actual = int(packed.hardware_metadata[0, 0, 0, 0, 0, 0]) & 0xFFFFFFFF
        self.assertEqual(actual, expected)

    def test_rejects_invalid_mask_and_layout(self):
        layout = HybridBlockSparseLayout(2, 4, 1, 2)
        weight = torch.randn(2, 8)
        with self.assertRaisesRegex(ValueError, "select exactly"):
            dense_to_hybrid_block_sparse(
                weight, torch.zeros_like(weight, dtype=torch.bool), layout
            )
        with self.assertRaisesRegex(ValueError, "exactly 2:4"):
            invalid_mask = make_valid_mask(weight, layout)
            invalid_mask[0, 2] = False
            dense_to_hybrid_block_sparse(weight, invalid_mask, layout)
        with self.assertRaisesRegex(ValueError, "divisible by block_h"):
            dense_to_hybrid_block_sparse(
                torch.randn(3, 8), torch.zeros(3, 8, dtype=torch.bool), layout
            )

    def test_rejects_corrupt_metadata(self):
        layout = HybridBlockSparseLayout(2, 4, 1, 2)
        weight = torch.randn(2, 8)
        packed = dense_to_hybrid_block_sparse(
            weight, make_valid_mask(weight, layout), layout
        )
        corrupt_metadata = packed.sparse_metadata.clone()
        corrupt_metadata.fill_(6)
        corrupt = HybridBlockSparseWeight(
            packed.original_shape,
            packed.layout,
            packed.block_selector,
            packed.dense_values,
            packed.sparse_values,
            corrupt_metadata,
        )
        with self.assertRaisesRegex(ValueError, "invalid 2:4 pair code"):
            corrupt.to_dense()


class TestHybridSparseReference(unittest.TestCase):
    def setUp(self):
        self.layout = HybridBlockSparseLayout(2, 4, 1, 2)

    def pack(self, weight):
        mask = make_valid_mask(weight, self.layout)
        return dense_to_hybrid_block_sparse(weight, mask, self.layout), mask

    def test_gemm_and_accumulation(self):
        a = torch.randn(5, 8, generator=torch.Generator().manual_seed(1))
        weight = torch.randn(4, 8, generator=torch.Generator().manual_seed(2))
        c = torch.randn(5, 4, generator=torch.Generator().manual_seed(3))
        packed, mask = self.pack(weight)
        dense_weight = weight.masked_fill(mask, 0)

        actual = hybrid_block_sparse_gemm_ref(a, packed)
        accumulated = hybrid_block_sparse_gemm_ref(
            a, packed, c=c, out_dtype=torch.float64
        )

        torch.testing.assert_close(actual, a.float() @ dense_weight.float().t())
        self.assertEqual(accumulated.dtype, torch.float64)
        torch.testing.assert_close(
            accumulated.float(), a.float() @ dense_weight.float().t() + c.float()
        )

    def test_grouped_contiguous_psum(self):
        a = torch.randn(12, 8, generator=torch.Generator().manual_seed(4))
        weight = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(5))
        packed, mask = self.pack(weight)
        dense_weight = weight.masked_fill(mask, 0)
        grouped_layout = torch.tensor([2, 4, 10], dtype=torch.int32)

        actual = hybrid_block_sparse_grouped_contiguous_ref(
            a, packed, grouped_layout, m_alignment=4
        )
        expected = torch.zeros(12, 4)
        expected[0:2] = a[0:2] @ dense_weight[0].t()
        expected[4:10] = a[4:10] @ dense_weight[2].t()

        torch.testing.assert_close(actual, expected)

    def test_grouped_masked(self):
        a = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(6))
        weight = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(7))
        packed, mask = self.pack(weight)
        dense_weight = weight.masked_fill(mask, 0)
        masked_m = torch.tensor([0, 3, 5], dtype=torch.int64)

        actual = hybrid_block_sparse_grouped_masked_ref(a, packed, masked_m)
        expected = torch.zeros(3, 5, 4)
        expected[1, :3] = a[1, :3] @ dense_weight[1].t()
        expected[2] = a[2] @ dense_weight[2].t()

        torch.testing.assert_close(actual, expected)

    def test_reference_validation(self):
        weight = torch.randn(3, 4, 8)
        packed, _ = self.pack(weight)
        with self.assertRaisesRegex(ValueError, "before group start"):
            hybrid_block_sparse_grouped_contiguous_ref(
                torch.randn(12, 8),
                packed,
                torch.tensor([3, 2, 8], dtype=torch.int32),
                m_alignment=4,
            )
        with self.assertRaisesRegex(ValueError, "final psum alignment"):
            hybrid_block_sparse_grouped_contiguous_ref(
                torch.randn(9, 8),
                packed,
                torch.tensor([2, 6, 9], dtype=torch.int32),
                m_alignment=4,
            )
        with self.assertRaisesRegex(ValueError, "must be in"):
            hybrid_block_sparse_grouped_masked_ref(
                torch.randn(3, 5, 8),
                packed,
                torch.tensor([0, 6, 5], dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
