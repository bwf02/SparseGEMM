import itertools
import unittest

import torch

from moe_batch_baselines import (
    dense_to_fixed_csr,
    prune_2_of_8,
    slide_activation_2_of_8,
    slide_weight_2_of_8,
)


class FormatTest(unittest.TestCase):
    def test_every_two_zero_pattern_preserves_dot_product(self):
        activation = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 8)
        for zeros in itertools.combinations(range(8), 2):
            weight = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 8)
            weight[..., list(zeros)] = 0
            expected = torch.bmm(activation, weight.transpose(1, 2))
            actual = torch.bmm(
                slide_activation_2_of_8(activation),
                slide_weight_2_of_8(weight).transpose(1, 2),
            )
            torch.testing.assert_close(actual, expected)

    def test_pruning_and_csr_density(self):
        weight = prune_2_of_8(torch.randn(2, 4, 16))
        rows, values, offsets, columns, nonzeros = dense_to_fixed_csr(weight)
        self.assertEqual(nonzeros, 48)
        self.assertEqual(tuple(rows.shape), (2, 4))
        self.assertEqual(tuple(values.shape), (2, 48))
        self.assertEqual(tuple(offsets.shape), (2, 5))
        self.assertEqual(columns.dtype, torch.int16)


if __name__ == "__main__":
    unittest.main()
