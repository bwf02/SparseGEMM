"""Run on the benchmark GPU: python baselines/moe_batch/test_cublas_grouped.py."""

import unittest

import torch

from cublas_grouped import CublasGrouped


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class CublasGroupedTest(unittest.TestCase):
    def test_actual_m_and_empty_experts(self):
        torch.manual_seed(1234)
        for rows in ([0, 1, 7, 65], [0, 0, 0, 0], [128, 127, 2, 0]):
            with self.subTest(rows=rows):
                x = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16)
                y = torch.full((4, 128, 256), float("nan"),
                               device="cuda", dtype=torch.bfloat16)
                counts = torch.tensor(rows, device="cuda", dtype=torch.int32)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    fn = CublasGrouped(x, w, counts, y)
                    try:
                        fn()
                        stream.synchronize()
                        for expert, m in enumerate(rows):
                            if m:
                                reference = (x[expert, :m].float()
                                             @ w[expert].float().T).bfloat16()
                                torch.testing.assert_close(y[expert, :m], reference,
                                                           rtol=2e-2, atol=2e-2)
                            self.assertTrue(torch.isnan(y[expert, m:]).all().item())
                    finally:
                        fn.close()
                    with self.assertRaises(RuntimeError):
                        fn()


if __name__ == "__main__":
    unittest.main()
