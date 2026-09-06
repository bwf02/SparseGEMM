"""Remote GPU checks: python -m baselines.moe_batch.test_slidesparse_moe."""

import os
import unittest

import torch

from .slidesparse_moe import SlideSparseProjection


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class SlideSparseServingTest(unittest.TestCase):
    def test_chunked_expansion_with_tail_and_graph(self):
        from unittest.mock import patch

        weight = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"SLIDESPARSE_ACTIVATION_CHUNK_M": "32"}):
            projection = SlideSparseProjection(weight, offload_source=True)
        try:
            x = torch.randn(4, 70, 128, device="cuda", dtype=torch.bfloat16)
            expected_weight = projection.weight.cuda().float().transpose(1, 2)
            expected = torch.bmm(x.float(), expected_weight).bfloat16()
            torch.testing.assert_close(projection(x), expected, rtol=2e-2, atol=5e-2)
            self.assertEqual(set(projection.plans), {16, 32, 256})
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = projection(x)
            x.normal_()
            graph.replay()
            expected = torch.bmm(x.float(), expected_weight).bfloat16()
            torch.testing.assert_close(output, expected, rtol=2e-2, atol=5e-2)
            del graph
        finally:
            projection.close()

    def test_projection_and_graph_replay(self):
        torch.manual_seed(1234)
        weight = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16)
        projection = SlideSparseProjection(weight)
        self.assertTrue((projection.weight.reshape(4, 256, -1, 8).eq(0).sum(-1) >= 2).all().item())
        try:
            for m in (16, 64, 128):
                with self.subTest(m=m):
                    x = torch.randn(4, m, 128, device="cuda", dtype=torch.bfloat16)
                    expected = torch.bmm(x.float(), projection.weight.float().transpose(1, 2)).bfloat16()
                    torch.testing.assert_close(projection(x), expected, rtol=2e-2, atol=5e-2)
                    self.assertEqual(projection.plans[m].compressed.data_ptr(), projection.compressed.data_ptr())
                    self.assertEqual(projection.plans[m].workspace.data_ptr(), projection.workspace.data_ptr())
                    released = projection(x)
                    released.set_(torch.empty(0, device="cuda", dtype=released.dtype))
                    torch.testing.assert_close(projection(x), expected, rtol=2e-2, atol=5e-2)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            projection(x)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        y = projection(x)
                    for _ in range(2):
                        x.normal_()
                        graph.replay()
                        expected = torch.bmm(x.float(), projection.weight.float().transpose(1, 2)).bfloat16()
                        torch.testing.assert_close(y, expected, rtol=2e-2, atol=5e-2)
                    del graph
        finally:
            projection.close()


if __name__ == "__main__":
    unittest.main()
