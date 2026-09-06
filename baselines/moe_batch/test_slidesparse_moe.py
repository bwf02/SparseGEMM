"""Remote GPU checks: python -m baselines.moe_batch.test_slidesparse_moe."""

import unittest

import torch

from .slidesparse_moe import SlideSparseProjection


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class SlideSparseServingTest(unittest.TestCase):
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
