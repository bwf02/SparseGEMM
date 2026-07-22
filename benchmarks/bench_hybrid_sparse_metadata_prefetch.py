"""Compare shared-memory metadata prefetch with the 64x64 TMA baseline."""

import argparse

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_wgmma_tma,
    hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


BASELINE_KERNEL_NAMES = (
    "hybrid_sparse_dense_wgmma_tma",
    "hybrid_sparse_2_4_wgmma_tma",
    "hybrid_sparse_reduce_wgmma_tma",
)
PREFETCH_KERNEL_NAMES = (
    "hybrid_sparse_dense_wgmma_tma",
    "hybrid_sparse_2_4_wgmma_tma_metadata_prefetch",
    "hybrid_sparse_reduce_wgmma_tma",
)
STANDARD_M = (128, 256, 512, 1024, 2048, 4096)


def benchmark_shape(
    shape: Shape,
    layout: HybridBlockSparseLayout,
    num_tests: int,
    flush_l2: bool,
) -> None:
    torch.manual_seed(0)
    activation = torch.randn(
        shape.m, shape.k, device="cuda", dtype=torch.bfloat16
    )
    source_weight = torch.randn(
        shape.n, shape.k, device="cuda", dtype=torch.bfloat16
    )
    mask = make_hybrid_mask(source_weight, layout)
    packed_weight = dense_to_hybrid_block_sparse(source_weight, mask, layout)
    dense_weight = packed_weight.to_dense().contiguous()

    baseline_out = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    prefetch_out = torch.empty_like(baseline_out)
    deepgemm_out = torch.empty_like(baseline_out)

    hybrid_block_sparse_gemm_wgmma_tma(
        activation, packed_weight, out=baseline_out
    )
    hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
        activation, packed_weight, out=prefetch_out
    )
    deep_gemm.bf16_gemm_nt(activation, dense_weight, deepgemm_out)
    torch.cuda.synchronize()
    torch.testing.assert_close(prefetch_out, baseline_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(prefetch_out, deepgemm_out, rtol=2e-2, atol=2e-2)

    baseline_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma(
            activation, packed_weight, out=baseline_out
        ),
        BASELINE_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    prefetch_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
            activation, packed_weight, out=prefetch_out
        ),
        PREFETCH_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    deepgemm_time = bench_kineto(
        lambda: deep_gemm.bf16_gemm_nt(activation, dense_weight, deepgemm_out),
        "bf16_gemm",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    baseline_total = sum(baseline_times)
    prefetch_total = sum(prefetch_times)
    print(
        f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
        f"{baseline_total * 1e6:9.2f} {prefetch_total * 1e6:12.2f} "
        f"{deepgemm_time * 1e6:11.2f} "
        f"{baseline_total / prefetch_total:9.3f}x "
        f"{deepgemm_time / prefetch_total:8.3f}x | "
        f"{prefetch_times[0] * 1e6:9.2f} "
        f"{prefetch_times[1] * 1e6:10.2f} "
        f"{prefetch_times[2] * 1e6:9.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=list(STANDARD_M))
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--block-n", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=2)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--no-flush-l2", action="store_true")
    args = parser.parse_args()
    if (args.n is None) != (args.k is None):
        parser.error("--n and --k must be provided together")
    if args.n is not None and len(args.m) != 1:
        parser.error("a custom --n/--k shape requires exactly one --m")
    if args.num_tests <= 0:
        parser.error("--num-tests must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    layout = HybridBlockSparseLayout(64, 64, args.block_n, args.block_m)
    shapes = (
        [Shape(args.m[0], args.n, args.k)]
        if args.n is not None
        else qwen_moe_shapes(args.m)
    )
    for shape in shapes:
        layout.validate_shape((shape.n, shape.k))

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Hybrid layout: 64x64, sparse blocks {layout.block_n}:"
        f"{layout.block_m}, element sparsity {layout.sparsity:.1%}"
    )
    print(
        "     M      N      K | baseline(us) prefetch(us) deepgemm(us) "
        "base/pref dg/pref | dense(us) sparse(us) reduce(us)"
    )
    for shape in shapes:
        benchmark_shape(
            shape,
            layout,
            num_tests=args.num_tests,
            flush_l2=not args.no_flush_l2,
        )


if __name__ == "__main__":
    main()
