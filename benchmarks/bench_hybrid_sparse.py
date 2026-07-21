"""Benchmark versioned hybrid sparse GEMM kernels against DeepGEMM."""

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_naive,
    hybrid_block_sparse_gemm_tensorcore,
    hybrid_block_sparse_gemm_wgmma_sync,
    hybrid_block_sparse_gemm_wgmma_tma,
)


HYBRID_KERNEL_NAMES = (
    "hybrid_sparse_dense_naive",
    "hybrid_sparse_2_4_naive",
    "hybrid_sparse_reduce_naive",
)
TENSORCORE_KERNEL_NAMES = (
    "hybrid_sparse_dense_tensorcore",
    "hybrid_sparse_2_4_tensorcore",
    "hybrid_sparse_reduce_tensorcore",
)
WGMMA_SYNC_KERNEL_NAMES = (
    "hybrid_sparse_dense_wgmma_sync",
    "hybrid_sparse_2_4_wgmma_sync",
    "hybrid_sparse_reduce_wgmma_sync",
)
WGMMA_TMA_KERNEL_NAMES = (
    "hybrid_sparse_dense_wgmma_tma",
    "hybrid_sparse_2_4_wgmma_tma",
    "hybrid_sparse_reduce_wgmma_tma",
)


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int


def make_hybrid_mask(weight: torch.Tensor, layout: HybridBlockSparseLayout) -> torch.Tensor:
    """Select the first block_n blocks in each block group for 2:4 sparsity."""
    mask = torch.zeros_like(weight, dtype=torch.bool)
    block_rows = weight.shape[0] // layout.block_h
    block_columns = weight.shape[1] // layout.block_w
    for block_row in range(block_rows):
        row_start = block_row * layout.block_h
        for group_start in range(0, block_columns, layout.block_m):
            for local_block in range(layout.block_n):
                column_start = (group_start + local_block) * layout.block_w
                block = mask[
                    row_start : row_start + layout.block_h,
                    column_start : column_start + layout.block_w,
                ].reshape(layout.block_h, -1, 4)
                block[..., 2:] = True
    return mask


def qwen_moe_shapes(ms: Iterable[int]) -> list[Shape]:
    # Qwen1.5-MoE routed expert gate/up and down projections.
    return [
        Shape(m, n, k)
        for m in ms
        for n, k in ((1408, 2048), (2048, 1408))
    ]


def benchmark_shape(
    shape: Shape,
    layout: HybridBlockSparseLayout,
    num_tests: int,
    flush_l2: bool,
) -> None:
    torch.manual_seed(0)
    a = torch.randn(shape.m, shape.k, device="cuda", dtype=torch.bfloat16)
    source_weight = torch.randn(
        shape.n, shape.k, device="cuda", dtype=torch.bfloat16
    )
    mask = make_hybrid_mask(source_weight, layout)
    packed_weight = dense_to_hybrid_block_sparse(source_weight, mask, layout)
    dense_weight = packed_weight.to_dense().contiguous()

    hybrid_out = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    tensorcore_out = torch.empty_like(hybrid_out)
    wgmma_sync_out = torch.empty_like(hybrid_out)
    wgmma_tma_out = torch.empty_like(hybrid_out)
    deepgemm_out = torch.empty_like(hybrid_out)

    hybrid_block_sparse_gemm_naive(a, packed_weight, out=hybrid_out)
    hybrid_block_sparse_gemm_tensorcore(a, packed_weight, out=tensorcore_out)
    hybrid_block_sparse_gemm_wgmma_sync(a, packed_weight, out=wgmma_sync_out)
    hybrid_block_sparse_gemm_wgmma_tma(a, packed_weight, out=wgmma_tma_out)
    deep_gemm.bf16_gemm_nt(a, dense_weight, deepgemm_out)
    torch.cuda.synchronize()
    torch.testing.assert_close(hybrid_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tensorcore_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(wgmma_sync_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(wgmma_tma_out, deepgemm_out, rtol=2e-2, atol=2e-2)

    hybrid_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_naive(a, packed_weight, out=hybrid_out),
        HYBRID_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    tensorcore_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_tensorcore(
            a, packed_weight, out=tensorcore_out
        ),
        TENSORCORE_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    wgmma_sync_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_sync(
            a, packed_weight, out=wgmma_sync_out
        ),
        WGMMA_SYNC_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    wgmma_tma_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma(
            a, packed_weight, out=wgmma_tma_out
        ),
        WGMMA_TMA_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    deepgemm_time = bench_kineto(
        lambda: deep_gemm.bf16_gemm_nt(a, dense_weight, deepgemm_out),
        "bf16_gemm",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )

    hybrid_time = sum(hybrid_times)
    tensorcore_time = sum(tensorcore_times)
    wgmma_sync_time = sum(wgmma_sync_times)
    wgmma_tma_time = sum(wgmma_tma_times)
    dense_flops = 2 * shape.m * shape.n * shape.k
    executed_flops = dense_flops * (1.0 - layout.sparsity)
    sync_to_tma = wgmma_sync_time / wgmma_tma_time
    dense_speedup = deepgemm_time / wgmma_tma_time

    print(
        f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
        f"{hybrid_time * 1e6:9.1f} "
        f"{tensorcore_time * 1e6:7.1f} "
        f"{wgmma_sync_time * 1e6:8.1f} "
        f"{wgmma_tma_time * 1e6:7.1f} "
        f"{deepgemm_time * 1e6:11.1f} "
        f"{sync_to_tma:8.3f}x {dense_speedup:7.3f}x | "
        f"{dense_flops / wgmma_tma_time / 1e12:9.2f} "
        f"{executed_flops / wgmma_tma_time / 1e12:10.2f} | "
        f"{wgmma_tma_times[0] * 1e6:8.1f} "
        f"{wgmma_tma_times[1] * 1e6:9.1f} "
        f"{wgmma_tma_times[2] * 1e6:8.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument(
        "--n", type=int, help="output dimension; requires exactly one --m and --k"
    )
    parser.add_argument(
        "--k", type=int, help="reduction dimension; requires exactly one --m and --n"
    )
    parser.add_argument("--block-n", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=2)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument(
        "--no-flush-l2",
        action="store_true",
        help="disable the repository-standard L2 flush between profiler iterations",
    )
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
        f"Hybrid layout: 64x64, sparse blocks {layout.block_n}:{layout.block_m}, "
        f"element sparsity {layout.sparsity:.1%}"
    )
    print("Timing: CUDA kernel time from bench_kineto; packing is excluded")
    print(
        "     M      N      K | naive(us) mma(us) sync(us) tma(us) deepgemm(us) "
        "sync/tma dg/tma | tma-eff-TF tma-exec-TF | dense(us) sparse(us) reduce(us)"
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
