"""Compare TMA metadata with a producer-warp vector copy."""

import argparse

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


STANDARD_M = (128, 256, 512, 1024, 2048, 4096)
TMA_METADATA_KERNEL = (
    "hybrid_sparse_fused_wgmma_tma_stsm_persistent_lane_ready_tma_metadata"
)
PRODUCER_COPY_KERNEL = (
    "hybrid_sparse_fused_wgmma_tma_stsm_persistent_lane_ready_producer_metadata_copy"
)
CUBLASLT_KERNEL_NAMES = ("nvjet", "gemv", "gemm")


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def benchmark_shape(shape: Shape, num_tests: int, flush_l2: bool) -> None:
    torch.manual_seed(0)
    layout = HybridBlockSparseLayout(64, 64, 1, 2)
    activation = torch.randn(
        shape.m, shape.k, device="cuda", dtype=torch.bfloat16
    )
    source_weight = torch.randn(
        shape.n, shape.k, device="cuda", dtype=torch.bfloat16
    )
    mask = make_hybrid_mask(source_weight, layout)
    packed = dense_to_hybrid_block_sparse(source_weight, mask, layout)
    dense_weight = packed.to_dense().contiguous()
    tma_metadata_out = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    producer_copy_out = torch.empty_like(tma_metadata_out)
    deepgemm_out = torch.empty_like(tma_metadata_out)
    cublas_out = torch.empty_like(tma_metadata_out)

    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata(
        activation, packed, out=tma_metadata_out
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
        activation, packed, out=producer_copy_out
    )
    deep_gemm.bf16_gemm_nt(activation, dense_weight, deepgemm_out)
    deep_gemm.cublaslt_gemm_nt(activation, dense_weight, cublas_out, c=None)
    torch.cuda.synchronize()
    torch.testing.assert_close(producer_copy_out, tma_metadata_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tma_metadata_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(cublas_out, deepgemm_out, rtol=2e-2, atol=2e-2)

    tma_metadata_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_tma_metadata(
            activation, packed, out=tma_metadata_out
        ),
        TMA_METADATA_KERNEL,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    producer_copy_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
            activation, packed, out=producer_copy_out
        ),
        PRODUCER_COPY_KERNEL,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    deepgemm_time = bench_kineto(
        lambda: deep_gemm.bf16_gemm_nt(
            activation, dense_weight, deepgemm_out
        ),
        "bf16_gemm",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    cublas_time = sum(
        bench_kineto(
            lambda: deep_gemm.cublaslt_gemm_nt(
                activation, dense_weight, cublas_out, c=None
            ),
            CUBLASLT_KERNEL_NAMES,
            num_tests=num_tests,
            suppress_kineto_output=True,
            flush_l2=flush_l2,
        )
    )
    print(
        f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
        f"{tma_metadata_time * 1e6:12.2f} {producer_copy_time * 1e6:13.2f} "
        f"{deepgemm_time * 1e6:10.2f} {cublas_time * 1e6:10.2f} "
        f"{safe_ratio(tma_metadata_time, producer_copy_time):13.3f}x "
        f"{safe_ratio(deepgemm_time, producer_copy_time):10.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=list(STANDARD_M))
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--no-flush-l2", action="store_true")
    args = parser.parse_args()
    if (args.n is None) != (args.k is None):
        parser.error("--n and --k must be provided together")
    if args.n is not None and len(args.m) != 1:
        parser.error("a custom --n/--k shape requires exactly one --m")

    shapes = (
        [Shape(args.m[0], args.n, args.k)]
        if args.n is not None
        else qwen_moe_shapes(args.m)
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "     M      N      K | tma_meta(us) producer(us) deepgemm(us) "
        "cublas(us) tma/producer dg/producer"
    )
    for shape in shapes:
        benchmark_shape(shape, args.num_tests, not args.no_flush_l2)


if __name__ == "__main__":
    main()
