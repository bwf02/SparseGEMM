"""Compare fused hybrid sparse kernels with the separate-kernel baseline."""

import argparse

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_direct,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready,
    hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


SEPARATE_KERNEL_NAMES = (
    "hybrid_sparse_dense_wgmma_tma",
    "hybrid_sparse_2_4_wgmma_tma_metadata_prefetch",
    "hybrid_sparse_reduce_wgmma_tma",
)
FUSED_DIRECT_KERNEL_NAME = "hybrid_sparse_fused_wgmma_tma_direct"
FUSED_STSM_KERNEL_NAME = "hybrid_sparse_fused_wgmma_tma_stsm"
PERSISTENT_KERNEL_NAME = "hybrid_sparse_fused_wgmma_tma_stsm_persistent"
LANE_READY_KERNEL_NAME = (
    "hybrid_sparse_fused_wgmma_tma_stsm_persistent_lane_ready"
)
CUBLASLT_KERNEL_NAMES = ("nvjet", "gemv", "gemm")
STANDARD_M = (128, 256, 512, 1024, 2048, 4096)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


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
    separate_out = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    fused_out = torch.empty_like(separate_out)
    stsm_out = torch.empty_like(separate_out)
    persistent_out = torch.empty_like(separate_out)
    lane_ready_out = torch.empty_like(separate_out)
    deepgemm_out = torch.empty_like(separate_out)
    cublaslt_out = torch.empty_like(separate_out)

    hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
        activation, packed_weight, out=separate_out
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_direct(
        activation, packed_weight, out=fused_out
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm(
        activation, packed_weight, out=stsm_out
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent(
        activation, packed_weight, out=persistent_out
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
        activation, packed_weight, out=lane_ready_out
    )
    deep_gemm.bf16_gemm_nt(activation, dense_weight, deepgemm_out)
    deep_gemm.cublaslt_gemm_nt(
        activation, dense_weight, cublaslt_out, c=None
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_out, separate_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(stsm_out, fused_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(persistent_out, stsm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lane_ready_out, persistent_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fused_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        cublaslt_out, deepgemm_out, rtol=2e-2, atol=2e-2
    )

    separate_times = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_metadata_prefetch(
            activation, packed_weight, out=separate_out
        ),
        SEPARATE_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    fused_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_direct(
            activation, packed_weight, out=fused_out
        ),
        FUSED_DIRECT_KERNEL_NAME,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    stsm_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm(
            activation, packed_weight, out=stsm_out
        ),
        FUSED_STSM_KERNEL_NAME,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    persistent_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent(
            activation, packed_weight, out=persistent_out
        ),
        PERSISTENT_KERNEL_NAME,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    lane_ready_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready(
            activation, packed_weight, out=lane_ready_out
        ),
        LANE_READY_KERNEL_NAME,
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
    cublaslt_times = bench_kineto(
        lambda: deep_gemm.cublaslt_gemm_nt(
            activation, dense_weight, cublaslt_out, c=None
        ),
        CUBLASLT_KERNEL_NAMES,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    separate_total = sum(separate_times)
    cublaslt_time = sum(cublaslt_times)
    print(
        f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
        f"{separate_total * 1e6:11.2f} {fused_time * 1e6:15.2f} "
        f"{stsm_time * 1e6:14.2f} {persistent_time * 1e6:14.2f} "
        f"{lane_ready_time * 1e6:14.2f} "
        f"{deepgemm_time * 1e6:11.2f} {cublaslt_time * 1e6:10.2f} "
        f"{safe_divide(fused_time, stsm_time):10.3f}x "
        f"{safe_divide(deepgemm_time, lane_ready_time):8.3f}x "
        f"{safe_divide(cublaslt_time, lane_ready_time):8.3f}x"
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
        "     M      N      K | separate(us) fused-direct(us) fused-stsm(us) "
        "persistent(us) lane-ready(us) deepgemm(us) cublas(us) "
        "direct/stsm dg/lane-ready cublas/lane-ready"
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
