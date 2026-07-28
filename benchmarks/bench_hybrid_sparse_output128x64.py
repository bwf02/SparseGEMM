"""Compare 64x64 and 128x64 hybrid sparse output tiles."""

import argparse

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


STANDARD_M = (128, 256, 512, 1024, 2048, 4096)
KERNEL_64X64 = (
    "hybrid_sparse_fused_wgmma_tma_stsm_persistent_lane_ready_"
    "producer_metadata_copy"
)
KERNEL_128X64 = (
    "hybrid_sparse_fused_wgmma_tma_stsm_persistent_lane_ready_"
    "producer_metadata_copy_output128x64"
)
KERNEL_STAGE_KIND = "hybrid_sparse_output128x64_stage_kind"
KERNEL_STAGE_SELECTOR = "hybrid_sparse_output128x64_stage_selector"


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
    output_64x64 = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    output_128x64 = torch.empty_like(output_64x64)
    output_stage_kind = torch.empty_like(output_64x64)
    output_stage_selector = torch.empty_like(output_64x64)
    deepgemm_output = torch.empty_like(output_64x64)

    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
        activation, packed, out=output_64x64
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64(
        activation, packed, out=output_128x64
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
        activation, packed, out=output_stage_kind
    )
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector(
        activation, packed, out=output_stage_selector
    )
    deep_gemm.bf16_gemm_nt(activation, dense_weight, deepgemm_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output_128x64, output_64x64, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        output_stage_kind, output_128x64, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        output_stage_selector, output_128x64, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        output_128x64, deepgemm_output, rtol=2e-2, atol=2e-2
    )

    time_64x64 = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy(
            activation, packed, out=output_64x64
        ),
        KERNEL_64X64,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    time_128x64 = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64(
            activation, packed, out=output_128x64
        ),
        KERNEL_128X64,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    stage_kind_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
            activation, packed, out=output_stage_kind
        ),
        KERNEL_STAGE_KIND,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    stage_selector_time = bench_kineto(
        lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_selector(
            activation, packed, out=output_stage_selector
        ),
        KERNEL_STAGE_SELECTOR,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    deepgemm_time = bench_kineto(
        lambda: deep_gemm.bf16_gemm_nt(
            activation, dense_weight, deepgemm_output
        ),
        "bf16_gemm",
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=flush_l2,
    )
    print(
        f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
        f"{time_64x64 * 1e6:10.2f} {time_128x64 * 1e6:11.2f} "
        f"{stage_kind_time * 1e6:10.2f} "
        f"{stage_selector_time * 1e6:12.2f} "
        f"{deepgemm_time * 1e6:10.2f} "
        f"{safe_ratio(stage_kind_time, stage_selector_time):10.3f}x "
        f"{safe_ratio(deepgemm_time, stage_selector_time):10.3f}x"
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
        "     M      N      K | old64(us) new128(us) kind(us) selector(us) "
        "deepgemm(us) kind/selector dg/selector"
    )
    for shape in shapes:
        benchmark_shape(shape, args.num_tests, not args.no_flush_l2)


if __name__ == "__main__":
    main()
