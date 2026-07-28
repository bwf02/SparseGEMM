"""Benchmark the 32x64 small-M kernel against the current kernel and DeepGEMM."""

import argparse
import json
import statistics
from pathlib import Path

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


STANDARD_M = (128, 256, 512, 1024)
SMALL_M_KERNEL = "hybrid_sparse_output32x64_stage_kind"
MERGED_KERNEL = "hybrid_sparse_output32x64_stage_kind_merge_k3_stage6"
WIDE_KERNEL = "hybrid_sparse_output64x64_stage_kind"
CURRENT_KERNEL = "hybrid_sparse_output128x64_stage_kind"


def measure(function, kernel_name: str, num_tests: int) -> float:
    return bench_kineto(
        function,
        kernel_name,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=False,
    ) * 1e6


def benchmark_shape(shape: Shape, repeats: int, num_tests: int) -> dict:
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
    small_output = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    current_output = torch.empty_like(small_output)
    merged_output = torch.empty_like(small_output)
    wide_output = torch.empty_like(small_output)
    deepgemm_output = torch.empty_like(small_output)

    small_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind(
        activation, packed, out=small_output
    )
    current_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
        activation, packed, out=current_output
    )
    merged_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output32x64_stage_kind_merge_k3_stage6(
        activation, packed, out=merged_output
    )
    wide_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output64x64_stage_kind(
        activation, packed, out=wide_output
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_output
    )

    small_call()
    merged_call()
    wide_call()
    current_call()
    deepgemm_call()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        small_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        current_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        merged_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        wide_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )

    timings = {
        "small": [],
        "merged": [],
        "wide": [],
        "current": [],
        "deepgemm": [],
    }
    measurements = (
        ("small", small_call, SMALL_M_KERNEL),
        ("merged", merged_call, MERGED_KERNEL),
        ("wide", wide_call, WIDE_KERNEL),
        ("current", current_call, CURRENT_KERNEL),
        ("deepgemm", deepgemm_call, "bf16_gemm"),
    )
    for repeat in range(repeats):
        ordered = measurements if repeat % 2 == 0 else tuple(reversed(measurements))
        for name, function, kernel_name in ordered:
            timings[name].append(measure(function, kernel_name, num_tests))

    speedups = [
        deepgemm / wide
        for wide, deepgemm in zip(timings["wide"], timings["deepgemm"])
    ]
    return {
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "small_us": timings["small"],
        "merged_us": timings["merged"],
        "wide_us": timings["wide"],
        "current_us": timings["current"],
        "deepgemm_us": timings["deepgemm"],
        "small_median_us": statistics.median(timings["small"]),
        "merged_median_us": statistics.median(timings["merged"]),
        "wide_median_us": statistics.median(timings["wide"]),
        "current_median_us": statistics.median(timings["current"]),
        "deepgemm_median_us": statistics.median(timings["deepgemm"]),
        "speedup": speedups,
        "speedup_median": statistics.median(speedups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=list(STANDARD_M))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.num_tests <= 0:
        parser.error("--repeats and --num-tests must be positive")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "     M      N      K | small32 median (us) merged median (us) wide64 median (us) current median (us) "
        "deepgemm median (us) speedup | per-run speedups"
    )
    results = []
    for shape in qwen_moe_shapes(args.m):
        result = benchmark_shape(shape, args.repeats, args.num_tests)
        results.append(result)
        run_text = ", ".join(f"{value:.3f}x" for value in result["speedup"])
        print(
            f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
            f"{result['small_median_us']:19.2f} "
            f"{result['merged_median_us']:18.2f} "
            f"{result['wide_median_us']:18.2f} "
            f"{result['current_median_us']:19.2f} "
            f"{result['deepgemm_median_us']:20.2f} "
            f"{result['speedup_median']:7.3f}x | {run_text}",
            flush=True,
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "repeats": args.repeats,
                    "num_tests": args.num_tests,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
