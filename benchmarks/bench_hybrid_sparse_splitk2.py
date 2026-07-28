"""Benchmark split-K2 fused reduction against output64 and DeepGEMM."""

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
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


BASELINE_KERNEL = "hybrid_sparse_group_stage_output64x64"
SPLITK2_KERNEL = "hybrid_sparse_group_stage_output64x64_splitk2_fused_reduce"


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

    baseline_output = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    splitk2_output = torch.empty_like(baseline_output)
    deepgemm_output = torch.empty_like(baseline_output)
    partial = torch.empty(
        2, shape.m, shape.n, device="cuda", dtype=torch.float32
    )
    counters = torch.zeros(
        ((shape.m + 63) // 64) * ((shape.n + 63) // 64),
        device="cuda",
        dtype=torch.int32,
    )

    baseline_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64(
        activation, packed, out=baseline_output
    )
    splitk2_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_splitk2_fused_reduce(
        activation,
        packed,
        out=splitk2_output,
        partial=partial,
        tile_counters=counters,
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_output
    )

    baseline_call()
    splitk2_call()
    splitk2_call()
    deepgemm_call()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        baseline_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        splitk2_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )
    if torch.count_nonzero(counters).item() != 0:
        raise AssertionError("split-K2 tile counters were not reset")

    timings = {"baseline": [], "splitk2": [], "deepgemm": []}
    measurements = (
        ("baseline", baseline_call, BASELINE_KERNEL),
        ("splitk2", splitk2_call, SPLITK2_KERNEL),
        ("deepgemm", deepgemm_call, "bf16_gemm"),
    )
    for repeat in range(repeats):
        ordered = measurements if repeat % 2 == 0 else tuple(reversed(measurements))
        for name, function, kernel_name in ordered:
            timings[name].append(measure(function, kernel_name, num_tests))

    baseline_us = statistics.median(timings["baseline"])
    splitk2_us = statistics.median(timings["splitk2"])
    deepgemm_us = statistics.median(timings["deepgemm"])
    return {
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "baseline_us": timings["baseline"],
        "splitk2_us": timings["splitk2"],
        "deepgemm_us": timings["deepgemm"],
        "baseline_median_us": baseline_us,
        "splitk2_median_us": splitk2_us,
        "deepgemm_median_us": deepgemm_us,
        "splitk2_over_baseline": baseline_us / splitk2_us,
        "splitk2_speedup_over_deepgemm": deepgemm_us / splitk2_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--num-tests", type=int, default=300)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.num_tests <= 0:
        parser.error("--repeats and --num-tests must be positive")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "     M      N      K | baseline us split-K2 us DeepGEMM us "
        "split/base DG/split"
    )
    results = []
    for shape in qwen_moe_shapes(args.m):
        result = benchmark_shape(shape, args.repeats, args.num_tests)
        results.append(result)
        print(
            f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
            f"{result['baseline_median_us']:11.2f} "
            f"{result['splitk2_median_us']:12.2f} "
            f"{result['deepgemm_median_us']:11.2f} "
            f"{result['splitk2_over_baseline']:10.3f}x "
            f"{result['splitk2_speedup_over_deepgemm']:8.3f}x",
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
