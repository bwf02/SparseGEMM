"""Benchmark two outer K groups per WGMMA commit against the baseline."""

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
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2,
)

from bench_hybrid_sparse import make_hybrid_mask, qwen_moe_shapes


BASELINE_KERNEL = "hybrid_sparse_group_stage_output48x64_nm12_fastpath"
MERGE2_KERNEL = "hybrid_sparse_group_stage_output48x64_nm12_fastpath_merge2"


def measure(function, kernel_name: str, num_tests: int) -> float:
    return bench_kineto(
        function,
        kernel_name,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=False,
    ) * 1e6


def benchmark_shape(shape, repeats: int, num_tests: int) -> dict:
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
    merge2_output = torch.empty_like(baseline_output)
    deepgemm_output = torch.empty_like(baseline_output)

    baseline_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath(
        activation, packed, out=baseline_output
    )
    merge2_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output48x64_nm12_fastpath_merge2(
        activation, packed, out=merge2_output
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_output
    )
    baseline_call()
    merge2_call()
    deepgemm_call()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        merge2_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )

    timings = {"baseline": [], "merge2": [], "deepgemm": []}
    measurements = (
        ("baseline", baseline_call, BASELINE_KERNEL),
        ("merge2", merge2_call, MERGE2_KERNEL),
        ("deepgemm", deepgemm_call, "bf16_gemm"),
    )
    for repeat in range(repeats):
        ordered = measurements if repeat % 2 == 0 else tuple(reversed(measurements))
        for name, function, kernel_name in ordered:
            timings[name].append(measure(function, kernel_name, num_tests))

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    return {
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        **{f"{name}_us": values for name, values in timings.items()},
        **{f"{name}_median_us": value for name, value in medians.items()},
        "speedup_over_deepgemm": statistics.median(
            deepgemm / merge2
            for merge2, deepgemm in zip(
                timings["merge2"], timings["deepgemm"]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-tests", type=int, default=200)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    results = []
    print("     M      N      K | baseline merge2 DeepGEMM speedup")
    for shape in qwen_moe_shapes(args.m):
        result = benchmark_shape(shape, args.repeats, args.num_tests)
        results.append(result)
        print(
            f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
            f"{result['baseline_median_us']:8.2f} "
            f"{result['merge2_median_us']:6.2f} "
            f"{result['deepgemm_median_us']:8.2f} "
            f"{result['speedup_over_deepgemm']:7.3f}x",
            flush=True,
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
