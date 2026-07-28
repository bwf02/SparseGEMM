"""Benchmark the current-best hybrid sparse kernel against DeepGEMM."""

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
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


STANDARD_M = (128, 256, 512, 1024, 2048, 4096)
HYBRID_KERNEL = "hybrid_sparse_output128x64_stage_kind"
DEEPGEMM_KERNEL = "bf16_gemm"


def benchmark_shape(
    shape: Shape,
    repeats: int,
    num_tests: int,
    flush_l2: bool,
) -> dict:
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
    hybrid_output = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    deepgemm_output = torch.empty_like(hybrid_output)

    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
        activation, packed, out=hybrid_output
    )
    deep_gemm.bf16_gemm_nt(activation, dense_weight, deepgemm_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        hybrid_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )

    hybrid_times = []
    deepgemm_times = []
    for repeat in range(repeats):
        measurements = [
            (
                "hybrid",
                lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
                    activation, packed, out=hybrid_output
                ),
                HYBRID_KERNEL,
            ),
            (
                "deepgemm",
                lambda: deep_gemm.bf16_gemm_nt(
                    activation, dense_weight, deepgemm_output
                ),
                DEEPGEMM_KERNEL,
            ),
        ]
        if repeat % 2:
            measurements.reverse()
        current = {}
        for name, function, kernel_name in measurements:
            current[name] = bench_kineto(
                function,
                kernel_name,
                num_tests=num_tests,
                suppress_kineto_output=True,
                flush_l2=flush_l2,
            )
        hybrid_times.append(current["hybrid"] * 1e6)
        deepgemm_times.append(current["deepgemm"] * 1e6)

    speedups = [
        deepgemm / hybrid
        for hybrid, deepgemm in zip(hybrid_times, deepgemm_times)
    ]
    return {
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "hybrid_us": hybrid_times,
        "deepgemm_us": deepgemm_times,
        "speedup": speedups,
        "hybrid_median_us": statistics.median(hybrid_times),
        "deepgemm_median_us": statistics.median(deepgemm_times),
        "speedup_median": statistics.median(speedups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=list(STANDARD_M))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.num_tests <= 0:
        parser.error("--repeats and --num-tests must be positive")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "     M      N      K | hybrid median (us) deepgemm median (us) "
        "median speedup | per-run speedups"
    )
    results = []
    for shape in qwen_moe_shapes(args.m):
        result = benchmark_shape(
            shape, args.repeats, args.num_tests, args.flush_l2
        )
        results.append(result)
        run_text = ", ".join(f"{value:.3f}x" for value in result["speedup"])
        print(
            f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
            f"{result['hybrid_median_us']:18.2f} "
            f"{result['deepgemm_median_us']:20.2f} "
            f"{result['speedup_median']:14.3f}x | {run_text}",
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
                    "flush_l2": args.flush_l2,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
