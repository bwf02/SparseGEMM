"""Benchmark output96 GMMA descriptor reuse for Qwen M=256."""

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
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse,
)

from bench_hybrid_sparse import make_hybrid_mask


def measure(function, kernel_name: str, num_tests: int) -> float:
    return bench_kineto(
        function,
        kernel_name,
        num_tests=num_tests,
        suppress_kineto_output=True,
        flush_l2=False,
    ) * 1e6


def benchmark(repeats: int, num_tests: int) -> dict:
    torch.manual_seed(0)
    m, n, k = 256, 1408, 2048
    layout = HybridBlockSparseLayout(64, 64, 1, 2)
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    source_weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    packed = dense_to_hybrid_block_sparse(
        source_weight, make_hybrid_mask(source_weight, layout), layout
    )
    dense_weight = packed.to_dense().contiguous()
    baseline_out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    reuse_out = torch.empty_like(baseline_out)
    output88_out = torch.empty_like(baseline_out)
    deepgemm_out = torch.empty_like(baseline_out)
    baseline_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath(
        activation, packed, out=baseline_out
    )
    reuse_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output96x64_nm12_fastpath_desc_reuse(
        activation, packed, out=reuse_out
    )
    output88_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output88x64_nm12_fastpath_desc_reuse(
        activation, packed, out=output88_out
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_out
    )

    for function in (baseline_call, reuse_call, output88_call, deepgemm_call):
        function()
    torch.cuda.synchronize()
    torch.testing.assert_close(baseline_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(reuse_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(output88_out, deepgemm_out, rtol=2e-2, atol=2e-2)

    measurements = (
        (
            "baseline",
            baseline_call,
            "hybrid_sparse_group_stage_output96x64_nm12_fastpath",
        ),
        (
            "desc_reuse",
            reuse_call,
            "hybrid_sparse_group_stage_output96x64_nm12_fastpath_desc_reuse",
        ),
        (
            "output88",
            output88_call,
            "hybrid_sparse_group_stage_output88x64_nm12_fastpath_desc_reuse",
        ),
        ("deepgemm", deepgemm_call, "bf16_gemm"),
    )
    timings = {name: [] for name, _, _ in measurements}
    for repeat in range(repeats):
        ordered = measurements if repeat % 2 == 0 else tuple(reversed(measurements))
        for name, function, kernel_name in ordered:
            timings[name].append(measure(function, kernel_name, num_tests))
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "m": m,
        "n": n,
        "k": k,
        "timings_us": timings,
        "medians_us": medians,
        "reuse_over_baseline": medians["baseline"] / medians["desc_reuse"],
        "output88_over_output96": medians["desc_reuse"] / medians["output88"],
        "speedup_over_deepgemm": medians["deepgemm"] / medians["output88"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--num-tests", type=int, default=300)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.repeats, args.num_tests)
    medians = result["medians_us"]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("     M      N      K | baseline output96 output88 DeepGEMM 88/96 DG/88")
    print(
        f"{result['m']:6d} {result['n']:6d} {result['k']:6d} | "
        f"{medians['baseline']:8.2f} {medians['desc_reuse']:8.2f} "
        f"{medians['output88']:8.2f} {medians['deepgemm']:8.2f} "
        f"{result['output88_over_output96']:5.3f}x "
        f"{result['speedup_over_deepgemm']:8.3f}x"
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "repeats": args.repeats,
                    "num_tests": args.num_tests,
                    "results": [result],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
