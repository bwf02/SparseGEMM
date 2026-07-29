"""Benchmark output128 stage-kind GMMA descriptor reuse for Qwen M=256."""

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
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_output_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_async_group2,
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
    m, n, k = 256, 2048, 1408
    layout = HybridBlockSparseLayout(64, 64, 1, 2)
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    source_weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    packed = dense_to_hybrid_block_sparse(
        source_weight, make_hybrid_mask(source_weight, layout), layout
    )
    dense_weight = packed.to_dense().contiguous()
    baseline_out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    reuse_out = torch.empty_like(baseline_out)
    compact_out = torch.empty_like(baseline_out)
    group_stage_out = torch.empty_like(baseline_out)
    fused_mma_group_out = torch.empty_like(baseline_out)
    fixed_shape_unroll_k_out = torch.empty_like(baseline_out)
    stage5_output_reuse_out = torch.empty_like(baseline_out)
    stage5_async_group2_out = torch.empty_like(baseline_out)
    deepgemm_out = torch.empty_like(baseline_out)
    baseline_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind(
        activation, packed, out=baseline_out
    )
    reuse_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse(
        activation, packed, out=reuse_out
    )
    compact_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_producer_metadata_copy_output128x64_stage_kind_desc_reuse_compact(
        activation, packed, out=compact_out
    )
    group_stage_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse(
        activation, packed, out=group_stage_out
    )
    fused_mma_group_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group(
        activation, packed, out=fused_mma_group_out
    )
    fixed_shape_unroll_k_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k(
        activation, packed, out=fixed_shape_unroll_k_out
    )
    stage5_output_reuse_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_output_reuse(
        activation, packed, out=stage5_output_reuse_out
    )
    stage5_async_group2_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group_fixed_shape_unroll_k_stage5_async_group2(
        activation, packed, out=stage5_async_group2_out
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_out
    )

    for function in (
        baseline_call,
        reuse_call,
        compact_call,
        group_stage_call,
        fused_mma_group_call,
        fixed_shape_unroll_k_call,
        stage5_output_reuse_call,
        stage5_async_group2_call,
        deepgemm_call,
    ):
        function()
    torch.cuda.synchronize()
    torch.testing.assert_close(baseline_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(reuse_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(compact_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(group_stage_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fused_mma_group_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fixed_shape_unroll_k_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(stage5_output_reuse_out, deepgemm_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(stage5_async_group2_out, deepgemm_out, rtol=2e-2, atol=2e-2)

    measurements = (
        ("baseline", baseline_call, "hybrid_sparse_output128x64_stage_kind"),
        (
            "desc_reuse",
            reuse_call,
            "hybrid_sparse_output128x64_stage_kind_desc_reuse",
        ),
        (
            "compact",
            compact_call,
            "hybrid_sparse_output128x64_stage_kind_desc_reuse_compact",
        ),
        (
            "group_stage",
            group_stage_call,
            "hybrid_sparse_group_stage_output128x64_nm12_fastpath_desc_reuse",
        ),
        (
            "fused_mma_group",
            fused_mma_group_call,
            "hybrid_sparse_group_stage_output128x64_nm12_fastpath_desc_reuse_fused_mma_group",
        ),
        (
            "fixed_shape_unroll_k",
            fixed_shape_unroll_k_call,
            "hybrid_sparse_group_stage_output128x64",
        ),
        (
            "stage5_output_reuse",
            stage5_output_reuse_call,
            "hybrid_sparse_group_stage_output128x64",
        ),
        (
            "stage5_async_group2",
            stage5_async_group2_call,
            "hybrid_sparse_group_stage_output128x64",
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
        "compact_over_reuse": medians["desc_reuse"] / medians["compact"],
        "group_stage_over_reuse": medians["desc_reuse"] / medians["group_stage"],
        "speedup_over_deepgemm": medians["deepgemm"] / medians["group_stage"],
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
    print("     M      N      K | baseline reuse compact group DeepGEMM group/reuse DG/group")
    print(
        f"{result['m']:6d} {result['n']:6d} {result['k']:6d} | "
        f"{medians['baseline']:8.2f} {medians['desc_reuse']:5.2f} "
        f"{medians['compact']:7.2f} {medians['group_stage']:5.2f} "
        f"{medians['deepgemm']:8.2f} {result['group_stage_over_reuse']:11.3f}x "
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
