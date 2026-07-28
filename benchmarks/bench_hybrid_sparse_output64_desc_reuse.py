"""Benchmark output64 GMMA descriptor reuse against prior kernels."""

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
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7,
    hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask


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
    packed = dense_to_hybrid_block_sparse(
        source_weight, make_hybrid_mask(source_weight, layout), layout
    )
    dense_weight = packed.to_dense().contiguous()

    generic_out = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    fast_out = torch.empty_like(generic_out)
    reuse_out = torch.empty_like(generic_out)
    fixed_shape_out = torch.empty_like(generic_out)
    stage7_out = torch.empty_like(generic_out)
    async_group3_out = torch.empty_like(generic_out)
    deepgemm_out = torch.empty_like(generic_out)
    generic_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64(
        activation, packed, out=generic_out
    )
    fast_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath(
        activation, packed, out=fast_out
    )
    reuse_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse(
        activation, packed, out=reuse_out
    )
    fixed_shape_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape(
        activation, packed, out=fixed_shape_out
    )
    stage7_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7(
        activation, packed, out=stage7_out
    )
    async_group3_call = lambda: hybrid_block_sparse_gemm_wgmma_tma_fused_stsm_persistent_lane_ready_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7_async_group3(
        activation, packed, out=async_group3_out
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_out
    )

    for function in (
        generic_call,
        fast_call,
        reuse_call,
        fixed_shape_call,
        stage7_call,
        async_group3_call,
        deepgemm_call,
    ):
        function()
    torch.cuda.synchronize()
    for actual in (
        generic_out,
        fast_out,
        reuse_out,
        fixed_shape_out,
        stage7_out,
        async_group3_out,
    ):
        torch.testing.assert_close(actual, deepgemm_out, rtol=2e-2, atol=2e-2)

    measurements = (
        ("generic", generic_call, "hybrid_sparse_group_stage_output64x64"),
        (
            "fastpath",
            fast_call,
            "hybrid_sparse_group_stage_output64x64_nm12_fastpath",
        ),
        (
            "desc_reuse",
            reuse_call,
            "hybrid_sparse_group_stage_output64x64_nm12_fastpath_desc_reuse",
        ),
        (
            "fixed_shape",
            fixed_shape_call,
            "hybrid_sparse_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape",
        ),
        (
            "stage7",
            stage7_call,
            "hybrid_sparse_group_stage_output64x64_nm12_fastpath_desc_reuse_fixed_shape_stage7",
        ),
        (
            "async_group3",
            async_group3_call,
            "hybrid_sparse_group_stage_output64x64",
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
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "timings_us": timings,
        "medians_us": medians,
        "reuse_over_generic": medians["generic"] / medians["desc_reuse"],
        "reuse_over_fastpath": medians["fastpath"] / medians["desc_reuse"],
        "speedup_over_deepgemm": medians["deepgemm"] / medians["desc_reuse"],
        "fixed_shape_over_reuse": medians["desc_reuse"] / medians["fixed_shape"],
        "fixed_shape_speedup_over_deepgemm": medians["deepgemm"] / medians["fixed_shape"],
        "stage7_over_fixed_shape": medians["fixed_shape"] / medians["stage7"],
        "stage7_speedup_over_deepgemm": medians["deepgemm"] / medians["stage7"],
        "async_group3_over_stage7": medians["stage7"] / medians["async_group3"],
        "async_group3_speedup_over_deepgemm": medians["deepgemm"] / medians["async_group3"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[128])
    parser.add_argument("--n", type=int, choices=(1408, 2048), default=2048)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--num-tests", type=int, default=300)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    k = 2048 if args.n == 1408 else 1408

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "     M      N      K | generic fastpath reuse fixed stage7 async3 "
        "DeepGEMM async3/stage7 DG/async3"
    )
    results = []
    for m in args.m:
        shape = Shape(m, args.n, k)
        result = benchmark_shape(shape, args.repeats, args.num_tests)
        results.append(result)
        medians = result["medians_us"]
        print(
            f"{m:6d} {args.n:6d} {k:6d} | "
            f"{medians['generic']:7.2f} {medians['fastpath']:8.2f} "
            f"{medians['desc_reuse']:5.2f} {medians['fixed_shape']:5.2f} "
            f"{medians['stage7']:6.2f} "
            f"{medians['async_group3']:6.2f} "
            f"{medians['deepgemm']:8.2f} "
            f"{result['async_group3_over_stage7']:13.3f}x "
            f"{result['async_group3_speedup_over_deepgemm']:9.3f}x",
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
