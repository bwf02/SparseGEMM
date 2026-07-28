"""Benchmark the tuned hybrid sparse dispatch against DeepGEMM BF16."""

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
    hybrid_block_sparse_gemm_wgmma_tuned,
)

from bench_hybrid_sparse import Shape, make_hybrid_mask, qwen_moe_shapes


STANDARD_M = (128, 256, 512, 1024)
TUNED_KERNEL_NAMES = {
    (128, 1408, 2048): "hybrid_sparse_group_stage_output48x64_nm12_fastpath",
    (256, 1408, 2048): "hybrid_sparse_group_stage_output88x64_nm12_fastpath_desc_reuse",
    (512, 1408, 2048): "hybrid_sparse_group_stage_output80x64_nm12_fastpath_desc_reuse",
    (1024, 1408, 2048): "hybrid_sparse_group_stage_output80x64_nm12_fastpath_desc_reuse",
    (128, 2048, 1408): "hybrid_sparse_group_stage_output64x64",
    (256, 2048, 1408): "hybrid_sparse_group_stage_output128x64_nm12_fastpath_desc_reuse",
    (512, 2048, 1408): "hybrid_sparse_group_stage_output128x64_nm12_fastpath_desc_reuse",
    (1024, 2048, 1408): "hybrid_sparse_group_stage_output96x64_nm12_fastpath",
}


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
    tuned_output = torch.empty(
        shape.m, shape.n, device="cuda", dtype=torch.bfloat16
    )
    deepgemm_output = torch.empty_like(tuned_output)

    tuned_call = lambda: hybrid_block_sparse_gemm_wgmma_tuned(
        activation, packed, out=tuned_output
    )
    deepgemm_call = lambda: deep_gemm.bf16_gemm_nt(
        activation, dense_weight, deepgemm_output
    )
    tuned_call()
    deepgemm_call()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        tuned_output, deepgemm_output, rtol=2e-2, atol=2e-2
    )

    tuned_us = []
    deepgemm_us = []
    tuned_kernel_name = TUNED_KERNEL_NAMES[(shape.m, shape.n, shape.k)]
    for repeat in range(repeats):
        measurements = (
            (tuned_call, tuned_kernel_name, tuned_us),
            (deepgemm_call, "bf16_gemm", deepgemm_us),
        )
        if repeat % 2:
            measurements = tuple(reversed(measurements))
        for function, kernel_name, values in measurements:
            values.append(measure(function, kernel_name, num_tests))

    speedups = [
        deepgemm / tuned
        for tuned, deepgemm in zip(tuned_us, deepgemm_us)
    ]
    return {
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "tuned_us": tuned_us,
        "deepgemm_us": deepgemm_us,
        "tuned_median_us": statistics.median(tuned_us),
        "deepgemm_median_us": statistics.median(deepgemm_us),
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
        "     M      N      K | tuned median (us) "
        "DeepGEMM median (us) speedup"
    )
    results = []
    for shape in qwen_moe_shapes(args.m):
        result = benchmark_shape(shape, args.repeats, args.num_tests)
        results.append(result)
        print(
            f"{shape.m:6d} {shape.n:6d} {shape.k:6d} | "
            f"{result['tuned_median_us']:17.2f} "
            f"{result['deepgemm_median_us']:20.2f} "
            f"{result['speedup_median']:7.3f}x",
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
