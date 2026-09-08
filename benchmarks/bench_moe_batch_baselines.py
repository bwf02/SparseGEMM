"""Benchmark fixed-shape MoE batch kernels at 25% weight sparsity."""

import argparse
import sys
from pathlib import Path

import torch


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baselines/moe_batch"
sys.path.insert(0, str(BASELINE_DIR))

from moe_batch_baselines import (  # noqa: E402
    SlideSparseBatch,
    SputnikBatch,
    prune_2_of_8,
    slide_activation_2_of_8,
    slide_weight_2_of_8,
)


def time_cuda(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print(f"# correctness,{name},pass")


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dense_weight = torch.randn(
        args.experts, args.n, args.k, device="cuda", dtype=torch.float16
    )
    sparse_weight_fp16 = prune_2_of_8(dense_weight)
    sparse_weight_bf16 = sparse_weight_fp16.to(torch.bfloat16)
    slided_weight = slide_weight_2_of_8(sparse_weight_bf16)

    print("backend,experts,M,N,K,physical_K,sparsity,latency_us,dense_tflops")
    for m in args.m:
        activation_fp16 = torch.randn(
            args.experts, m, args.k, device="cuda", dtype=torch.float16
        )
        activation_bf16 = activation_fp16.to(torch.bfloat16)
        reference = torch.bmm(
            activation_bf16, sparse_weight_bf16.transpose(1, 2)
        )

        slidesparse = SlideSparseBatch(slided_weight, m)
        slided_activation = slide_activation_2_of_8(activation_bf16)
        check_close("slidesparse", slidesparse(slided_activation), reference)
        slide_us = time_cuda(
            lambda: slide_activation_2_of_8(activation_bf16), args.warmup, args.iterations
        )
        slidesparse_us = time_cuda(
            lambda: slidesparse(slided_activation), args.warmup, args.iterations
        )

        sputnik = SputnikBatch(sparse_weight_fp16, m)
        sputnik_activation = sputnik.prepare_activation(activation_fp16)
        sputnik_reference = torch.bmm(
            activation_fp16, sparse_weight_fp16.transpose(1, 2)
        )
        check_close("sputnik", sputnik.run_prepared(sputnik_activation), sputnik_reference)
        sputnik_us = time_cuda(
            lambda: sputnik.run_prepared(sputnik_activation), args.warmup, args.iterations
        )
        transpose_us = time_cuda(
            lambda: sputnik.prepare_activation(activation_fp16), args.warmup, args.iterations
        )

        logical_flops = 2 * args.experts * m * args.n * args.k
        for backend, physical_k, latency in (
            ("slidesparse_cusparselt", slided_weight.shape[-1], slidesparse_us),
            ("sputnik_batch", args.k, sputnik_us),
        ):
            tflops = logical_flops / (latency * 1e-6) / 1e12
            print(
                f"{backend},{args.experts},{m},{args.n},{args.k},{physical_k},"
                f"0.25,{latency:.3f},{tflops:.3f}"
            )
        print(f"slide_activation,{args.experts},{m},{args.n},{args.k},"
              f"{slided_weight.shape[-1]},0.25,{slide_us:.3f},0.000")
        print(f"sputnik_transpose,{args.experts},{m},{args.n},{args.k},"
              f"{args.k},0.25,{transpose_us:.3f},0.000")
        slidesparse.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--m", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--n", type=int, default=1408)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.experts <= 0 or min(args.m) <= 0:
        parser.error("--experts and every --m value must be positive")
    if args.k % 16 or args.n % 8 or any(m % 2 for m in args.m):
        parser.error("K must be divisible by 16, N by 8, and M by 2")
    return args


if __name__ == "__main__":
    run(parse_args())
