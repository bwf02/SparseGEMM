"""Benchmark TP1 MoE projection shapes across model-level token batches."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/moe_batch"))

import deep_gemm  # noqa: E402
from moe_batch_baselines import (  # noqa: E402
    SlideSparseBatch,
    SputnikBatch,
    prune_2_of_8,
    slide_activation_2_of_8,
    slide_weight_2_of_8,
)
from sparse_gemm.hybrid_sparse import (  # noqa: E402
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_grouped_masked_wgmma_tma,
)


@dataclass(frozen=True)
class ModelSpec:
    experts: int
    top_k: int
    hidden: int
    expert_intermediate: int
    config_url: str


MODEL_SPECS = {
    "qwen15": ModelSpec(
        60, 4, 2048, 1408,
        "https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json",
    ),
    "deepseek_v2_lite": ModelSpec(
        64, 6, 2048, 1408,
        "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json",
    ),
    "qwen3_30b": ModelSpec(
        128, 8, 2048, 768,
        "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json",
    ),
    "mixtral_8x7b": ModelSpec(
        8, 2, 4096, 14336,
        "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json",
    ),
    "llama4_scout": ModelSpec(
        16, 1, 5120, 8192,
        "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json",
    ),
}

DEFAULT_MODELS = ["qwen15", "deepseek_v2_lite", "qwen3_30b", "llama4_scout"]

PROJECTIONS = {
    "gate_up": lambda spec: (2 * spec.expert_intermediate, spec.hidden),
    "down": lambda spec: (spec.hidden, spec.expert_intermediate),
}


def balanced_counts(tokens: int, spec: ModelSpec, device: str) -> torch.Tensor:
    """Distribute the B * top-k routed rows evenly across experts."""
    routed_rows = tokens * spec.top_k
    base, remainder = divmod(routed_rows, spec.experts)
    counts = torch.full((spec.experts,), base, device=device, dtype=torch.int32)
    if remainder:
        counts[:remainder] += 1
    return counts


def make_hybrid_mask(weight: torch.Tensor, layout: HybridBlockSparseLayout) -> torch.Tensor:
    mask = torch.zeros_like(weight, dtype=torch.bool)
    block_rows = weight.shape[-2] // layout.block_h
    block_columns = weight.shape[-1] // layout.block_w
    for block_row in range(block_rows):
        row_start = block_row * layout.block_h
        for group_start in range(0, block_columns, layout.block_m):
            for local_block in range(layout.block_n):
                column_start = (group_start + local_block) * layout.block_w
                block = mask[
                    :, row_start : row_start + layout.block_h,
                    column_start : column_start + layout.block_w,
                ].reshape(weight.shape[0], layout.block_h, -1, 4)
                block[..., 2:] = True
    return mask


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


def test_count(tokens: int, args: argparse.Namespace) -> tuple[int, int]:
    return args.warmup, args.iterations


def append_result(
    writer, model: str, projection: str, backend: str, tokens: int,
    spec: ModelSpec, n: int, k: int, counts: torch.Tensor, capacity: int,
    latency_us: float, scheduler: str, dtype: str, computed_m: int | None = None,
) -> None:
    valid_m = int(counts.sum().item())
    logical_flops = 2 * valid_m * n * k
    writer.writerow({
        "model": model,
        "projection": projection,
        "backend": backend,
        "token_bs": tokens,
        "num_experts": spec.experts,
        "active_experts": int(counts.ne(0).sum().item()),
        "top_k": spec.top_k,
        "valid_expert_m": valid_m,
        "computed_expert_m": valid_m if computed_m is None else computed_m,
        "max_expert_m": int(counts.max().item()),
        "capacity_m": capacity,
        "N": n,
        "K": k,
        "dtype": dtype,
        "scheduler": scheduler,
        "latency_us": f"{latency_us:.3f}",
        "effective_tflops": f"{logical_flops / (latency_us * 1e-6) / 1e12:.3f}",
    })


def run_external_baselines(
    model: str, projection: str, spec: ModelSpec, n: int, k: int,
    args: argparse.Namespace, writer,
) -> None:
    torch.manual_seed(args.seed)
    weight = prune_2_of_8(torch.randn(
        spec.experts, n, k, device="cuda", dtype=torch.float16
    ))
    slided_weight = slide_weight_2_of_8(weight)
    for tokens in args.batch_sizes:
        counts = balanced_counts(tokens, spec, "cuda")
        max_expert_m = int(counts.max().item())
        capacity = max(8, math.ceil(max_expert_m / 8) * 8)
        activation = torch.randn(
            spec.experts, capacity, k, device="cuda", dtype=torch.float16
        )
        warmup, iterations = test_count(tokens, args)

        if "cusparselt" in args.external_backends:
            slidesparse = SlideSparseBatch(slided_weight, capacity)
            slided_activation = slide_activation_2_of_8(activation)
            if tokens == args.batch_sizes[0]:
                reference = torch.bmm(activation, weight.transpose(1, 2))
                torch.testing.assert_close(
                    slidesparse(slided_activation), reference, rtol=2e-2, atol=2e-2
                )
                del reference
            slide_us = time_cuda(
                lambda: slidesparse(slided_activation), warmup, iterations
            )
            append_result(
                writer, model, projection, "slidesparse_cusparselt", tokens,
                spec, n, k, counts, capacity, slide_us, "strided_batch", "fp16",
                computed_m=spec.experts * capacity,
            )
            slidesparse.close()
            del slidesparse, slided_activation

        if "sputnik" in args.external_backends:
            sputnik = SputnikBatch(weight, capacity)
            prepared = sputnik.prepare_activation(activation)
            if tokens == args.batch_sizes[0]:
                reference = torch.bmm(activation, weight.transpose(1, 2))
                torch.testing.assert_close(
                    sputnik.run_prepared(prepared), reference, rtol=2e-2, atol=2e-2
                )
                del reference
            sputnik_us = time_cuda(
                lambda: sputnik.run_prepared(prepared), warmup, iterations
            )
            append_result(
                writer, model, projection, "sputnik_batch", tokens,
                spec, n, k, counts, capacity, sputnik_us, "grid_z_batch", "fp16",
                computed_m=spec.experts * capacity,
            )
            del sputnik, prepared
        del activation
        torch.cuda.empty_cache()
    del weight, slided_weight
    torch.cuda.empty_cache()


def run_native_baselines(
    model: str, projection: str, spec: ModelSpec, n: int, k: int,
    args: argparse.Namespace, writer,
) -> None:
    torch.manual_seed(args.seed)
    layout = HybridBlockSparseLayout(64, 64, 1, 2)
    source = torch.randn(spec.experts, n, k, device="cuda", dtype=torch.bfloat16)
    packed = dense_to_hybrid_block_sparse(source, make_hybrid_mask(source, layout), layout)
    dense_weight = packed.to_dense().contiguous()
    del source
    for tokens in args.batch_sizes:
        counts = balanced_counts(tokens, spec, "cuda")
        expected_m = int(counts.max().item())
        capacity_alignment = 64 if expected_m <= 64 else 128
        capacity = max(64, math.ceil(expected_m / capacity_alignment) * capacity_alignment)
        activation = torch.randn(
            spec.experts, capacity, k, device="cuda", dtype=torch.bfloat16
        )
        sparse_out = torch.empty(
            spec.experts, capacity, n, device="cuda", dtype=torch.bfloat16
        )
        deepgemm_out = torch.empty_like(sparse_out)
        request_prebind = expected_m <= args.prebind_max_m
        sparse_fn = lambda: hybrid_block_sparse_grouped_masked_wgmma_tma(
            activation, packed, counts, out=sparse_out, expected_m=expected_m,
            use_active_expert_prebind=request_prebind,
        )
        deepgemm_fn = lambda: deep_gemm.m_grouped_bf16_gemm_nt_masked(
            activation, dense_weight, deepgemm_out, counts, expected_m
        )
        sparse_fn()
        deepgemm_fn()
        torch.cuda.synchronize()
        for expert, rows in enumerate(counts.tolist()):
            if rows:
                torch.testing.assert_close(
                    sparse_out[expert, :rows], deepgemm_out[expert, :rows],
                    rtol=2e-2, atol=2e-2,
                )
        warmup, iterations = test_count(tokens, args)
        sparse_us = time_cuda(sparse_fn, warmup, iterations)
        deepgemm_us = time_cuda(deepgemm_fn, warmup, iterations)
        prebind_eligible = (
            request_prebind and capacity == 64 and expected_m <= 64
        )
        append_result(
            writer, model, projection, "sparse_gemm", tokens, spec, n, k,
            counts, capacity, sparse_us,
            "active_expert_prebind" if prebind_eligible else "default_dispatch",
            "bf16",
        )
        append_result(
            writer, model, projection, "deepgemm_grouped", tokens, spec, n, k,
            counts, capacity, deepgemm_us, "masked_grouped", "bf16",
        )
        del activation, sparse_out, deepgemm_out
        torch.cuda.empty_cache()
    del packed, dense_weight
    torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    fields = [
        "model", "projection", "backend", "token_bs", "num_experts",
        "active_experts", "top_k",
        "valid_expert_m", "computed_expert_m", "max_expert_m", "capacity_m",
        "N", "K", "dtype",
        "scheduler", "latency_us", "effective_tflops",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in args.models:
            spec = MODEL_SPECS[model]
            for projection in args.projections:
                n, k = PROJECTIONS[projection](spec)
                print(f"Running {model} {projection}: E={spec.experts}, N={n}, K={k}")
                if not args.native_only:
                    run_external_baselines(model, projection, spec, n, k, args, writer)
                    handle.flush()
                if not args.external_only:
                    run_native_baselines(model, projection, spec, n, k, args, writer)
                    handle.flush()
    print(f"Results: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=DEFAULT_MODELS)
    parser.add_argument("--projections", nargs="+", choices=PROJECTIONS, default=list(PROJECTIONS))
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[32, 64, 128, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--prebind-max-m", type=int, default=64)
    parser.add_argument(
        "--external-backends", nargs="+", choices=("cusparselt", "sputnik"),
        default=["cusparselt"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="/tmp/moe_tp1_kernel_baselines.csv")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--external-only", action="store_true")
    args = parser.parse_args()
    if args.native_only and args.external_only:
        parser.error("--native-only and --external-only are mutually exclusive")
    if min(args.batch_sizes) <= 0:
        parser.error("batch sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
