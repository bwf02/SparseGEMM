"""Small-batch ablation for active-expert prebinding and selector dispatch."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_gemm.hybrid_sparse import (  # noqa: E402
    HybridBlockSparseLayout,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_grouped_masked_wgmma_tma,
)


SHAPES = {
    "gate_up": (2816, 2048),
    "down": (2048, 1408),
}
NUM_EXPERTS = 60
TOP_K = 4


def balanced_counts(tokens: int) -> torch.Tensor:
    routed_rows = tokens * TOP_K
    base, remainder = divmod(routed_rows, NUM_EXPERTS)
    counts = torch.full(
        (NUM_EXPERTS,), base, device="cuda", dtype=torch.int32
    )
    if remainder:
        counts[:remainder] += 1
    return counts


def make_alternating_mask(
    weight: torch.Tensor, layout: HybridBlockSparseLayout
) -> torch.Tensor:
    """Exercise both legal 1:2 selectors without changing the packed volume."""
    mask = torch.zeros_like(weight, dtype=torch.bool)
    block_rows = weight.shape[-2] // layout.block_h
    block_columns = weight.shape[-1] // layout.block_w
    for block_row in range(block_rows):
        row_start = block_row * layout.block_h
        for group_id, group_start in enumerate(
            range(0, block_columns, layout.block_m)
        ):
            local_block = (block_row + group_id) & 1
            column_start = (group_start + local_block) * layout.block_w
            block = mask[
                :, row_start : row_start + layout.block_h,
                column_start : column_start + layout.block_w,
            ].reshape(weight.shape[0], layout.block_h, -1, 4)
            block[..., 2:] = True
    return mask


def time_sample(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def build_case(projection: str, tokens: int, seed: int):
    torch.manual_seed(seed)
    n, k = SHAPES[projection]
    layout = HybridBlockSparseLayout(64, 64, 1, 2)
    weight = torch.randn(
        NUM_EXPERTS, n, k, device="cuda", dtype=torch.bfloat16
    )
    mask = make_alternating_mask(weight, layout)
    packed = dense_to_hybrid_block_sparse(weight, mask, layout)
    del weight, mask
    counts = balanced_counts(tokens)
    activation = torch.randn(
        NUM_EXPERTS, 64, k, device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty(
        NUM_EXPERTS, 64, n, device="cuda", dtype=torch.bfloat16
    )
    return activation, packed, counts, output


def make_fn(activation, packed, counts, output, prebind: bool, bitmask: bool):
    expected_m = max(1, int(counts.max().item()))
    return lambda: hybrid_block_sparse_grouped_masked_wgmma_tma(
        activation,
        packed,
        counts,
        out=output,
        expected_m=expected_m,
        use_active_expert_prebind=prebind,
        use_bitmask_selector_fast_path=bitmask,
    )


def validate_variants(activation, packed, counts, output) -> None:
    reference = torch.empty_like(output)
    baseline = make_fn(
        activation, packed, counts, reference, prebind=False, bitmask=False
    )
    baseline()
    for prebind in (False, True):
        for bitmask in (False, True):
            candidate = make_fn(
                activation, packed, counts, output, prebind, bitmask
            )
            candidate()
            torch.cuda.synchronize()
            for expert, rows in enumerate(counts.tolist()):
                if rows:
                    torch.testing.assert_close(
                        output[expert, :rows],
                        reference[expert, :rows],
                        rtol=0,
                        atol=0,
                    )


def run_profile(args: argparse.Namespace) -> None:
    activation, packed, counts, output = build_case(
        args.profile_projection, args.profile_bs, args.seed
    )
    fn = make_fn(
        activation,
        packed,
        counts,
        output,
        bool(args.profile_prebind),
        bool(args.profile_bitmask),
    )
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()


def run_benchmark(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "projection",
        "token_bs",
        "routed_rows",
        "active_experts",
        "max_expert_m",
        "N",
        "K",
        "prebind",
        "bitmask_selector",
        "repeat",
        "inner_iterations",
        "latency_us",
        "seed",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for projection in args.projections:
            n, k = SHAPES[projection]
            for tokens in args.batch_sizes:
                activation, packed, counts, output = build_case(
                    projection, tokens, args.seed
                )
                validate_variants(activation, packed, counts, output)
                variants = {
                    (prebind, bitmask): make_fn(
                        activation,
                        packed,
                        counts,
                        output,
                        prebind,
                        bitmask,
                    )
                    for prebind in (False, True)
                    for bitmask in (False, True)
                }
                for fn in variants.values():
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                order = list(variants)
                rng = random.Random(args.seed + tokens + n)
                for repeat in range(args.repeats):
                    rng.shuffle(order)
                    for prebind, bitmask in order:
                        latency_us = time_sample(
                            variants[(prebind, bitmask)], args.iterations
                        )
                        writer.writerow(
                            {
                                "projection": projection,
                                "token_bs": tokens,
                                "routed_rows": int(counts.sum().item()),
                                "active_experts": int(counts.ne(0).sum().item()),
                                "max_expert_m": int(counts.max().item()),
                                "N": n,
                                "K": k,
                                "prebind": int(prebind),
                                "bitmask_selector": int(bitmask),
                                "repeat": repeat,
                                "inner_iterations": args.iterations,
                                "latency_us": f"{latency_us:.6f}",
                                "seed": args.seed,
                            }
                        )
                        handle.flush()
                del activation, packed, counts, output, variants
                torch.cuda.empty_cache()
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/prebind_selector_ablation_raw.csv")
    parser.add_argument("--projections", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-projection", choices=SHAPES, default="gate_up")
    parser.add_argument("--profile-bs", type=int, default=8)
    parser.add_argument("--profile-prebind", type=int, choices=(0, 1), default=0)
    parser.add_argument("--profile-bitmask", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.profile_only:
        run_profile(parsed)
    else:
        run_benchmark(parsed)
