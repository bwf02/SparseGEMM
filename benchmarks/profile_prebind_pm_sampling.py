"""Prepare and profile one Qwen1.5-MoE kernel for NCU PM sampling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_gemm.hybrid_sparse import (  # noqa: E402
    HybridBlockSparseLayout,
    HybridBlockSparseWeight,
    hybrid_block_sparse_grouped_masked_wgmma_tma,
)

from bench_prebind_selector_ablation import build_case  # noqa: E402


def prepare(path: Path, token_bs: int, seed: int) -> None:
    activation, packed, counts, _ = build_case("gate_up", token_bs, seed)
    payload = {
        "original_shape": packed.original_shape,
        "layout": (
            packed.layout.block_h,
            packed.layout.block_w,
            packed.layout.block_n,
            packed.layout.block_m,
        ),
        "block_selector": packed.block_selector.cpu(),
        "dense_values": packed.dense_values.cpu(),
        "sparse_values": packed.sparse_values.cpu(),
        "sparse_metadata": packed.sparse_metadata.cpu(),
        "hardware_metadata": (
            None
            if packed.hardware_metadata is None
            else packed.hardware_metadata.cpu()
        ),
        "activation": activation.cpu(),
        "counts": counts.cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(path)


def load_case(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    layout = HybridBlockSparseLayout(*payload["layout"])

    def to_cuda(name: str):
        tensor = payload[name]
        return None if tensor is None else tensor.cuda(non_blocking=False)

    packed = HybridBlockSparseWeight(
        original_shape=tuple(payload["original_shape"]),
        layout=layout,
        block_selector=to_cuda("block_selector"),
        dense_values=to_cuda("dense_values"),
        sparse_values=to_cuda("sparse_values"),
        sparse_metadata=to_cuda("sparse_metadata"),
        hardware_metadata=to_cuda("hardware_metadata"),
    )
    activation = to_cuda("activation")
    counts = to_cuda("counts")
    output = torch.empty(
        activation.shape[0], activation.shape[1], packed.original_shape[-2],
        device="cuda", dtype=torch.bfloat16,
    )
    return activation, packed, counts, output


def profile(path: Path, prebind: bool, warmup: int) -> None:
    activation, packed, counts, output = load_case(path)
    expected_m = max(1, int(counts.max().item()))

    def launch() -> None:
        hybrid_block_sparse_grouped_masked_wgmma_tma(
            activation,
            packed,
            counts,
            out=output,
            expected_m=expected_m,
            use_active_expert_prebind=prebind,
            use_bitmask_selector_fast_path=False,
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    launch()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prebind", type=int, choices=(0, 1), default=0)
    parser.add_argument("--token-bs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prepare:
        prepare(args.data, args.token_bs, args.seed)
    else:
        profile(args.data, bool(args.prebind), args.warmup)
