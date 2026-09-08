"""Capture CTA consumer intervals for the prepacked PM-sampling workload."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from profile_prebind_pm_sampling import load_case
from trace_hybrid_sparse_scheduler import FIELDS, rows_from_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prebind", type=int, choices=(0, 1), required=True)
    parser.add_argument("--max-tasks", type=int, default=64)
    args = parser.parse_args()

    activation, packed, counts, output = load_case(args.data)
    workers = 2 * torch.cuda.get_device_properties(0).multi_processor_count
    trace = torch.full(
        (workers, args.max_tasks, 7, len(FIELDS)),
        -1,
        device="cuda",
        dtype=torch.int64,
    )

    from sparse_gemm.hybrid_sparse import hybrid_block_sparse_grouped_masked_wgmma_tma

    def launch() -> None:
        hybrid_block_sparse_grouped_masked_wgmma_tma(
            activation,
            packed,
            counts,
            out=output,
            expected_m=max(1, int(counts.max().item())),
            use_active_expert_prebind=bool(args.prebind),
            use_bitmask_selector_fast_path=False,
            scheduler_trace=trace,
        )

    launch()
    torch.cuda.synchronize()
    trace.fill_(-1)
    launch()
    torch.cuda.synchronize()

    rows = rows_from_trace("prebind" if args.prebind else "global", trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("mode", "event_name", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.output}: {len(rows)} events")


if __name__ == "__main__":
    main()
