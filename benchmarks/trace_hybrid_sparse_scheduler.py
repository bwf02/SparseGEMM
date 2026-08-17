#!/usr/bin/env python3
"""Capture CTA-level scheduler timelines for the TP2 masked SparseGEMM kernel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch


EVENT_NAMES = (
    "task_assigned",
    "producer_begin",
    "producer_end",
    "consumer_begin",
    "consumer_end",
    "empty_begin",
    "empty_end",
)
FIELDS = (
    "timestamp_ns",
    "sm_id",
    "cta_id",
    "task",
    "event",
    "expert",
    "m_tile",
    "n_tile",
    "valid_rows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--projection", choices=("w13", "down"), default="w13")
    parser.add_argument("--bs", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=16)
    parser.add_argument("--pass-id", type=int, default=32)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--model-layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-tasks", type=int, default=32)
    return parser.parse_args()


def load_bs_counts(args: argparse.Namespace) -> torch.Tensor:
    records = torch.load(args.routing, map_location="cpu", weights_only=False)["records"]
    record = next(
        (item for item in records if item["forward_pass_id"] == args.pass_id), None
    )
    if record is None:
        raise ValueError(f"forward pass {args.pass_id} is absent from {args.routing}")
    base = record["global_physical_count"][args.layer].to(torch.float32)
    probabilities = (base + 0.25) / (base.sum() + 0.25 * base.numel())
    generator = torch.Generator().manual_seed(42)
    counts = None
    for batch_size in (1, 2, 4, 8, 16, 32, 64):
        counts = torch.bincount(
            torch.multinomial(
                probabilities,
                batch_size * 4,
                replacement=True,
                generator=generator,
            ),
            minlength=60,
        ).to(torch.int32)
        if batch_size == args.bs:
            return counts
    raise AssertionError("unreachable")


def load_weight(args: argparse.Namespace):
    os.environ["SGLANG_SPARSE_GEMM_MOE_PATH"] = args.weight_path
    from sglang.srt.layers.moe.moe_runner.sparse_gemm import (
        load_sparse_gemm_moe_weight,
    )

    projection = "w13_weight" if args.projection == "w13" else "down_proj"
    return load_sparse_gemm_moe_weight(
        layer_id=args.model_layer,
        projection=projection,
        device=torch.device("cuda"),
        expert_start=0,
        num_local_experts=60,
        moe_tp_rank=args.tp_rank,
        moe_tp_size=2,
    )


def rows_from_trace(mode: str, trace: torch.Tensor) -> list[dict[str, int | str]]:
    rows = []
    for values in trace.cpu().reshape(-1, len(FIELDS)).tolist():
        if values[0] < 0:
            continue
        row = dict(zip(FIELDS, values))
        row["mode"] = mode
        row["event_name"] = EVENT_NAMES[row["event"]]
        rows.append(row)
    rows.sort(key=lambda row: row["timestamp_ns"])
    return rows


def task_records(rows: list[dict[str, int | str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    for row in rows:
        key = (int(row["cta_id"]), int(row["task"]))
        task = grouped.setdefault(
            key,
            {
                "cta": int(row["cta_id"]),
                "task": int(row["task"]),
                "sm": int(row["sm_id"]),
                "expert": int(row["expert"]),
                "m_tile": int(row["m_tile"]),
                "n_tile": int(row["n_tile"]),
                "valid_rows": int(row["valid_rows"]),
                "events": {},
            },
        )
        task["events"][str(row["event_name"])] = int(row["timestamp_ns"])
    return sorted(grouped.values(), key=lambda task: (task["sm"], task["cta"], task["task"]))


def interval(task: dict[str, object], begin: str, end: str):
    events = task["events"]
    if begin not in events or end not in events:
        return None
    return events[begin], events[end]


def write_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("mode", "event_name", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_perfetto(path: Path, mode: str, tasks: list[dict[str, object]]) -> None:
    if not tasks:
        raise ValueError(f"no trace events were captured for {mode}")
    origin = min(min(task["events"].values()) for task in tasks)
    events: list[dict[str, object]] = []
    lanes = sorted({(task["sm"], task["cta"]) for task in tasks})
    for sm, cta in lanes:
        for role_id, role in enumerate(("task", "producer", "consumer")):
            events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": int(sm),
                    "tid": int(cta) * 3 + role_id,
                    "args": {"name": f"CTA {cta} {role}"},
                }
            )
    for task in tasks:
        args = {
            "expert": task["expert"],
            "m_tile": task["m_tile"],
            "n_tile": task["n_tile"],
            "valid_rows": task["valid_rows"],
        }
        candidates = []
        for begin, end in (
            ("empty_begin", "empty_end"),
            ("producer_begin", "producer_end"),
            ("consumer_begin", "consumer_end"),
        ):
            span = interval(task, begin, end)
            if span:
                candidates.extend(span)
        if candidates:
            events.append(
                {
                    "ph": "X",
                    "name": f"E{task['expert']} m{task['m_tile']} n{task['n_tile']}",
                    "cat": "empty" if task["valid_rows"] == 0 else "task",
                    "pid": task["sm"],
                    "tid": task["cta"] * 3,
                    "ts": (min(candidates) - origin) / 1000.0,
                    "dur": (max(candidates) - min(candidates)) / 1000.0,
                    "args": args,
                }
            )
        for role_id, begin, end in (
            (1, "producer_begin", "producer_end"),
            (2, "consumer_begin", "consumer_end"),
        ):
            span = interval(task, begin, end)
            if span:
                events.append(
                    {
                        "ph": "X",
                        "name": f"E{task['expert']} {begin.removesuffix('_begin')}",
                        "cat": begin.removesuffix("_begin"),
                        "pid": task["sm"],
                        "tid": task["cta"] * 3 + role_id,
                        "ts": (span[0] - origin) / 1000.0,
                        "dur": (span[1] - span[0]) / 1000.0,
                        "args": args,
                    }
                )
    path.write_text(json.dumps({"traceEvents": events}, separators=(",", ":")))


def write_plot(path: Path, mode: str, tasks: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.patches import Patch

    intervals = []
    for task in tasks:
        span = interval(
            task,
            "empty_begin" if task["valid_rows"] == 0 else "consumer_begin",
            "empty_end" if task["valid_rows"] == 0 else "consumer_end",
        )
        if span:
            intervals.append((task, *span))
    origin = min(begin for _, begin, _ in intervals)
    lanes = sorted({(task["sm"], task["cta"]) for task, _, _ in intervals})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    colors = {
        expert: hsv_to_rgb(((expert * 0.61803398875) % 1.0, 0.65, 0.82))
        for expert in range(60)
    }
    figure, axis = plt.subplots(figsize=(16, max(10, len(lanes) * 0.12)))
    for task, begin, end in intervals:
        y = lane_index[(task["sm"], task["cta"])]
        color = "#d0d0d0" if task["valid_rows"] == 0 else colors[task["expert"]]
        axis.broken_barh(
            [((begin - origin) / 1000.0, max(0.01, (end - begin) / 1000.0))],
            (y - 0.42, 0.84),
            facecolors=color,
            edgecolors="none",
        )
    tick_step = max(1, len(lanes) // 24)
    ticks = list(range(0, len(lanes), tick_step))
    axis.set_yticks(ticks, [f"SM {lanes[i][0]} / CTA {lanes[i][1]}" for i in ticks])
    axis.set_xlabel("Time from first recorded task (us)")
    axis.set_ylabel("Resident CTA")
    axis.set_title(f"{mode}: CTA expert/tile consumer timeline")
    axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    axis.invert_yaxis()
    axis.legend(
        handles=[
            Patch(facecolor="#d0d0d0", label="empty/padded tile"),
            Patch(facecolor="#3f8f6b", label="active expert tile"),
        ],
        loc="upper right",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize(mode: str, tasks: list[dict[str, object]]) -> dict[str, object]:
    useful = [task for task in tasks if task["valid_rows"] > 0]
    empty = [task for task in tasks if task["valid_rows"] == 0]
    intervals = []
    for task in tasks:
        span = interval(
            task,
            "empty_begin" if task["valid_rows"] == 0 else "consumer_begin",
            "empty_end" if task["valid_rows"] == 0 else "consumer_end",
        )
        if span:
            intervals.append((task, span[0], span[1]))
    starts = [begin for _, begin, _ in intervals]
    ends = [end for _, _, end in intervals]
    cta_ends: dict[int, int] = {}
    for task, _, end in intervals:
        cta_ends[task["cta"]] = max(cta_ends.get(task["cta"], 0), end)
    sorted_ends = sorted(cta_ends.values())
    median_end = sorted_ends[len(sorted_ends) // 2]
    return {
        "mode": mode,
        "recorded_tasks": len(tasks),
        "useful_tasks": len(useful),
        "empty_tasks": len(empty),
        "active_ctas": len(cta_ends),
        "timeline_us": (max(ends) - min(starts)) / 1000.0,
        "cta_tail_after_median_us": (max(sorted_ends) - median_end) / 1000.0,
    }


def main() -> None:
    args = parse_args()
    if args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    counts_cpu = load_bs_counts(args)
    counts = counts_cpu.cuda()
    weight = load_weight(args)
    _, n, k = weight.original_shape
    activation = torch.randn(60, 64, k, device="cuda", dtype=torch.bfloat16)
    expected_m = math.ceil(int(counts.sum()) / counts.numel())
    workers = 2 * torch.cuda.get_device_properties(0).multi_processor_count

    from sparse_gemm.hybrid_sparse import hybrid_block_sparse_grouped_masked_wgmma_tma

    outputs = {}
    summaries = []
    for mode, use_prebind in (("global", False), ("prebind", True)):
        trace = torch.full(
            (workers, args.max_tasks, len(EVENT_NAMES), len(FIELDS)),
            -1,
            device="cuda",
            dtype=torch.int64,
        )
        output = torch.empty(60, 64, n, device="cuda", dtype=torch.bfloat16)

        def call():
            hybrid_block_sparse_grouped_masked_wgmma_tma(
                activation,
                weight,
                counts,
                out=output,
                expected_m=expected_m,
                use_active_expert_prebind=use_prebind,
                scheduler_trace=trace,
            )

        call()
        torch.cuda.synchronize()
        trace.fill_(-1)
        call()
        torch.cuda.synchronize()
        outputs[mode] = output.clone()
        rows = rows_from_trace(mode, trace)
        tasks = task_records(rows)
        write_csv(output_dir / f"{mode}.csv", rows)
        write_perfetto(output_dir / f"{mode}.perfetto.json", mode, tasks)
        try:
            write_plot(output_dir / f"{mode}.png", mode, tasks)
        except ImportError:
            pass
        summaries.append(summarize(mode, tasks))

    torch.testing.assert_close(outputs["global"], outputs["prebind"], rtol=0, atol=0)
    result = {
        "shape": {
            "batch_size": args.bs,
            "experts": 60,
            "active_experts": int((counts > 0).sum()),
            "sum_m": int(counts.sum()),
            "max_expert_m": int(counts.max()),
            "n": n,
            "k": k,
            "workers": workers,
        },
        "traces": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
