"""Plot freshly paired Grouped GEMM measurements without changing paper figures."""

import argparse
import csv
import math
from pathlib import Path
from statistics import geometric_mean

import matplotlib.pyplot as plt
import numpy as np


MODELS = [("qwen15", "Qwen1.5"), ("deepseek_v2_lite", "DeepSeek-V2"),
          ("qwen3_30b", "Qwen3"), ("llama4_scout", "Llama 4")]
SERIES = [("sparse_gemm", "LoSparse", "#E7A28D", "xx"),
          ("deepgemm_grouped", "DeepGEMM", "#B9C0C5", ""),
          ("cublas_grouped", "cuBLAS Grouped", "#CAB5D7", "//")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len({source.resolve() for source in args.inputs}) != len(args.inputs):
        raise ValueError("Each input file must occur exactly once")
    paired = []
    expected = {(model, projection, batch) for model, _ in MODELS
                for projection in ("gate_up", "down")
                for batch in (8, 16, 32, 64, 128, 4096, 8192, 16384, 32768)}
    for source in args.inputs:
        groups = {}
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["backend"] not in {s[0] for s in SERIES}:
                    continue
                key = row["model"], row["projection"], int(row["token_bs"])
                group = groups.setdefault(key, {})
                if row["backend"] in group:
                    raise ValueError(f"Duplicate row in {source}: {key}")
                group[row["backend"]] = row
        if set(groups) != expected:
            raise ValueError(f"Missing or mismatched configurations: {source}")
        for key, group in sorted(groups.items()):
            if set(group) != {s[0] for s in SERIES}:
                raise ValueError(f"Incomplete pair: {source}, {key}")
            for field in ("N", "K", "num_experts", "active_experts", "top_k",
                          "valid_expert_m", "dtype", "capacity_m"):
                if len({r[field] for r in group.values()}) != 1:
                    raise ValueError(f"Mismatch: {key}, {field}")
            latency = {b: float(r["latency_us"]) for b, r in group.items()}
            if any(not math.isfinite(v) or v <= 0 for v in latency.values()):
                raise ValueError("Latency must be finite and positive")
            paired.append(dict(source=source.name, model=key[0], projection=key[1],
                               token_bs=key[2], **latency))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "hatch.linewidth": 0.4,
                         "axes.spines.top": False, "axes.spines.right": False})
    for scale, batches in [("small", sorted({r["token_bs"] for r in paired if r["token_bs"] < 4096})),
                           ("prefill", sorted({r["token_bs"] for r in paired if r["token_bs"] >= 4096}))]:
        if not batches:
            continue
        fig, axes = plt.subplots(len(batches), 1, figsize=(7.1, 1.28 * len(batches) + 0.5),
                                 squeeze=False)
        for ax, batch in zip(axes[:, 0], batches):
            keys = [(m, p) for p in ("gate_up", "down") for m, _ in MODELS]
            x = np.arange(8, dtype=float)
            for idx, (backend, label, color, hatch) in enumerate(SERIES):
                values = []
                for model, projection in keys:
                    rows = [r for r in paired if (r["model"], r["projection"], r["token_bs"])
                            == (model, projection, batch)]
                    if not rows:
                        raise ValueError(f"Missing {model}/{projection}/BS={batch}")
                    values.append(geometric_mean(r["deepgemm_grouped"] / r[backend] for r in rows))
                ax.bar(x + (idx - 1) * 0.22, values, width=0.22, color=color,
                       edgecolor="#555555", linewidth=0.4, hatch=hatch, label=label)
            ax.axhline(1, color="#555555", linewidth=0.9, linestyle="--")
            ax.axvline(3.5, color="#999999", linewidth=0.6, linestyle=":")
            ax.set_xticks(x, [name for _, name in MODELS] * 2)
            ax.tick_params(axis="x", labelsize=7, pad=2)
            ax.set_xlim(-0.5, 7.5)
            ax.set_ylim(bottom=0)
            ax.set_title(f"BS={batch:,}    |    Gate/Up (left)    /    Down (right)",
                         fontsize=8, fontweight="bold", pad=3)
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.supylabel("Speedup over DeepGEMM", fontsize=9)
        fig.tight_layout(rect=(0.025, 0, 1, 0.94), h_pad=1.0)
        for ext in ("pdf", "png"):
            fig.savefig(args.output_dir / f"h20_cublas_grouped_{scale}.{ext}", dpi=220)
        plt.close(fig)
    summary = []
    for scale, rows in [("all", paired), ("small", [r for r in paired if r["token_bs"] < 4096]),
                        ("prefill", [r for r in paired if r["token_bs"] >= 4096])]:
        if rows:
            configurations = {(r["model"], r["projection"], r["token_bs"]) for r in rows}
            values = [geometric_mean(r["cublas_grouped"] / r["sparse_gemm"]
                                     for r in rows
                                     if (r["model"], r["projection"], r["token_bs"]) == key)
                      for key in sorted(configurations)]
            summary.append(f"{scale}: LoSparse/cuBLAS speedup geometric mean "
                           f"{geometric_mean(values):.3f}x, max {max(values):.3f}x; "
                           f"{len(values)} configurations, {len(rows)} paired observations; "
                           "each configuration aggregated geometrically across seeds")
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
