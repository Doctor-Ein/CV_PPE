from __future__ import annotations

import argparse
from pathlib import Path

from step2_analyze import DEFAULT_EXPERIMENT2_DIR, SUPPORTED_PARAMS, collect_records as collect_step2_records
from step3_analyze import (
    DEFAULT_EXPERIMENT3_DIR,
    SUPPORTED_GROUPS,
    SUPPORTED_OBJECTIVES,
    collect_records as collect_step3_records,
    pick_best as pick_best_step3,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "summary_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate summary plots for Step 2 and Step 3 experiments:\n"
            "1) Step 2 parameter-value vs mAP50-95 line charts.\n"
            "2) Step 3 grouped bar charts comparing group bests and final compose."
        )
    )
    parser.add_argument(
        "--experiment2-root",
        default=str(DEFAULT_EXPERIMENT2_DIR),
        help="Path to experiment2 directory.",
    )
    parser.add_argument(
        "--experiment3-root",
        default=str(DEFAULT_EXPERIMENT3_DIR),
        help="Path to experiment3 directory.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for plot images.",
    )
    return parser.parse_args()


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for summary_plots.py") from exc
    return plt


def plot_step2_map95_curves(experiment2_root: Path, outdir: Path) -> Path | None:
    plt = ensure_matplotlib()
    records = collect_step2_records(experiment2_root)
    if not records:
        return None

    grouped: dict[str, list[tuple[float, float]]] = {param: [] for param in SUPPORTED_PARAMS}
    for record in records:
        if record.map50_95 is None:
            continue
        grouped[record.param].append((record.value, record.map50_95))

    if not any(grouped.values()):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_list = list(axes.flatten())
    for idx, param in enumerate(SUPPORTED_PARAMS):
        ax = axes_list[idx]
        points = sorted(grouped[param], key=lambda item: item[0])
        if not points:
            ax.set_title(f"{param} (no data)")
            ax.axis("off")
            continue
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8)
        for x, y in points:
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_title(f"{param} vs mAP50-95")
        ax.set_xlabel(param)
        ax.set_ylabel("mAP50-95")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Step 2 Single-Parameter Scan", fontsize=14)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "step2_param_vs_map50_95.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_step3_group_bars(experiment3_root: Path, outdir: Path, objective: str) -> Path | None:
    plt = ensure_matplotlib()
    records = collect_step3_records(experiment3_root)
    if not records:
        return None

    best_by_objective = pick_best_step3(records)
    grouped = best_by_objective.get(objective, {})
    valid_groups = [group for group in SUPPORTED_GROUPS if group in grouped]
    if not valid_groups:
        return None

    metric_key = "map50" if objective == "map50" else "map50_95"

    sorted_records = sorted(
        [grouped[group] for group in valid_groups],
        key=lambda record: getattr(record, metric_key) if getattr(record, metric_key) is not None else -1.0,
        reverse=True,
    )

    labels = [record.group for record in sorted_records]
    map50_values = [record.map50 or 0.0 for record in sorted_records]
    map95_values = [record.map50_95 or 0.0 for record in sorted_records]

    x_positions = list(range(len(sorted_records)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    left_positions = [x - width / 2 for x in x_positions]
    right_positions = [x + width / 2 for x in x_positions]

    bars1 = ax.bar(left_positions, map50_values, width=width, label="mAP50")
    bars2 = ax.bar(right_positions, map95_values, width=width, label="mAP50-95")

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Step 3 Best Results Sorted by {metric_key}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"step3_group_compare_{objective}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    experiment2_root = Path(args.experiment2_root).expanduser().resolve()
    experiment3_root = Path(args.experiment3_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    print(f"Experiment2 : {experiment2_root}")
    print(f"Experiment3 : {experiment3_root}")
    print(f"Output dir  : {outdir}")

    try:
        step2_plot = plot_step2_map95_curves(experiment2_root, outdir)
    except RuntimeError as exc:
        print(exc)
        print("Install it first, for example: `pip install matplotlib`")
        return
    if step2_plot:
        print(f"Wrote: {step2_plot}")
    else:
        print("Skip Step 2 plot: no usable experiment2 records found.")

    for objective in SUPPORTED_OBJECTIVES:
        step3_plot = plot_step3_group_bars(experiment3_root, outdir, objective)
        if step3_plot:
            print(f"Wrote: {step3_plot}")
        else:
            print(f"Skip Step 3 plot for {objective}: no usable experiment3 records found.")


if __name__ == "__main__":
    main()
