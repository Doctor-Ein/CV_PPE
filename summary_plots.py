from __future__ import annotations

import argparse
from pathlib import Path

from step2_analyze import DEFAULT_EXPERIMENT2_DIR, SUPPORTED_PARAMS, collect_records as collect_step2_records
from step2_fine_scan import EXPERIMENT_DIR as DEFAULT_EXPERIMENT2_FINE_DIR
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
            "1) Step 2 parameter-value vs mAP50-95 line charts (one image per parameter).\n"
            "2) Step 3 grouped bar charts comparing group bests and final compose."
        )
    )
    parser.add_argument(
        "--experiment2-root",
        default=str(DEFAULT_EXPERIMENT2_FINE_DIR),
        help="Path to Step 2 directory. Defaults to fine-grained experiment2_fine.",
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


def plot_step2_map95_curves(experiment2_root: Path, outdir: Path) -> list[Path]:
    plt = ensure_matplotlib()
    records = collect_step2_records(experiment2_root)
    if not records:
        return []

    grouped: dict[str, list[tuple[float, float]]] = {param: [] for param in SUPPORTED_PARAMS}
    for record in records:
        if record.map50_95 is None:
            continue
        grouped[record.param].append((record.value, record.map50_95))

    if not any(grouped.values()):
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for param in SUPPORTED_PARAMS:
        points = sorted(grouped[param], key=lambda item: item[0])
        if not points:
            continue
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, marker="o", linewidth=1.8)
        for x, y in points:
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_title(f"{param} vs mAP50-95")
        ax.set_xlabel(param)
        ax.set_ylabel("mAP50-95")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = outdir / f"step2_{param}_vs_map50_95.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)
    return output_paths


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

    # Prefer the fine-grained Step 2 directory when present, otherwise fall back to the original Step 2 root.
    if not experiment2_root.exists() or not collect_step2_records(experiment2_root):
        fallback_root = Path(DEFAULT_EXPERIMENT2_DIR).expanduser().resolve()
        if fallback_root != experiment2_root and fallback_root.exists():
            experiment2_root = fallback_root

    print(f"Experiment2 : {experiment2_root}")
    print(f"Experiment3 : {experiment3_root}")
    print(f"Output dir  : {outdir}")

    try:
        step2_plots = plot_step2_map95_curves(experiment2_root, outdir)
    except RuntimeError as exc:
        print(exc)
        print("Install it first, for example: `pip install matplotlib`")
        return
    if step2_plots:
        for path in step2_plots:
            print(f"Wrote: {path}")
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
