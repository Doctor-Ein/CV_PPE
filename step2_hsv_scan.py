from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from exp_manager import DATA_YAML_PATH, MODEL_NAME, find_metric_value, validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "experiment2"
PLANS_DIR = EXPERIMENT_DIR / "plans"
SUMMARY_CSV_PATH = EXPERIMENT_DIR / "step2_summary.csv"

DEFAULT_SCAN_VALUES: dict[str, list[float]] = {
    "hsv_h": [0.0, 0.01, 0.02, 0.03],
    "hsv_s": [0.0, 0.35, 0.7, 1.00],
    "hsv_v": [0.0, 0.2, 0.4, 0.6],
    "bgr": [0.0, 0.1, 0.2, 0.3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: single-parameter color/illumination scan (YOLOv8n only).\n"
            "Each execution scans one parameter with built-in recommended values."
        )
    )
    parser.add_argument(
        "--param",
        required=True,
        choices=sorted(DEFAULT_SCAN_VALUES.keys()),
        help="Which single parameter to scan in this execution.",
    )
    parser.add_argument(
        "--values",
        help="Comma-separated values to scan for --param. If omitted, uses built-in values.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument(
        "--device",
        default="",
        help="Training device passed to Ultralytics, for example '0' or 'cpu'.",
    )
    parser.add_argument(
        "--run-prefix",
        default="hsv_scan",
        help="Prefix for run names under the experiment directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch training. Without this flag, only validation and plan export run.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Treat extra unmatched label files as an error instead of a warning.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip trend plot generation.",
    )
    return parser.parse_args()


def parse_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("No values parsed from --values")
    return values


def slugify_value(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def export_plan(run_name: str, train_args: dict[str, Any], dataset_report: Any) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = PLANS_DIR / f"step2_{run_name}.json"
    payload = {
        "stage": "step2-single-param",
        "model": MODEL_NAME,
        "data_yaml": str(DATA_YAML_PATH.resolve()),
        "train_args": train_args,
        "dataset_report": asdict(dataset_report),
    }
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path


def append_summary_row(row: dict[str, str]) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV_PATH.exists()
    with SUMMARY_CSV_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def extract_last_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "precision": find_metric_value(last, ["precision"]),
        "recall": find_metric_value(last, ["recall"]),
        "mAP50": find_metric_value(last, ["map50("]),
        "mAP50-95": find_metric_value(last, ["map50-95("]),
    }


def plot_trend(param: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not SUMMARY_CSV_PATH.exists():
        return None

    rows: list[dict[str, str]] = []
    with SUMMARY_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("param") == param:
                rows.append(row)
    if not rows:
        return None

    def to_float(value: str) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    points: list[tuple[float, float]] = []
    for row in rows:
        x = to_float(row.get("value", ""))
        y = to_float(row.get("mAP50", ""))
        if x is None or y is None:
            continue
        points.append((x, y))
    points.sort(key=lambda item: item[0])
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(param)
    plt.ylabel("mAP50")
    plt.title(f"Step2 scan: {param}")
    plt.grid(True, alpha=0.3)
    plot_path = EXPERIMENT_DIR / f"trend_{param}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def build_train_args(args: argparse.Namespace, run_name: str, param: str, value: float) -> dict[str, Any]:
    train_args: dict[str, Any] = {
        "data": str(DATA_YAML_PATH.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(EXPERIMENT_DIR.resolve()),
        "name": run_name,
        "exist_ok": False,
        "pretrained": True,
        "seed": 42,
        "deterministic": True,
        "verbose": True,
        param: value,
    }
    if args.device:
        train_args["device"] = args.device
    return train_args


def release_runtime_memory(model: Any | None = None) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run() -> None:
    args = parse_args()
    scan_plan = {
        args.param: parse_values(args.values) if args.values else DEFAULT_SCAN_VALUES[args.param]
    }

    dataset_report = validate_dataset(DATA_YAML_PATH, strict_data=args.strict_data)
    print("Dataset validation passed.")
    print(f"Data YAML : {DATA_YAML_PATH.resolve()}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Epochs    : {args.epochs}")
    print(f"Workers   : {args.workers}")
    print(f"Output    : {EXPERIMENT_DIR.resolve()}")
    print(f"Scan plan : {scan_plan}")

    for param, values in scan_plan.items():
        print(f"\n=== Scan {param}: {values} ===")
        for value in values:
            value_slug = slugify_value(value)
            run_name = f"{args.run_prefix}_{param}_{value_slug}"
            train_args = build_train_args(args, run_name, param, value)
            plan_path = export_plan(run_name, train_args, dataset_report)

            print(f"\nRun name  : {run_name}")
            print(f"Plan file : {plan_path}")
            print("Train args:")
            print(json.dumps(train_args, indent=2, ensure_ascii=False))

            if not args.execute:
                continue

            from ultralytics import YOLO

            model = YOLO(MODEL_NAME)
            result = model.train(**train_args)
            save_dir = Path(result.save_dir)
            metrics = extract_last_metrics(save_dir / "results.csv")
            summary_row = {
                "stage": "step2-single-param",
                "param": param,
                "value": str(value),
                "run_name": run_name,
                "model": MODEL_NAME,
                "epochs": str(args.epochs),
                "imgsz": str(args.imgsz),
                "batch": str(args.batch),
                "save_dir": str(save_dir),
                **metrics,
            }
            append_summary_row(summary_row)
            print(f"Saved summary row to {SUMMARY_CSV_PATH}")
            release_runtime_memory(model)

        if args.execute and not args.no_plot:
            plot_path = plot_trend(param)
            if plot_path:
                print(f"Saved trend plot: {plot_path}")

    if not args.execute:
        print("\nDry-run only. Add `--execute` to actually start Step 2 scan runs.")


if __name__ == "__main__":
    run()
