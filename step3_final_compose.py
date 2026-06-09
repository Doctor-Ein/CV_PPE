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
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "experiment3"
PLANS_DIR = EXPERIMENT_DIR / "plans"
SUMMARY_CSV_PATH = EXPERIMENT_DIR / "step3_summary.csv"
DEFAULT_RECOMMENDATIONS_PATH = EXPERIMENT_DIR / "step3_recommendations.json"
SOURCE_GROUPS = ("g1", "g2", "g3", "g4", "g5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the best Step 3 group-wise results into one final joint configuration "
            "and optionally train it (YOLOv8n only)."
        )
    )
    parser.add_argument("--objective", default="map50", choices=["map50", "map50_95"])
    parser.add_argument(
        "--recommendations",
        default=str(DEFAULT_RECOMMENDATIONS_PATH),
        help="Path to step3_recommendations.json produced by step3_analyze.py.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default="", help="Training device passed to Ultralytics.")
    parser.add_argument("--run-prefix", default="joint", help="Run name prefix.")
    parser.add_argument("--execute", action="store_true", help="Actually launch training.")
    parser.add_argument("--strict-data", action="store_true", help="Fail on unmatched extra labels.")
    parser.add_argument("--show-only", action="store_true", help="Only print the merged config and exit.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Recommendations file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


def load_composed_params(recommendations_path: Path, objective: str) -> tuple[dict[str, Any], dict[str, str]]:
    payload = read_json(recommendations_path)
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, dict):
        raise ValueError(f"`recommendations` missing in {recommendations_path}")

    objective_map = recommendations.get(objective)
    if not isinstance(objective_map, dict):
        raise ValueError(f"Objective `{objective}` missing in {recommendations_path}")

    merged: dict[str, Any] = {}
    source_configs: dict[str, str] = {}
    for group in SOURCE_GROUPS:
        entry = objective_map.get(group)
        if not isinstance(entry, dict):
            raise ValueError(
                f"Group `{group}` missing under objective `{objective}` in {recommendations_path}. "
                "Run `step3_analyze.py` after completing Step 3 group experiments first."
            )
        params = entry.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"`params` missing for {group}/{objective} in {recommendations_path}")
        merged.update(params)
        config_name = entry.get("config_name")
        if isinstance(config_name, str) and config_name:
            source_configs[group] = config_name
    return merged, source_configs


def build_train_args(args: argparse.Namespace, run_name: str, params: dict[str, Any]) -> dict[str, Any]:
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
        **params,
    }
    if args.device:
        train_args["device"] = args.device
    return train_args


def export_plan(
    run_name: str,
    objective: str,
    params: dict[str, Any],
    source_configs: dict[str, str],
    train_args: dict[str, Any],
    dataset_report: Any,
) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = PLANS_DIR / f"step3_{run_name}.json"
    payload = {
        "stage": "step3-final-compose",
        "group": "final_compose",
        "objective": objective,
        "config_name": "final_compose",
        "params": params,
        "source_groups": list(SOURCE_GROUPS),
        "source_configs": source_configs,
        "model": MODEL_NAME,
        "data_yaml": str(DATA_YAML_PATH.resolve()),
        "train_args": train_args,
        "dataset_report": asdict(dataset_report),
    }
    plan_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path


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


def append_summary_row(row: dict[str, str]) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV_PATH.exists()
    with SUMMARY_CSV_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def release_runtime_memory(model: Any | None = None) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    recommendations_path = Path(args.recommendations).expanduser().resolve()
    try:
        params, source_configs = load_composed_params(recommendations_path, args.objective)
    except FileNotFoundError as exc:
        print(exc)
        print("Run `python step3_analyze.py` on the machine with finished Step 3 results first.")
        return
    except ValueError as exc:
        print(exc)
        return

    print(f"Objective    : {args.objective}")
    print(f"Source file  : {recommendations_path}")
    print(f"Source groups: {source_configs}")
    print("Composed params:")
    print(json.dumps(params, indent=2, ensure_ascii=False, sort_keys=True))

    if args.show_only:
        return

    dataset_report = validate_dataset(DATA_YAML_PATH, strict_data=args.strict_data)
    run_name = f"{args.run_prefix}_final_compose_{args.objective}"
    train_args = build_train_args(args, run_name, params)
    plan_path = export_plan(run_name, args.objective, params, source_configs, train_args, dataset_report)

    print(f"Data YAML    : {DATA_YAML_PATH.resolve()}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Epochs       : {args.epochs}")
    print(f"Workers      : {args.workers}")
    print(f"Output       : {EXPERIMENT_DIR.resolve()}")
    print(f"Plan file    : {plan_path}")

    if not args.execute:
        print("\nDry-run only. Add `--execute` to actually start final compose training.")
        return

    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)
    result = model.train(**train_args)
    save_dir = Path(result.save_dir)
    metrics = extract_last_metrics(save_dir / "results.csv")
    summary_row = {
        "stage": "step3-final-compose",
        "group": "final_compose",
        "objective": args.objective,
        "config_name": "final_compose",
        "run_name": run_name,
        "model": MODEL_NAME,
        "epochs": str(args.epochs),
        "imgsz": str(args.imgsz),
        "batch": str(args.batch),
        "save_dir": str(save_dir),
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "source_configs_json": json.dumps(source_configs, ensure_ascii=False, sort_keys=True),
        **metrics,
    }
    append_summary_row(summary_row)
    print(f"Saved summary row to {SUMMARY_CSV_PATH}")
    release_runtime_memory(model)


if __name__ == "__main__":
    main()
