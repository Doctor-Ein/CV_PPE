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


STEP3_SEARCH_CONFIGS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "g1": {
        "map50": [
            {
                "name": "g1_map50_center",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.012,
                    "weight_decay": 0.00010,
                    "lrf": 0.01,
                },
            },
            {
                "name": "g1_map50_wd_up",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.012,
                    "weight_decay": 0.00014,
                    "lrf": 0.02,
                },
            },
            {
                "name": "g1_map50_lr_up",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.013,
                    "weight_decay": 0.00010,
                    "lrf": 0.02,
                },
            },
            {
                "name": "g1_map50_balanced",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.011,
                    "weight_decay": 0.00012,
                    "lrf": 0.03,
                },
            },
        ],
        "map50_95": [
            {
                "name": "g1_map95_center",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.012,
                    "weight_decay": 0.00022,
                    "lrf": 0.01,
                },
            },
            {
                "name": "g1_map95_wd_mid",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.012,
                    "weight_decay": 0.00018,
                    "lrf": 0.02,
                },
            },
            {
                "name": "g1_map95_lr_low",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.011,
                    "weight_decay": 0.00022,
                    "lrf": 0.02,
                },
            },
            {
                "name": "g1_map95_balanced",
                "params": {
                    "optimizer": "SGD",
                    "cos_lr": False,
                    "warmup_epochs": 3.0,
                    "lr0": 0.0125,
                    "weight_decay": 0.00020,
                    "lrf": 0.03,
                },
            },
        ],
    },
    "g2": {
        "map50": [
            {
                "name": "g2_map50_center",
                "params": {"hsv_h": 0.015, "hsv_s": 1.00, "hsv_v": 0.20, "bgr": 0.0},
            },
            {
                "name": "g2_map50_h_low",
                "params": {"hsv_h": 0.010, "hsv_s": 1.00, "hsv_v": 0.20, "bgr": 0.0},
            },
            {
                "name": "g2_map50_s_mid",
                "params": {"hsv_h": 0.015, "hsv_s": 0.85, "hsv_v": 0.20, "bgr": 0.0},
            },
            {
                "name": "g2_map50_v_mid",
                "params": {"hsv_h": 0.015, "hsv_s": 1.00, "hsv_v": 0.30, "bgr": 0.0},
            },
        ],
        "map50_95": [
            {
                "name": "g2_map95_center",
                "params": {"hsv_h": 0.015, "hsv_s": 0.35, "hsv_v": 0.40, "bgr": 0.0},
            },
            {
                "name": "g2_map95_h_low",
                "params": {"hsv_h": 0.010, "hsv_s": 0.35, "hsv_v": 0.40, "bgr": 0.0},
            },
            {
                "name": "g2_map95_s_mid",
                "params": {"hsv_h": 0.015, "hsv_s": 0.50, "hsv_v": 0.40, "bgr": 0.0},
            },
            {
                "name": "g2_map95_v_low",
                "params": {"hsv_h": 0.015, "hsv_s": 0.35, "hsv_v": 0.30, "bgr": 0.0},
            },
        ],
    },
    "g3": {
        "map50": [
            {
                "name": "g3_map50_center",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.05,
                    "scale": 0.95,
                    "shear": 2.0,
                    "perspective": 0.0001,
                    "flipud": 0.25,
                    "fliplr": 0.75,
                },
            },
            {
                "name": "g3_map50_scale_flip",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.05,
                    "scale": 1.00,
                    "shear": 2.0,
                    "perspective": 0.0001,
                    "flipud": 0.25,
                    "fliplr": 0.80,
                },
            },
            {
                "name": "g3_map50_translate_down",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.03,
                    "scale": 0.95,
                    "shear": 2.0,
                    "perspective": 0.0001,
                    "flipud": 0.20,
                    "fliplr": 0.75,
                },
            },
            {
                "name": "g3_map50_shear_mid",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.05,
                    "scale": 0.95,
                    "shear": 1.5,
                    "perspective": 0.0001,
                    "flipud": 0.25,
                    "fliplr": 0.70,
                },
            },
        ],
        "map50_95": [
            {
                "name": "g3_map95_center",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.05,
                    "scale": 0.95,
                    "shear": 2.0,
                    "perspective": 0.0001,
                    "flipud": 0.25,
                    "fliplr": 0.75,
                },
            },
            {
                "name": "g3_map95_translate_up",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.07,
                    "scale": 0.95,
                    "shear": 2.0,
                    "perspective": 0.0001,
                    "flipud": 0.25,
                    "fliplr": 0.70,
                },
            },
            {
                "name": "g3_map95_perspective_up",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.05,
                    "scale": 0.95,
                    "shear": 2.0,
                    "perspective": 0.0002,
                    "flipud": 0.25,
                    "fliplr": 0.75,
                },
            },
            {
                "name": "g3_map95_flipud_mid",
                "params": {
                    "degrees": 0.0,
                    "translate": 0.05,
                    "scale": 0.90,
                    "shear": 2.0,
                    "perspective": 0.0001,
                    "flipud": 0.30,
                    "fliplr": 0.75,
                },
            },
        ],
    },
    "g4": {
        "map50": [
            {
                "name": "g4_map50_center",
                "params": {
                    "mosaic": 0.10,
                    "close_mosaic": 10,
                    "mixup": 0.10,
                    "cutmix": 0.00,
                    "copy_paste": 0.00,
                    "erasing": 0.40,
                },
            },
            {
                "name": "g4_map50_mixup_up",
                "params": {
                    "mosaic": 0.10,
                    "close_mosaic": 10,
                    "mixup": 0.15,
                    "cutmix": 0.00,
                    "copy_paste": 0.00,
                    "erasing": 0.40,
                },
            },
            {
                "name": "g4_map50_mosaic_mid",
                "params": {
                    "mosaic": 0.20,
                    "close_mosaic": 10,
                    "mixup": 0.10,
                    "cutmix": 0.05,
                    "copy_paste": 0.00,
                    "erasing": 0.40,
                },
            },
            {
                "name": "g4_map50_balanced",
                "params": {
                    "mosaic": 0.15,
                    "close_mosaic": 10,
                    "mixup": 0.10,
                    "cutmix": 0.05,
                    "copy_paste": 0.00,
                    "erasing": 0.20,
                },
            },
        ],
        "map50_95": [
            {
                "name": "g4_map95_center",
                "params": {
                    "mosaic": 0.15,
                    "close_mosaic": 10,
                    "mixup": 0.15,
                    "cutmix": 0.00,
                    "copy_paste": 0.00,
                    "erasing": 0.40,
                },
            },
            {
                "name": "g4_map95_cutmix_low",
                "params": {
                    "mosaic": 0.15,
                    "close_mosaic": 10,
                    "mixup": 0.15,
                    "cutmix": 0.05,
                    "copy_paste": 0.00,
                    "erasing": 0.40,
                },
            },
            {
                "name": "g4_map95_mosaic_up",
                "params": {
                    "mosaic": 0.25,
                    "close_mosaic": 10,
                    "mixup": 0.10,
                    "cutmix": 0.00,
                    "copy_paste": 0.00,
                    "erasing": 0.40,
                },
            },
            {
                "name": "g4_map95_erasing_low",
                "params": {
                    "mosaic": 0.15,
                    "close_mosaic": 10,
                    "mixup": 0.15,
                    "cutmix": 0.00,
                    "copy_paste": 0.00,
                    "erasing": 0.20,
                },
            },
        ],
    },
    "g5": {
        "map50": [
            {"name": "g5_map50_large_box", "params": {"box": 10.0, "cls": 1.25, "dfl": 1.5}},
            {"name": "g5_map50_center", "params": {"box": 8.75, "cls": 1.20, "dfl": 1.5}},
            {"name": "g5_map50_small_box", "params": {"box": 5.0, "cls": 1.25, "dfl": 1.5}},
            {"name": "g5_map50_balanced", "params": {"box": 6.25, "cls": 1.0, "dfl": 1.0}},
        ],
        "map50_95": [
            {"name": "g5_map95_high_precision", "params": {"box": 10.0, "cls": 1.0, "dfl": 0.5}},
            {"name": "g5_map95_low_bias", "params": {"box": 3.75, "cls": 1.0, "dfl": 0.5}},
            {"name": "g5_map95_mid_box", "params": {"box": 5.0, "cls": 1.0, "dfl": 0.75}},
            {"name": "g5_map95_balanced", "params": {"box": 6.25, "cls": 1.0, "dfl": 1.0}},
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: within-group joint tuning for one group and one objective per execution "
            "(YOLOv8n only)."
        )
    )
    parser.add_argument("--group", required=True, choices=sorted(STEP3_SEARCH_CONFIGS.keys()))
    parser.add_argument("--objective", default="map50", choices=["map50", "map50_95"])
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per config.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default="", help="Training device passed to Ultralytics.")
    parser.add_argument("--run-prefix", default="joint", help="Run name prefix.")
    parser.add_argument("--execute", action="store_true", help="Actually launch training.")
    parser.add_argument("--strict-data", action="store_true", help="Fail on unmatched extra labels.")
    parser.add_argument("--list", action="store_true", help="Only print built-in configs and exit.")
    return parser.parse_args()


def build_train_args(
    args: argparse.Namespace, run_name: str, config_name: str, group: str, objective: str, params: dict[str, Any]
) -> dict[str, Any]:
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
    group: str,
    objective: str,
    config_name: str,
    params: dict[str, Any],
    train_args: dict[str, Any],
    dataset_report: Any,
) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = PLANS_DIR / f"step3_{run_name}.json"
    payload = {
        "stage": "step3-joint-tune",
        "group": group,
        "objective": objective,
        "config_name": config_name,
        "params": params,
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
    configs = STEP3_SEARCH_CONFIGS[args.group][args.objective]

    if args.list:
        print(json.dumps(configs, indent=2, ensure_ascii=False))
        return

    dataset_report = validate_dataset(DATA_YAML_PATH, strict_data=args.strict_data)
    print("Dataset validation passed.")
    print(f"Data YAML : {DATA_YAML_PATH.resolve()}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Group     : {args.group}")
    print(f"Objective : {args.objective}")
    print(f"Epochs    : {args.epochs}")
    print(f"Workers   : {args.workers}")
    print(f"Output    : {EXPERIMENT_DIR.resolve()}")

    for config in configs:
        config_name = str(config["name"])
        params = dict(config["params"])
        run_name = f"{args.run_prefix}_{args.group}_{args.objective}_{config_name}"
        train_args = build_train_args(args, run_name, config_name, args.group, args.objective, params)
        plan_path = export_plan(run_name, args.group, args.objective, config_name, params, train_args, dataset_report)

        print(f"\nRun name   : {run_name}")
        print(f"Config     : {config_name}")
        print(f"Plan file  : {plan_path}")
        print("Joint args:")
        print(json.dumps(params, indent=2, ensure_ascii=False))

        if not args.execute:
            continue

        from ultralytics import YOLO

        model = YOLO(MODEL_NAME)
        result = model.train(**train_args)
        save_dir = Path(result.save_dir)
        metrics = extract_last_metrics(save_dir / "results.csv")
        summary_row = {
            "stage": "step3-joint-tune",
            "group": args.group,
            "objective": args.objective,
            "config_name": config_name,
            "run_name": run_name,
            "model": MODEL_NAME,
            "epochs": str(args.epochs),
            "imgsz": str(args.imgsz),
            "batch": str(args.batch),
            "save_dir": str(save_dir),
            "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
            **metrics,
        }
        append_summary_row(summary_row)
        print(f"Saved summary row to {SUMMARY_CSV_PATH}")
        release_runtime_memory(model)

    if not args.execute:
        print("\nDry-run only. Add `--execute` to actually start Step 3 joint tuning runs.")


if __name__ == "__main__":
    main()

