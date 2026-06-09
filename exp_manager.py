from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML_PATH = PROJECT_ROOT / "ppe_data.yaml"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
PLANS_DIR = EXPERIMENT_ROOT / "plans"
SUMMARY_CSV_PATH = EXPERIMENT_ROOT / "experiment_summary.csv"

MODEL_NAME = "yolov8n.pt"
EXPECTED_CLASSES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitValidation:
    split: str
    image_dir: str
    label_dir: str
    image_count: int
    label_count: int
    matched_pairs: int
    missing_labels: list[str] = field(default_factory=list)
    extra_labels: list[str] = field(default_factory=list)
    invalid_label_entries: list[str] = field(default_factory=list)


@dataclass
class DatasetValidationReport:
    dataset_root: str
    data_yaml: str
    class_names: list[str]
    split_reports: list[SplitValidation]
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PPE experiment manager. Defaults to Step 1 Baseline and keeps "
            "YOLOv8n as the only supported model."
        )
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="step1-baseline",
        choices=["step1-baseline", "step2-hsv", "step3-group", "step4-validate"],
        help="Which experiment stage to prepare or execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch training. Without this flag, only validation and plan export run.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Baseline training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--device",
        default="",
        help="Training device passed to Ultralytics, for example '0' or 'cpu'.",
    )
    parser.add_argument(
        "--run-name",
        default="baseline_yolov8n_default",
        help="Run name under the experiment project directory.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Treat extra unmatched label files as an error instead of a warning.",
    )
    return parser.parse_args()


def read_yaml(yaml_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    active_section: str | None = None

    with yaml_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line_without_comment = raw_line.split("#", 1)[0].rstrip()
            if not line_without_comment.strip():
                continue

            if raw_line[:1].isspace():
                if active_section != "names":
                    raise ValueError(
                        f"Unsupported nested YAML content at {yaml_path}:{line_number}. "
                        "Only the `names` mapping is expected."
                    )
                entry = line_without_comment.strip()
                if ":" not in entry:
                    raise ValueError(f"Invalid names entry at {yaml_path}:{line_number}")
                key, value = entry.split(":", 1)
                data["names"][key.strip()] = value.strip().strip("'\"")
                continue

            active_section = None
            if ":" not in line_without_comment:
                raise ValueError(f"Invalid YAML line at {yaml_path}:{line_number}")

            key, value = line_without_comment.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if value:
                data[key] = value
                continue

            if key == "names":
                data[key] = {}
                active_section = key
                continue

            data[key] = ""

    return data


def normalize_class_names(names: Any) -> list[str]:
    if isinstance(names, list):
        normalized = [str(name) for name in names]
    elif isinstance(names, dict):
        normalized = [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    else:
        raise ValueError("`names` must be a list or mapping in ppe_data.yaml")
    return normalized


def resolve_dataset_root(yaml_path: Path, yaml_data: dict[str, Any]) -> Path:
    dataset_path = yaml_data.get("path")
    if not dataset_path:
        raise ValueError("`path` is required in ppe_data.yaml")
    return (yaml_path.parent / dataset_path).resolve()


def resolve_split_dirs(dataset_root: Path, split_path: str) -> tuple[Path, Path]:
    image_dir = (dataset_root / split_path).resolve()
    split_parts = Path(split_path).parts
    if "images" not in split_parts:
        raise ValueError(f"Split path '{split_path}' must contain an 'images' segment")
    label_parts = ["labels" if part == "images" else part for part in split_parts]
    label_dir = (dataset_root / Path(*label_parts)).resolve()
    return image_dir, label_dir


def collect_image_stems(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def collect_label_stems(label_dir: Path) -> set[str]:
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    return {
        path.stem
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def validate_label_file(label_path: Path, class_count: int) -> list[str]:
    issues: list[str] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"{label_path.name}:{line_number} expected 5 columns, got {len(parts)}")
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                issues.append(f"{label_path.name}:{line_number} invalid class id '{parts[0]}'")
                continue
            if not 0 <= class_id < class_count:
                issues.append(
                    f"{label_path.name}:{line_number} class id {class_id} out of range 0-{class_count - 1}"
                )
            for index, value in enumerate(parts[1:], start=2):
                try:
                    float(value)
                except ValueError:
                    issues.append(f"{label_path.name}:{line_number} column {index} is not numeric")
    return issues


def validate_dataset(data_yaml_path: Path, strict_data: bool) -> DatasetValidationReport:
    yaml_data = read_yaml(data_yaml_path)
    class_names = normalize_class_names(yaml_data.get("names"))
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "Class names in ppe_data.yaml do not match the expected Construction-PPE 11-class layout.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Actual:   {class_names}"
        )

    nc = yaml_data.get("nc")
    if nc is not None and int(nc) != len(EXPECTED_CLASSES):
        raise ValueError(f"`nc` must be {len(EXPECTED_CLASSES)} when present, got {nc}")

    dataset_root = resolve_dataset_root(data_yaml_path, yaml_data)
    split_reports: list[SplitValidation] = []
    warnings: list[str] = []
    hard_failures: list[str] = []

    for split in ("train", "val", "test"):
        split_path = yaml_data.get(split)
        if not split_path:
            raise ValueError(f"`{split}` is required in ppe_data.yaml")

        image_dir, label_dir = resolve_split_dirs(dataset_root, str(split_path))
        image_stems = collect_image_stems(image_dir)
        label_stems = collect_label_stems(label_dir)
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)

        invalid_entries: list[str] = []
        for label_path in sorted(label_dir.glob("*.txt")):
            invalid_entries.extend(validate_label_file(label_path, len(class_names)))
            if len(invalid_entries) >= 20:
                invalid_entries = invalid_entries[:20]
                break

        if missing_labels:
            hard_failures.append(
                f"{split} split has {len(missing_labels)} images without labels: {missing_labels[:10]}"
            )
        if invalid_entries:
            hard_failures.append(
                f"{split} split has invalid label rows, sample: {invalid_entries[:10]}"
            )
        if extra_labels:
            message = (
                f"{split} split has {len(extra_labels)} unmatched label files that do not map to images: "
                f"{extra_labels[:10]}"
            )
            if strict_data:
                hard_failures.append(message)
            else:
                warnings.append(message)

        split_reports.append(
            SplitValidation(
                split=split,
                image_dir=str(image_dir),
                label_dir=str(label_dir),
                image_count=len(image_stems),
                label_count=len(label_stems),
                matched_pairs=len(image_stems & label_stems),
                missing_labels=missing_labels[:20],
                extra_labels=extra_labels[:20],
                invalid_label_entries=invalid_entries[:20],
            )
        )

    if hard_failures:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(hard_failures))

    return DatasetValidationReport(
        dataset_root=str(dataset_root),
        data_yaml=str(data_yaml_path.resolve()),
        class_names=class_names,
        split_reports=split_reports,
        warnings=warnings,
    )


def build_baseline_train_args(args: argparse.Namespace) -> dict[str, Any]:
    train_args: dict[str, Any] = {
        "data": str(DATA_YAML_PATH.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str((EXPERIMENT_ROOT / "step1_baseline").resolve()),
        "name": args.run_name,
        "exist_ok": False,
        "pretrained": True,
        "seed": 42,
        "deterministic": True,
        "verbose": True,
    }
    if args.device:
        train_args["device"] = args.device
    return train_args


def export_plan(stage: str, train_args: dict[str, Any], report: DatasetValidationReport) -> Path:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = PLANS_DIR / f"{stage}_{train_args['name']}.json"
    plan_payload = {
        "stage": stage,
        "model": MODEL_NAME,
        "data_yaml": str(DATA_YAML_PATH.resolve()),
        "train_args": train_args,
        "dataset_report": asdict(report),
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path


def print_dataset_report(report: DatasetValidationReport) -> None:
    print("Dataset validation passed.")
    print(f"Data YAML : {report.data_yaml}")
    print(f"Dataset   : {report.dataset_root}")
    print(f"Classes   : {', '.join(report.class_names)}")
    for split_report in report.split_reports:
        print(
            f"[{split_report.split}] images={split_report.image_count}, "
            f"labels={split_report.label_count}, matched={split_report.matched_pairs}"
        )
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")


def find_metric_value(row: dict[str, str], fragments: list[str]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for key, value in lowered.items():
        if all(fragment in key for fragment in fragments):
            return value
    return ""


def append_summary(stage: str, train_args: dict[str, Any], save_dir: Path) -> Path | None:
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        return None

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    last_row = rows[-1]
    summary_row = {
        "stage": stage,
        "run_name": train_args["name"],
        "model": MODEL_NAME,
        "data_yaml": str(DATA_YAML_PATH.resolve()),
        "epochs": str(train_args["epochs"]),
        "imgsz": str(train_args["imgsz"]),
        "batch": str(train_args["batch"]),
        "save_dir": str(save_dir),
        "precision": find_metric_value(last_row, ["precision"]),
        "recall": find_metric_value(last_row, ["recall"]),
        "mAP50": find_metric_value(last_row, ["map50("]),
        "mAP50-95": find_metric_value(last_row, ["map50-95("]),
    }

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV_PATH.exists()
    with SUMMARY_CSV_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    return SUMMARY_CSV_PATH


def run_baseline(args: argparse.Namespace) -> None:
    report = validate_dataset(DATA_YAML_PATH, strict_data=args.strict_data)
    train_args = build_baseline_train_args(args)
    plan_path = export_plan("step1-baseline", train_args, report)

    print_dataset_report(report)
    print(f"Model     : {MODEL_NAME}")
    print(f"Plan file : {plan_path}")
    print("Train args:")
    print(json.dumps(train_args, indent=2, ensure_ascii=False))

    if not args.execute:
        print(
            "\nDry-run only. Add `--execute` after your confirmation to actually start "
            "the Step 1 baseline training."
        )
        return

    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)
    result = model.train(**train_args)
    save_dir = Path(result.save_dir)
    summary_path = append_summary("step1-baseline", train_args, save_dir)
    print(f"\nTraining finished. Output directory: {save_dir}")
    if summary_path:
        print(f"Summary updated: {summary_path}")
    else:
        print("Training finished, but no results.csv summary could be extracted.")


def main() -> None:
    args = parse_args()
    if args.stage == "step1-baseline":
        run_baseline(args)
        return

    raise NotImplementedError(
        f"{args.stage} is intentionally not enabled yet. "
        "Complete Step 1 baseline first, then proceed to the next stage."
    )


if __name__ == "__main__":
    main()
