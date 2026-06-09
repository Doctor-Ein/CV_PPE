from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp_manager import find_metric_value


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT2_DIR = PROJECT_ROOT / "experiments" / "experiment2"

SUPPORTED_PARAMS = ("hsv_h", "hsv_s", "hsv_v", "bgr")


@dataclass(frozen=True)
class RunRecord:
    param: str
    value: float
    run_name: str
    run_dir: str
    results_csv: str
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Step 2 runs under experiments/experiment2 and recommend best values per param."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_EXPERIMENT2_DIR),
        help="Experiment2 directory path (default: experiments/experiment2).",
    )
    parser.add_argument(
        "--metric",
        default="map50",
        choices=["map50", "map50_95"],
        help="Primary metric used for selecting best value.",
    )
    parser.add_argument(
        "--out-prefix",
        default="step2_recommendations",
        help="Output filename prefix (written under experiment2 root).",
    )
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


def find_param_value(train_args: dict[str, Any]) -> tuple[str, float] | None:
    found: list[tuple[str, float]] = []
    for key in SUPPORTED_PARAMS:
        if key in train_args:
            value = to_float(train_args.get(key))
            if value is None:
                continue
            found.append((key, value))
    if len(found) != 1:
        return None
    return found[0]


def choose_run_dir(root: Path, run_name: str) -> Path | None:
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "plans":
            continue
        if not child.name.startswith(run_name):
            continue
        if (child / "results.csv").exists():
            candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def read_last_metrics(results_csv: Path) -> tuple[float | None, float | None, float | None, float | None]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, None, None, None
    last = rows[-1]
    precision = to_float(find_metric_value(last, ["precision"]))
    recall = to_float(find_metric_value(last, ["recall"]))
    map50 = to_float(find_metric_value(last, ["map50("]))
    map50_95 = to_float(find_metric_value(last, ["map50-95("]))
    return precision, recall, map50, map50_95


def collect_records(experiment2_dir: Path) -> list[RunRecord]:
    plans_dir = experiment2_dir / "plans"
    plan_paths = sorted(plans_dir.glob("*.json")) if plans_dir.exists() else []

    records: list[RunRecord] = []
    for plan_path in plan_paths:
        payload = read_json(plan_path)
        train_args = payload.get("train_args")
        if not isinstance(train_args, dict):
            continue

        run_name = train_args.get("name")
        if not isinstance(run_name, str) or not run_name:
            continue

        param_value = find_param_value(train_args)
        if not param_value:
            continue
        param, value = param_value

        run_dir = choose_run_dir(experiment2_dir, run_name)
        if not run_dir:
            continue

        results_csv = run_dir / "results.csv"
        precision, recall, map50, map50_95 = read_last_metrics(results_csv)
        records.append(
            RunRecord(
                param=param,
                value=value,
                run_name=run_name,
                run_dir=str(run_dir),
                results_csv=str(results_csv),
                precision=precision,
                recall=recall,
                map50=map50,
                map50_95=map50_95,
            )
        )

    return records


def pick_best(records: list[RunRecord], metric: str) -> dict[str, RunRecord]:
    best_by_param: dict[str, RunRecord] = {}
    for record in records:
        score = record.map50 if metric == "map50" else record.map50_95
        if score is None:
            continue
        current = best_by_param.get(record.param)
        if current is None:
            best_by_param[record.param] = record
            continue
        current_score = current.map50 if metric == "map50" else current.map50_95
        if current_score is None or score > current_score:
            best_by_param[record.param] = record
    return best_by_param


def write_outputs(
    experiment2_dir: Path,
    out_prefix: str,
    metric: str,
    records: list[RunRecord],
    best_by_param: dict[str, RunRecord],
) -> tuple[Path, Path]:
    out_json = experiment2_dir / f"{out_prefix}.json"
    out_csv = experiment2_dir / f"{out_prefix}.csv"

    payload = {
        "metric": metric,
        "recommendations": {
            param: {
                "value": best.value,
                "run_name": best.run_name,
                "run_dir": best.run_dir,
                "precision": best.precision,
                "recall": best.recall,
                "map50": best.map50,
                "map50_95": best.map50_95,
            }
            for param, best in sorted(best_by_param.items(), key=lambda item: item[0])
        },
        "runs": [record.__dict__ for record in records],
    }
    experiment2_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for param, best in sorted(best_by_param.items(), key=lambda item: item[0]):
        rows.append(
            {
                "param": param,
                "recommended_value": str(best.value),
                "run_name": best.run_name,
                "run_dir": best.run_dir,
                "precision": "" if best.precision is None else str(best.precision),
                "recall": "" if best.recall is None else str(best.recall),
                "map50": "" if best.map50 is None else str(best.map50),
                "map50_95": "" if best.map50_95 is None else str(best.map50_95),
            }
        )
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["param"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_json, out_csv


def main() -> None:
    args = parse_args()
    experiment2_dir = Path(args.root).expanduser().resolve()
    records = collect_records(experiment2_dir)
    best_by_param = pick_best(records, metric=args.metric)

    print(f"Experiment2: {experiment2_dir}")
    print(f"Runs found : {len(records)}")
    print(f"Metric     : {args.metric}")
    if not records:
        print("No runs found. Ensure experiment2 contains run folders with results.csv and plans/*.json.")
        return

    for param in SUPPORTED_PARAMS:
        best = best_by_param.get(param)
        if not best:
            continue
        score = best.map50 if args.metric == "map50" else best.map50_95
        print(f"{param}: recommend {best.value} ({args.metric}={score})  dir={best.run_dir}")

    out_json, out_csv = write_outputs(
        experiment2_dir=experiment2_dir,
        out_prefix=args.out_prefix,
        metric=args.metric,
        records=records,
        best_by_param=best_by_param,
    )
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

