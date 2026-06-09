from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp_manager import find_metric_value


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT3_DIR = PROJECT_ROOT / "experiments" / "experiment3"
SUPPORTED_GROUPS = ("g1", "g2", "g3", "g4", "g5", "final_compose")
SUPPORTED_OBJECTIVES = ("map50", "map50_95")


@dataclass(frozen=True)
class RunRecord:
    group: str
    objective: str
    config_name: str
    run_name: str
    run_dir: str
    results_csv: str
    params: dict[str, Any]
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Step 3 runs under experiments/experiment3 and recommend best joint config per group."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_EXPERIMENT3_DIR),
        help="Experiment3 directory path (default: experiments/experiment3).",
    )
    parser.add_argument(
        "--out-prefix",
        default="step3_recommendations",
        help="Output filename prefix under experiment3 root.",
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


def collect_records(experiment3_dir: Path) -> list[RunRecord]:
    plans_dir = experiment3_dir / "plans"
    plan_paths = sorted(plans_dir.glob("*.json")) if plans_dir.exists() else []

    records: list[RunRecord] = []
    for plan_path in plan_paths:
        payload = read_json(plan_path)
        group = payload.get("group")
        objective = payload.get("objective")
        config_name = payload.get("config_name")
        params = payload.get("params")
        train_args = payload.get("train_args")

        if group not in SUPPORTED_GROUPS or objective not in SUPPORTED_OBJECTIVES:
            continue
        if not isinstance(config_name, str) or not config_name:
            continue
        if not isinstance(params, dict):
            continue
        if not isinstance(train_args, dict):
            continue

        run_name = train_args.get("name")
        if not isinstance(run_name, str) or not run_name:
            continue

        run_dir = choose_run_dir(experiment3_dir, run_name)
        if not run_dir:
            continue

        results_csv = run_dir / "results.csv"
        precision, recall, map50, map50_95 = read_last_metrics(results_csv)
        records.append(
            RunRecord(
                group=group,
                objective=objective,
                config_name=config_name,
                run_name=run_name,
                run_dir=str(run_dir),
                results_csv=str(results_csv),
                params=params,
                precision=precision,
                recall=recall,
                map50=map50,
                map50_95=map50_95,
            )
        )
    return records


def pick_best(records: list[RunRecord]) -> dict[str, dict[str, RunRecord]]:
    best_by_objective: dict[str, dict[str, RunRecord]] = {objective: {} for objective in SUPPORTED_OBJECTIVES}
    for record in records:
        score = record.map50 if record.objective == "map50" else record.map50_95
        if score is None:
            continue
        current = best_by_objective[record.objective].get(record.group)
        if current is None:
            best_by_objective[record.objective][record.group] = record
            continue
        current_score = current.map50 if record.objective == "map50" else current.map50_95
        if current_score is None or score > current_score:
            best_by_objective[record.objective][record.group] = record
    return best_by_objective


def write_outputs(
    experiment3_dir: Path,
    out_prefix: str,
    records: list[RunRecord],
    best_by_objective: dict[str, dict[str, RunRecord]],
) -> tuple[Path, Path]:
    out_json = experiment3_dir / f"{out_prefix}.json"
    out_csv = experiment3_dir / f"{out_prefix}.csv"

    payload = {
        "recommendations": {
            objective: {
                group: {
                    "config_name": best.config_name,
                    "run_name": best.run_name,
                    "run_dir": best.run_dir,
                    "params": best.params,
                    "precision": best.precision,
                    "recall": best.recall,
                    "map50": best.map50,
                    "map50_95": best.map50_95,
                }
                for group, best in sorted(best_by_objective[objective].items(), key=lambda item: item[0])
            }
            for objective in SUPPORTED_OBJECTIVES
        },
        "runs": [
            {
                **record.__dict__,
                "params": record.params,
            }
            for record in records
        ],
    }
    experiment3_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict[str, str]] = []
    for objective in SUPPORTED_OBJECTIVES:
        for group, best in sorted(best_by_objective[objective].items(), key=lambda item: item[0]):
            rows.append(
                {
                    "objective": objective,
                    "group": group,
                    "config_name": best.config_name,
                    "run_name": best.run_name,
                    "run_dir": best.run_dir,
                    "params_json": json.dumps(best.params, ensure_ascii=False, sort_keys=True),
                    "precision": "" if best.precision is None else str(best.precision),
                    "recall": "" if best.recall is None else str(best.recall),
                    "map50": "" if best.map50 is None else str(best.map50),
                    "map50_95": "" if best.map50_95 is None else str(best.map50_95),
                }
            )
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["objective", "group"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_json, out_csv


def main() -> None:
    args = parse_args()
    experiment3_dir = Path(args.root).expanduser().resolve()
    records = collect_records(experiment3_dir)
    best_by_objective = pick_best(records)

    print(f"Experiment3: {experiment3_dir}")
    print(f"Runs found : {len(records)}")
    if not records:
        print("No runs found. Ensure experiment3 contains run folders with results.csv and plans/*.json.")
        return

    for objective in SUPPORTED_OBJECTIVES:
        print(f"\nObjective: {objective}")
        for group in SUPPORTED_GROUPS:
            best = best_by_objective[objective].get(group)
            if not best:
                continue
            score = best.map50 if objective == "map50" else best.map50_95
            print(f"{group}: {best.config_name} ({objective}={score})  dir={best.run_dir}")

    out_json, out_csv = write_outputs(
        experiment3_dir=experiment3_dir,
        out_prefix=args.out_prefix,
        records=records,
        best_by_objective=best_by_objective,
    )
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
