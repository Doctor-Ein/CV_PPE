from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from exp_manager import DATA_YAML_PATH, read_yaml, resolve_dataset_root


PROJECT_ROOT = Path(__file__).resolve().parent


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: qualitative visual comparison on test images.\n"
            "Compares 4 models: raw, baseline, best single-group joint result, and final compose."
        )
    )
    parser.add_argument("--n", type=int, default=5, help="Number of random test images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="", help="Device string passed to Ultralytics.")
    parser.add_argument("--raw-weights", default="yolov8n.pt", help="Raw YOLOv8n weights.")

    parser.add_argument(
        "--baseline-weights",
        default="",
        help="Baseline weights path. If empty, auto-detect latest best.pt under experiments/step1_baseline.",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(PROJECT_ROOT / "experiments" / "step1_baseline"),
        help="Baseline experiment root used for auto-detection.",
    )

    parser.add_argument(
        "--joint-weights",
        default="",
        help="Single-group joint weights path. If empty, auto-detect from step3 recommendations or experiment3 outputs.",
    )
    parser.add_argument(
        "--final-weights",
        default="",
        help="Final compose weights path. If empty, auto-detect from step3 recommendations or experiment3 outputs.",
    )
    parser.add_argument(
        "--step3-root",
        default=str(PROJECT_ROOT / "experiments" / "experiment3"),
        help="Step 3 experiment root.",
    )
    parser.add_argument(
        "--objective",
        default="map50",
        choices=["map50", "map50_95"],
        help="Objective used when selecting joint and final compose runs from step3 recommendations.",
    )
    parser.add_argument(
        "--joint-group",
        default="overall",
        choices=["overall", "g1", "g2", "g3", "g4", "g5"],
        help="Which single group to use from step3 recommendations. 'overall' picks best among g1-g5 only.",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Output directory. Default: experiments/visual_compare/<timestamp>.",
    )
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    images.sort()
    if not images:
        raise ValueError(f"No images found under: {image_dir}")
    return images


def find_latest_matching(root: Path, relative_pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = list(root.glob(relative_pattern))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_baseline_weights(baseline_weights: str, baseline_root: Path) -> Path:
    if baseline_weights:
        return Path(baseline_weights).expanduser().resolve()
    detected = find_latest_matching(baseline_root, "**/weights/best.pt")
    if detected:
        return detected
    raise FileNotFoundError(
        "Could not auto-detect baseline weights. Provide --baseline-weights "
        "or ensure experiments/step1_baseline contains */weights/best.pt."
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return payload


def pick_best_joint_from_recommendations(
    step3_root: Path, objective: str, group: str
) -> Path | None:
    rec_path = step3_root / "step3_recommendations.json"
    if not rec_path.exists():
        return None
    payload = read_json(rec_path)
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, dict):
        return None

    objective_map = recommendations.get(objective)
    if not isinstance(objective_map, dict):
        return None

    def score_from(entry: Any) -> float | None:
        if not isinstance(entry, dict):
            return None
        key = "map50" if objective == "map50" else "map50_95"
        value = entry.get(key)
        try:
            return float(value)
        except Exception:
            return None

    if group != "overall":
        entry = objective_map.get(group)
        if not isinstance(entry, dict):
            return None
        run_dir = entry.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            return None
        return Path(run_dir).expanduser().resolve()

    best_dir: Path | None = None
    best_score: float | None = None
    for group_key, entry in objective_map.items():
        if group_key not in {"g1", "g2", "g3", "g4", "g5"}:
            continue
        score = score_from(entry)
        run_dir = entry.get("run_dir") if isinstance(entry, dict) else None
        if score is None or not isinstance(run_dir, str) or not run_dir:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_dir = Path(run_dir).expanduser().resolve()
    return best_dir


def resolve_joint_weights(joint_weights: str, step3_root: Path, objective: str, group: str) -> Path:
    if joint_weights:
        return Path(joint_weights).expanduser().resolve()
    best_dir = pick_best_joint_from_recommendations(step3_root, objective=objective, group=group)
    if best_dir:
        best = best_dir / "weights" / "best.pt"
        if best.exists():
            return best.resolve()
        last = best_dir / "weights" / "last.pt"
        if last.exists():
            return last.resolve()
    detected = find_latest_matching(step3_root, "**/weights/best.pt")
    if detected:
        return detected
    raise FileNotFoundError(
        "Could not auto-detect joint weights. Provide --joint-weights or "
        "ensure experiments/experiment3 contains */weights/best.pt."
    )


def resolve_final_compose_weights(final_weights: str, step3_root: Path, objective: str) -> Path:
    if final_weights:
        return Path(final_weights).expanduser().resolve()
    best_dir = pick_best_joint_from_recommendations(step3_root, objective=objective, group="final_compose")
    if best_dir:
        best = best_dir / "weights" / "best.pt"
        if best.exists():
            return best.resolve()
        last = best_dir / "weights" / "last.pt"
        if last.exists():
            return last.resolve()
    detected = find_latest_matching(step3_root, "**/joint_final_compose_*/weights/best.pt")
    if detected:
        return detected
    raise FileNotFoundError(
        "Could not auto-detect final compose weights. Provide --final-weights or "
        "ensure experiments/experiment3 contains joint_final_compose_*/weights/best.pt."
    )


def find_saved_prediction_image(save_dir: Path, source: Path) -> Path | None:
    candidates: list[Path] = []
    for suffix in IMAGE_SUFFIXES | {".jpg"}:
        candidates.extend(save_dir.glob(f"{source.stem}{suffix}"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        candidates = [p for p in save_dir.glob("*") if p.is_file() and p.stem == source.stem]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_predict(weight_path: Path, images: list[Path], outdir: Path, tag: str, args: argparse.Namespace) -> dict[str, str]:
    from ultralytics import YOLO

    model = YOLO(str(weight_path))
    save_dir = outdir / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, str] = {}
    for image_path in images:
        _ = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            save=True,
            project=str(outdir),
            name=tag,
            exist_ok=True,
            verbose=False,
        )
        saved = find_saved_prediction_image(save_dir, image_path)
        if saved:
            predictions[str(image_path)] = str(saved)
        else:
            predictions[str(image_path)] = ""
    return predictions


def write_index_html(outdir: Path, images: list[Path], mapping: dict[str, dict[str, str]]) -> Path:
    lines: list[str] = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'><title>YOLO Visual Compare</title></head>")
    lines.append("<body>")
    lines.append("<h2>Visual Compare (raw vs baseline vs single-group joint vs final compose)</h2>")
    lines.append("<table border='1' cellspacing='0' cellpadding='6'>")
    lines.append(
        "<tr><th>Image</th><th>Raw</th><th>Baseline</th><th>Joint</th><th>Final Compose</th></tr>"
    )

    def rel(path_str: str) -> str:
        if not path_str:
            return ""
        try:
            return str(Path(path_str).resolve().relative_to(outdir.resolve()))
        except Exception:
            return path_str

    for image_path in images:
        key = str(image_path)
        raw_img = rel(mapping["raw"].get(key, ""))
        base_img = rel(mapping["baseline"].get(key, ""))
        joint_img = rel(mapping["joint"].get(key, ""))
        final_img = rel(mapping["final_compose"].get(key, ""))
        lines.append("<tr>")
        lines.append(f"<td>{image_path.name}</td>")
        for item in (raw_img, base_img, joint_img, final_img):
            if item:
                lines.append(
                    f"<td><a href='{item}'><img src='{item}' style='max-width: 360px; height: auto;'></a></td>"
                )
            else:
                lines.append("<td>(missing)</td>")
        lines.append("</tr>")
    lines.append("</table>")
    lines.append("</body></html>")

    path = outdir / "index.html"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()

    yaml_data = read_yaml(DATA_YAML_PATH)
    dataset_root = resolve_dataset_root(DATA_YAML_PATH, yaml_data)
    test_dir = (dataset_root / str(yaml_data.get("test", "images/test"))).resolve()

    all_images = list_images(test_dir)
    rng = random.Random(args.seed)
    selected = rng.sample(all_images, k=min(args.n, len(all_images)))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (PROJECT_ROOT / "experiments" / "visual_compare" / timestamp)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(args.baseline_root).expanduser().resolve()
    step3_root = Path(args.step3_root).expanduser().resolve()

    raw_weights = Path(args.raw_weights).expanduser().resolve() if Path(args.raw_weights).suffix else Path(args.raw_weights)
    baseline_weights = resolve_baseline_weights(args.baseline_weights, baseline_root)
    joint_weights = resolve_joint_weights(args.joint_weights, step3_root, objective=args.objective, group=args.joint_group)
    final_weights = resolve_final_compose_weights(args.final_weights, step3_root, objective=args.objective)

    print(f"Test dir         : {test_dir}")
    print(f"Selected images  : {len(selected)}")
    print(f"Output dir       : {outdir}")
    print(f"Raw weights      : {raw_weights}")
    print(f"Baseline weights : {baseline_weights}")
    print(f"Joint weights    : {joint_weights}")
    print(f"Final weights    : {final_weights}")

    mapping: dict[str, dict[str, str]] = {}
    mapping["raw"] = run_predict(raw_weights, selected, outdir, "raw", args)
    mapping["baseline"] = run_predict(baseline_weights, selected, outdir, "baseline", args)
    mapping["joint"] = run_predict(joint_weights, selected, outdir, "joint", args)
    mapping["final_compose"] = run_predict(final_weights, selected, outdir, "final_compose", args)

    meta = {
        "test_dir": str(test_dir),
        "selected_images": [str(p) for p in selected],
        "outdir": str(outdir),
        "weights": {
            "raw": str(raw_weights),
            "baseline": str(baseline_weights),
            "joint": str(joint_weights),
            "final_compose": str(final_weights),
        },
        "objective": args.objective,
        "joint_group": args.joint_group,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "outputs": mapping,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    index_path = write_index_html(outdir, selected, mapping)
    print(f"Wrote: {index_path}")


if __name__ == "__main__":
    main()
