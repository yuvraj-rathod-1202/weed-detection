#!/usr/bin/env python
"""Evaluate a trained run on a split and save every shot, for choosing the threshold offline.

Writes <out>/eval_<split>/shots.jsonl and summary.json (see src/split_eval.py). Resumable.

Usage:
    python scripts/evaluate_split.py --run experiments/yolo11s_seg_cropandweed_1280_aug --split val \
        --prepared-dir /tmp/cropandweed_seg
    python scripts/evaluate_split.py --run models/released/yolo11s_seg_cropandweed_1280_aug --split val \
        --prepared-dir /tmp/cropandweed_seg --out experiments/yolo11s_seg_cropandweed_1280_aug
    python scripts/evaluate_split.py --baseline --run experiments/baseline_exg --split val \
        --prepared-dir /tmp/cropandweed_seg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline import baseline_prediction
from src.split_eval import evaluate_split
from src.utils.io import load_json, save_yaml

BASELINE_CONFIG = {
    "name": "baseline_exg",
    "description": "ExG - ExR vegetation filter; every green blob is a weed (src/baseline.py)",
    "eval": {"imgsz": None, "iou": None, "match_iou": 0.5},
    "targeting": {"crop_classes": ["Crop"], "crop_margin_px": 20},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run folder with config.yaml and best.pt (or train/weights/best.pt)")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--prepared-dir", required=True, help="Prepared YOLO-seg dataset (prepare_data.py output)")
    parser.add_argument("--conf-floor", type=float, default=0.05, help="Lowest confidence saved (default 0.05)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output folder (default: --run)")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the excess-green baseline; --run is its output folder")
    args = parser.parse_args()

    predictor = None
    if args.baseline:
        Path(args.run).mkdir(parents=True, exist_ok=True)
        save_yaml(BASELINE_CONFIG, Path(args.run) / "config.yaml")
        predictor = baseline_prediction
    eval_dir = evaluate_split(Path(args.run), args.split, Path(args.prepared_dir), args.conf_floor, args.max_images,
                              predictor=predictor, out_dir=Path(args.out) if args.out else None)
    summary = load_json(eval_dir / "summary.json")
    print(f"{summary['run']} on {args.split}: {summary['num_images']} images -> {eval_dir}")
    if args.split == "test":
        print("test: report it only at the threshold chosen on validation (results/final_selection.json)")
    elif args.baseline:
        print("baseline: one operating point (every blob fires), see curve[0]:", summary["curve"][0])
    else:
        best = summary["optimal"]
        print("optimal (<= 2 % crop shots):", "none" if best is None else
              f"threshold {best['threshold']:.2f}, weed hit rate {best['weed_hit_rate']:.3f}, crop shots {best['crop_shot_share']:.3f}")


if __name__ == "__main__":
    main()
