#!/usr/bin/env python
"""Copy every result the website reports into results/runs/<run>/ (experiments/ is not version-controlled).

Per run: config.yaml, train_results.csv (per-epoch losses and validation metrics), train_summary.json,
results.json (development-time evaluation on test), and for the final protocol eval_val/ + eval_test/
(summary.json and shots.jsonl: every predicted weed with its confidence and where its aim point landed).
Nothing here contains image pixels. Development-time predictions.jsonl files are left out: they hold target
points computed from CropAndWeed's own labels (the dataset licence forbids redistributing its annotations).

Usage: python scripts/collect_results.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

RUNS = ["baseline_yolo11n_seg", "yolo11s_seg", "yolo11s_seg_cropandweed", "yolo11s_seg_cropandweed_1280",
        "yolo11s_seg_cropandweed_1280_aug", "baseline_exg"]
FILES = {"config.yaml": "config.yaml", "train/results.csv": "train_results.csv", "train_summary.json": "train_summary.json",
         "results.json": "results.json", "eval/segmentation_metrics.json": "segmentation_metrics.json"}


def main() -> None:
    out_root = Path("results/runs")
    for run in RUNS:
        src, out = Path("experiments") / run, out_root / run
        for name, target in FILES.items():
            if (src / name).exists():
                out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / name, out / target)
        for split in ("val", "test"):
            for name in ("summary.json", "shots.jsonl"):
                if (src / f"eval_{split}" / name).exists():
                    (out / f"eval_{split}").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src / f"eval_{split}" / name, out / f"eval_{split}" / name)
        print(run, "->", out, sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
