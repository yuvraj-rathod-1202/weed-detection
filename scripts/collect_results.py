#!/usr/bin/env python
"""Copy every result the website reports into results/runs/<run>/ (experiments/ is not version-controlled).

Per run: config.yaml, train_results.csv (per-epoch losses and validation metrics), train_summary.json,
results.json (development-time evaluation on test), and for the final protocol eval_val/ + eval_test/
(summary.json and shots.jsonl: every predicted weed with its confidence and which labelled weed or crop its aim
point landed on). Nothing here contains image pixels; the public shots drop the aim points and keep only a size
bin per labelled weed (src/site_data.py public_record), so no CropAndWeed annotation can be rebuilt from them.
Development-time predictions.jsonl files are left out: they hold target points computed from CropAndWeed's own
labels (the dataset licence forbids redistributing its annotations).

Usage: python scripts/collect_results.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.site_data import public_record
from src.utils.io import read_jsonl

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
            eval_src, eval_out = src / f"eval_{split}", out / f"eval_{split}"
            if not (eval_src / "summary.json").exists():
                continue
            eval_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(eval_src / "summary.json", eval_out / "summary.json")
            records = [public_record(r) for r in read_jsonl(eval_src / "shots.jsonl")]
            with open(eval_out / "shots.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        print(run, "->", out, sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
