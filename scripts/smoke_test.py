#!/usr/bin/env python
"""Smoke test: run each experiment on a tiny data subset (16 images, 1 epoch, 320 px) and check
that every stage runs, every output file is written, and a re-run resumes instead of retraining.

Outputs go to experiments/_smoke/ and data/processed/_smoke/ (wiped at the start of each smoke test).

Usage:
    python scripts/smoke_test.py                       # every configs/experiments/*.yaml
    python scripts/smoke_test.py --config configs/experiments/baseline_yolo11n_seg.yaml
    python scripts/smoke_test.py --override dataset.raw_dir=/kaggle/input/cottonweeddet12
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_dir_for, run_experiment
from src.utils.config import load_experiment_config

SMOKE_OUTPUT_ROOT = Path("experiments/_smoke")
SMOKE_DATA_ROOT = Path("data/processed/_smoke")
SMOKE_OVERRIDES = [
    f"output_root={SMOKE_OUTPUT_ROOT.as_posix()}",
    "dataset.max_images=16",
    "train.epochs=1",
    "train.imgsz=320",
    "train.batch=4",
    "train.workers=0",
    "train.plots=false",
    "eval.imgsz=320",
    "eval.conf=0.001",  # a 1-epoch model is unconfident; keep predictions so point extraction/matching run
    "eval.num_visualizations=2",
]
EXPECTED_FILES = [
    "config.yaml",
    "environment.json",
    "run.log",
    "train/weights/last.pt",
    "train/weights/best.pt",
    "train_summary.json",
    "eval/segmentation_metrics.json",
    "eval/predictions.jsonl",
    "results.json",
]
EXPECTED_RESULT_KEYS = ["experiment", "dataset", "model", "training", "segmentation", "targeting", "environment"]


def smoke_one(config_path: Path, extra_overrides: list[str]) -> list[str]:
    """Run one experiment in smoke mode. Returns a list of problems (empty = passed)."""
    cfg = load_experiment_config(config_path, SMOKE_OVERRIDES + extra_overrides)
    cfg.dataset.prepared_dir = (SMOKE_DATA_ROOT / cfg.dataset.name).as_posix()
    run_dir = run_dir_for(cfg)

    results = run_experiment(cfg)

    problems = [f"missing output: {f}" for f in EXPECTED_FILES if not (run_dir / f).exists()]
    problems += [f"results.json has no '{k}'" for k in EXPECTED_RESULT_KEYS if k not in results]
    problems += [
        f"no targeting metrics for method '{m}'"
        for m in cfg.targeting.methods
        if m not in results.get("targeting", {}).get("methods", {})
    ]

    # Auto-save / resume check: a second run must reuse every saved stage, not retrain.
    last_pt = run_dir / "train" / "weights" / "last.pt"
    mtime = last_pt.stat().st_mtime if last_pt.exists() else None
    start = time.time()
    run_experiment(cfg)
    if last_pt.exists() and last_pt.stat().st_mtime != mtime:
        problems.append("re-run retrained the model instead of resuming")
    print(f"  resume check: re-run took {time.time() - start:.1f}s")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="*", help="Experiment configs (default: all in configs/experiments/)")
    parser.add_argument("--override", nargs="*", default=[], help="Extra dotlist overrides, applied after the smoke ones")
    args = parser.parse_args()

    configs = [Path(c) for c in args.config] if args.config else sorted(Path("configs/experiments").glob("*.yaml"))
    shutil.rmtree(SMOKE_OUTPUT_ROOT, ignore_errors=True)
    shutil.rmtree(SMOKE_DATA_ROOT, ignore_errors=True)

    summary = {}
    for config_path in configs:
        print(f"\n===== SMOKE TEST: {config_path} =====")
        try:
            summary[config_path] = smoke_one(config_path, args.override)
        except Exception:  # noqa: BLE001 - report and continue with the next experiment
            summary[config_path] = [traceback.format_exc()]

    print("\n===== SMOKE TEST SUMMARY =====")
    for config_path, problems in summary.items():
        print(f"[{'PASS' if not problems else 'FAIL'}] {config_path}")
        for problem in problems:
            print(f"    - {problem}")
    sys.exit(0 if all(not p for p in summary.values()) else 1)


if __name__ == "__main__":
    main()
