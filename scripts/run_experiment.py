#!/usr/bin/env python
"""Run one experiment: prepare data -> train -> evaluate -> results.json.

Re-running the same command resumes an interrupted run (see src/experiment.py).

Usage:
    python scripts/run_experiment.py --config configs/experiments/baseline_yolo11n_seg.yaml
    python scripts/run_experiment.py --config configs/experiments/baseline_yolo11n_seg.yaml \
        --override train.batch=32 dataset.raw_dir=/kaggle/input/cottonweeddet12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment
from src.utils.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment config YAML (configs/experiments/*.yaml)")
    parser.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. train.epochs=50")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config, args.override)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
