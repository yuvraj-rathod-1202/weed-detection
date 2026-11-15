#!/usr/bin/env python
"""Create the YOLO-seg dataset (SAM masks from GT boxes + train/val/test split).

run_experiment.py does this automatically; run it on its own to inspect the data first
(notebooks/01_data_exploration.ipynb). Resumable: re-run to continue after an interruption.

Usage:
    python scripts/prepare_data.py --dataset configs/datasets/cottonweeddet12.yaml
    python scripts/prepare_data.py --dataset configs/datasets/cottonweeddet12.yaml --override raw_dir=/kaggle/input/cwd12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from src.data.prepare import prepare_dataset
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset config YAML (configs/datasets/*.yaml)")
    parser.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. max_images=50")
    args = parser.parse_args()

    ds_cfg = OmegaConf.merge(load_yaml(args.dataset), OmegaConf.from_dotlist(args.override))
    prepare_dataset(ds_cfg)


if __name__ == "__main__":
    main()
