#!/usr/bin/env python
"""Export the website's data files (site/data/*.json). No image pixels, only statistics and model outputs.

    dataset.json   CropAndWeed statistics per split (from the official annotations + the training split rule)

Usage: python scripts/site/export_site_data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.prepare import split_images
from src.site_data import dataset_stats
from src.utils.io import load_yaml_dict, save_json


def cropandweed_splits(bbox_dir: Path, dataset_cfg: Path) -> dict[str, list[str]]:
    """The exact training split: same rule, ratios, seed and session grouping as prepare_data.py."""
    cfg = load_yaml_dict(dataset_cfg)
    names = [Path(p.stem + ".jpg") for p in sorted(bbox_dir.glob("*.csv"))]
    splits = split_images(names, cfg["split"], cfg["split_seed"], cfg.get("max_images"), cfg.get("split_group_pattern"))
    return {s: [p.stem for p in paths] for s, paths in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/raw/cropandweed_src", help="Official CropAndWeed folder (bboxes/)")
    parser.add_argument("--dataset-config", default="configs/datasets/cropandweed.yaml")
    parser.add_argument("--out", default="site/data")
    args = parser.parse_args()
    out = Path(args.out)

    bbox_dir = Path(args.source) / "bboxes"
    splits = cropandweed_splits(bbox_dir, Path(args.dataset_config))
    stats = dataset_stats(bbox_dir, splits)
    save_json(stats, out / "dataset.json")
    print("dataset.json:", {s: v["images"] for s, v in stats["splits"].items()})


if __name__ == "__main__":
    main()
