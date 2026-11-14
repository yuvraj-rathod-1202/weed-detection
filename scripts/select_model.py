#!/usr/bin/env python
"""Pick the final model and threshold from the candidates' VALIDATION summaries (src/selection.py).

Writes results/final_selection.json. Evaluate only the selected run on test afterwards.

Usage:
    python scripts/select_model.py experiments/yolo11s_seg_cropandweed experiments/yolo11s_seg_cropandweed_1280 \
        experiments/yolo11s_seg_cropandweed_1280_aug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.selection import select_final
from src.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run folders, each with eval_val/summary.json")
    parser.add_argument("--out", default="results/final_selection.json")
    args = parser.parse_args()

    summaries = {Path(r).name: load_json(Path(r) / "eval_val" / "summary.json") for r in args.runs}
    for run, s in summaries.items():
        best = s["optimal"]
        print(f"{run:40s}", "no threshold within the 2 % crop limit" if best is None else
              f"threshold {best['threshold']:.2f}  weeds hit {best['weed_hit_rate']:.3f}  crop shots {best['crop_shot_share']:.3f}"
              f"  {s['inference_ms_per_image']:.1f} ms/img")
    chosen = select_final(summaries)
    if chosen is None:
        sys.exit("No candidate meets the 2 % crop limit on validation.")
    save_json({**chosen, "candidates": list(summaries), "rule": "max weed hit rate with <= 2 % crop shots (validation)"},
              Path(args.out))
    print(f"\nSelected: {chosen['run']} at threshold {chosen['threshold']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
