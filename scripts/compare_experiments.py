#!/usr/bin/env python
"""Collect <experiments-root>/*/results.json into one flat table + two summary plots.

Writes:
    results/comparison.csv / comparison.json   one row per experiment
    results/plots/targeting_methods.png        shot precision & weed hit rate per targeting method
    results/plots/accuracy_vs_latency.png      mask mAP50-95 vs end-to-end latency

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --experiments-root experiments/_smoke --output-dir results/_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import load_json, save_json


def flatten_results(r: dict) -> dict:
    row = {
        "experiment": r["experiment"]["name"],
        "dataset": r["dataset"]["name"],
        "model": r["model"]["init_weights"],
        "mask_map50": r["segmentation"]["mask"]["map50"],
        "mask_map50_95": r["segmentation"]["mask"]["map50_95"],
        "mask_precision": r["segmentation"]["mask"]["precision"],
        "mask_recall": r["segmentation"]["mask"]["recall"],
        "box_map50": r["segmentation"]["box"]["map50"],
        "box_map50_95": r["segmentation"]["box"]["map50_95"],
        "primary_method": r["targeting"]["primary_method"],
        "mean_matched_mask_iou": r["targeting"]["mean_matched_mask_iou"],
        "inference_ms": r["targeting"]["inference_ms_per_image"],
        "end_to_end_ms": r["targeting"]["end_to_end_ms_per_image"],
        "fps": r["targeting"]["fps"],
        "num_parameters": r["model"]["num_parameters"],
        "gflops": r["model"]["gflops"],
        "weights_size_mb": r["model"]["weights_size_mb"],
        "epochs_completed": r["training"]["epochs_completed"],
        "best_epoch": r["training"]["best_epoch"],
        "train_time_h": (r["training"]["train_time_s"] or 0) / 3600,
    }
    for method, m in r["targeting"]["methods"].items():
        row[f"{method}/shot_precision"] = m["shot_precision"]
        row[f"{method}/weed_hit_rate"] = m["weed_hit_rate"]
        row[f"{method}/shot_f1"] = m["shot_f1"]
        row[f"{method}/point_error_px_median"] = m["point_error_px"]["median"]
        row[f"{method}/point_error_rel_median"] = m["point_error_rel"]["median"]
        row[f"{method}/clearance_px_median"] = m.get("clearance_px", {}).get("median")
        row[f"{method}/clearance_px_p10"] = m.get("clearance_px", {}).get("p10")
        row[f"{method}/off_center_rate"] = m.get("off_center_rate")
        row[f"{method}/crop_hit_rate"] = m.get("crop_hit_rate")
        row[f"{method}/crop_hit_rate_ungated"] = m.get("crop_hit_rate_ungated")
        row[f"{method}/veto_rate"] = m.get("veto_rate")
    return row


def plot_targeting_methods(results: list[dict], path: Path) -> None:
    rows = [
        {"experiment": r["experiment"]["name"], "method": method, "shot precision": m["shot_precision"], "weed hit rate": m["weed_hit_rate"]}
        for r in results
        for method, m in r["targeting"]["methods"].items()
    ]
    df = pd.DataFrame(rows).set_index(["experiment", "method"])
    ax = df.plot(kind="barh", figsize=(8, 0.5 * len(df) + 1.5), xlim=(0, 1))
    ax.set_xlabel("fraction")
    ax.set_title("Laser targeting quality per point-extraction method")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_accuracy_vs_latency(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["end_to_end_ms"], df["mask_map50_95"], s=60)
    for _, row in df.iterrows():
        ax.annotate(row["experiment"], (row["end_to_end_ms"], row["mask_map50_95"]), fontsize=8)
    ax.set_xlabel("end-to-end latency per image (ms)")
    ax.set_ylabel("mask mAP50-95")
    ax.set_title("Accuracy vs. latency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    result_files = sorted(Path(args.experiments_root).glob("*/results.json"))
    if not result_files:
        sys.exit(f"No results.json under {args.experiments_root}/*/ - run scripts/run_experiment.py first.")
    results = [load_json(f) for f in result_files]

    output_dir = Path(args.output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([flatten_results(r) for r in results])
    df.to_csv(output_dir / "comparison.csv", index=False)
    save_json(df.to_dict(orient="records"), output_dir / "comparison.json")
    plot_targeting_methods(results, output_dir / "plots" / "targeting_methods.png")
    plot_accuracy_vs_latency(df, output_dir / "plots" / "accuracy_vs_latency.png")

    print(df.set_index("experiment").T.to_string())
    print(f"\nWrote {output_dir}/comparison.csv, comparison.json and plots/")


if __name__ == "__main__":
    main()
