#!/usr/bin/env python
"""Export the website's data files (site/data/*.json). No image pixels, only statistics and model outputs.

    dataset.json      CropAndWeed statistics per split (official annotations + the training split rule)
    curves.json       validation threshold sweep of every candidate + the baseline's operating point
    comparison.json   candidates and baseline on validation at their chosen thresholds (+ paired bootstraps)
    test.json         final model and baseline on test at the threshold chosen on validation (+ session CIs)
    errors.json       final model on test: hit rate by plant size / field type, crop shots by species, sessions
    training.json     training curves of the final model

Usage: python scripts/site/export_site_data.py   (reads results/runs/ and results/final_selection.json)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cropandweed import map_label
from src.data.prepare import split_images
from src.site_data import CROP_NAME, dataset_stats, hit_rate_by, paired_bootstrap_diff, size_bin
from src.targeting.sweep import bootstrap_ci, fired, sweep
from src.utils.io import load_json, load_yaml_dict, read_jsonl, save_json

CANDIDATES = {
    "yolo11s_seg_cropandweed": ("Exp 3 · 1024 px", "input 1024 px wide"),
    "yolo11s_seg_cropandweed_1280": ("Exp 4 · 1280 px", "input 1280 px: small weeds get ~1.25x more pixels"),
    "yolo11s_seg_cropandweed_1280_aug": ("Exp 5 · 1280 px + flips/rotation", "as exp 4 + vertical flips and ±15° rotation"),
}
RUNS_DIR = Path("results/runs")  # filled by scripts/collect_results.py (experiments/ is not version-controlled)
BASELINE = ("baseline_exg", "Baseline · green blobs", "ExG−ExR vegetation filter; every green blob is a weed")


def cropandweed_splits(bbox_dir: Path, dataset_cfg: Path) -> dict[str, list[str]]:
    """The exact training split: same rule, ratios, seed and session grouping as prepare_data.py."""
    cfg = load_yaml_dict(dataset_cfg)
    names = [Path(p.stem + ".jpg") for p in sorted(bbox_dir.glob("*.csv"))]
    splits = split_images(names, cfg["split"], cfg["split_seed"], cfg.get("max_images"), cfg.get("split_group_pattern"))
    return {s: [p.stem for p in paths] for s, paths in splits.items()}


def at(records: list[dict], threshold: float) -> dict:
    return sweep(records, [threshold])[0]


def best_epoch_metrics(run: str) -> dict:
    history = pd.read_csv(RUNS_DIR / run / "train_results.csv")
    history.columns = history.columns.str.strip()
    best = history.loc[history["metrics/mAP50-95(M)"].idxmax()]
    return {"epochs": int(history["epoch"].max()), "best_epoch": int(best["epoch"]),
            "val_mask_map50_95": float(best["metrics/mAP50-95(M)"]), "val_mask_map50": float(best["metrics/mAP50(M)"])}


def crop_species_of_shots(records: list[dict], threshold: float, bbox_dir: Path) -> tuple[dict, dict]:
    """(crop shots per species, test crop plants per species); GT crop index -> species via annotation order."""
    shots, plants = {}, {}
    for r in records:
        with open(bbox_dir / f"{r['image_id']}.csv", newline="", encoding="utf-8") as f:
            species = [int(row[4]) for row in csv.reader(f) if row and map_label(int(row[4])) == 0]
        if len(species) != r["num_gt_crops"]:
            print(f"warning: {r['image_id']}: {len(species)} crops in annotations vs {r['num_gt_crops']} in results, skipped")
            continue
        for sid in species:
            plants[CROP_NAME.get(sid, "Other crop")] = plants.get(CROP_NAME.get(sid, "Other crop"), 0) + 1
        for s in fired(r["shots"], threshold):
            if s["on_crop"]:
                name = CROP_NAME.get(species[s["gt_crop"]], "Other crop")
                shots[name] = shots.get(name, 0) + 1
    return shots, plants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/raw/cropandweed_src", help="Official CropAndWeed folder (bboxes/)")
    parser.add_argument("--dataset-config", default="configs/datasets/cropandweed.yaml")
    parser.add_argument("--out", default="site/data")
    parser.add_argument("--skip-annotations", action="store_true",
                        help="Without the official CropAndWeed annotations: keep the existing dataset.json and the "
                             "crop-species counts in errors.json, recompute everything else from results/")
    args = parser.parse_args()
    out, bbox_dir = Path(args.out), Path(args.source) / "bboxes"

    if args.skip_annotations:
        print("dataset.json: kept (needs the official annotations to rebuild)")
    else:
        stats = dataset_stats(bbox_dir, cropandweed_splits(bbox_dir, Path(args.dataset_config)))
        save_json(stats, out / "dataset.json")
        print("dataset.json:", {s: v["images"] for s, v in stats["splits"].items()})

    selection = load_json("results/final_selection.json")
    final, threshold = selection["run"], selection["threshold"]
    val = {run: read_jsonl(RUNS_DIR / run / "eval_val" / "shots.jsonl") for run in CANDIDATES}
    summaries = {run: load_json(RUNS_DIR / run / "eval_val" / "summary.json") for run in CANDIDATES}
    base_val = read_jsonl(RUNS_DIR / BASELINE[0] / "eval_val" / "shots.jsonl")
    base_val_summary = load_json(RUNS_DIR / BASELINE[0] / "eval_val" / "summary.json")

    save_json({
        "selected": final, "threshold": threshold, "rule": selection["rule"], "max_crop_share": 0.02,
        "runs": {run: {"label": CANDIDATES[run][0], "curve": summaries[run]["curve"], "optimal": summaries[run]["optimal"]}
                 for run in CANDIDATES},
        "baseline": {"label": BASELINE[1], **base_val_summary["curve"][0]},
    }, out / "curves.json")

    rows = []
    for run, (label, change) in CANDIDATES.items():
        opt = summaries[run]["optimal"]
        rows.append({"run": run, "label": label, "change": change, "selected": run == final, **opt,
                     "inference_ms": summaries[run]["inference_ms_per_image"], **best_epoch_metrics(run),
                     "vs_selected": None if run == final else
                     paired_bootstrap_diff(val[final], val[run], threshold, opt["threshold"])})
    rows.append({"run": BASELINE[0], "label": BASELINE[1], "change": BASELINE[2], "selected": False,
                 **base_val_summary["curve"][0], "inference_ms": base_val_summary["inference_ms_per_image"],
                 "vs_selected": paired_bootstrap_diff(val[final], base_val, threshold, 0.05)})
    save_json(rows, out / "comparison.json")
    print("comparison.json:", [(r["label"], round(r["weed_hit_rate"], 3)) for r in rows])

    test = read_jsonl(RUNS_DIR / final / "eval_test" / "shots.jsonl")
    base_test = read_jsonl(RUNS_DIR / BASELINE[0] / "eval_test" / "shots.jsonl")
    save_json({
        "run": final, "label": CANDIDATES[final][0], "threshold": threshold, "num_images": len(test),
        "model": {**at(test, threshold), "ci": bootstrap_ci(test, threshold),
                  "inference_ms": load_json(RUNS_DIR / final / "eval_test" / "summary.json")["inference_ms_per_image"]},
        "baseline": {**at(base_test, 0.05), "ci": bootstrap_ci(base_test, 0.05)},
        "num_gt_weeds": sum(len(r["gt_weeds"]) for r in test), "num_gt_crops": sum(r["num_gt_crops"] for r in test),
    }, out / "test.json")

    size_label = lambda record, weed: weed["size_bin"] if "size_bin" in weed else size_bin(weed["size_px"])  # noqa: E731
    if args.skip_annotations:
        kept = load_json(out / "errors.json")["crop_shots_by_species"]
        shot_species = {k: v["shots"] for k, v in kept.items()}
        plant_species = {k: v["plants"] for k, v in kept.items()}
    else:
        shot_species, plant_species = crop_species_of_shots(test, threshold, bbox_dir)
    sessions = sorted(((r["image_id"][:8], sum(s["on_crop"] for s in fired(r["shots"], threshold))) for r in test),
                      key=lambda x: x[0])
    per_session: dict[str, int] = {}
    for name, n in sessions:
        per_session[name] = per_session.get(name, 0) + n
    total_crop_shots = sum(per_session.values())
    save_json({
        "threshold": threshold,
        "by_size": hit_rate_by(test, threshold, size_label),
        "by_set": hit_rate_by(test, threshold, lambda record, weed: "Real fields (ave)" if record["image_id"].startswith("ave") else "Test plots (vwg)"),
        "crop_shots_by_species": {k: {"shots": shot_species.get(k, 0), "plants": v, "per_100_plants": 100 * shot_species.get(k, 0) / v}
                                  for k, v in sorted(plant_species.items(), key=lambda kv: -shot_species.get(kv[0], 0))},
        "top_sessions": [{"session": k, "crop_shots": v} for k, v in sorted(per_session.items(), key=lambda kv: -kv[1])[:5]],
        "total_crop_shots": total_crop_shots,
        "missed_weeds": sum(len(r["gt_weeds"]) for r in test) - sum(
            len({g for s in fired(r["shots"], threshold) for g in s["gt_weeds"]}) for r in test),
    }, out / "errors.json")

    history = pd.read_csv(RUNS_DIR / final / "train_results.csv")
    history.columns = history.columns.str.strip()
    cols = {"epoch": "epoch", "train/seg_loss": "train_seg", "val/seg_loss": "val_seg", "train/cls_loss": "train_cls",
            "val/cls_loss": "val_cls", "metrics/mAP50-95(M)": "val_map50_95", "metrics/mAP50(M)": "val_map50"}
    save_json({"run": final, **best_epoch_metrics(final),
               "history": history[list(cols)].rename(columns=cols).round(4).to_dict("records")}, out / "training.json")
    print("test.json / errors.json / training.json written for", final, "at threshold", threshold)


if __name__ == "__main__":
    main()
