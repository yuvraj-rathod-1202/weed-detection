#!/usr/bin/env python
"""Build the website gallery: 5 Sugar Beets 2016 photos with the final CropAndWeed model and 5 CottonWeedDet12
test photos with the CottonWeedDet12 model (random picks, seed 0). Never CropAndWeed photos (licence).

Writes site/gallery/<id>_{raw,output,gt}.jpg (1000 px wide) and site/data/gallery.json.

Usage:
    python scripts/site/fetch_sugarbeets_sample.py        # once
    python scripts/site/make_gallery.py                   # model + threshold from results/final_selection.json
    python scripts/site/make_gallery.py --cropandweed-run yolo11s_seg_cropandweed_1280_aug --threshold 0.15
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.labels import list_images
from src.evaluation import load_gt_masks, predict_instances, split_crops, subset
from src.gallery import render_gt, render_output, shot_outcome, sugarbeets_masks
from src.targeting.points import point_in_mask
from src.targeting.sweep import shot_records
from src.utils.io import load_json, load_yaml_dict, save_json

WIDTH = 1000
CROP_MARGIN_PX = 20
SUGARBEETS = {"dataset": "Sugar Beets 2016", "licence": "CC BY-SA 4.0",
              "attribution_url": "https://www.ipb.uni-bonn.de/data/sugarbeets2016/",
              "note": "Farm (Bonn, Germany) and camera never seen in training. The camera records dark images; they are brightened here for viewing only, the model saw the originals."}
COTTON = {"dataset": "CottonWeedDet12", "licence": "CC BY-NC 4.0", "attribution_url": "https://zenodo.org/records/7535814",
          "note": "Model trained on CottonWeedDet12 (weeds only): this dataset labels no crops, so crop safety is not shown."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cropandweed-run", help="Folder name in models/released (default: results/final_selection.json)")
    parser.add_argument("--threshold", type=float, help="Confidence threshold (default: results/final_selection.json)")
    parser.add_argument("--sugarbeets-dir", default="data/raw/sugarbeets2016_sample")
    parser.add_argument("--cotton-dir", default="data/processed/cottonweeddet12_seg")
    parser.add_argument("--site", default="site")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    selection = Path("results/final_selection.json")
    chosen = load_json(selection) if selection.exists() else {}
    run = args.cropandweed_run or chosen.get("run")
    threshold = args.threshold if args.threshold is not None else chosen.get("threshold")
    if run is None or threshold is None:
        sys.exit("No results/final_selection.json yet: pass --cropandweed-run and --threshold.")

    site = Path(args.site)
    (site / "gallery").mkdir(parents=True, exist_ok=True)
    items = []

    model, cfg = _model(Path("models/released") / run)
    for path in sorted(Path(args.sugarbeets_dir, "rgb").glob("*.png")):
        weeds, crops = sugarbeets_masks(cv2.imread(str(Path(args.sugarbeets_dir, "labels", path.name))))
        items.append(_item(f"sugarbeets-{path.stem}", path, model, cfg, threshold, weeds, crops, [0], site,
                           {**SUGARBEETS, "model": run, "threshold": threshold}))

    cotton_model, cotton_cfg = _model(Path("models/released/yolo11s_seg"))
    test_images = list_images(Path(args.cotton_dir) / "images" / "test")
    for path in sorted(random.Random(args.seed).sample(test_images, 5)):
        image = cv2.imread(str(path))
        gt, _ = load_gt_masks(Path(args.cotton_dir) / "labels" / "test" / f"{path.stem}.txt", *image.shape[:2])
        conf = float(cotton_cfg["eval"]["conf"])
        items.append(_item(f"cotton-{path.stem}", path, cotton_model, cotton_cfg, conf, gt, [], [], site,
                           {**COTTON, "model": "yolo11s_seg", "threshold": conf}))

    save_json(items, site / "data" / "gallery.json")
    print(f"{len(items)} gallery items -> {site / 'data' / 'gallery.json'}")


def _model(folder: Path):
    from ultralytics import YOLO

    return YOLO(str(folder / "best.pt")), load_yaml_dict(folder / "config.yaml")


def _item(item_id, path, model, cfg, threshold, gt_weeds, gt_crops, crop_ids, site, meta) -> dict:
    image = cv2.imread(str(path))
    pred = predict_instances(model, image, conf=threshold, iou=float(cfg["eval"]["iou"]), imgsz=int(cfg["eval"]["imgsz"]))
    weed_idx, crop_idx = split_crops(pred.class_ids, set(crop_ids))
    weeds, crops = subset(pred, weed_idx), subset(pred, crop_idx)
    records = shot_records(weeds.masks, weeds.confidences, crops.masks, crops.confidences, gt_weeds, gt_crops,
                           ignored=set(), margin_px=CROP_MARGIN_PX)
    shots = [{"x": r["x"], "y": r["y"], "vetoed": r["max_nearby_crop_conf"] >= threshold,
              "outcome": shot_outcome((r["x"], r["y"]), gt_weeds, gt_crops)} for r in records]
    fired = [s for s in shots if not s["vetoed"]]
    rgb, brightened = _for_display(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    files = {}
    for kind, picture in (("raw", rgb), ("output", render_output(rgb, weeds.masks, crops.masks, shots)),
                          ("gt", render_gt(rgb, gt_weeds, gt_crops))):
        files[kind] = f"gallery/{item_id}_{kind}.jpg"
        scale = WIDTH / picture.shape[1]
        small = cv2.resize(picture, (WIDTH, round(picture.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(site / files[kind]), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        "id": item_id, **meta, **files,
        "weeds": len(gt_weeds),
        "weeds_hit": sum(any(point_in_mask((s["x"], s["y"]), w) for s in fired) for w in gt_weeds),
        "crops": len(gt_crops),
        "crop_shots": sum(s["outcome"] == "crop" for s in fired),
        "soil_shots": sum(s["outcome"] == "soil" for s in fired),
        "vetoed": sum(s["vetoed"] for s in shots),
        "brightened_for_display": brightened,
    }


def _for_display(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """Brighten dark photos for viewing only (the model saw the original): 99th percentile -> 255."""
    if rgb.mean() >= 70:
        return rgb, False
    gain = 255.0 / max(float(np.percentile(rgb, 99)), 1.0)
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8), True


if __name__ == "__main__":
    main()
