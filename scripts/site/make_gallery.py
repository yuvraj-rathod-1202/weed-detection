#!/usr/bin/env python
"""Build the website gallery: 5 Sugar Beets 2016 and 5 CottonWeedDet12 photos, each run through every model the
site compares (baseline, exp 2, exp 3, exp 4, exp 5). Never CropAndWeed photos (licence).

The photos are hand-picked to show the models working (src/gallery.py SUGARBEETS_FRAMES, COTTON_IMAGES: frames
with clearly visible weeds where the final model does well); the site says so. For honesty the selected model is
also scored on 5 random Sugar Beets frames (RANDOM_SUGARBEETS_FRAMES, drawn with seed 0), reported as
"random_sugarbeets".

Writes site/gallery/<id>_{raw,gt}.jpg, site/gallery/<id>_<model>.jpg (800 px wide) and site/data/gallery.json.

Usage:
    python scripts/site/fetch_sugarbeets_sample.py        # once
    python scripts/site/make_gallery.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baseline import baseline_prediction
from src.evaluation import load_gt_masks, predict_instances, split_crops, subset
from src.gallery import COTTON_IMAGES, RANDOM_SUGARBEETS_FRAMES, SUGARBEETS_FRAMES, render_gt, render_output, shot_outcome, sugarbeets_masks
from src.targeting.points import point_in_mask
from src.targeting.sweep import shot_records
from src.utils.io import load_json, load_yaml_dict, save_json

WIDTH = 800
CROP_MARGIN_PX = 20
RELEASED = Path("models/released")
# key, label, released folder (None = non-learned baseline), class ids that are crops
MODELS = [
    ("baseline", "Baseline · green blobs (no learning)", None, set()),
    ("exp2", "Exp 2 · trained on CottonWeedDet12 (weeds only)", "yolo11s_seg", set()),
    ("exp3", "Exp 3 · CropAndWeed, 1024 px", "yolo11s_seg_cropandweed", {0}),
    ("exp4", "Exp 4 · CropAndWeed, 1280 px", "yolo11s_seg_cropandweed_1280", {0}),
    ("exp5", "Exp 5 · CropAndWeed, 1280 px + flips/rotation", "yolo11s_seg_cropandweed_1280_aug", {0}),
]
SUGARBEETS = {"dataset": "Sugar Beets 2016", "licence": "CC BY-SA 4.0",
              "attribution_url": "https://www.ipb.uni-bonn.de/data/sugarbeets2016/",
              "note": "Farm (Bonn, Germany) and camera never seen by any model. The camera records dark images; "
                      "they are brightened here for viewing only, the models saw the originals."}
COTTON = {"dataset": "CottonWeedDet12", "licence": "CC BY-NC 4.0", "attribution_url": "https://zenodo.org/records/7535814",
          "note": "This dataset labels weeds only (no crops), so crop safety cannot be checked here. Exp 2 was "
                  "trained on it (these are its test photos); exp 3–5 never saw it."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sugarbeets-dir", default="data/raw/sugarbeets2016_sample")
    parser.add_argument("--cotton-dir", default="data/processed/cottonweeddet12_seg")
    parser.add_argument("--site", default="site")
    args = parser.parse_args()
    selected_run = load_json(Path("results/final_selection.json"))["run"]
    models = [_load(key, label, folder, crops) for key, label, folder, crops in MODELS]
    default = next(m["key"] for m in models if m["folder"] == selected_run)

    site = Path(args.site)
    (site / "gallery").mkdir(parents=True, exist_ok=True)
    for old in (site / "gallery").glob("*.jpg"):
        old.unlink()
    sb_dir, cotton_dir = Path(args.sugarbeets_dir), Path(args.cotton_dir)
    items = []
    for frame in SUGARBEETS_FRAMES:
        image, gt_weeds, gt_crops = _sugarbeets(sb_dir, frame)
        items.append(_item(f"sugarbeets-{frame:05d}", image, gt_weeds, gt_crops, models, site, SUGARBEETS))
    for stem in COTTON_IMAGES:
        path = cotton_dir / "images" / "test" / f"{stem}.jpg"
        image = cv2.imread(str(path))
        gt, _ = load_gt_masks(cotton_dir / "labels" / "test" / f"{stem}.txt", *image.shape[:2])
        items.append(_item(f"cotton-{_slug(stem)}", image, gt, [], models, site, COTTON))

    # Honesty check: the selected model on random Sugar Beets frames, not hand-picked ones.
    chosen = next(m for m in models if m["key"] == default)
    random_frames = list(RANDOM_SUGARBEETS_FRAMES)
    totals = {"weeds": 0, "weeds_hit": 0, "crop_shots": 0, "soil_shots": 0}
    for frame in random_frames:
        image, gt_weeds, gt_crops = _sugarbeets(sb_dir, frame)
        result = _run(chosen, image, gt_weeds, gt_crops)
        totals["weeds"] += len(gt_weeds)
        for k in ("weeds_hit", "crop_shots", "soil_shots"):
            totals[k] += result[k]

    gallery = {
        "models": [{"key": m["key"], "label": m["label"], "threshold": m["threshold"]} for m in models],
        "default": default,
        "items": items,
        "random_sugarbeets": {"model": default, "seed": 0, "frames": random_frames, **totals},
    }
    save_json(gallery, site / "data" / "gallery.json")
    print(f"{len(items)} photos x {len(models)} models -> {site / 'data' / 'gallery.json'}; random check: {totals}")


def _load(key, label, folder, crop_ids) -> dict:
    if folder is None:
        return {"key": key, "label": label, "folder": None, "threshold": None, "crop_ids": crop_ids}
    from ultralytics import YOLO

    cfg = load_yaml_dict(RELEASED / folder / "config.yaml")
    summary = Path("results/runs") / folder / "eval_val" / "summary.json"
    # CropAndWeed models: the threshold chosen for them on validation; exp 2: the threshold it was evaluated at
    threshold = load_json(summary)["optimal"]["threshold"] if summary.exists() else float(cfg["eval"]["conf"])
    return {"key": key, "label": label, "folder": folder, "threshold": threshold, "crop_ids": crop_ids,
            "model": YOLO(str(RELEASED / folder / "best.pt")), "iou": float(cfg["eval"]["iou"]),
            "imgsz": int(cfg["eval"]["imgsz"])}


def _run(m: dict, image, gt_weeds, gt_crops) -> dict:
    """Predict, aim, apply the crop gate; return the shots and per-photo counts."""
    if m["folder"] is None:
        pred, threshold = baseline_prediction(image), 1.0
    else:
        threshold = m["threshold"]
        pred = predict_instances(m["model"], image, conf=threshold, iou=m["iou"], imgsz=m["imgsz"])
    weed_idx, crop_idx = split_crops(pred.class_ids, m["crop_ids"])
    weeds, crops = subset(pred, weed_idx), subset(pred, crop_idx)
    records = shot_records(weeds.masks, weeds.confidences, crops.masks, crops.confidences, gt_weeds, gt_crops,
                           ignored=set(), margin_px=CROP_MARGIN_PX)
    shots = [{"x": r["x"], "y": r["y"], "vetoed": r["max_nearby_crop_conf"] >= threshold,
              "outcome": shot_outcome((r["x"], r["y"]), gt_weeds, gt_crops)} for r in records]
    fired = [s for s in shots if not s["vetoed"]]
    return {
        "weeds": weeds, "crops": crops, "shots": shots,
        "weeds_hit": sum(any(point_in_mask((s["x"], s["y"]), w) for s in fired) for w in gt_weeds),
        "crop_shots": sum(s["outcome"] == "crop" for s in fired),
        "soil_shots": sum(s["outcome"] == "soil" for s in fired),
        "vetoed": sum(s["vetoed"] for s in shots),
    }


def _item(item_id, image, gt_weeds, gt_crops, models, site, meta) -> dict:
    rgb, brightened = _for_display(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    item = {"id": item_id, **meta, "raw": _save(rgb, site, f"gallery/{item_id}_raw.jpg"),
            "gt": _save(render_gt(rgb, gt_weeds, gt_crops), site, f"gallery/{item_id}_gt.jpg"),
            "weeds": len(gt_weeds), "crops": len(gt_crops), "brightened_for_display": brightened, "outputs": {}}
    for m in models:
        r = _run(m, image, gt_weeds, gt_crops)
        picture = render_output(rgb, r["weeds"].masks, r["crops"].masks, r["shots"])
        item["outputs"][m["key"]] = {"file": _save(picture, site, f"gallery/{item_id}_{m['key']}.jpg"),
                                     **{k: r[k] for k in ("weeds_hit", "crop_shots", "soil_shots", "vetoed")}}
    print(item_id, {k: (o["weeds_hit"], item["weeds"]) for k, o in item["outputs"].items()})
    return item


def _save(picture_rgb: np.ndarray, site: Path, rel: str) -> str:
    scale = WIDTH / picture_rgb.shape[1]
    small = cv2.resize(picture_rgb, (WIDTH, round(picture_rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(site / rel), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return rel


def _sugarbeets(folder: Path, frame: int):
    image = cv2.imread(str(folder / "rgb" / f"{frame:05d}.png"))
    weeds, crops = sugarbeets_masks(cv2.imread(str(folder / "labels" / f"{frame:05d}.png")))
    return image, weeds, crops


def _slug(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def _for_display(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """Brighten dark photos for viewing only (the model saw the original): 99th percentile -> 255."""
    if rgb.mean() >= 70:
        return rgb, False
    gain = 255.0 / max(float(np.percentile(rgb, 99)), 1.0)
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8), True


if __name__ == "__main__":
    main()
