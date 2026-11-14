"""Evaluate one trained run (or the baseline) on a split and save every shot for offline threshold sweeps.

    <out_dir>/eval_<split>/shots.jsonl   one line per image: GT weed sizes, number of GT crops, and every
                                         predicted weed above the confidence floor (src/targeting/sweep.py)
    <out_dir>/eval_<split>/summary.json  sweep curve (0.05..0.95), optimal threshold (2 % crop rule; not for
                                         test, where results/final_selection.json's threshold is used), ms/image

Re-running skips images already in shots.jsonl, so an interrupted Kaggle session resumes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.data.labels import list_images
from src.evaluation import (Prediction, crop_class_ids, load_gt_masks, load_ignore_masks, predict_instances,
                            split_crops, subset)
from src.targeting.metrics import ignored_predictions, match_instances
from src.targeting.sweep import optimal_threshold, shot_records, sweep
from src.utils.io import append_jsonl, load_yaml_dict, read_jsonl, save_json

THRESHOLDS = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 .. 0.95
MAX_CROP_SHARE = 0.02


def evaluate_split(run_dir: Path, split: str, prepared_dir: Path, conf_floor: float = 0.05,
                   max_images: int | None = None, predictor: Callable[[np.ndarray], Prediction] | None = None,
                   out_dir: Path | None = None) -> Path:
    """Write shots.jsonl + summary.json for `split`; returns the eval directory.

    `run_dir` holds config.yaml and train/weights/best.pt (or best.pt directly, for released weights).
    `predictor` replaces the YOLO model (tests, baseline). Outputs go to `out_dir` (default: run_dir).
    """
    run_dir, prepared_dir = Path(run_dir), Path(prepared_dir)
    cfg = load_yaml_dict(run_dir / "config.yaml")
    eval_dir = Path(out_dir or run_dir) / f"eval_{split}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    jsonl = eval_dir / "shots.jsonl"
    class_names = list(yaml.safe_load((prepared_dir / "data.yaml").read_text(encoding="utf-8"))["names"].values())
    crop_ids = crop_class_ids(class_names, list(cfg["targeting"].get("crop_classes") or []))
    margin = float(cfg["targeting"].get("crop_margin_px") or 0.0)

    images = list_images(prepared_dir / "images" / split)[:max_images]
    done = {r["image_id"] for r in read_jsonl(jsonl)}
    todo = [p for p in images if p.stem not in done]
    if todo and predictor is None:
        predictor = _yolo_predictor(run_dir, cfg, conf_floor)
        predictor(cv2.imread(str(todo[0])))  # warm-up, not timed

    for path in tqdm(todo, desc=f"eval {split}", unit="img"):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        gt_masks, gt_classes = load_gt_masks(prepared_dir / "labels" / split / f"{path.stem}.txt", h, w)
        gt_weed_idx, gt_crop_idx = split_crops(gt_classes, crop_ids)
        gt_weeds, gt_crops = [gt_masks[i] for i in gt_weed_idx], [gt_masks[i] for i in gt_crop_idx]
        pred = predictor(image)
        weed_idx, crop_idx = split_crops(pred.class_ids, crop_ids)
        weeds, crops = subset(pred, weed_idx), subset(pred, crop_idx)
        matched = {p for p, _, _ in match_instances(weeds.masks, gt_weeds, float(cfg["eval"]["match_iou"]))}
        ignored = set(ignored_predictions(weeds.masks, load_ignore_masks(prepared_dir, split, path.stem, h, w), matched))
        append_jsonl({
            "image_id": path.stem,
            "session": path.stem[:8],
            "inference_ms": round(pred.inference_ms, 2),
            "gt_weeds": [{"size_px": round(math.sqrt(float(m.sum())), 1)} for m in gt_weeds],
            "num_gt_crops": len(gt_crops),
            "shots": shot_records(weeds.masks, weeds.confidences, crops.masks, crops.confidences,
                                  gt_weeds, gt_crops, ignored, margin),
        }, jsonl)

    wanted = {p.stem for p in images}
    records = [r for r in read_jsonl(jsonl) if r["image_id"] in wanted]
    curve = sweep(records, THRESHOLDS)
    save_json({
        "run": cfg.get("name", run_dir.name),
        "split": split,
        "conf_floor": conf_floor,
        "num_images": len(records),
        "curve": curve,
        # the threshold is chosen on validation only: a test summary offers no optimum of its own
        "optimal": None if split == "test" else optimal_threshold(curve, MAX_CROP_SHARE),
        "inference_ms_per_image": float(np.mean([r["inference_ms"] for r in records])) if records else None,
    }, eval_dir / "summary.json")
    return eval_dir


def _yolo_predictor(run_dir: Path, cfg: dict, conf_floor: float) -> Callable[[np.ndarray], Prediction]:
    from ultralytics import YOLO

    weights = run_dir / "train" / "weights" / "best.pt"
    model = YOLO(str(weights if weights.exists() else run_dir / "best.pt"))
    args = dict(conf=conf_floor, iou=float(cfg["eval"]["iou"]), imgsz=int(cfg["eval"]["imgsz"]))
    return lambda image: predict_instances(model, image, **args)
