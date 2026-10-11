"""Evaluation stage on the test split. Two parts, each auto-saved:

1. Segmentation metrics (ultralytics val): box + mask precision / recall / mAP50 / mAP50-95.
   Cached in eval/segmentation_metrics.json.
2. Targeting metrics (laser points, see src/targeting/metrics.py). One JSON line per image is
   appended to eval/predictions.jsonl as soon as it is computed; a restarted run skips
   images already in that file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data.label_fixes import read_ignore_regions
from src.data.labels import box_to_polygon, list_images, polygon_to_mask, read_seg_labels
from src.targeting.metrics import aggregate_targeting, evaluate_image
from src.utils.io import append_jsonl, load_json, read_jsonl, save_json
from src.visualization import GT_COLOR, IGNORE_COLOR, PRED_COLOR, draw_instances, save_rgb, side_by_side


@dataclass
class Prediction:
    masks: list[np.ndarray]    # bool (H, W) at original image resolution, sorted by confidence
    class_ids: list[int]
    confidences: list[float]
    inference_ms: float        # preprocess + model + NMS/mask postprocess
    boxes: list[list[float]] = field(default_factory=list)  # xyxy pixels, same order as masks


def predict_instances(model, image_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> Prediction:
    """Run a YOLO-seg model on one image and return full-resolution instance masks."""
    h, w = image_bgr.shape[:2]
    start = time.perf_counter()
    result = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros(0)  # .cpu() syncs GPU
    inference_ms = (time.perf_counter() - start) * 1000.0

    if result.masks is None or len(confidences) == 0:
        return Prediction([], [], [], inference_ms)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    masks = [polygon_to_mask(p, h, w, normalized=False) for p in result.masks.xy]
    boxes = result.boxes.xyxy.cpu().numpy()
    order = np.argsort(-confidences)
    return Prediction(
        masks=[masks[i] for i in order],
        class_ids=[int(class_ids[i]) for i in order],
        confidences=[round(float(confidences[i]), 4) for i in order],
        inference_ms=inference_ms,
        boxes=[[round(float(v), 1) for v in boxes[i]] for i in order],
    )


def load_gt_masks(label_path: Path, height: int, width: int) -> tuple[list[np.ndarray], list[int]]:
    instances = read_seg_labels(label_path)
    return [polygon_to_mask(inst.polygon, height, width) for inst in instances], [inst.class_id for inst in instances]


def load_ignore_masks(prepared_dir: Path, split: str, image_id: str, height: int, width: int) -> list[np.ndarray]:
    """Ignore regions from label fixes as bool masks (none for most images)."""
    regions = read_ignore_regions(Path(prepared_dir) / "ignore" / split / f"{image_id}.txt")
    return [polygon_to_mask(box_to_polygon(*box), height, width) for box in regions]


def segmentation_metrics(cfg, weights: Path, data_yaml: Path, run_dir: Path, logger) -> dict:
    out_path = run_dir / "eval" / "segmentation_metrics.json"
    if out_path.exists():
        logger.info("Segmentation metrics already computed: %s", out_path)
        return load_json(out_path)

    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split=cfg.eval.split,
        imgsz=cfg.eval.imgsz,
        iou=cfg.eval.iou,
        project=str((run_dir / "eval").resolve()),
        name="ultralytics_val",
        exist_ok=True,
        verbose=False,
    )
    names = metrics.names
    result = {
        "split": cfg.eval.split,
        "box": _summary(metrics.box),
        "mask": _summary(metrics.seg),
        "per_class_mask": {
            names[c]: {"map50": float(metrics.seg.ap50[i]), "map50_95": float(metrics.seg.ap[i])}
            for i, c in enumerate(metrics.seg.ap_class_index)
        },
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
    }
    save_json(result, out_path)
    return result


def targeting_metrics(cfg, weights: Path, run_dir: Path, class_names: list[str], logger) -> dict:
    split = cfg.eval.split
    prepared_dir = Path(cfg.dataset.prepared_dir)
    methods = list(cfg.targeting.methods)
    jsonl_path = run_dir / "eval" / "predictions.jsonl"

    image_paths = list_images(prepared_dir / "images" / split)
    if cfg.eval.max_images:
        image_paths = image_paths[: cfg.eval.max_images]
    vis_ids = {p.stem for p in image_paths[: cfg.eval.num_visualizations]}

    done = {r["image_id"] for r in read_jsonl(jsonl_path)}
    todo = [p for p in image_paths if p.stem not in done]
    logger.info("Targeting eval: %d images (%d already saved, %d to do)", len(image_paths), len(done), len(todo))

    if todo:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        predict_args = dict(conf=cfg.eval.conf, iou=cfg.eval.iou, imgsz=cfg.eval.imgsz)
        predict_instances(model, cv2.imread(str(todo[0])), **predict_args)  # warm-up, not timed

        for image_path in tqdm(todo, desc="Targeting eval", unit="img"):
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            gt_masks, gt_classes = load_gt_masks(prepared_dir / "labels" / split / f"{image_path.stem}.txt", h, w)
            ignore_masks = load_ignore_masks(prepared_dir, split, image_path.stem, h, w)
            pred = predict_instances(model, image, **predict_args)

            record = {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "height": h,
                "width": w,
                "inference_ms": round(pred.inference_ms, 2),
                "gt_classes": gt_classes,
                "pred_classes": pred.class_ids,
                "pred_confidences": pred.confidences,
                **evaluate_image(pred.masks, gt_masks, methods, cfg.eval.match_iou, ignore_masks),
            }
            append_jsonl(record, jsonl_path)

            if image_path.stem in vis_ids:
                save_visualization(image, gt_masks, pred, record, methods[0], class_names,
                                   run_dir / "eval" / "visualizations" / f"{image_path.stem}.jpg", ignore_masks)

    wanted = {p.stem for p in image_paths}
    records = [r for r in read_jsonl(jsonl_path) if r["image_id"] in wanted]
    summary = aggregate_targeting(records, methods)
    inference_ms = [r["inference_ms"] for r in records]
    primary_extraction = summary["methods"][methods[0]]["extraction_ms_per_image"] or 0.0
    summary["inference_ms_per_image"] = float(np.mean(inference_ms)) if inference_ms else None
    summary["fps"] = 1000.0 / summary["inference_ms_per_image"] if inference_ms else None
    summary["end_to_end_ms_per_image"] = summary["inference_ms_per_image"] + primary_extraction if inference_ms else None
    return {"primary_method": methods[0], "conf": cfg.eval.conf, "match_iou": cfg.eval.match_iou, **summary}


def save_visualization(image_bgr, gt_masks, pred: Prediction, record: dict, method: str, class_names, path: Path,
                       ignore_masks=None) -> None:
    """Save [ground truth | prediction] panels with the laser target points of `method` (ignore regions grey)."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    points = record["methods"][method]
    gt_panel = draw_instances(image, ignore_masks or [], color=IGNORE_COLOR, fill=False)
    gt_panel = draw_instances(gt_panel, gt_masks, points["gt_points"], color=GT_COLOR, fill=False)
    labels = [f"{class_names[c]} {s:.2f}" for c, s in zip(pred.class_ids, pred.confidences)]
    pred_panel = draw_instances(image, pred.masks, points["pred_points"], labels, color=PRED_COLOR)
    save_rgb(side_by_side(gt_panel, pred_panel), path)


def _summary(m) -> dict:
    return {"precision": float(m.mp), "recall": float(m.mr), "map50": float(m.map50), "map50_95": float(m.map)}
