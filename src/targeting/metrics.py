"""Targeting metrics: how good are the laser points derived from predicted masks?

Per method, over all evaluated images:

    shot_precision   fraction of fired points (one per predicted weed) that land inside a GT weed mask
    weed_hit_rate    fraction of GT weeds that contain at least one fired point (recall: weeds destroyed)
    shot_f1          harmonic mean of the two
    point_error_px   distance between predicted and GT point, for prediction/GT pairs matched by mask IoU.
                     The GT point is the same method applied to the GT mask.
    point_error_rel  point_error_px / GT equivalent radius sqrt(area / pi); < 1 is roughly "within the weed"

Predictions that match no GT weed but lie mostly inside an ignore region (label fixes: "weed,
species unsure") are left out: they are neither shots nor misses.

shot_precision and weed_hit_rate compare methods fairly (same GT masks for all methods);
point_error compares each method with its own GT definition.
"""
from __future__ import annotations

import math
import time

import numpy as np

from src.targeting.points import point_in_mask, target_point

IGNORE_OVERLAP = 0.5  # fraction of a prediction's mask inside ignore regions to leave it out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def match_instances(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float) -> list[tuple[int, int, float]]:
    """Greedy, class-agnostic matching. `pred_masks` must be sorted by confidence (highest first).

    Returns [(pred_index, gt_index, iou), ...].
    """
    matches, used_gt = [], set()
    for p, pred in enumerate(pred_masks):
        best_iou, best_g = 0.0, -1
        for g, gt in enumerate(gt_masks):
            if g in used_gt:
                continue
            iou = mask_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_g = iou, g
        if best_g >= 0 and best_iou >= iou_threshold:
            matches.append((p, best_g, best_iou))
            used_gt.add(best_g)
    return matches


def ignored_predictions(pred_masks: list[np.ndarray], ignore_masks: list[np.ndarray], matched: set[int]) -> list[int]:
    """Unmatched predictions with at least IGNORE_OVERLAP of their area inside the ignore regions."""
    if not ignore_masks:
        return []
    ignore = np.logical_or.reduce(ignore_masks)
    return [
        p for p, mask in enumerate(pred_masks)
        if p not in matched and mask.any() and np.logical_and(mask, ignore).sum() >= IGNORE_OVERLAP * mask.sum()
    ]


def evaluate_image(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], methods: list[str], match_iou: float,
    ignore_masks: list[np.ndarray] | None = None,
) -> dict:
    """Targeting record for one image (JSON-serializable; one line of predictions.jsonl)."""
    matches = match_instances(pred_masks, gt_masks, match_iou)
    ignored = ignored_predictions(pred_masks, ignore_masks or [], {p for p, _, _ in matches})
    record = {
        "num_gt": len(gt_masks),
        "num_pred": len(pred_masks),
        "matches": [[p, g, round(iou, 4)] for p, g, iou in matches],
        "ignored_preds": ignored,
        "methods": {},
    }
    for method in methods:
        start = time.perf_counter()
        pred_points = [target_point(m, method) for m in pred_masks]
        extraction_ms = (time.perf_counter() - start) * 1000.0
        gt_points = [target_point(m, method) for m in gt_masks]
        fired = [pt for p, pt in enumerate(pred_points) if p not in ignored]

        errors_px, errors_rel = [], []
        for p, g, _ in matches:
            if pred_points[p] is None or gt_points[g] is None:
                continue
            error = math.dist(pred_points[p], gt_points[g])
            errors_px.append(round(error, 2))
            errors_rel.append(round(error / math.sqrt(gt_masks[g].sum() / math.pi), 4))

        record["methods"][method] = {
            "pred_points": _round_points(pred_points),
            "gt_points": _round_points(gt_points),
            "shots": sum(p is not None for p in fired),
            "shots_on_weed": sum(any(point_in_mask(p, gt) for gt in gt_masks) for p in fired),
            "weeds_hit": sum(any(point_in_mask(p, gt) for p in fired) for gt in gt_masks),
            "point_errors_px": errors_px,
            "point_errors_rel": errors_rel,
            "extraction_ms": round(extraction_ms, 3),
        }
    return record


def aggregate_targeting(records: list[dict], methods: list[str]) -> dict:
    """Aggregate per-image records (from predictions.jsonl) into the targeting metrics."""
    matched_ious = [m[2] for r in records for m in r["matches"]]
    summary = {
        "num_images": len(records),
        "num_gt_weeds": sum(r["num_gt"] for r in records),
        "num_pred_weeds": sum(r["num_pred"] for r in records),
        "num_ignored_preds": sum(len(r.get("ignored_preds", [])) for r in records),
        "num_matched": len(matched_ious),
        "mean_matched_mask_iou": _mean(matched_ious),
        "methods": {},
    }
    for method in methods:
        per_image = [r["methods"][method] for r in records]
        shots = sum(m["shots"] for m in per_image)
        precision = _ratio(sum(m["shots_on_weed"] for m in per_image), shots)
        recall = _ratio(sum(m["weeds_hit"] for m in per_image), summary["num_gt_weeds"])
        errors_px = [e for m in per_image for e in m["point_errors_px"]]
        errors_rel = [e for m in per_image for e in m["point_errors_rel"]]
        summary["methods"][method] = {
            "shot_precision": precision,
            "weed_hit_rate": recall,
            "shot_f1": _ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
            "point_error_px": _describe(errors_px),
            "point_error_rel": _describe(errors_rel),
            "extraction_ms_per_image": _mean([m["extraction_ms"] for m in per_image]),
        }
    return summary


def _round_points(points: list) -> list:
    return [None if p is None else [round(p[0], 1), round(p[1], 1)] for p in points]


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _mean(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def _describe(values: list) -> dict:
    if not values:
        return {"mean": None, "median": None, "p90": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "n": len(values),
    }
