"""Threshold sweep over saved shots: pick the laser's confidence threshold without re-running the model.

Every predicted weed above a low confidence floor is saved as a shot record (where its target point
lands and how sure the nearest crop prediction is). At threshold X a shot fires when

    conf >= X, it is not ignored, and no crop prediction with confidence >= X lies within the crop margin,

which is the same rule as evaluation at eval.conf = X (one threshold for weeds and crops).

    weed_hit_rate    GT weeds containing at least one fired shot / all GT weeds (as src/targeting/metrics.py)
    shot_precision   fired shots inside a GT weed / fired shots
    crop_shot_share  fired shots inside a GT crop / fired shots
    soil_shot_share  fired shots inside neither / fired shots

The optimal threshold is the one with the most weeds hit while at most `max_crop_share` of shots hit crops.
"""
from __future__ import annotations

import numpy as np

from src.targeting.points import point_in_mask, target_point


def shot_records(weed_masks: list[np.ndarray], weed_confs: list[float], crop_masks: list[np.ndarray],
                 crop_confs: list[float], gt_weed_masks: list[np.ndarray], gt_crop_masks: list[np.ndarray],
                 ignored: set[int], margin_px: float, method: str = "center_weighted_dt") -> list[dict]:
    """One record per weed prediction that has a target point (JSON-serializable)."""
    records = []
    for i, (mask, conf) in enumerate(zip(weed_masks, weed_confs)):
        point = target_point(mask, method) if mask.any() else None
        if point is None:
            continue
        gt_weeds = [g for g, gt in enumerate(gt_weed_masks) if point_in_mask(point, gt)]
        gt_crop = next((c for c, gt in enumerate(gt_crop_masks) if point_in_mask(point, gt)), -1)
        near = [c for m, c in zip(crop_masks, crop_confs) if _within(point, m, margin_px)]
        records.append({
            "conf": round(float(conf), 4),
            "x": round(float(point[0]), 1),
            "y": round(float(point[1]), 1),
            "ignored": i in ignored,
            "gt_weed": gt_weeds[0] if gt_weeds else -1,  # first GT weed containing the point (-1: none)
            "gt_weeds": gt_weeds,                        # all of them: overlapping weeds are all hit
            "gt_crop": gt_crop,                          # first GT crop containing the point (-1: none)
            "on_crop": gt_crop >= 0,
            "max_nearby_crop_conf": round(float(max(near, default=0.0)), 4),
        })
    return records


def fired(shots: list[dict], threshold: float) -> list[dict]:
    return [s for s in shots if s["conf"] >= threshold and not s["ignored"] and s["max_nearby_crop_conf"] < threshold]


def sweep(records: list[dict], thresholds: list[float]) -> list[dict]:
    """Targeting rates at each threshold; `records` are per-image {"image_id", "gt_weeds", "shots"}."""
    num_gt = sum(len(r["gt_weeds"]) for r in records)
    rows = []
    for threshold in thresholds:
        n = on_weed = on_crop = soil = hit = 0
        for r in records:
            shots = fired(r["shots"], threshold)
            n += len(shots)
            on_weed += sum(s["gt_weed"] >= 0 for s in shots)
            on_crop += sum(s["on_crop"] for s in shots)
            soil += sum(s["gt_weed"] < 0 and not s["on_crop"] for s in shots)
            hit += len({g for s in shots for g in _hit_weeds(s)})
        rows.append({
            "threshold": round(float(threshold), 2),
            "shots": n,
            "weed_hit_rate": _ratio(hit, num_gt),
            "shot_precision": _ratio(on_weed, n),
            "crop_shot_share": _ratio(on_crop, n),
            "soil_shot_share": _ratio(soil, n),
        })
    return rows


def optimal_threshold(curve: list[dict], max_crop_share: float = 0.02) -> dict | None:
    """Row with the highest weed hit rate among rows within the crop limit (ties: the higher threshold)."""
    ok = [r for r in curve if r["shots"] > 0 and r["crop_shot_share"] is not None
          and r["crop_shot_share"] <= max_crop_share and r["weed_hit_rate"] is not None]
    return max(ok, key=lambda r: (r["weed_hit_rate"], r["threshold"])) if ok else None


def session_counts(records: list[dict], threshold: float) -> tuple[list[str], np.ndarray]:
    """Per recording session (image_id[:8]): [weeds hit, GT weeds, fired shots, shots on crops] at `threshold`."""
    totals: dict[str, np.ndarray] = {}
    for r in records:
        shots = fired(r["shots"], threshold)
        row = totals.setdefault(r["image_id"][:8], np.zeros(4))
        row += (len({g for s in shots for g in _hit_weeds(s)}), len(r["gt_weeds"]), len(shots), sum(s["on_crop"] for s in shots))
    names = sorted(totals)
    return names, np.array([totals[k] for k in names]).reshape(-1, 4)


def bootstrap_ci(records: list[dict], threshold: float, n: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile intervals from resampling recording sessions (image_id[:8]) with replacement."""
    _, counts = session_counts(records, threshold)
    rng = np.random.default_rng(seed)
    sums = counts[rng.integers(0, len(counts), (n, len(counts)))].sum(axis=1)   # (n, 4)
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    with np.errstate(invalid="ignore", divide="ignore"):
        hit, crop = sums[:, 0] / sums[:, 1], sums[:, 3] / sums[:, 2]
    return {"weed_hit_rate": _percentiles(list(hit[np.isfinite(hit)]), q),
            "crop_shot_share": _percentiles(list(crop[np.isfinite(crop)]), q)}


def _hit_weeds(shot: dict) -> list[int]:
    return shot.get("gt_weeds", [shot["gt_weed"]] if shot["gt_weed"] >= 0 else [])


def _within(point, mask: np.ndarray, margin_px: float) -> bool:
    """Is any mask pixel within margin_px (Euclidean) of the point? Same rule as src/targeting/safety.py."""
    h, w = mask.shape
    x, y = min(max(int(round(point[0])), 0), w - 1), min(max(int(round(point[1])), 0), h - 1)
    r = int(np.ceil(margin_px))
    y0, x0 = max(y - r, 0), max(x - r, 0)
    ys, xs = np.nonzero(mask[y0:y + r + 1, x0:x + r + 1])
    return bool(len(ys)) and float(np.min((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2)) <= margin_px ** 2


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _percentiles(values: list[float], q: list[float]) -> list[float] | None:
    return [float(v) for v in np.percentile(values, q)] if values else None
