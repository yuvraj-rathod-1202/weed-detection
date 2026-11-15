"""Small JSON summaries for the website (no image pixels): dataset statistics, curves, errors.

Plant size = sqrt(box area) in pixels of the 1920 x 1088 CropAndWeed images (~0.4 mm/px, so 25 px ~ 1 cm).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np

from src.data.cropandweed import map_label
from src.targeting.sweep import fired, session_counts

SIZE_BINS = [("0–16", 0, 16), ("16–24", 16, 24), ("24–32", 24, 32), ("32–48", 32, 48), ("48–64", 48, 64),
             ("64–128", 64, 128), ("128+", 128, math.inf)]
# CropsOrWeed9 grouping of CropAndWeed crop species ids (official cnw/utilities/datasets.py)
CROP_GROUPS = {"Maize": [1, 2, 3, 4, 5, 6], "Sugar beet": [7, 8, 9, 10, 11, 12], "Soybean": [94], "Sunflower": [24],
               "Potato": [18], "Pea": [13], "Bean": [26, 27], "Pumpkin": [15]}
CROP_NAME = {sid: name for name, ids in CROP_GROUPS.items() for sid in ids}
CLASS_NAMES = ["Crop", "Weed"]


def size_bin(size: float) -> str:
    return next(label for label, lo, hi in SIZE_BINS if lo <= size < hi)


def size_bins(sizes: list[float]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in SIZE_BINS}
    for s in sizes:
        counts[size_bin(s)] += 1
    return counts


def dataset_stats(bbox_dir: Path, splits: dict[str, list[str]]) -> dict:
    """Per split: images, sessions, crop/weed/ignored plants, ave/vwg images; plant-size bins; crop species."""
    out = {"splits": {}, "sizes": {}, "crop_species": {}}
    sizes = {name: [] for name in CLASS_NAMES}
    for split, stems in splits.items():
        row = {"images": len(stems), "sessions": len({s[:8] for s in stems}), "crop": 0, "weed": 0, "ignored": 0,
               "ave": sum(s.startswith("ave") for s in stems), "vwg": sum(s.startswith("vwg") for s in stems)}
        for stem in stems:
            with open(Path(bbox_dir) / f"{stem}.csv", newline="", encoding="utf-8") as f:
                for left, top, right, bottom, species, *_ in (tuple(int(v) for v in r) for r in csv.reader(f) if r):
                    cls = map_label(species)
                    if cls is None:
                        row["ignored"] += 1
                        continue
                    row[CLASS_NAMES[cls].lower()] += 1
                    sizes[CLASS_NAMES[cls]].append(math.sqrt(max(right - left, 0) * max(bottom - top, 0)))
                    if cls == 0:
                        name = CROP_NAME.get(species, "Other crop")
                        out["crop_species"][name] = out["crop_species"].get(name, 0) + 1
        out["splits"][split] = row
    out["sizes"] = {name: size_bins(values) for name, values in sizes.items()}
    out["crop_species"] = dict(sorted(out["crop_species"].items(), key=lambda kv: -kv[1]))
    return out


def hit_rate_by(records: list[dict], threshold: float, key: Callable[[dict, dict], str]) -> dict[str, dict]:
    """Weed hit rate at `threshold`, grouped by key(image record, GT weed) -> {group: {"weeds", "hit", "rate"}}."""
    groups: dict[str, dict] = {}
    for record in records:
        hit = {g for shot in fired(record["shots"], threshold) for g in shot.get("gt_weeds", [])}
        for g, weed in enumerate(record["gt_weeds"]):
            row = groups.setdefault(key(record, weed), {"weeds": 0, "hit": 0})
            row["weeds"] += 1
            row["hit"] += g in hit
    return {k: {**v, "rate": v["hit"] / v["weeds"]} for k, v in groups.items()}


def paired_bootstrap_diff(records_a: list[dict], records_b: list[dict], threshold_a: float, threshold_b: float,
                          n: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Weed hit rate of run A minus run B on the same images; CI from resampling sessions (both runs together).

    Answers one question: could the gap between two candidates be explained by which fields happen to be in
    the validation split? A CI that contains 0 means the data cannot tell them apart.
    """
    names_a, a = session_counts(records_a, threshold_a)
    names_b, b = session_counts(records_b, threshold_b)
    if names_a != names_b:
        raise ValueError("paired bootstrap needs both runs on the same images")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    samples = sa[:, 0] / sa[:, 1] - sb[:, 0] / sb[:, 1]
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    diff = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    return {"diff": float(diff), "ci": [float(lo), float(hi)]}


PUBLIC_SHOT_KEYS = ("conf", "ignored", "gt_weed", "gt_weeds", "gt_crop", "on_crop", "max_nearby_crop_conf")


def public_record(record: dict) -> dict:
    """A shots.jsonl record reduced to what the website needs, for the public repository.

    Drops the aim points (x, y) and replaces each labelled weed's exact size by its size bin, so the files
    cannot be turned back into point-and-size annotations of CropAndWeed (its licence forbids that).
    """
    return {
        "image_id": record["image_id"],
        "inference_ms": record["inference_ms"],
        "num_gt_crops": record["num_gt_crops"],
        "gt_weeds": [{"size_bin": weed["size_bin"] if "size_bin" in weed else size_bin(weed["size_px"])}
                     for weed in record["gt_weeds"]],
        "shots": [{k: shot[k] for k in PUBLIC_SHOT_KEYS if k in shot} for shot in record["shots"]],
    }
