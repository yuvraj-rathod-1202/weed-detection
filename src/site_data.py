"""Small JSON summaries for the website (no image pixels): dataset statistics, curves, errors.

Plant size = sqrt(box area) in pixels of the 1920 x 1088 CropAndWeed images (~0.4 mm/px, so 25 px ~ 1 cm).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from src.data.cropandweed import map_label

SIZE_BINS = [("0–16", 0, 16), ("16–24", 16, 24), ("24–32", 24, 32), ("32–48", 32, 48), ("48–64", 48, 64),
             ("64–128", 64, 128), ("128+", 128, math.inf)]
# CropsOrWeed9 grouping of CropAndWeed crop species ids (official cnw/utilities/datasets.py)
CROP_GROUPS = {"Maize": [1, 2, 3, 4, 5, 6], "Sugar beet": [7, 8, 9, 10, 11, 12], "Soybean": [94], "Sunflower": [24],
               "Potato": [18], "Pea": [13], "Bean": [26, 27], "Pumpkin": [15]}
CROP_NAME = {sid: name for name, ids in CROP_GROUPS.items() for sid in ids}
CLASS_NAMES = ["Crop", "Weed"]


def size_bins(sizes: list[float]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in SIZE_BINS}
    for s in sizes:
        counts[next(label for label, lo, hi in SIZE_BINS if lo <= s < hi)] += 1
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
