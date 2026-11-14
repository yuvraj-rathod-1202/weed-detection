"""Non-learned baseline: "every green blob is a weed".

Vegetation index ExG - ExR > 0 (Meyer & Neto 2008) on chromaticity-normalised channels, 3x3 opening,
then connected components of at least `min_area_px` pixels. Each component is a Weed prediction with
confidence 1.0; there are no crop predictions, so the crop safety gate never fires.
Fixed a priori, no tuning: the index's zero threshold is the published one (plain ExG + Otsu turned
greenish-tinted CropAndWeed soil into thousands of fragments), and 64 px (8x8) removes noise specks
while keeping the smallest labelled plants (~16 px wide, ~256 px^2).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from src.evaluation import Prediction

WEED_CLASS = 1  # CropOrWeed2: 0 = Crop, 1 = Weed


def vegetation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Bool plant mask: ExG - ExR > 0, with ExG = 2g - r - b and ExR = 1.4r - g, then a 3x3 opening."""
    bgr = image_bgr.astype(np.float32)
    total = bgr.sum(axis=2) + 1e-6
    b, g, r = (bgr[..., i] / total for i in range(3))
    mask = ((2 * g - r - b) - (1.4 * r - g) > 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def baseline_prediction(image_bgr: np.ndarray, min_area_px: int = 64) -> Prediction:
    start = time.perf_counter()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(vegetation_mask(image_bgr).astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    keep.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    masks = [labels == i for i in keep]
    boxes = [[float(stats[i, 0]), float(stats[i, 1]), float(stats[i, 0] + stats[i, 2]), float(stats[i, 1] + stats[i, 3])]
             for i in keep]
    return Prediction(masks, [WEED_CLASS] * len(masks), [1.0] * len(masks), (time.perf_counter() - start) * 1000.0, boxes)
