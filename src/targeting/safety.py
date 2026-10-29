"""Crop safety gate: do not fire at a weed point that lies on or too close to a crop plant.

The gate uses the PREDICTED crop masks (in the field there is nothing else). A shot is vetoed when
its distance to the nearest crop pixel is <= margin_px, so margin_px = 0 vetoes only shots that
land on a crop. Choose the margin from laser spot radius + aiming error, in pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.targeting.points import Point


def vetoed_shots(points: list[Point | None], crop_masks: list[np.ndarray], margin_px: float) -> list[int]:
    """Indices of `points` within `margin_px` of any crop mask (missing points are never vetoed)."""
    if not crop_masks:
        return []
    crop = np.logical_or.reduce(crop_masks)
    dist = cv2.distanceTransform((~crop).astype(np.uint8), cv2.DIST_L2, 5)  # 0 on crop pixels
    h, w = crop.shape
    vetoed = []
    for i, point in enumerate(points):
        if point is None:
            continue
        x, y = min(max(int(round(point[0])), 0), w - 1), min(max(int(round(point[1])), 0), h - 1)
        if dist[y, x] <= margin_px:
            vetoed.append(i)
    return vetoed
