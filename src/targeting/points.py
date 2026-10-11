"""Laser target point extraction: one (x, y) pixel per weed instance mask.

Methods (all take a bool mask of shape (H, W) and return (x, y) or None for an empty mask):

    centroid            mask centre of mass; can fall OUTSIDE a non-convex weed
    distance_transform  pixel farthest from the mask border; always inside, the "safest" point
    skeleton            skeleton pixel closest to the centroid; approximates the stem/plant centre
    box_center          centre of the mask's bounding box = what a box-only detector would aim at

To add a method: write `def my_method(mask) -> (x, y) | None` and add it to POINT_METHODS.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

Point = tuple[float, float]


def centroid(mask: np.ndarray) -> Point | None:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def distance_transform(mask: np.ndarray) -> Point | None:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() == 0:
        return None
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x), float(y)


def skeleton(mask: np.ndarray) -> Point | None:
    center = centroid(mask)
    if center is None:
        return None
    ys, xs = np.nonzero(skeletonize(mask.astype(bool)))
    if len(xs) == 0:  # tiny masks can have an empty skeleton
        return distance_transform(mask)
    i = np.argmin((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    return float(xs[i]), float(ys[i])


def box_center(mask: np.ndarray) -> Point | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0


POINT_METHODS = {
    "centroid": centroid,
    "distance_transform": distance_transform,
    "skeleton": skeleton,
    "box_center": box_center,
}


def target_point(mask: np.ndarray, method: str) -> Point | None:
    """Target point of `mask` in image pixel coordinates.

    The mask is cropped to its bounding box (+1 px background border) first: same result, much
    faster, and the image border counts as background (a weed cut by the frame is aimed inward).
    """
    if method not in POINT_METHODS:
        raise ValueError(f"Unknown targeting method '{method}'. Available: {sorted(POINT_METHODS)}")
    rows, cols = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    y0, x0 = rows[0], cols[0]
    crop = np.pad(mask[y0 : rows[-1] + 1, x0 : cols[-1] + 1], 1)
    point = POINT_METHODS[method](crop)
    return None if point is None else (float(point[0] + x0 - 1), float(point[1] + y0 - 1))


def point_in_mask(point: Point | None, mask: np.ndarray) -> bool:
    if point is None:
        return False
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape
    return 0 <= x < w and 0 <= y < h and bool(mask[y, x])
