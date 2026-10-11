"""Drawing helpers for qualitative inspection (used by evaluation and the notebooks).

Convention: images are RGB uint8 arrays. Ground truth = red, prediction = green,
laser target point = crosshair.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GT_COLOR = (230, 50, 50)
PRED_COLOR = (40, 210, 40)
METHOD_COLORS = {  # target points when several methods are drawn on one image
    "distance_transform": (255, 215, 0),
    "centroid": (0, 150, 255),
    "skeleton": (255, 0, 255),
    "box_center": (255, 255, 255),
}


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def draw_instances(
    image: np.ndarray,
    masks: list[np.ndarray],
    points: list | None = None,
    labels: list[str] | None = None,
    color: tuple[int, int, int] = PRED_COLOR,
    fill: bool = True,
) -> np.ndarray:
    """Overlay instance masks (filled + outline), optional text labels and target points."""
    out = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 400))
    for i, mask in enumerate(masks):
        if fill:
            out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness)
        if labels and mask.any():
            ys, xs = np.nonzero(mask)
            _put_label(out, labels[i], (int(xs.min()), int(ys.min())), color, thickness)
    for point in points or []:
        if point is not None:
            draw_target(out, point, color, thickness)
    return out


def draw_target(image: np.ndarray, point, color, thickness: int = 2) -> None:
    """Laser target: white-outlined crosshair + circle."""
    center = (int(round(point[0])), int(round(point[1])))
    size = thickness * 16
    for c, t in (((255, 255, 255), thickness + 2), (color, thickness)):
        cv2.drawMarker(image, center, c, cv2.MARKER_CROSS, size, t)
        cv2.circle(image, center, size // 2, c, t)


def side_by_side(*images: np.ndarray, gap: int = 8) -> np.ndarray:
    """Concatenate same-height images horizontally with a white gap."""
    spacer = np.full((images[0].shape[0], gap, 3), 255, dtype=np.uint8)
    parts = []
    for image in images:
        parts += [image, spacer]
    return np.hstack(parts[:-1])


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], color, thickness: int) -> None:
    scale = thickness / 4
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin[0], max(h + 4, origin[1])
    cv2.rectangle(image, (x, y - h - 4), (x + w + 4, y), color, -1)
    cv2.putText(image, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
