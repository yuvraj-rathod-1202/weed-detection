"""Label file I/O and polygon <-> mask conversion.

Raw labels are YOLO boxes:          `<cls> <cx> <cy> <w> <h>`       (normalized)
Prepared labels are YOLO-seg:       `<cls> <x1> <y1> <x2> <y2> ...` (normalized polygon)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Instance:
    class_id: int
    polygon: np.ndarray  # (N, 2) normalized x, y in [0, 1]


def list_images(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_class_names(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_yolo_boxes(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO box label file -> [(class_id, cx, cy, w, h), ...]. Missing file = no objects."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_no}: expected 5 values, got {len(parts)}")
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def read_seg_labels(label_path: str | Path) -> list[Instance]:
    """Read a YOLO-seg label file -> list of Instance with normalized polygons."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    instances = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:  # class + at least 3 points
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            instances.append(Instance(class_id=int(parts[0]), polygon=coords))
    return instances


def write_seg_labels(label_path: str | Path, instances: list[Instance]) -> None:
    """Write YOLO-seg labels atomically (tmp file + rename), so a crash never leaves a half file."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for inst in instances:
        coords = " ".join(f"{v:.6f}" for v in inst.polygon.reshape(-1))
        lines.append(f"{inst.class_id} {coords}")
    tmp_path = label_path.with_suffix(".tmp")
    tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp_path, label_path)


def polygon_to_mask(polygon: np.ndarray, height: int, width: int, normalized: bool = True) -> np.ndarray:
    """Rasterize one polygon into a bool mask of shape (height, width)."""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if normalized:
        points = points * np.array([width, height], dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def box_to_polygon(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """Normalized YOLO box -> 4-corner normalized polygon (fallback when SAM returns nothing)."""
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.clip(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32), 0.0, 1.0)
