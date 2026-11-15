"""CropAndWeed (Steininger et al., WACV 2023) -> this project's raw dataset layout.

Source (as extracted from the official tar files):
    images/<stem>.jpg                 1920x1088 RGB
    bboxes/<stem>.csv                 left,top,right,bottom,label_id,stem_x,stem_y  (pixels, species ids)
    labelIds/CropAndWeed/<stem>.png   semantic mask, pixel = species id (0 = soil)

Output (what src/data/prepare.py reads):
    images/<stem>.jpg      hard link (or copy)
    labels/<stem>.txt      YOLO boxes: 0 = Crop, 1 = Weed
    classes.txt
    ignore/<stem>.txt      `cx cy w h` of plants outside the two classes (only images that have some)
    semantic/<stem>.png    pixel = class id + 1, 0 = soil, 255 = other plant  (mask_source: semantic)
    stems/<stem>.txt       `x y` stem position per label line (normalized), `-1 -1` if unknown

Classes follow the official CropOrWeed2 variant (cnw/utilities/datasets.py in the CropAndWeed repo).
Species outside it, and the "Vegetation" label (255: tiny, excluded or unclear plants), become
ignore regions: neither targets nor false positives.
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["Crop", "Weed"]
_CROP_OR_WEED2 = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 94, 24, 18, 13, 26, 27, 15],
    1: [31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37, 49, 30, 44, 66, 87, 89, 91, 61, 79, 34, 41, 52, 35,
        36, 78, 38, 39, 71, 72, 88, 42, 45, 70, 47, 51, 54, 58, 60, 80, 83, 96, 22, 63, 85, 56, 57, 64, 77, 50, 59, 67,
        76],
}
_SOURCE_TO_CLASS = {source: cls for cls, sources in _CROP_OR_WEED2.items() for source in sources}
OTHER_PLANT = 255  # semantic value for plants outside the classes

Box = tuple[float, float, float, float]


def map_label(source_id: int) -> int | None:
    """CropOrWeed2 class id of a CropAndWeed species id, None if the species is not in it."""
    return _SOURCE_TO_CLASS.get(int(source_id))


def convert_rows(rows: list[tuple], width: int, height: int) -> tuple[list[tuple], list[Box], list[tuple | None]]:
    """CSV rows (pixels) -> (YOLO boxes, ignore regions, stem point per box), all normalized."""
    boxes, ignore, stems = [], [], []
    for left, top, right, bottom, label, stem_x, stem_y in rows:
        box = ((left + right) / 2 / width, (top + bottom) / 2 / height, (right - left) / width, (bottom - top) / height)
        cls = map_label(label)
        if cls is None:
            ignore.append(box)
            continue
        boxes.append((cls, *box))
        stems.append((stem_x / width, stem_y / height) if stem_x >= 0 and stem_y >= 0 else None)
    return boxes, ignore, stems


def convert_semantic(source: np.ndarray) -> np.ndarray:
    """Species-id mask -> class id + 1 per pixel (0 = soil, 255 = other plant)."""
    lut = np.full(256, OTHER_PLANT, dtype=np.uint8)
    lut[0] = 0
    for source_id, cls in _SOURCE_TO_CLASS.items():
        lut[source_id] = cls + 1
    return lut[source]


def convert_image(src_dir: Path, stem: str, out_dir: Path) -> None:
    """Convert one image. The label file is written last, so it marks the image as done."""
    source_mask = cv2.imread(str(src_dir / "labelIds" / "CropAndWeed" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if source_mask is None:
        raise FileNotFoundError(f"Missing semantic mask for {stem}")
    height, width = source_mask.shape
    with open(src_dir / "bboxes" / f"{stem}.csv", newline="", encoding="utf-8") as f:
        rows = [tuple(int(v) for v in row) for row in csv.reader(f) if row]
    boxes, ignore, stems = convert_rows(rows, width, height)

    _link_or_copy(src_dir / "images" / f"{stem}.jpg", out_dir / "images" / f"{stem}.jpg")
    _write_png(out_dir / "semantic" / f"{stem}.png", convert_semantic(source_mask))
    if ignore:
        _write_text(out_dir / "ignore" / f"{stem}.txt", [" ".join(f"{v:.6f}" for v in box) for box in ignore])
    _write_text(out_dir / "stems" / f"{stem}.txt", ["-1 -1" if s is None else f"{s[0]:.6f} {s[1]:.6f}" for s in stems])
    _write_text(out_dir / "labels" / f"{stem}.txt", [f"{b[0]} " + " ".join(f"{v:.6f}" for v in b[1:]) for b in boxes])


def convert_dataset(src_dir: str | Path, out_dir: str | Path, progress=None) -> dict:
    """Convert every image that has both a JPEG and annotations; already converted images are skipped."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    stems = sorted(p.stem for p in (src_dir / "images").glob("*.jpg") if (src_dir / "bboxes" / f"{p.stem}.csv").exists())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    todo = [s for s in stems if not (out_dir / "labels" / f"{s}.txt").exists()]
    for stem in progress(todo) if progress else todo:
        convert_image(src_dir, stem, out_dir)
    return {"images": len(stems), "converted_now": len(todo)}


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    os.replace(tmp, path)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")  # cv2 picks the format from the extension
    cv2.imwrite(str(tmp), image)
    os.replace(tmp, path)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
