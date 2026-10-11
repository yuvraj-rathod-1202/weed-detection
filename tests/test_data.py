from pathlib import Path

import numpy as np

from src.data.labels import Instance, box_to_polygon, polygon_to_mask, read_seg_labels, read_yolo_boxes, write_seg_labels
from src.data.prepare import split_images


def test_seg_label_roundtrip(tmp_path):
    polygon = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]], dtype=np.float32)
    path = tmp_path / "img.txt"
    write_seg_labels(path, [Instance(3, polygon)])
    [inst] = read_seg_labels(path)
    assert inst.class_id == 3
    np.testing.assert_allclose(inst.polygon, polygon, atol=1e-6)


def test_empty_seg_label_file(tmp_path):
    path = tmp_path / "empty.txt"
    write_seg_labels(path, [])
    assert path.exists() and read_seg_labels(path) == []


def test_read_yolo_boxes(tmp_path):
    path = tmp_path / "boxes.txt"
    path.write_text("0 0.5 0.5 0.2 0.4\n\n2 0.1 0.1 0.1 0.1\n")
    assert read_yolo_boxes(path) == [(0, 0.5, 0.5, 0.2, 0.4), (2, 0.1, 0.1, 0.1, 0.1)]
    assert read_yolo_boxes(tmp_path / "missing.txt") == []


def test_box_polygon_to_mask_area():
    mask = polygon_to_mask(box_to_polygon(0.5, 0.5, 0.5, 0.5), height=100, width=100)
    assert abs(mask.sum() - 50 * 50) < 150  # rasterization includes the border pixels


def test_split_is_deterministic_and_disjoint():
    paths = [Path(f"img_{i:03d}.jpg") for i in range(100)]
    ratios = {"train": 0.65, "val": 0.2, "test": 0.15}
    a = split_images(paths, ratios, seed=42)
    b = split_images(list(reversed(paths)), ratios, seed=42)
    assert a == b
    assert [len(a[s]) for s in ("train", "val", "test")] == [65, 20, 15]
    assert not set(a["train"]) & set(a["test"])


def test_split_max_images():
    paths = [Path(f"img_{i:03d}.jpg") for i in range(100)]
    split = split_images(paths, {"train": 0.65, "val": 0.2, "test": 0.15}, seed=0, max_images=16)
    assert sum(len(v) for v in split.values()) == 16
