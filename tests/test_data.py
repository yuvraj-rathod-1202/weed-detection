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


def test_group_split_keeps_each_session_in_one_split():
    paths = [Path(f"vwg-{s:04d}-{i:04d}.jpg") for s in range(20) for i in range(5)]
    ratios = {"train": 0.65, "val": 0.2, "test": 0.15}
    split = split_images(paths, ratios, seed=42, group_pattern=r"^\w+-\d{4}")
    sessions = {s: {p.name[:8] for p in split[s]} for s in split}
    assert not sessions["train"] & sessions["val"] and not sessions["train"] & sessions["test"]
    assert not sessions["val"] & sessions["test"]
    assert [len(split[s]) for s in ("train", "val", "test")] == [65, 20, 15]  # equal sessions: exact ratios


def test_group_split_is_deterministic():
    paths = [Path(f"ave-{s:04d}-{i:04d}.jpg") for s in range(10) for i in range(s + 1)]
    ratios = {"train": 0.65, "val": 0.2, "test": 0.15}
    a = split_images(paths, ratios, seed=1, group_pattern=r"^\w+-\d{4}")
    assert a == split_images(list(reversed(paths)), ratios, seed=1, group_pattern=r"^\w+-\d{4}")
    assert sum(len(v) for v in a.values()) == len(paths)


def test_semantic_instance_takes_class_pixels_inside_box():
    from src.data.prepare import semantic_instances

    semantic = np.zeros((100, 100), np.uint8)
    semantic[20:40, 20:40] = 1    # crop (class 0) plant
    semantic[60:90, 60:70] = 2    # weed (class 1) plant
    boxes = [(0, 0.30, 0.30, 0.24, 0.24), (1, 0.65, 0.75, 0.14, 0.34)]
    instances, ambiguous = semantic_instances(semantic, boxes)
    assert ambiguous == []
    crop = polygon_to_mask(instances[0].polygon, 100, 100)
    assert instances[0].class_id == 0 and abs(crop.sum() - 400) < 60
    assert not crop[60:90, 60:70].any()


def test_overlapping_same_class_boxes_are_ambiguous():
    from src.data.prepare import semantic_instances

    semantic = np.zeros((100, 100), np.uint8)
    semantic[20:60, 20:60] = 2
    boxes = [(1, 0.35, 0.35, 0.3, 0.3), (1, 0.45, 0.45, 0.3, 0.3), (0, 0.8, 0.8, 0.1, 0.1)]
    _, ambiguous = semantic_instances(semantic, boxes)
    assert ambiguous == [0, 1, 2]  # box 2 has no crop pixels at all: also left to SAM


class _FakeSam:
    """Stands in for ultralytics SAM: returns each prompt box as its mask polygon."""

    def __init__(self):
        self.prompts = []

    def __call__(self, image, bboxes, **kwargs):
        from types import SimpleNamespace

        h, w = image.shape[:2]
        self.prompts.append(bboxes)
        polys = [np.array([[x0 / w, y0 / h], [x1 / w, y0 / h], [x1 / w, y1 / h], [x0 / w, y1 / h]]) for x0, y0, x1, y1 in bboxes]
        return [SimpleNamespace(masks=SimpleNamespace(xyn=polys))]


def test_gt_instances_uses_sam_only_for_ambiguous_boxes_and_cuts_it_to_class_pixels(tmp_path):
    import cv2

    from src.data.prepare import gt_instances

    image_path, semantic_path = tmp_path / "img.jpg", tmp_path / "img.png"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), np.uint8))
    semantic = np.zeros((100, 100), np.uint8)
    semantic[10:30, 10:30] = 1      # one crop plant
    semantic[50:90, 50:90] = 2      # two touching weeds: their boxes overlap
    cv2.imwrite(str(semantic_path), semantic)
    boxes = [(0, 0.2, 0.2, 0.2, 0.2), (1, 0.6, 0.7, 0.3, 0.5), (1, 0.75, 0.7, 0.3, 0.5)]
    sam = _FakeSam()

    instances, fallbacks, sam_boxes = gt_instances(lambda: sam, image_path, boxes, semantic_path)

    assert sam_boxes == [1, 2] and fallbacks == [] and len(sam.prompts[0]) == 2
    for inst in instances[1:]:  # SAM gave the whole box; only weed pixels may remain
        mask = polygon_to_mask(inst.polygon, 100, 100)
        assert mask.any() and not mask[:50, :].any() and not mask[:, :50].any()
