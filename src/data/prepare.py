"""Build a YOLO-seg dataset (GT instance masks) from a YOLO box dataset with box-prompted SAM.

    raw_dir/{images/, labels/, classes.txt}  (+ optional ignore/, semantic/, see below)
        -> prepared_dir/
            images/{train,val,test}/   hard links (or copies) of the raw images
            labels/{train,val,test}/   YOLO-seg polygons, one line per weed
            splits.json                settings + image names per split (written first)
            ignore/{train,val,test}/   ignore regions from label fixes (only images that have some)
            sam_fallbacks.txt          boxes where SAM gave no mask (box polygon used instead)
            label_fixes_applied.json   which label fixes each image was built with
            data.yaml                  ultralytics dataset file
            prepare_info.json          settings + counts (written last = "done" marker)

GT boxes = raw boxes + manual corrections from `label_fixes` (src/data/label_fixes.py).

Optional, per dataset config (absent = the default behaviour above):
    split_group_pattern    regex on the file name; images of one group (e.g. recording session) stay in one split
    mask_source: semantic  masks from raw_dir/semantic/<stem>.png (pixel = class_id + 1, 0 = background)
                           where a box's pixels are unambiguous; SAM only for the rest (see semantic_instances)
    raw_dir/ignore/<stem>.txt  extra ignore regions (`cx cy w h`), e.g. unlabelled plants

Resumable: an image whose label file already exists is skipped, so an interrupted run
continues where it stopped. When the label fixes change, only the affected images are redone.
"""
from __future__ import annotations

import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.data.label_fixes import (
    apply_fixes,
    changes_by_image,
    fingerprint,
    load_fixes,
    read_ignore_regions,
    split_hashes,
    summarize_fixes,
    write_ignore_regions,
)
from src.data.labels import (
    Instance,
    box_to_polygon,
    list_images,
    polygon_to_mask,
    read_class_names,
    read_seg_labels,
    read_yolo_boxes,
    write_seg_labels,
)
from src.utils.io import load_json, save_json
from src.utils.logging_utils import get_logger

SPLITS = ("train", "val", "test")


def split_images(
    image_paths: list[Path], ratios: dict, seed: int, max_images: int | None = None, group_pattern: str | None = None
) -> dict[str, list[Path]]:
    """Seeded random split. `max_images` takes a seeded subset first (for smoke tests).

    `group_pattern`: regex whose match on the file name is the image's group (e.g. its recording
    session). Whole groups go to one split, so near-duplicate images never sit in train and test.
    """
    paths = sorted(image_paths, key=lambda p: p.name)
    random.Random(seed).shuffle(paths)
    if max_images:
        paths = paths[:max_images]
    n_train = round(len(paths) * ratios["train"])
    n_val = round(len(paths) * ratios["val"])
    if group_pattern:
        return _group_split(paths, n_train, n_val, group_pattern, seed)
    return {
        "train": sorted(paths[:n_train]),
        "val": sorted(paths[n_train : n_train + n_val]),
        "test": sorted(paths[n_train + n_val :]),
    }


def _group_split(paths: list[Path], n_train: int, n_val: int, pattern: str, seed: int) -> dict[str, list[Path]]:
    """Shuffled groups fill train up to n_train images, then val up to n_val; the rest is test."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        match = re.match(pattern, path.name)
        if match is None:
            raise ValueError(f"split_group_pattern {pattern!r} does not match image name {path.name!r}")
        groups.setdefault(match.group(0), []).append(path)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    split = {"train": [], "val": [], "test": []}
    for key in keys:
        name = "train" if len(split["train"]) < n_train else "val" if len(split["val"]) < n_val else "test"
        split[name] += groups[key]
    return {s: sorted(v) for s, v in split.items()}


def sam_instances(sam, image_path: Path, boxes: list) -> tuple[list[Instance], list[int]]:
    """Prompt SAM with every GT box of one image. Returns instances + indices of fallback boxes."""
    if not boxes:
        return [], []
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    xyxy = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h] for _, cx, cy, bw, bh in boxes]
    result = sam(image, bboxes=xyxy, verbose=False, save=False)[0]
    polygons = result.masks.xyn if result.masks is not None else []
    if len(polygons) != len(boxes):  # unexpected output shape: fall back for every box
        polygons = [np.zeros((0, 2))] * len(boxes)

    instances, fallbacks = [], []
    for i, ((class_id, cx, cy, bw, bh), polygon) in enumerate(zip(boxes, polygons)):
        if len(polygon) < 3:
            polygon = box_to_polygon(cx, cy, bw, bh)
            fallbacks.append(i)
        instances.append(Instance(class_id=class_id, polygon=np.asarray(polygon, dtype=np.float32)))
    return instances, fallbacks


def gt_instances(get_sam, image_path: Path, boxes: list, semantic_path: Path | None) -> tuple[list[Instance], list[int], list[int]]:
    """GT masks for one image: semantic masks where possible (if `semantic_path`), SAM for the rest.

    Returns (instances, SAM fallback box indices, indices of boxes that went to SAM). With a semantic
    mask, SAM's mask is cut to the box's class pixels, so it can't leak onto soil or the other class.
    """
    if not boxes:
        return [], [], []
    semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE) if semantic_path else None
    if semantic is None:
        instances, fallbacks = sam_instances(get_sam(), image_path, boxes)
        return instances, fallbacks, list(range(len(boxes))) if semantic_path else []

    instances, todo = semantic_instances(semantic, boxes)
    fallbacks = []
    if todo:
        sam_result, sam_fallbacks = sam_instances(get_sam(), image_path, [boxes[i] for i in todo])
        h, w = semantic.shape
        for k, (i, inst) in enumerate(zip(todo, sam_result)):
            polygon = None
            if k not in sam_fallbacks:
                polygon = mask_to_polygon(polygon_to_mask(inst.polygon, h, w) & (semantic == inst.class_id + 1))
            instances[i] = inst if polygon is None else Instance(class_id=inst.class_id, polygon=polygon)
        fallbacks = [todo[k] for k in sam_fallbacks]
    return instances, fallbacks, todo


def semantic_instances(semantic: np.ndarray, boxes: list) -> tuple[list[Instance | None], list[int]]:
    """Instance masks from a human-labelled semantic mask (pixel = class_id + 1, 0 = background).

    Box i gets the pixels of its class inside the box (largest connected part). Boxes that overlap
    another box of the same class (pixels can't be assigned) or contain no pixels of their class
    are returned as None and listed as ambiguous, for SAM.
    """
    h, w = semantic.shape
    rects = [_pixel_rect(box, h, w) for box in boxes]
    instances, ambiguous = [], []
    for i, (class_id, *_) in enumerate(boxes):
        x0, y0, x1, y1 = rects[i]
        overlaps = any(
            j != i and boxes[j][0] == class_id and min(x1, r[2]) > max(x0, r[0]) and min(y1, r[3]) > max(y0, r[1])
            for j, r in enumerate(rects)
        )
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = semantic[y0:y1, x0:x1] == class_id + 1
        polygon = None if overlaps else mask_to_polygon(mask)
        if polygon is None:
            ambiguous.append(i)
            instances.append(None)
        else:
            instances.append(Instance(class_id=class_id, polygon=polygon))
    return instances, ambiguous


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Normalized polygon of the largest connected part of `mask`, or None if it has none."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if len(c) >= 3]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return largest / np.array([mask.shape[1], mask.shape[0]], dtype=np.float32)


def _pixel_rect(box, h: int, w: int) -> tuple[int, int, int, int]:
    _, cx, cy, bw, bh = box
    x0, y0 = max(0, int(np.floor((cx - bw / 2) * w))), max(0, int(np.floor((cy - bh / 2) * h)))
    return x0, y0, min(w, int(np.ceil((cx + bw / 2) * w))), min(h, int(np.ceil((cy + bh / 2) * h)))


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # no extra disk space; fails across drives / read-only mounts
    except OSError:
        shutil.copy2(src, dst)


def dataset_stats(prepared_dir: Path, class_names: list[str]) -> dict:
    stats = {}
    for split in SPLITS:
        label_files = sorted((prepared_dir / "labels" / split).glob("*.txt"))
        per_class = Counter()
        for label_file in label_files:
            per_class.update(class_names[inst.class_id] for inst in read_seg_labels(label_file))
        stats[split] = {
            "num_images": len(label_files),
            "num_instances": sum(per_class.values()),
            "instances_per_class": {name: per_class.get(name, 0) for name in class_names},
            "num_ignore_regions": sum(len(read_ignore_regions(f)) for f in (prepared_dir / "ignore" / split).glob("*.txt")),
        }
    return stats


def write_data_yaml(prepared_dir: Path, class_names: list[str]) -> Path:
    data_yaml = prepared_dir / "data.yaml"
    content = {
        "path": str(prepared_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)
    return data_yaml


def prepare_dataset(ds_cfg, logger=None) -> dict:
    """Create (or resume creating) the YOLO-seg dataset for `ds_cfg`. Returns prepare_info."""
    logger = logger or get_logger("prepare_data")
    raw_dir, out_dir = Path(ds_cfg.raw_dir), Path(ds_cfg.prepared_dir)
    settings = {
        "split": dict(ds_cfg.split),
        "split_seed": ds_cfg.split_seed,
        "max_images": ds_cfg.get("max_images"),
        "sam_model": ds_cfg.sam_model,
    }
    for key in ("split_group_pattern", "mask_source"):  # only when set: older prepared datasets stay valid
        if ds_cfg.get(key):
            settings[key] = ds_cfg[key]
    if settings.get("mask_source", "sam") not in ("sam", "semantic"):
        raise ValueError(f"mask_source must be 'sam' or 'semantic', got {settings['mask_source']!r}")
    class_names = read_class_names(raw_dir / "classes.txt")

    info_path = out_dir / "prepare_info.json"
    info = load_json(info_path) if info_path.exists() else None
    if info:
        _check_settings(info["settings"], settings, out_dir)

    splits_path = out_dir / "splits.json"
    if splits_path.exists():
        saved = load_json(splits_path)
        _check_settings(saved["settings"], settings, out_dir)
        splits = {s: [raw_dir / "images" / name for name in names] for s, names in saved["splits"].items()}
    else:
        splits = split_images(list_images(raw_dir / "images"), settings["split"], settings["split_seed"], settings["max_images"],
                              settings.get("split_group_pattern"))
        save_json({"settings": settings, "splits": {s: [p.name for p in paths] for s, paths in splits.items()}}, splits_path)

    changes = changes_by_image(load_fixes(ds_cfg.get("label_fixes")))
    fixes_info = summarize_fixes(changes, {s: [p.stem for p in paths] for s, paths in splits.items()}, ds_cfg.get("label_fixes"))
    if info:
        if split_hashes(info.get("label_fixes"), splits) == split_hashes(fixes_info, splits):
            write_data_yaml(out_dir, class_names)  # refresh absolute path in case the folder moved
            logger.info("Dataset already prepared: %s", out_dir)
            return info
        logger.info("Label fixes changed (%s): updating the affected images", fixes_info["file"])
    elif splits_path.exists():
        logger.info("Resuming dataset preparation in %s", out_dir)

    sam = None

    def get_sam():
        nonlocal sam
        if sam is None:
            from ultralytics import SAM

            sam = SAM(ds_cfg.sam_model)
        return sam

    fallback_log = out_dir / "sam_fallbacks.txt"
    sam_log = out_dir / "sam_boxes.txt"
    applied_path = out_dir / "label_fixes_applied.json"
    applied = load_json(applied_path) if applied_path.exists() else {}
    for split, image_paths in splits.items():
        for image_path in tqdm(image_paths, desc=f"GT masks [{split}]", unit="img"):
            stem = image_path.stem
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            fixes_fp = fingerprint(changes.get(stem, []))
            if label_path.exists() and applied.get(stem, "") == fixes_fp:
                continue
            _link_or_copy(image_path, out_dir / "images" / split / image_path.name)
            boxes, ignore = apply_fixes(read_yolo_boxes(raw_dir / "labels" / f"{stem}.txt"), changes.get(stem, []), class_names)
            ignore = ignore + read_ignore_regions(raw_dir / "ignore" / f"{stem}.txt")
            semantic_path = raw_dir / "semantic" / f"{stem}.png" if settings.get("mask_source") == "semantic" else None
            instances, fallbacks, sam_boxes = gt_instances(get_sam, image_path, boxes, semantic_path)
            for log, indices in ((fallback_log, fallbacks), (sam_log, sam_boxes)):
                if indices:
                    with open(log, "a", encoding="utf-8") as f:
                        f.writelines(f"{split}/{stem} box={i}\n" for i in indices)
            write_ignore_regions(out_dir / "ignore" / split / f"{stem}.txt", ignore)
            write_seg_labels(label_path, instances)  # written last: marks this image as done
            if applied.get(stem, "") != fixes_fp:
                if fixes_fp:
                    applied[stem] = fixes_fp
                else:
                    applied.pop(stem, None)
                save_json(applied, applied_path)
                for cache in (out_dir / "labels").glob("*.cache"):  # ultralytics label cache would be stale
                    cache.unlink()
    if sam is not None:
        sam = None
        release_gpu_memory()

    write_data_yaml(out_dir, class_names)
    num_fallbacks = len(set(fallback_log.read_text(encoding="utf-8").splitlines())) if fallback_log.exists() else 0
    num_sam = len(set(sam_log.read_text(encoding="utf-8").splitlines())) if sam_log.exists() else 0
    info = {
        "dataset": ds_cfg.name,
        "settings": settings,
        "class_names": class_names,
        "splits": dataset_stats(out_dir, class_names),
        "sam_fallback_boxes": num_fallbacks,
        **({"sam_boxes": num_sam} if settings.get("mask_source") == "semantic" else {}),
        "label_fixes": fixes_info,
    }
    save_json(info, info_path)
    logger.info(
        "Prepared %s: %s (SAM fallbacks: %d boxes)",
        ds_cfg.name, {s: v["num_images"] for s, v in info["splits"].items()}, num_fallbacks,
    )
    return info


def release_gpu_memory() -> None:
    """Hand cached CUDA memory back to the driver. Multi-GPU training runs in new processes, which
    can't use what this process keeps reserved (SAM left ~10 GB of a 15 GB T4 reserved)."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_settings(saved: dict, current: dict, out_dir: Path) -> None:
    if saved != current:
        raise ValueError(
            f"{out_dir} was prepared with different settings.\n  saved:   {saved}\n  current: {current}\n"
            f"Delete that folder or point `prepared_dir` somewhere else."
        )
