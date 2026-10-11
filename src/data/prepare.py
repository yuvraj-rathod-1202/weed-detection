"""Build a YOLO-seg dataset (GT instance masks) from a YOLO box dataset with box-prompted SAM.

    raw_dir/{images/, labels/, classes.txt}
        -> prepared_dir/
            images/{train,val,test}/   hard links (or copies) of the raw images
            labels/{train,val,test}/   YOLO-seg polygons, one line per weed
            splits.json                settings + image names per split (written first)
            sam_fallbacks.txt          boxes where SAM gave no mask (box polygon used instead)
            data.yaml                  ultralytics dataset file
            prepare_info.json          settings + counts (written last = "done" marker)

Resumable: an image whose label file already exists is skipped, so an interrupted run
continues where it stopped.
"""
from __future__ import annotations

import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.data.labels import (
    Instance,
    box_to_polygon,
    list_images,
    read_class_names,
    read_seg_labels,
    read_yolo_boxes,
    write_seg_labels,
)
from src.utils.io import load_json, save_json
from src.utils.logging_utils import get_logger

SPLITS = ("train", "val", "test")


def split_images(
    image_paths: list[Path], ratios: dict, seed: int, max_images: int | None = None
) -> dict[str, list[Path]]:
    """Seeded random split. `max_images` takes a seeded subset first (for smoke tests)."""
    paths = sorted(image_paths, key=lambda p: p.name)
    random.Random(seed).shuffle(paths)
    if max_images:
        paths = paths[:max_images]
    n_train = round(len(paths) * ratios["train"])
    n_val = round(len(paths) * ratios["val"])
    return {
        "train": sorted(paths[:n_train]),
        "val": sorted(paths[n_train : n_train + n_val]),
        "test": sorted(paths[n_train + n_val :]),
    }


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
    class_names = read_class_names(raw_dir / "classes.txt")

    info_path = out_dir / "prepare_info.json"
    if info_path.exists():
        info = load_json(info_path)
        _check_settings(info["settings"], settings, out_dir)
        write_data_yaml(out_dir, class_names)  # refresh absolute path in case the folder moved
        logger.info("Dataset already prepared: %s", out_dir)
        return info

    splits_path = out_dir / "splits.json"
    if splits_path.exists():
        saved = load_json(splits_path)
        _check_settings(saved["settings"], settings, out_dir)
        splits = {s: [raw_dir / "images" / name for name in names] for s, names in saved["splits"].items()}
        logger.info("Resuming dataset preparation in %s", out_dir)
    else:
        splits = split_images(list_images(raw_dir / "images"), settings["split"], settings["split_seed"], settings["max_images"])
        save_json({"settings": settings, "splits": {s: [p.name for p in paths] for s, paths in splits.items()}}, splits_path)

    sam = None
    fallback_log = out_dir / "sam_fallbacks.txt"
    for split, image_paths in splits.items():
        for image_path in tqdm(image_paths, desc=f"SAM masks [{split}]", unit="img"):
            label_path = out_dir / "labels" / split / f"{image_path.stem}.txt"
            if label_path.exists():
                continue
            _link_or_copy(image_path, out_dir / "images" / split / image_path.name)
            boxes = read_yolo_boxes(raw_dir / "labels" / f"{image_path.stem}.txt")
            if sam is None and boxes:
                from ultralytics import SAM

                sam = SAM(ds_cfg.sam_model)
            instances, fallbacks = sam_instances(sam, image_path, boxes)
            if fallbacks:
                with open(fallback_log, "a", encoding="utf-8") as f:
                    f.writelines(f"{split}/{image_path.stem} box={i}\n" for i in fallbacks)
            write_seg_labels(label_path, instances)  # written last: marks this image as done

    write_data_yaml(out_dir, class_names)
    num_fallbacks = len(fallback_log.read_text(encoding="utf-8").splitlines()) if fallback_log.exists() else 0
    info = {
        "dataset": ds_cfg.name,
        "settings": settings,
        "class_names": class_names,
        "splits": dataset_stats(out_dir, class_names),
        "sam_fallback_boxes": num_fallbacks,
    }
    save_json(info, info_path)
    logger.info(
        "Prepared %s: %s (SAM fallbacks: %d boxes)",
        ds_cfg.name, {s: v["num_images"] for s, v in info["splits"].items()}, num_fallbacks,
    )
    return info


def _check_settings(saved: dict, current: dict, out_dir: Path) -> None:
    if saved != current:
        raise ValueError(
            f"{out_dir} was prepared with different settings.\n  saved:   {saved}\n  current: {current}\n"
            f"Delete that folder or point `prepared_dir` somewhere else."
        )
