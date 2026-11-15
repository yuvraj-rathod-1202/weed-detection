import cv2
import numpy as np
import yaml

from src.evaluation import Prediction
from src.split_eval import evaluate_split
from src.utils.io import load_json, read_jsonl


def _square(x0, y0, x1, y1, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def _prepared(tmp_path):
    root = tmp_path / "prepared"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": {0: "Crop", 1: "Weed"}}))
    for stem in ("ave-0001-0001", "ave-0002-0001"):
        cv2.imwrite(str(root / "images" / "val" / f"{stem}.jpg"), np.zeros((64, 64, 3), np.uint8))
        # weed square 8-24, crop square 40-56 (normalized polygons)
        (root / "labels" / "val" / f"{stem}.txt").write_text(
            "1 0.125 0.125 0.375 0.125 0.375 0.375 0.125 0.375\n"
            "0 0.625 0.625 0.875 0.625 0.875 0.875 0.625 0.875\n")
    return root


def _run_dir(tmp_path, prepared):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({
        "name": "fake", "dataset": {"prepared_dir": str(prepared)},
        "eval": {"imgsz": 64, "iou": 0.7, "match_iou": 0.5},
        "targeting": {"crop_classes": ["Crop"], "crop_margin_px": 20}}))
    return run


def test_test_split_summary_has_no_test_chosen_threshold(tmp_path):
    """The threshold is chosen on validation only; a test summary must not offer its own optimum."""
    prepared = _prepared(tmp_path)
    (prepared / "images" / "val").rename(prepared / "images" / "test")
    (prepared / "labels" / "val").rename(prepared / "labels" / "test")
    run = _run_dir(tmp_path, prepared)
    out = evaluate_split(run, "test", prepared, predictor=lambda image: Prediction([_square(8, 8, 24, 24)], [1], [0.9], 5.0))
    summary = load_json(out / "summary.json")
    assert summary["optimal"] is None and len(summary["curve"]) == 91


def test_evaluate_split_writes_records_and_resumes(tmp_path):
    prepared = _prepared(tmp_path)
    run = _run_dir(tmp_path, prepared)
    calls = []

    def predictor(image_bgr):
        calls.append(1)
        return Prediction([_square(8, 8, 24, 24)], [1], [0.9], 5.0)

    out = evaluate_split(run, "val", prepared, predictor=predictor)
    lines = read_jsonl(out / "shots.jsonl")
    assert len(lines) == 2 and lines[0]["num_gt_crops"] == 1 and len(lines[0]["gt_weeds"]) == 1
    summary = load_json(out / "summary.json")
    assert summary["curve"][0]["threshold"] == 0.05 and summary["curve"][0]["weed_hit_rate"] == 1.0
    assert len(summary["curve"]) == 91 and summary["optimal"]["weed_hit_rate"] == 1.0

    n = len(calls)
    evaluate_split(run, "val", prepared, predictor=predictor)
    assert len(calls) == n and len(read_jsonl(out / "shots.jsonl")) == 2
