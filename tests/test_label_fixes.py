import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

import src.data.prepare as prepare
from src.data.label_fixes import (
    NO_FIXES,
    apply_fixes,
    changes_by_image,
    load_fixes,
    save_fixes,
    split_hashes,
    summarize_fixes,
)
from src.data.labels import box_to_polygon, read_seg_labels
from src.targeting.metrics import evaluate_image

CLASSES = ["Waterhemp", "Carpetweed"]


def fix(image="a.jpg", kind="extra", decision="add", label="Carpetweed", box=(0.2, 0.2, 0.1, 0.1)):
    return {"image": image, "split": "test", "kind": kind, "decision": decision, "label": label,
            **{k: f"{v:.6f}" for k, v in zip(("cx", "cy", "w", "h"), box)}, "conf": "", "source": "t"}


def test_apply_fixes_add_remove_ignore():
    raw = [(0, 0.5, 0.5, 0.2, 0.2), (0, 0.8, 0.8, 0.1, 0.1)]
    changes = [
        fix(decision="add", box=(0.2, 0.2, 0.1, 0.1)),
        fix(kind="missed", decision="remove", label="Waterhemp", box=(0.5, 0.5, 0.2, 0.2)),
        fix(kind="missed", decision="ignore", label="Waterhemp", box=(0.8, 0.8, 0.1, 0.1)),
        fix(kind="extra", decision="ignore", box=(0.1, 0.9, 0.05, 0.05)),
    ]
    boxes, ignore = apply_fixes(raw, changes, CLASSES)
    assert boxes == [(1, 0.2, 0.2, 0.1, 0.1)]
    assert ignore == [(0.8, 0.8, 0.1, 0.1), (0.1, 0.9, 0.05, 0.05)]


def test_extra_ignore_never_deletes_a_gt_box():
    raw = [(0, 0.5, 0.5, 0.2, 0.2)]
    boxes, ignore = apply_fixes(raw, [fix(decision="ignore", box=(0.5, 0.5, 0.2, 0.2))], CLASSES)
    assert boxes == raw and len(ignore) == 1


def test_apply_fixes_errors():
    with pytest.raises(ValueError, match="no GT box"):
        apply_fixes([], [fix(kind="missed", decision="remove")], CLASSES)
    with pytest.raises(ValueError, match="unknown class"):
        apply_fixes([], [fix(label="Dandelion")], CLASSES)


def test_fix_file_roundtrip_and_validation(tmp_path):
    path = tmp_path / "fixes.csv"
    rows = [fix(image="b.jpg"), fix(image="a.jpg", decision="")]
    save_fixes(rows, path)
    loaded = load_fixes(path)
    assert [r["image"] for r in loaded] == ["a.jpg", "b.jpg"]
    assert list(changes_by_image(loaded)) == ["b"]  # undecided rows change nothing
    assert load_fixes(tmp_path / "missing.csv") == []
    path.write_text(path.read_text().replace(",add,", ",maybe,"))
    with pytest.raises(ValueError, match="unknown decision"):
        load_fixes(path)


def test_split_hash_changes_only_for_affected_split():
    splits = {"train": ["t1"], "test": ["a"]}
    empty = summarize_fixes({}, splits, None)
    assert split_hashes(empty, splits) == split_hashes(None, splits) == {"train": NO_FIXES, "test": NO_FIXES}
    one = summarize_fixes(changes_by_image([fix()]), splits, None)
    assert one["splits"]["test"]["add"] == 1
    assert split_hashes(one, splits)["train"] == NO_FIXES
    assert split_hashes(one, splits)["test"] != NO_FIXES


def test_prediction_in_ignore_region_is_not_a_shot():
    gt = np.zeros((100, 100), bool)
    gt[10:30, 10:30] = True
    on_weed = gt.copy()
    unlabeled = np.zeros((100, 100), bool)
    unlabeled[60:80, 60:80] = True
    ignore = np.zeros((100, 100), bool)
    ignore[55:85, 55:85] = True

    without = evaluate_image([on_weed, unlabeled], [gt], ["centroid"], 0.5)["methods"]["centroid"]
    record = evaluate_image([on_weed, unlabeled], [gt], ["centroid"], 0.5, [ignore])
    assert (without["shots"], without["shots_on_weed"]) == (2, 1)
    assert record["ignored_preds"] == [1]
    assert (record["methods"]["centroid"]["shots"], record["methods"]["centroid"]["shots_on_weed"]) == (1, 1)


def test_prepare_applies_fixes_and_redoes_only_changed_images(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir()
    (raw / "classes.txt").write_text("\n".join(CLASSES))
    for i in range(10):
        cv2.imwrite(str(raw / "images" / f"img{i}.jpg"), np.zeros((40, 40, 3), np.uint8))
        (raw / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    calls = []

    def fake_sam(sam, image_path, boxes):
        calls.append(image_path.stem)
        return [prepare.Instance(b[0], box_to_polygon(*b[1:])) for b in boxes], []

    monkeypatch.setattr(prepare, "sam_instances", fake_sam)
    monkeypatch.setattr("ultralytics.SAM", lambda path: object())
    cfg = OmegaConf.create({
        "name": "toy", "raw_dir": str(raw), "prepared_dir": str(tmp_path / "prep"),
        "label_fixes": str(tmp_path / "fixes.csv"), "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        "split_seed": 0, "max_images": None, "sam_model": "unused.pt",
    })
    info = prepare.prepare_dataset(cfg)
    assert len(calls) == 10 and split_hashes(info["label_fixes"], info["splits"])["test"] == NO_FIXES

    test_image = (tmp_path / "prep" / "images" / "test").iterdir().__next__().name
    stem = test_image.rsplit(".", 1)[0]
    save_fixes([fix(image=test_image, box=(0.1, 0.1, 0.1, 0.1)),
                fix(image=test_image, decision="ignore", box=(0.9, 0.9, 0.1, 0.1))], cfg.label_fixes)
    calls.clear()
    info = prepare.prepare_dataset(cfg)
    assert calls == [stem]
    assert len(read_seg_labels(tmp_path / "prep" / "labels" / "test" / f"{stem}.txt")) == 2
    assert info["splits"]["test"]["num_ignore_regions"] == 1
    assert split_hashes(info["label_fixes"], info["splits"])["test"] != NO_FIXES

    calls.clear()
    prepare.prepare_dataset(cfg)  # unchanged fixes: nothing to redo
    assert calls == []

    save_fixes([], cfg.label_fixes)  # fixes withdrawn: back to the raw labels
    info = prepare.prepare_dataset(cfg)
    assert calls == [stem]
    assert len(read_seg_labels(tmp_path / "prep" / "labels" / "test" / f"{stem}.txt")) == 1
    assert info["splits"]["test"]["num_ignore_regions"] == 0


def _run(tmp_path, trained=True, finished=True):
    from src.experiment import check_label_fixes

    run_dir = tmp_path / "run"
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "results.json").write_text("{}")
    if trained:
        (run_dir / "train").mkdir()
    if trained and finished:
        (run_dir / "train_summary.json").write_text("{}")
    cfg = OmegaConf.create({"name": "exp", "eval": {"split": "test"}})
    splits = {"train": [], "val": [], "test": []}

    def check(changed_split=None):
        summary = {"splits": {s: {"hash": "abc" if s == changed_split else NO_FIXES} for s in splits}}
        return check_label_fixes(cfg, run_dir, {"splits": splits, "label_fixes": summary}, logger=__import__("logging").getLogger())

    return run_dir, check


def test_changed_test_fixes_recompute_evaluation(tmp_path):
    run_dir, check = _run(tmp_path)
    check()  # run from before label fixes existed: nothing changed
    assert (run_dir / "results.json").exists()
    check("test")
    assert not (run_dir / "eval").exists() and not (run_dir / "results.json").exists()
    assert (run_dir / "results_test-labels-none.json").exists()  # original-label results are kept


def test_changed_train_fixes(tmp_path):
    run_dir, check = _run(tmp_path)  # finished: evaluated as it is, original train labels stay recorded
    state = check("train")
    assert state["trained_with"]["train"] == NO_FIXES and (run_dir / "results.json").exists()

    _, check_running = _run(tmp_path / "running", finished=False)  # resuming would mix labels
    with pytest.raises(ValueError, match="changed while 'exp' was training"):
        check_running("train")

    _, check_new = _run(tmp_path / "new", trained=False)  # not trained yet: trains on the new labels
    assert check_new("train")["trained_with"]["train"] == "abc"


def test_old_flat_label_fix_state_is_read(tmp_path):
    run_dir, check = _run(tmp_path)
    (run_dir / "label_fixes.json").write_text('{"train": "none", "val": "none", "test": "abc"}')
    state = check("test")  # test hash "abc" already evaluated: nothing to redo
    assert state == {"trained_with": {"train": NO_FIXES, "val": NO_FIXES}, "evaluated_with": "abc"}
    assert (run_dir / "results.json").exists()


def test_add_duplicating_a_gt_box_is_refused():
    raw = [(0, 0.5, 0.5, 0.2, 0.2)]
    with pytest.raises(ValueError, match="duplicate detection"):
        apply_fixes(raw, [fix(box=(0.501, 0.5, 0.2, 0.2))], CLASSES)
