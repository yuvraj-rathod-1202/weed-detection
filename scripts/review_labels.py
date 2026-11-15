#!/usr/bin/env python
"""Find and fix GT label errors with a trained model. Decisions go to the dataset's label-fix file
(`label_fixes` in the dataset config), never into the raw data. Guide: docs/label_fixes.md.

    find    run the model on a split; every disagreement with the GT becomes an undecided row:
              extra  = predicted weed with no GT weed (missing label?)
              missed = GT weed the model did not find (wrong label?)
            a candidate already in the file (same image, overlapping box) is not added again
    review  show the undecided rows one by one (GT and model prediction, full image + zoom);
            decide with one key, saved immediately

Then re-run the experiment: the fixes are applied (SAM masks for added weeds) and the evaluation
is recomputed without retraining.

Usage:
    python scripts/review_labels.py find   --config configs/experiments/baseline_yolo11n_seg.yaml
    python scripts/review_labels.py review --config configs/experiments/baseline_yolo11n_seg.yaml
    python scripts/review_labels.py find   --config ... --split train --conf 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from tqdm import tqdm

from src.data.label_fixes import apply_fixes, box_iou, changes_by_image, load_fixes, row_box, save_fixes
from src.data.labels import list_images, read_yolo_boxes
from src.data.prepare import prepare_dataset
from src.evaluation import load_gt_masks, load_ignore_masks, predict_instances
from src.experiment import run_dir_for
from src.targeting.metrics import evaluate_image
from src.utils.config import load_experiment_config
from src.utils.io import load_json
from src.visualization import GT_COLOR, IGNORE_COLOR, PRED_COLOR, draw_instances, load_rgb

DUPLICATE_IOU = 0.5  # a candidate overlapping a row of the same image this much is already reviewed
CANDIDATE_COLOR = (255, 200, 0)
KEYS = {
    "extra": {"a": "add", "i": "ignore", "n": "none"},
    "missed": {"k": "none", "r": "remove", "i": "ignore"},
}
HELP = {
    "extra": "model found a weed the GT does not have\n[a] add as {label}    [i] ignore (weed, species unsure)    [n] not a weed",
    "missed": "GT weed the model did not find\n[k] keep, GT is right    [r] remove, not a weed    [i] ignore (unsure)",
}


def find(cfg, split: str, conf: float, weights: Path) -> None:
    from ultralytics import YOLO

    ds, fixes_path = cfg.dataset, cfg.dataset.label_fixes
    class_names = prepare_dataset(ds)["class_names"]  # applies the current fixes, so the GT below is up to date
    prepared_dir, raw_dir = Path(ds.prepared_dir), Path(ds.raw_dir)
    rows = load_fixes(fixes_path)
    changes = changes_by_image(rows)
    seen: dict[str, list] = {}
    for row in rows:
        seen.setdefault(Path(row["image"]).stem, []).append(row_box(row))

    model = YOLO(str(weights))
    new_rows = []
    for image_path in tqdm(list_images(prepared_dir / "images" / split), desc=f"Finding candidates [{split}]", unit="img"):
        stem = image_path.stem
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        gt_masks, _ = load_gt_masks(prepared_dir / "labels" / split / f"{stem}.txt", h, w)
        gt_boxes, _ = apply_fixes(read_yolo_boxes(raw_dir / "labels" / f"{stem}.txt"), changes.get(stem, []), class_names)
        if len(gt_boxes) != len(gt_masks):
            raise RuntimeError(f"{stem}: {len(gt_boxes)} GT boxes but {len(gt_masks)} prepared masks; delete {prepared_dir} and prepare again")
        pred = predict_instances(model, image, conf=conf, iou=cfg.eval.iou, imgsz=cfg.eval.imgsz)
        record = evaluate_image(pred.masks, gt_masks, [], cfg.eval.match_iou, load_ignore_masks(prepared_dir, split, stem, h, w))
        matched_pred = {p for p, _, _ in record["matches"]} | set(record["ignored_preds"])
        matched_gt = {g for _, g, _ in record["matches"]}

        candidates = [
            ("extra", xyxy_to_box(pred.boxes[p], w, h), class_names[pred.class_ids[p]], f"{pred.confidences[p]:.3f}")
            for p in range(len(pred.masks)) if p not in matched_pred
        ] + [("missed", tuple(b[1:]), class_names[b[0]], "") for g, b in enumerate(gt_boxes) if g not in matched_gt]
        for kind, box, label, score in candidates:
            if any(box_iou(box, other) >= DUPLICATE_IOU for other in seen.get(stem, [])):
                continue
            seen.setdefault(stem, []).append(box)
            new_rows.append({
                "image": image_path.name, "split": split, "kind": kind, "decision": "", "label": label,
                **{k: f"{v:.6f}" for k, v in zip(("cx", "cy", "w", "h"), box)}, "conf": score, "source": cfg.name,
            })

    save_fixes(rows + new_rows, fixes_path)
    kinds = [r["kind"] for r in new_rows]
    pending = sum(not r["decision"] for r in rows + new_rows)
    print(f"\n{len(new_rows)} new candidates ({kinds.count('extra')} extra, {kinds.count('missed')} missed) -> {fixes_path}")
    print(f"{pending} rows to review: python scripts/review_labels.py review --config {cfg.config_file}")


def review(cfg, weights: Path) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        sys.exit("review needs a desktop window (matplotlib backend is 'agg'); edit the CSV by hand instead.")
    fixes_path = cfg.dataset.label_fixes
    rows = load_fixes(fixes_path)
    pending = [r for r in rows if not r["decision"]]
    if not pending:
        print(f"Nothing to review in {fixes_path}. Run `find` first.")
        return
    used = {k for keys in KEYS.values() for k in keys} | {"z", "q"}
    for name in [k for k in plt.rcParams if k.startswith("keymap.")]:  # free our keys from matplotlib shortcuts
        plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in used]

    prepared_dir, raw_dir = Path(cfg.dataset.prepared_dir), Path(cfg.dataset.raw_dir)
    class_names = load_json(prepared_dir / "prepare_info.json")["class_names"]
    model = None
    if weights.exists():
        from ultralytics import YOLO

        model = YOLO(str(weights))
    else:
        print(f"Weights not found: {weights}; showing the GT only (pass --weights to see predictions)")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.83, wspace=0.03, hspace=0.08)
    state = {"row": pending[0], "history": [], "pred_key": None, "pred": None}

    def predictions(image_path: Path, image_rgb):
        """Model output for the shown image; reused while consecutive rows are on the same image."""
        if model is None:
            return [], []
        if state["pred_key"] != image_path:
            pred = predict_instances(model, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                                     conf=cfg.eval.conf, iou=cfg.eval.iou, imgsz=cfg.eval.imgsz)
            labels = [f"{class_names[c]} {s:.2f}" for c, s in zip(pred.class_ids, pred.confidences)]
            state["pred_key"], state["pred"] = image_path, (pred.masks, labels)
        return state["pred"]

    def show() -> None:
        row = state["row"]
        image_path = prepared_dir / "images" / row["split"] / row["image"]
        image = load_rgb(image_path)
        h, w = image.shape[:2]
        stem = Path(row["image"]).stem
        gt_masks, _ = load_gt_masks(prepared_dir / "labels" / row["split"] / f"{stem}.txt", h, w)
        gt_view = draw_instances(image, load_ignore_masks(prepared_dir, row["split"], stem, h, w), color=IGNORE_COLOR, fill=False)
        gt_view = draw_instances(gt_view, gt_masks, color=GT_COLOR, fill=False)
        pred_masks, pred_labels = predictions(image_path, image)
        pred_view = draw_instances(image, pred_masks, labels=pred_labels, color=PRED_COLOR)
        pred_outline = draw_instances(image, pred_masks, labels=pred_labels, color=PRED_COLOR, fill=False)  # zoom: plant stays visible

        cx, cy, bw, bh = (v * s for v, s in zip(row_box(row), (w, h, w, h)))
        thickness = max(2, round(max(h, w) / 300))
        for view in (gt_view, pred_view, pred_outline):
            cv2.rectangle(view, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), CANDIDATE_COLOR, thickness)
        half = max(bw, bh, 120)
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))

        pred_title = f"model prediction ({len(pred_masks)})" if model else "model prediction (no weights)"
        panels = (
            (axes[0, 0], gt_view, f"ground truth ({len(gt_masks)})"),
            (axes[0, 1], pred_view, pred_title),
            (axes[1, 0], gt_view[y1:y2, x1:x2], "ground truth: zoom"),
            (axes[1, 1], pred_outline[y1:y2, x1:x2], "model prediction: zoom"),
        )
        for ax, img, title in panels:
            ax.clear()
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        conf = f" {float(row['conf']):.2f}" if row["conf"] else ""
        done = sum(bool(r["decision"]) for r in rows)
        warning = ""
        if row["kind"] == "extra":
            overlap = max((box_iou(row_box(row), b[1:]) for b in read_yolo_boxes(raw_dir / "labels" / f"{stem}.txt")), default=0.0)
            if overlap >= DUPLICATE_IOU:
                warning = (f"\nWARNING: overlaps an already labelled weed (box IoU {overlap:.2f}). "
                           "If it is the same plant, this is a duplicate prediction: press [n], not [a].")
        fig.suptitle(
            f"[{done}/{len(rows)} decided]   {row['image']} ({row['split']})   {row['kind']}: {row['label']}{conf}\n"
            "red = GT, green = model, grey = ignore region, yellow = candidate\n"
            + HELP[row["kind"]].format(label=row["label"]) + "\n[z] undo last    [q] quit (progress is saved)" + warning,
            fontsize=11, color="firebrick" if warning else "black",
        )
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        row = state["row"]
        if event.key == "q":
            plt.close(fig)
            return
        if event.key == "z":
            if state["history"]:
                state["row"] = state["history"].pop()
                state["row"]["decision"] = ""
                save_fixes(rows, fixes_path)
                show()
            return
        decision = KEYS[row["kind"]].get(event.key)
        if decision is None:
            return
        row["decision"] = decision
        state["history"].append(row)
        save_fixes(rows, fixes_path)
        remaining = [r for r in rows if not r["decision"]]
        if not remaining:
            plt.close(fig)
            return
        state["row"] = remaining[0]
        show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()

    counts = {d: sum(r["decision"] == d for r in rows) for d in ("add", "remove", "ignore", "none", "")}
    print(f"{fixes_path}: add={counts['add']} remove={counts['remove']} ignore={counts['ignore']} "
          f"no change={counts['none']} still to review={counts['']}")
    if counts["add"] + counts["remove"] + counts["ignore"]:
        print(f"Apply them: python scripts/run_experiment.py --config {cfg.config_file}")


def xyxy_to_box(xyxy, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["find", "review"])
    parser.add_argument("--config", required=True, help="Experiment config YAML (its model proposes the candidates)")
    parser.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. dataset.raw_dir=...")
    parser.add_argument("--split", help="Split to search (find; default: eval.split)")
    parser.add_argument("--conf", type=float, help="Confidence threshold (find; default: eval.conf)")
    parser.add_argument("--weights", help="Model weights (default: <run_dir>/train/weights/best.pt)")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config, args.override)
    if not cfg.dataset.get("label_fixes"):
        sys.exit(f"Dataset config has no `label_fixes` path (e.g. label_fixes: data/label_fixes/{cfg.dataset.name}.csv)")
    weights = Path(args.weights) if args.weights else run_dir_for(cfg) / "train" / "weights" / "best.pt"
    if args.command == "find":
        if not weights.exists():
            sys.exit(f"Weights not found: {weights} (train the experiment first or pass --weights)")
        find(cfg, args.split or cfg.eval.split, args.conf if args.conf is not None else cfg.eval.conf, weights)
    else:
        review(cfg, weights)


if __name__ == "__main__":
    main()
