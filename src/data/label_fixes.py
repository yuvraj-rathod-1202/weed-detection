"""Manual label corrections, kept next to (not inside) the raw data.

data/label_fixes/<dataset>.csv has one row per reviewed candidate (written by scripts/review_labels.py):

    image     image file name
    split     split the image is in (informational)
    kind      extra  = the model predicted a weed where the GT has none (missing label?)
              missed = a GT weed the model did not find (wrong label?)
    decision  add    -> (extra)  new GT weed of class `label` with this box
              remove -> (missed) delete the GT box at this position: not a weed
              ignore -> weed, but species/extent unsure: an ignore region at this box (a `missed`
                        GT box is deleted). Predictions inside it count neither as hit nor as miss.
              none   -> reviewed, nothing to change
              empty  -> not reviewed yet
    label     class name (for `add`; edit it in the CSV if the predicted species is wrong)
    cx cy w h normalized YOLO box
    conf      model confidence (extra only)
    source    experiment that proposed the row
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

FIELDS = ["image", "split", "kind", "decision", "label", "cx", "cy", "w", "h", "conf", "source"]
DECISIONS = {"add", "remove", "ignore", "none", ""}
CHANGES = {"add", "remove", "ignore"}  # decisions that change the labels
NO_FIXES = "none"  # split hash when no fix applies to it
REMOVE_IOU = 0.9  # a remove/ignore row must overlap the GT box it deletes at least this much

Box = tuple[float, float, float, float]  # normalized cx, cy, w, h


def load_fixes(path: str | Path | None) -> list[dict]:
    """Read a fix file. A missing file or no path = no fixes."""
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = [{k: (row.get(k) or "").strip() for k in FIELDS} for row in csv.DictReader(f)]
    for i, row in enumerate(rows, start=2):
        if row["decision"] not in DECISIONS:
            raise ValueError(f"{path}:{i}: unknown decision '{row['decision']}' (use one of {sorted(DECISIONS - {''})})")
    return rows


def save_fixes(rows: list[dict], path: str | Path) -> None:
    """Write the fix file atomically, sorted by image so diffs stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["image"], r["kind"], float(r["cx"]), float(r["cy"]))))
    os.replace(tmp_path, path)


def row_box(row: dict) -> Box:
    return float(row["cx"]), float(row["cy"]), float(row["w"]), float(row["h"])


def changes_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows that change the labels, grouped by image stem."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row["decision"] in CHANGES:
            out.setdefault(Path(row["image"]).stem, []).append(row)
    return out


def apply_fixes(boxes: list[tuple], changes: list[dict], class_names: list[str]) -> tuple[list[tuple], list[Box]]:
    """Raw GT boxes [(class_id, cx, cy, w, h)] + one image's changes -> (GT boxes, ignore regions).

    Order is kept (deleted boxes drop out, added boxes go last), so the i-th prepared GT instance
    is the i-th returned box.
    """
    boxes, ignore = list(boxes), []
    for row in changes:
        box = row_box(row)
        if row["decision"] == "remove" or (row["decision"] == "ignore" and row["kind"] == "missed"):
            ious = [box_iou(box, b[1:]) for b in boxes]
            best = max(range(len(ious)), key=ious.__getitem__, default=None)
            if best is None or ious[best] < REMOVE_IOU:
                raise ValueError(f"label fix for {row['image']}: no GT box at {box} to {row['decision']} (raw labels changed?)")
            boxes.pop(best)
        if row["decision"] == "ignore":
            ignore.append(box)
        elif row["decision"] == "add":
            if row["label"] not in class_names:
                raise ValueError(f"label fix for {row['image']}: unknown class '{row['label']}' (classes: {class_names})")
            if any(box_iou(box, b[1:]) >= REMOVE_IOU for b in boxes):
                raise ValueError(
                    f"label fix for {row['image']}: 'add' at {box} duplicates an existing GT box. A second prediction "
                    f"on an already labelled weed is a duplicate detection: set its decision to 'none'."
                )
            boxes.append((class_names.index(row["label"]), *box))
    return boxes, ignore


def fingerprint(changes: list[dict]) -> str:
    """Identifies one image's changes; '' = no changes. Used to redo only the images whose fixes changed."""
    return "|".join(sorted(
        f"{r['kind']}:{r['decision']}:{r['label']}:" + ",".join(f"{v:.6f}" for v in row_box(r)) for r in changes
    ))


def summarize_fixes(changes: dict[str, list[dict]], splits: dict[str, list[str]], path) -> dict:
    """Per split: number of changes by decision + a hash that changes whenever that split's labels do."""
    summary = {"file": str(path) if path else None, "splits": {}}
    for split, stems in splits.items():
        fps = sorted(f"{stem}={fingerprint(changes[stem])}" for stem in stems if stem in changes)
        counts = {d: sum(r["decision"] == d for stem in stems for r in changes.get(stem, [])) for d in sorted(CHANGES)}
        digest = hashlib.sha1("\n".join(fps).encode()).hexdigest()[:12] if fps else NO_FIXES
        summary["splits"][split] = {"hash": digest, **counts}
    return summary


def split_hashes(summary: dict | None, splits) -> dict[str, str]:
    """{split: hash}; a summary from before label fixes existed counts as 'no fixes'."""
    saved = (summary or {}).get("splits", {})
    return {s: saved.get(s, {}).get("hash", NO_FIXES) for s in splits}


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_ignore_regions(path: str | Path) -> list[Box]:
    path = Path(path)
    if not path.exists():
        return []
    return [tuple(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ignore_regions(path: str | Path, regions: list[Box]) -> None:
    """One `cx cy w h` line per region; no regions = no file."""
    path = Path(path)
    if not regions:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(" ".join(f"{v:.6f}" for v in box) + "\n" for box in regions), encoding="utf-8")
