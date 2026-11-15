# Fixing GT label errors

CottonWeedDet12's boxes are incomplete. Small seedlings next to the labelled plant are often
unlabelled. SAM only masks boxed weeds, so these seedlings have no GT mask. A model that finds
them is counted as wrong, which lowers `shot_precision`. The train split has the same gaps and
teaches the model to ignore small weeds.

Corrections are stored in `data/label_fixes/<dataset>.csv` (`label_fixes` in the dataset config).
That file is version-controlled. The raw data is never edited, so you can report results on both
the original and the corrected labels.

## Workflow

```bash
# 1. Let a trained model propose candidates on the test split (CPU is fine, ~1 min)
python scripts/review_labels.py find --config configs/experiments/baseline_yolo11n_seg.yaml

# 2. Decide each candidate with one key press (opens a window; progress is saved after every key)
python scripts/review_labels.py review --config configs/experiments/baseline_yolo11n_seg.yaml

# 3. Apply the fixes and re-evaluate (SAM runs only for changed images; no retraining)
python scripts/run_experiment.py --config configs/experiments/baseline_yolo11n_seg.yaml
```

`find` adds two kinds of rows:

| kind | What it is | Keys in `review` |
|---|---|---|
| `extra` | the model predicted a weed where the GT has none | **a** add as weed · **i** ignore · **n** not a weed |
| `missed` | a GT weed the model did not find | **k** keep (GT is right) · **r** remove (not a weed) · **i** ignore |

Other keys: **z** undoes the last decision, **q** quits. Run `review` again to continue.

The window shows the ground truth (left) and the model prediction (right), full image on top and
zoomed in on the candidate below. Red is the current GT, green is the model, grey is an ignore
region, and yellow is the candidate. The model is the experiment's `best.pt` (or `--weights`).

## Which decision when

- **add**: clearly a weed. It gets the predicted species. If the species is wrong, change `label`
  in the CSV to one of the names in `classes.txt`. SAM then makes its mask from the box, like
  every other GT weed.
- **ignore**: a weed, but you can't tell the species, or the box covers several plants or a
  half-visible one. It becomes an *ignore region*. A prediction mostly inside it is neither a shot
  nor a miss. A `missed` GT box marked ignore is deleted and replaced by the region.
- **remove**: the GT box is not a weed (soil, crop, mislabel).
- **none** (`n` / `k`): nothing changes. The row is kept, so the next `find` does not ask again.

When unsure, choose **ignore**. It is the honest option and changes the metrics least.

## What the fixes change

| Output | Effect |
|---|---|
| `labels/`, mask/box mAP (ultralytics) | added weeds become GT; removed ones disappear. Ultralytics has no ignore regions, so a prediction in an ignore region still counts as a false positive in mAP |
| targeting metrics | added/removed GT as above, and predictions inside ignore regions are left out (`num_ignored_preds` in `results.json`) |
| `results.json` → `dataset.label_fixes` | per split: number of add/remove/ignore and a hash, so every result records which labels it used |

Each run remembers the label-fix hash per split (`experiments/<name>/label_fixes.json`):

- **test labels changed:** the evaluation is recomputed, and the trained model is kept. The
  previous `results.json` is kept as `results_test-labels-<old hash>.json` (`none` = original labels).
- **train/val labels changed after training:** the run is refused. Train a new experiment (new
  `name`) on the corrected data instead.

## Reporting

- Report the original-label numbers too: `experiments/<name>/results_test-labels-none.json`.
- In the paper, state how many test images and boxes were reviewed, and how many were added,
  removed or ignored. `verify_setup.py` and `results.json` print these counts.
- Model-assisted review only finds weeds *some* model detects. Seedlings no model predicts stay
  unlabelled. Say so in the paper, or review a random sample of images fully by hand.

## Later: the train split

`find --split train` works the same way, but gives ~7× more rows. Corrected train labels need a
**new experiment** (new `name`), because a model trained on the old labels can't be compared fairly.
A cheaper option is to accept only confident candidates. Use `find --split train --conf 0.7`,
then review.
