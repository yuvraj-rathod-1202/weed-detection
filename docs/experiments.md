# Adding and changing experiments

One experiment = one YAML file in `configs/experiments/`. The file name is free; its `name`
decides the output folder `experiments/<name>/`.

## Add an experiment

1. Copy the baseline:
   `cp configs/experiments/baseline_yolo11n_seg.yaml configs/experiments/yolo11s_seg.yaml`
2. Change `name` (must be unique) and what you want to test, e.g.
   ```yaml
   name: yolo11s_seg
   model:
     weights: models/weights/yolo11s-seg.pt   # official names are auto-downloaded
   ```
3. Check and smoke test it:
   ```bash
   python scripts/verify_setup.py --config configs/experiments/yolo11s_seg.yaml
   python scripts/smoke_test.py --config configs/experiments/yolo11s_seg.yaml
   ```
4. Run it: `python scripts/run_experiment.py --config configs/experiments/yolo11s_seg.yaml`
5. Compare: `python scripts/compare_experiments.py`

Typical experiment variations, config-only:

| Idea | Change |
|---|---|
| Other YOLO-seg model | `model.weights: models/weights/yolov8n-seg.pt` (any official ultralytics name downloads there) or your own `.pt` |
| Species-agnostic "weed" model | `train.single_cls: true` |
| Image size / augmentation / optimizer | any key under `train:` ([ultralytics train args](https://docs.ultralytics.com/modes/train/#train-settings)) |
| Different operating point | `eval.conf`, `eval.iou` |
| Other dataset | `dataset: configs/datasets/<other>.yaml` |
| Different seed (variance) | `seed: 1`, and a new `name` |

## Change an existing experiment

| You change | What happens on the next run |
|---|---|
| `eval.*` or `targeting.*` | trained model is kept, evaluation is recomputed |
| `seed`, `dataset.*`, `model.*`, `train.*` | refused (the results would mix two configs). Use a new `name`, or delete `experiments/<name>/` to start over |
| `output_root`, `dataset.raw_dir`, `dataset.prepared_dir`, `description`, `train.device` | allowed (machine-specific) |
| label fixes for the `eval.split` ([docs/label_fixes.md](label_fixes.md)) | trained model is kept, evaluation is recomputed |
| label fixes for train/val after training started | refused, like a training config change |

## Add a targeting (laser point) method

1. In `src/targeting/points.py` write `def my_method(mask) -> (x, y) | None` (mask = bool array).
2. Add it to `POINT_METHODS`.
3. Add its name to `targeting.methods` in the experiment config (the first entry is the primary method).
4. Add a test to `tests/test_points.py`, then run `python -m pytest`.

To add it to an already-trained experiment, list it in `targeting.methods`: only the evaluation
is recomputed.

## Add a dataset

1. Put it in `data/raw/<name>/` as `images/`, `labels/` (YOLO boxes, same file stem) and `classes.txt`.
2. Copy `configs/datasets/cottonweeddet12.yaml` and set `name`, `raw_dir`, `prepared_dir`.
3. Point an experiment's `dataset:` at it. GT masks are created with SAM on the first run.

A dataset that already has polygon masks: put it directly in the prepared layout
(`images/{train,val,test}`, `labels/{train,val,test}` in YOLO-seg format), plus `data.yaml`
and `prepare_info.json`; see `src/data/prepare.py` for their format.

## Add a non-YOLO model (e.g. Mask R-CNN)

Training and evaluation call ultralytics in two places: `src/training.py:train_model` and
`src/evaluation.py:predict_instances` / `segmentation_metrics`. A new framework needs its own
versions of those three functions that return the same things (a trained weights file,
a `Prediction`, the same metrics dict). Targeting metrics and all outputs stay unchanged.
