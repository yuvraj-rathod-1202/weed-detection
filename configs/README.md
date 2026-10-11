# Configs

- `experiments/`: one file per experiment (what `scripts/*.py --config` takes)
- `datasets/`: one file per dataset, referenced by `dataset:` in an experiment

## Experiment keys

| Key | Meaning |
|---|---|
| `name` | unique id; output folder `<output_root>/<name>/` |
| `seed` | seed for python/numpy/torch and ultralytics training |
| `output_root` | where run folders go (default `experiments`) |
| `dataset` | path to a dataset config |
| `model.weights` | `.pt` path; official ultralytics names are auto-downloaded to that path |
| `train.*` | passed unchanged to ultralytics `YOLO.train()` |
| `eval.split` / `eval.imgsz` | split and image size for evaluation |
| `eval.conf` | confidence threshold for firing (targeting metrics) |
| `eval.iou` | NMS IoU |
| `eval.match_iou` | mask IoU for pairing prediction ↔ GT (point error) |
| `eval.max_images` | cap on evaluated images (`null` = all) |
| `eval.num_visualizations` | images saved to `eval/visualizations/` |
| `targeting.methods` | point methods to evaluate; first = primary |

## Dataset keys

| Key | Meaning |
|---|---|
| `raw_dir` | `images/`, `labels/` (YOLO boxes), `classes.txt` |
| `prepared_dir` | generated YOLO-seg dataset |
| `split`, `split_seed` | train/val/test fractions and seed |
| `max_images` | seeded subset (`null` = all) |
| `sam_model` | SAM checkpoint used to turn boxes into GT masks |

Override any key from the CLI: `--override train.epochs=50 dataset.raw_dir=/some/path`.
