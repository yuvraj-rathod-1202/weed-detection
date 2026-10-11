# Weed Instance Segmentation for Laser Targeting

A laser weeder needs **one point per weed**, not a box. This repo trains instance-segmentation
models (YOLO-seg) on weed images and turns every predicted weed mask into a single laser target
point (distance transform, centroid, skeleton, or box centre), then evaluates both the masks
and the points. Research context: [problem-statement.md](problem-statement.md).

```
image ─► YOLO-seg ─► weed masks ─► point extraction ─► (x, y) per weed ─► laser
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                     # unit tests
```

Data: put CottonWeedDet12 in `data/raw/cottonweeddet12/` (see [data/README.md](data/README.md)).
Model and SAM weights are downloaded automatically by ultralytics.

## Workflow

| Step | Command |
|---|---|
| 1. Check requirements | `python scripts/verify_setup.py --config configs/experiments/baseline_yolo11n_seg.yaml` |
| 2. Smoke test (16 images, 1 epoch) | `python scripts/smoke_test.py` |
| 3. Prepare data (optional, run by step 4 too) | `python scripts/prepare_data.py --dataset configs/datasets/cottonweeddet12.yaml` |
| 4. Explore data + GT | `notebooks/01_data_exploration.ipynb` |
| 5. Run experiment | `python scripts/run_experiment.py --config configs/experiments/baseline_yolo11n_seg.yaml` |
| 6. Inspect model input/output | `notebooks/02_model_outputs.ipynb` |
| 7. Compare experiments | `python scripts/compare_experiments.py` → `results/` |

Any config value can be overridden from the CLI: `--override train.batch=32 eval.conf=0.3`.

**Auto-save / resume:** every stage saves as it goes (SAM labels per image, `last.pt` every epoch,
one JSON line per evaluated image). If a run stops, run **the same command again** to continue.

## Outputs

```text
experiments/<name>/
├── config.yaml              config snapshot
├── results.json             ALL final metrics (structure below), used for plots/report
├── train/                   ultralytics training output (weights/best.pt, last.pt, results.csv)
├── train_summary.json       epochs, best epoch, train time, per-epoch history
├── eval/predictions.jsonl   per test image: predictions, target points, hits, errors, latency
├── eval/visualizations/     [ground truth | prediction] images with laser points
├── environment.json, run.log
results/comparison.csv|json, results/plots/   (from compare_experiments.py)
```

`results.json` keys: `experiment`, `dataset` (split sizes, per-class counts), `model` (params,
GFLOPs, size), `training`, `segmentation` (box/mask P, R, mAP50, mAP50-95, per class),
`targeting` (per method: `shot_precision`, `weed_hit_rate`, `shot_f1`, `point_error_px`,
`point_error_rel`, extraction time; plus inference ms and FPS), `environment`.

## Metrics

| Metric | Meaning |
|---|---|
| mask / box mAP50, mAP50-95 | standard COCO-style segmentation / detection quality (ultralytics val, test split) |
| `shot_precision` | fraction of fired points that land inside a GT weed mask |
| `weed_hit_rate` | fraction of GT weeds that receive at least one point |
| `point_error_px` | distance of predicted point from the GT point (same method on the GT mask), for IoU-matched weeds |
| `point_error_rel` | `point_error_px` / weed radius √(area/π); < 1 ≈ inside the weed |
| `end_to_end_ms_per_image` | inference + point extraction (primary method) |

## Running on Kaggle

```bash
# Kaggle notebook with GPU; add the dataset as input first
!git clone <this-repo> /kaggle/working/weed-detection
%cd /kaggle/working/weed-detection
!pip install -q -r requirements.txt
!python scripts/verify_setup.py --config configs/experiments/baseline_yolo11n_seg.yaml --override dataset.raw_dir=/kaggle/input/<dataset>
!python scripts/run_experiment.py --config configs/experiments/baseline_yolo11n_seg.yaml --override dataset.raw_dir=/kaggle/input/<dataset>
```

- Everything is written under `/kaggle/working/` and persists when you **Save Version**.
- Sessions are capped at 12 h: set `train.time: 10` so training stops in time for data prep + evaluation.
- To resume in a new session: attach the previous notebook output as input, copy its
  `experiments/` and `data/processed/` folders back to the same paths under
  `/kaggle/working/weed-detection/`, and run the same command.
- Two T4 GPUs: `--override train.device=[0,1]`.

## Docs

- [docs/experiments.md](docs/experiments.md): add or change experiments, targeting methods, datasets
- [configs/README.md](configs/README.md): config reference
