# AI-Based Weed Localization for Precision Weed Removal

**Question.** Given one top-down RGB photo of a crop field, can a model place one laser point on each weed,
so that most weeds are hit and almost no crop plants are?

**Website (full reasoning, figures and interactions):** https://yuvraj-rathod-1202.github.io/weed-detection/

**Answer on held-out test fields** (CropAndWeed, 1,201 images; model and threshold chosen on validation, test
used once afterwards — earlier development did look at test, see the website): YOLO11s-seg instance segmentation
at 1280 px with a crop safety gate hits **85.2 %** of weeds (95 % CI 82.8–87.4 %) and puts **1.0 %** of its
shots on crops (CI 0.4–2.1 %; at most 3.4 % of crop plants are hit); 28.7 % of shots land on bare soil. A non-learned
baseline that aims at every green blob hits 66.8 % of weeds and puts 23 % of its shots on crops.

```
image ─► YOLO11s-seg ─► crop + weed masks ─► one aim point per weed ─► crop safety gate ─► laser shots
```

ES 335 course project · solo · Yuvraj Rathod.

## Repository map

| Path | What it holds |
|---|---|
| [`site/`](site) | the Quarto website (`index.qmd`), its data (`site/data/*.json`) and gallery images |
| [`src/`](src) | all pipeline code: data conversion and preprocessing (`src/data/`), training and evaluation (`training.py`, `evaluation.py`, `split_eval.py`), aim points, crop gate, threshold sweep (`src/targeting/`), model selection (`selection.py`), baseline (`baseline.py`) |
| [`scripts/`](scripts) | command-line entry points (one per step below) |
| [`configs/`](configs) | dataset configs (split rule, seed) and one YAML per experiment |
| [`models/released/`](models/released) | trained weights of every compared model + their config snapshots |
| [`results/`](results) | **all results**: `final_selection.json`, and per run in `results/runs/<run>/` the config, training curves, development-time test metrics, and the final-protocol validation/test outputs |
| [`notebooks/`](notebooks) | data exploration, model outputs, results analysis |
| [`docs/`](docs) | Kaggle guides (training, validation/selection/test), experiment and label-fix notes, design and plan of the website |
| [`tests/`](tests) | unit tests (`python -m pytest`) |

## Setup

Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                  # unit tests (no data needed)
```

Training and the final evaluation need a GPU and ran on Kaggle (2 × Tesla T4); everything else (site data,
gallery, website) runs on a CPU. YOLO11 and SAM 2.1 base weights download automatically.

## Data access

| Dataset | Use | Licence | How to get it |
|---|---|---|---|
| **CropAndWeed** (Steininger et al., WACV 2023) | main dataset: train / validation / test | non-commercial, **no redistribution** | `python scripts/convert_cropandweed.py --download --src data/raw/cropandweed_src --out data/raw/cropandweed` downloads the official archives (~11.5 GB) and converts them; add `--parts 1` for a quarter |
| **CottonWeedDet12** (Lu et al., Zenodo 7535814) | earlier weeds-only experiments, gallery | CC BY-NC 4.0 | download from Zenodo into `data/raw/cottonweeddet12/` (layout in [data/README.md](data/README.md)) |
| **Sugar Beets 2016** (Chebrolu et al., IJRR 2017) | gallery photos from a new farm (5 hand-picked + 5 random as a check) | CC BY-SA 4.0 | `python scripts/site/fetch_sugarbeets_sample.py` (label zip + 10 photos read by HTTP range) |

Because of its licence, **no CropAndWeed image or annotation is in this repository or on the website** —
only statistics and model outputs. The committed shot files (`results/runs/*/eval_*/shots.jsonl`) keep each
prediction's confidence and which labelled plant (by index) it hit, but drop the aim points and keep only a
size bin per labelled weed, so no annotation can be rebuilt from them. Trained models
are allowed as derivative works for non-commercial use.

## Exact preprocessing and split

- **Labels:** the authors' CropOrWeed2 mapping (Crop / Weed); other species and "Vegetation" become ignore
  regions ([`src/data/cropandweed.py`](src/data/cropandweed.py)).
- **Per-plant masks:** class pixels inside each box from the dataset's semantic masks; SAM 2.1 only for
  ambiguous boxes, cut back to class pixels ([`src/data/prepare.py`](src/data/prepare.py)).
- **Split:** grouped by recording session (regex `^[a-z]+-\d{4}` on the file name), 65 / 20 / 15 % of images,
  seed 42 ([`configs/datasets/cropandweed.yaml`](configs/datasets/cropandweed.yaml)). It gives 5,224 / 1,609 /
  1,201 images in 602 / 190 / 137 sessions, and is rebuilt identically by `prepare_data.py` and by
  `scripts/site/export_site_data.py`.
- **No fitted preprocessing:** images are resized and divided by 255; augmentation is training-only.

## Reproduce the main result

| # | Step | Command | Where it ran |
|---|---|---|---|
| 1 | Download + convert CropAndWeed | see *Data access* | Kaggle `/tmp` |
| 2 | Build per-plant masks and the split | `python scripts/prepare_data.py --dataset configs/datasets/cropandweed.yaml` | Kaggle |
| 3 | Train exp 3 / 4 / 5 | `python scripts/run_experiment.py --config configs/experiments/yolo11s_seg_cropandweed{,_1280,_1280_aug}.yaml` ([guide](docs/kaggle_cropandweed.md)) | Kaggle, 2 × T4, 4–6 h each |
| 4 | Evaluate every candidate + baseline on **validation**, select by the pre-set rule, evaluate the selected model + baseline on **test** | cells in [docs/kaggle_validation.md](docs/kaggle_validation.md): `scripts/evaluate_split.py`, `scripts/select_model.py` | Kaggle, ~3 GPU-hours |
| 5 | Copy results into the repository | `python scripts/collect_results.py` | local |
| 6 | Website data (tables, curves, CIs, errors) | `python scripts/site/export_site_data.py` | local, ~4 min |
| 7 | Gallery (10 photos × 5 models: baseline, exp 2–5) | `python scripts/site/fetch_sugarbeets_sample.py` then `python scripts/site/make_gallery.py` | local CPU, ~5 min |
| 8 | Website | `cd site && quarto render` (Quarto ≥ 1.9); publish with `quarto publish gh-pages` | local |

Steps 1–4 can be skipped: the trained weights are in `models/released/` and all evaluation outputs are in
`results/runs/`. From this repository alone, `python scripts/site/export_site_data.py --skip-annotations`
rebuilds every result number (curves, comparison, test results with intervals, error analysis, training curves)
and keeps two annotation-derived files as committed: the dataset statistics (`site/data/dataset.json`) and the
crop-species counts in `errors.json`, which need the official CropAndWeed annotations (step 1). The gallery
(step 7) needs the CottonWeedDet12 download and the Sugar Beets fetch; `quarto render` then builds the site.

**Selection rule (fixed before looking at validation numbers):** for each candidate, the confidence
threshold with the most weeds hit while at most 2 % of shots land on crops; the final model is the
candidate whose best threshold hits the most weeds. Result: exp 4 (1280 px) at threshold 0.05 →
[`results/final_selection.json`](results/final_selection.json).

## Full results

| Where | What |
|---|---|
| `results/runs/<run>/eval_val/summary.json` | validation sweep (thresholds 0.05–0.95) and optimal threshold per candidate |
| `results/runs/<run>/eval_{val,test}/shots.jsonl` | every predicted weed: confidence, ignored, nearest crop prediction (for the gate), which labelled weed or crop it hit; labelled weeds as size bins (aim points dropped, see Data access) |
| `results/runs/<run>/train_results.csv` | per-epoch training and validation losses and metrics |
| `results/runs/<run>/results.json` | development-time evaluation on test (all aim-point methods, segmentation mAP); see the website's note on test use |
| `results/runs/baseline_yolo11n_seg`, `yolo11s_seg` | earlier CottonWeedDet12 experiments (exp 1, 2) |
| `site/data/*.json` | the exact numbers shown on the website |

Metric definitions (weeds hit, shots on crops, shots on soil, shot precision, mask mAP, time per image)
are on the website's *Evaluation* section; code in [`src/targeting/sweep.py`](src/targeting/sweep.py).

Extra material: a 1920 px run (exp 6, [config](configs/experiments/yolo11s_seg_cropandweed_1920_aug.yaml))
ran out of GPU memory and time and is not part of the comparison. Older developer docs:
[docs/experiments.md](docs/experiments.md), [docs/kaggle.md](docs/kaggle.md),
[docs/label_fixes.md](docs/label_fixes.md), [configs/README.md](configs/README.md).

## AI-use disclosure

I used **Claude Code** (Anthropic) as a coding assistant throughout the project: writing and debugging code
(data conversion, training and evaluation scripts, the threshold sweep and selection, the gallery and the
website), analysing results, and drafting documentation and website text. I reviewed and ran all code
myself, checked the key numbers against the raw outputs, and take responsibility for every result and
claim. No other AI tools were used.

## Citations

- D. Steininger, A. Trondl, G. Croonen, J. Simon, V. Widhalm. *The CropAndWeed Dataset: A Multi-Modal Learning
  Approach for Efficient Crop and Weed Manipulation.* WACV 2023.
- Y. Lu et al. *CottonWeedDet12: a 12-class weed dataset of cotton production systems.* Zenodo, 2023,
  doi:10.5281/zenodo.7535814; F. Dang, D. Chen, Y. Lu, Z. Li. *YOLOWeeds.* Computers and Electronics in
  Agriculture 205, 2023.
- N. Chebrolu, P. Lottes, A. Schaefer, W. Winterhalter, W. Burgard, C. Stachniss. *Agricultural robot dataset
  for plant classification, localization and mapping on sugar beet fields.* IJRR 36(10), 2017.
- G. Jocher, J. Qiu. *Ultralytics YOLO11*, 2024, https://github.com/ultralytics/ultralytics (AGPL-3.0).
- N. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024.
- G. E. Meyer, J. C. Neto. *Verification of color vegetation indices for automated crop imaging applications.*
  Computers and Electronics in Agriculture 63(2), 2008 (ExG − ExR baseline).
- Quarto (https://quarto.org) and Observable Plot (https://observablehq.com/plot/) for the website.
