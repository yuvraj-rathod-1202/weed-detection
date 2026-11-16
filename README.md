# AI-Based Weed Localization for Precision Weed Removal

> Given one top-down photo of a crop field, can a model place one laser point on each weed, so that most
> weeds are hit and almost no crop plants are?

**Website (full write-up, figures and interactive demos):**
**https://yuvraj-rathod-1202.github.io/weed-detection/**

ES 335 course project · Yuvraj Rathod

---

## Results at a glance

Held-out test fields of the CropAndWeed dataset: 1,201 images, 6,547 weeds, 3,555 crop plants. The model and
its confidence threshold were chosen on the validation split.

| | Weeds hit | Shots on crops | Shots on bare soil |
|---|---|---|---|
| **Final model** (YOLO11s-seg, 1280 px, threshold 0.05) | **85.2 %** (95 % CI 82.8–87.4) | **1.0 %** (CI 0.4–2.1) | 28.7 % |
| Baseline: aim at every green blob (no learning) | 66.8 % | 23.0 % | 32.8 % |

- **The model's real value is crop safety.** It hits more weeds than the baseline and puts about 20× fewer
  shots on crops. At most 3.4 % of crop plants are ever hit.
- **The cost is wasted shots.** More than a quarter of shots land on bare soil at the chosen threshold.
- **It does not transfer well.** On photos from a new farm and camera (Sugar Beets 2016), far fewer weeds
  are hit.

## How it works

```
photo ─► YOLO11s-seg ─► crop + weed masks ─► one aim point per weed ─► crop safety gate ─► laser shots
```

1. An instance-segmentation model outlines every **crop** and **weed** plant.
2. Each predicted weed gets one **aim point** inside its mask.
3. The **crop safety gate** cancels any shot within 20 px of a predicted crop.
4. A shot only fires if the model is at least as confident as the **threshold**.

## Repository layout

```
weed-detection/
├── site/                 website (Quarto): index.qmd, data/*.json, gallery images
├── src/                  all pipeline code
│   ├── data/             dataset conversion, per-plant masks, split
│   ├── targeting/        aim points, crop gate, threshold sweep
│   ├── training.py, evaluation.py, split_eval.py
│   ├── selection.py      model + threshold selection rule
│   └── baseline.py       non-learned green-blob baseline
├── scripts/              one command-line script per step (see "Reproduce")
├── configs/              dataset config (split rule, seed) and one YAML per experiment
├── models/released/      trained weights of every compared model + config snapshots
├── results/              all results: final_selection.json and results/runs/<run>/
├── notebooks/            data exploration, model outputs, results analysis
├── docs/                 Kaggle guides, experiment and label-fix notes
└── tests/                unit tests
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
python -m pytest                # unit tests, no data needed
```

Training and the final evaluation need a GPU; they ran on Kaggle (2 × Tesla T4). Everything else runs on a
CPU. YOLO11 and SAM 2.1 base weights download automatically.

## Data

| Dataset | Used for | Licence | How to get it |
|---|---|---|---|
| **CropAndWeed** (Steininger et al., WACV 2023) | main dataset: train / validation / test | non-commercial, no redistribution | `python scripts/convert_cropandweed.py --download --src data/raw/cropandweed_src --out data/raw/cropandweed` (~11.5 GB; add `--parts 1` for a quarter) |
| **CottonWeedDet12** (Lu et al., 2023) | earlier weeds-only experiments; gallery | CC BY-NC 4.0 | download from [Zenodo 7535814](https://zenodo.org/records/7535814) into `data/raw/cottonweeddet12/` (layout in [data/README.md](data/README.md)) |
| **Sugar Beets 2016** (Chebrolu et al., IJRR 2017) | gallery photos from a new farm | CC BY-SA 4.0 | `python scripts/site/fetch_sugarbeets_sample.py` (labels + 10 photos) |

**No CropAndWeed image or annotation is in this repository or on the website**, because its licence forbids
redistribution. Only statistics and model outputs are included. The shot files in `results/runs/` keep each
prediction's confidence and which labelled plant it hit, but no coordinates, so no annotation can be rebuilt
from them.

## Reproduce

**Fast path, no GPU:** the trained weights (`models/released/`) and all evaluation outputs (`results/runs/`)
are committed. These commands rebuild every number and figure on the website:

```bash
python scripts/site/export_site_data.py --skip-annotations   # website tables, curves, CIs, error analysis
python scripts/site/fetch_sugarbeets_sample.py               # gallery photos (needs CottonWeedDet12 too)
python scripts/site/make_gallery.py
cd site && quarto render                                     # Quarto >= 1.9
```

`--skip-annotations` keeps the two files that need the CropAndWeed annotations as committed: the dataset
statistics and the crop-species counts.

**Full pipeline:**

| # | Step | Command | Where |
|---|---|---|---|
| 1 | Download and convert CropAndWeed | see [Data](#data) | Kaggle |
| 2 | Per-plant masks and the split | `python scripts/prepare_data.py --dataset configs/datasets/cropandweed.yaml` | Kaggle |
| 3 | Train exp 3 / 4 / 5 | `python scripts/run_experiment.py --config configs/experiments/<name>.yaml` ([guide](docs/kaggle_cropandweed.md)) | Kaggle, 4–6 h each |
| 4 | Validate all candidates + baseline, select, then test | cells in [docs/kaggle_validation.md](docs/kaggle_validation.md) | Kaggle, ~3 GPU-hours |
| 5 | Copy results into the repo | `python scripts/collect_results.py` | local |
| 6 | Website data | `python scripts/site/export_site_data.py` | local, ~4 min |
| 7 | Gallery (10 photos × 5 models) | `python scripts/site/fetch_sugarbeets_sample.py`, then `python scripts/site/make_gallery.py` | local, ~10 min |
| 8 | Website | `cd site && quarto render`; publish with `quarto publish gh-pages` | local |

Experiment configs: `yolo11s_seg_cropandweed` (exp 3, 1024 px), `yolo11s_seg_cropandweed_1280` (exp 4),
`yolo11s_seg_cropandweed_1280_aug` (exp 5, + flips/rotation).

## Method details

**Labels.** The authors' CropOrWeed2 mapping (Crop / Weed). Other species and "Vegetation" become ignore
regions ([src/data/cropandweed.py](src/data/cropandweed.py)). Per-plant masks are the class pixels inside each
box, with SAM 2.1 only for ambiguous boxes ([src/data/prepare.py](src/data/prepare.py)).

**Split.** Grouped by recording session (regex `^[a-z]+-\d{4}` on the file name), 65 / 20 / 15 % of images,
seed 42 ([configs/datasets/cropandweed.yaml](configs/datasets/cropandweed.yaml)).

| | Train | Validation | Test |
|---|---|---|---|
| Images | 5,224 | 1,609 | 1,201 |
| Sessions | 602 | 190 | 137 |

**Preprocessing.** Images are resized and divided by 255. Nothing is fitted on the data. Augmentation is
used in training only.

**Selection rule (fixed in advance).** For each candidate, take the confidence threshold with the most weeds
hit while at most 2 % of shots land on crops. The final model is the candidate whose best threshold hits the
most weeds. Result: exp 4 at threshold 0.05 ([results/final_selection.json](results/final_selection.json)).
Test was looked at during development, so test numbers may be slightly optimistic.

**Uncertainty.** 95 % intervals come from resampling whole recording sessions (2,000 bootstrap draws).
Differences between models use a paired session bootstrap.

## Where the results are

| File | Contents |
|---|---|
| `results/runs/<run>/eval_val/summary.json` | validation sweep over thresholds 0.05–0.95, optimal threshold |
| `results/runs/<run>/eval_{val,test}/shots.jsonl` | every predicted weed: confidence, crop-gate input, which labelled plant it hit |
| `results/runs/<run>/train_results.csv` | per-epoch losses and validation metrics |
| `results/runs/<run>/results.json` | development-time evaluation (all aim-point methods, mask mAP) |
| `results/runs/baseline_yolo11n_seg`, `yolo11s_seg` | earlier CottonWeedDet12 experiments (exp 1, 2) |
| `site/data/*.json` | the exact numbers shown on the website |

Metric definitions are in the website's *Evaluation* section; the code is in
[src/targeting/sweep.py](src/targeting/sweep.py). A 1920 px run (exp 6) could not be completed with the
available GPU memory and time (30 of 80 epochs in about 6.5 h); its training log is in
`results/runs/yolo11s_seg_cropandweed_1920_aug/`, and it is not part of the comparison. Older notes: [docs/experiments.md](docs/experiments.md),
[docs/label_fixes.md](docs/label_fixes.md).

## AI-use disclosure

I used **Claude Code** (Anthropic) as a coding assistant throughout the project. I used it for writing and
debugging code, analysing results, and drafting documentation and website text. I reviewed and ran all code
myself, checked the key numbers against the raw outputs, and take responsibility for every result and claim.
No other AI tools were used.

## Citations

- D. Steininger, A. Trondl, G. Croonen, J. Simon, V. Widhalm. *The CropAndWeed Dataset: A Multi-Modal
  Learning Approach for Efficient Crop and Weed Manipulation.* WACV 2023.
- Y. Lu et al. *CottonWeedDet12: a 12-class weed dataset of cotton production systems.* Zenodo, 2023,
  doi:10.5281/zenodo.7535814.
- F. Dang, D. Chen, Y. Lu, Z. Li. *YOLOWeeds.* Computers and Electronics in Agriculture 205, 2023.
- N. Chebrolu, P. Lottes, A. Schaefer, W. Winterhalter, W. Burgard, C. Stachniss. *Agricultural robot
  dataset for plant classification, localization and mapping on sugar beet fields.* IJRR 36(10), 2017.
- G. Jocher, J. Qiu. *Ultralytics YOLO11*, 2024. https://github.com/ultralytics/ultralytics (AGPL-3.0)
- N. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024.
- G. E. Meyer, J. C. Neto. *Verification of color vegetation indices for automated crop imaging
  applications.* Computers and Electronics in Agriculture 63(2), 2008.
- [Quarto](https://quarto.org) and [Observable Plot](https://observablehq.com/plot/) for the website.
