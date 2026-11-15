# Project Website and Submission Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-select the model and threshold on the validation split, compute a non-learned baseline, and publish a Quarto website (GitHub Pages) plus README / AI disclosure that cover every ES 335 rubric item.

**Architecture:** New evaluation code saves, per image, every predicted weed shot (confidence, landing place, nearest-crop confidence) at a low confidence floor, so any threshold can be swept offline. One Kaggle session produces these files for the validation split (exp 3–5 + baseline) and, after selection, for test (final model + baseline). Local export scripts turn them into small JSON files and gallery images under `site/`; a single-page Quarto site with a left table of contents reads the JSON with Observable JS (OJS) and is published with `quarto publish gh-pages`.

**Tech Stack:** Python 3.12 (`.venv`), numpy, OpenCV, ultralytics 8.4.155, pytest; Quarto 1.9.38 with OJS / Observable Plot; GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-11-14-project-website-design.md`

## Global Constraints

- No CropAndWeed image pixels (raw or overlaid) anywhere in `site/`, the published site, or new commits. Only statistics and model outputs without image pixels.
- Test split is used only after selection: final model at its threshold, and the baseline. All choices come from validation.
- Optimal threshold rule: highest weed hit rate subject to crop-shot share ≤ **0.02** (2 % of fired shots land on a crop).
- One threshold for everything: at threshold X a shot fires if its confidence ≥ X, it is not ignored, and no crop prediction with confidence ≥ X lies within the crop margin (**20 px**).
- Confidence floor for saved predictions: **0.05**. Sweep grid: 0.05 to 0.95 in steps of 0.01.
- Targeting method for all shots: `center_weighted_dt`.
- Site look: white "clean paper", serif headings, greys; green only = shot hit a weed, red only = shot hit a crop, grey = soil / neutral.
- Gallery: 5 Sugar Beets 2016 (CC BY-SA 4.0) images with the final CropAndWeed model + 5 CottonWeedDet12 test images with `experiments/yolo11s_seg`; random pick with seed 0, stated on the page.
- Exp 6 is not a candidate (future work only).
- AI disclosure: only Claude Code (Anthropic) was used — coding, debugging, analysis, writing assistance; results checked by the author.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Deadline 2026-11-16. Order: Tasks 1–5 first (they unblock the Kaggle session); Tasks 7–8 while it runs; Tasks 6, 9–12 after its results.

## Review Focus

1. A CropAndWeed image slipping into `site/` (e.g. a stray visualization copied for convenience): the build must fail. → test in Task 8.
2. A validation sweep where no threshold meets the 2 % limit (e.g. the baseline, or a weak model): selection must skip it, not crash or pick it. → test in Task 2 and Task 5.
3. Images with zero predictions, zero GT weeds or zero crop predictions (many CropAndWeed images have no weeds): records and sweep must still be valid (rates over totals, no division by zero). → test in Task 2.
4. Re-running an interrupted evaluation on Kaggle: already evaluated images are skipped, not duplicated. → test in Task 3.
5. A prediction whose target point lies on the image border or whose mask is empty: point extraction returns `None`; such predictions are not shots. → test in Task 2.

---

### Task 1: Remove CropAndWeed photos from notebook 03

**Files:**
- Modify: `notebooks/03_results_analysis.ipynb` (section 11 "Look at the images" and all outputs)
- Test: `tests/test_notebooks.py`

**Interfaces:** none.

- [ ] **Step 1: Write the failing test** `test_results_notebook_embeds_no_dataset_photos`: load `notebooks/03_results_analysis.ipynb` as JSON; assert no code cell source contains `"visualizations"` inside an image-display call (`imread`/`imshow`), and no cell output has `image/png` or `image/jpeg` data.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_notebooks.py -v` — Expected: FAIL.
- [ ] **Step 3:** Replace section 11's code with a cell that only prints the visualization file paths and a note ("CropAndWeed licence: photos are not embedded; open the files locally"), then clear all outputs (`jupyter nbconvert --clear-output --inplace`). Charts may stay out; outputs are cleared entirely.
- [ ] **Step 4: Run** the test — Expected: PASS. Run the full suite `.venv/Scripts/python -m pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** `git add notebooks/03_results_analysis.ipynb tests/test_notebooks.py && git commit -m "Remove CropAndWeed photos from results notebook"`

### Task 2: Threshold sweep core (`src/targeting/sweep.py`)

**Files:**
- Create: `src/targeting/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Produces:
  - `shot_records(weed_masks: list[np.ndarray], weed_confs: list[float], crop_masks: list[np.ndarray], crop_confs: list[float], gt_weed_masks: list[np.ndarray], gt_crop_masks: list[np.ndarray], ignored: set[int], margin_px: float, method: str = "center_weighted_dt") -> list[dict]` — one dict per weed prediction with a non-`None` point: `{"conf", "x", "y", "ignored": bool, "gt_weed": int (index of first GT weed mask containing the point, -1 if none), "on_crop": bool (inside any GT crop mask), "max_nearby_crop_conf": float (max confidence of crop predictions whose mask is within margin_px of the point, by distance transform; 0.0 if none)}`.
  - `fired(shots: list[dict], threshold: float) -> list[dict]` — shots with `conf >= threshold`, not ignored, `max_nearby_crop_conf < threshold`.
  - `sweep(records: list[dict], thresholds: list[float]) -> list[dict]` — `records` are per-image dicts `{"image_id", "gt_weeds": [{"size_px"}], "shots": [...]}`; returns per threshold `{"threshold", "shots", "weed_hit_rate", "shot_precision", "crop_shot_share", "soil_shot_share"}` where weed_hit_rate = GT weeds hit by ≥ 1 fired shot / all GT weeds, shot_precision = fired with gt_weed ≥ 0 / fired, crop_shot_share = fired on_crop / fired, soil = fired with gt_weed = -1 and not on_crop / fired; rates are `None` when the denominator is 0.
  - `optimal_threshold(curve: list[dict], max_crop_share: float = 0.02) -> dict | None` — the curve row with the highest weed_hit_rate among rows with shots > 0 and crop_shot_share ≤ max_crop_share; ties → higher threshold; `None` if no row qualifies.
  - `bootstrap_ci(records: list[dict], threshold: float, n: int = 2000, seed: int = 0, level: float = 0.95) -> dict` — resample **sessions** (`image_id[:8]`) with replacement; returns `{"weed_hit_rate": [lo, hi], "crop_shot_share": [lo, hi]}` (percentile intervals).
- Consumes: `target_point`, `point_in_mask` (`src/targeting/points.py`).

- [ ] **Step 1: Write the failing tests** in `tests/test_sweep.py`, using 100×100 synthetic masks:
  - `test_shot_hits_weed_and_records_nearby_crop_conf`: one weed prediction (square 40–60) over a GT weed, one crop prediction (conf 0.3) 10 px away → `gt_weed == 0`, `on_crop is False`, `max_nearby_crop_conf == 0.3` with margin 20, and `0.0` with margin 5.
  - `test_fired_applies_threshold_ignore_and_gate`: shots with (conf 0.4, crop 0.0), (0.4, crop 0.5), (0.2, crop 0.0), (0.9, ignored) → `fired(shots, 0.3)` keeps only the first; at threshold 0.6 the second has conf < 0.6 → none fired.
  - `test_sweep_rates_and_empty_images`: two images, one with 2 GT weeds and 2 shots (one on weed 0, one on crop), one with no GT weeds and no shots → at threshold 0.05: weed_hit_rate 0.5, crop_shot_share 0.5, shots 2; an all-empty record list gives rates `None` and no exception.
  - `test_optimal_threshold_respects_crop_limit`: curve rows (0.1: hit 0.9, crop 0.05), (0.3: hit 0.8, crop 0.02), (0.5: hit 0.7, crop 0.0) → threshold 0.3; with every row's crop share 0.05 → `None`.
  - `test_empty_mask_gives_no_shot`: an all-false weed mask produces no record.
  - `test_bootstrap_ci_brackets_point_estimate`: 10 sessions of 1 image each → lo ≤ point estimate ≤ hi for weed_hit_rate.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_sweep.py -v` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the five functions in `src/targeting/sweep.py` (distance for the crop gate: `cv2.distanceTransform` of each crop mask's inverse, as in `src/targeting/safety.py`).
- [ ] **Step 4: Run** the tests — Expected: PASS; full suite passes.
- [ ] **Step 5: Commit** `git add src/targeting/sweep.py tests/test_sweep.py && git commit -m "Add threshold sweep, optimal-threshold rule and session bootstrap"`

### Task 3: Split evaluation script (`scripts/evaluate_split.py`)

**Files:**
- Create: `src/split_eval.py`, `scripts/evaluate_split.py`
- Test: `tests/test_split_eval.py`

**Interfaces:**
- Consumes: Task 2 functions; `predict_instances`, `split_crops`, `subset`, `crop_class_ids`, `load_gt_masks`, `load_ignore_masks` (`src/evaluation.py`); `match_instances`, `ignored_predictions` (`src/targeting/metrics.py`); `append_jsonl`, `read_jsonl`, `save_json` (`src/utils/io.py`).
- Produces:
  - `evaluate_split(run_dir: Path, split: str, prepared_dir: Path, conf_floor: float = 0.05, max_images: int | None = None, predictor=None) -> Path` — reads `run_dir/config.yaml` (imgsz, iou, match_iou, crop classes, crop margin) and `run_dir/train/weights/best.pt`; writes `run_dir/eval_<split>/shots.jsonl` (one line per image: `{"image_id", "session", "inference_ms", "gt_weeds": [{"size_px": sqrt(mask area)}], "num_gt_crops", "shots": [...]}`) and `run_dir/eval_<split>/summary.json` (`{"run", "split", "conf_floor", "num_images", "curve", "optimal", "inference_ms_per_image"}`); skips images already in `shots.jsonl`; returns the eval dir. `predictor(image_bgr) -> Prediction` lets tests and the baseline (Task 4) replace the YOLO model. Ignored predictions use the existing rule at the floor (unmatched at `match_iou` and ≥ 50 % inside ignore regions).
  - CLI: `python scripts/evaluate_split.py --run experiments/<name> --split val|test --prepared-dir /tmp/cropandweed_seg [--conf-floor 0.05] [--max-images N] [--baseline]`.

- [ ] **Step 1: Write the failing tests:**
  - `test_evaluate_split_writes_records_and_resumes(tmp_path)`: build a tiny prepared dir in `tmp_path` (2 images 64×64, one seg label with a weed square and a crop square, `data.yaml`/classes `Crop, Weed`), a minimal `run_dir/config.yaml` written by the test (`eval: {imgsz: 64, iou: 0.7, match_iou: 0.5}`, `targeting: {crop_classes: [Crop], crop_margin_px: 20}`, `dataset: {prepared_dir: <tmp>}`) — tests must not depend on the gitignored `experiments/`; a fake `predictor` returning the GT weed square with conf 0.9. Run twice → `shots.jsonl` has exactly 2 lines; `summary.json` curve at 0.05 has weed_hit_rate 1.0; second run does not call the predictor (count calls).
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_split_eval.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `evaluate_split` and the CLI (warm-up predict on the first image, untimed, like `targeting_metrics`).
- [ ] **Step 4: Run** the test — PASS. Then a real local check on the smoke data:
  `.venv/Scripts/python scripts/evaluate_split.py --run experiments/yolo11s_seg_cropandweed_1280_aug --split val --prepared-dir data/processed/_smoke/cropandweed --max-images 2`
  Expected: `eval_val/summary.json` with 2 images and a 91-row curve. Delete that `eval_val` folder afterwards (smoke data, not the real split).
- [ ] **Step 5: Commit** `git add src/split_eval.py scripts/evaluate_split.py tests/test_split_eval.py && git commit -m "Add per-shot split evaluation for offline threshold sweeps"`

### Task 4: Excess-green baseline (`src/baseline.py`)

**Files:**
- Create: `src/baseline.py`
- Modify: `scripts/evaluate_split.py` (`--baseline` flag)
- Test: `tests/test_baseline.py`

**Interfaces:**
- Produces:
  - `exg_mask(image_bgr: np.ndarray) -> np.ndarray` — bool mask: ExG = 2g − r − b on chromaticity-normalised channels (r = R/(R+G+B) etc.), Otsu threshold on ExG scaled to 0–255, then morphological opening with a 3×3 kernel.
  - `baseline_prediction(image_bgr: np.ndarray, min_area_px: int = 64) -> Prediction` — connected components (8-connectivity) of `exg_mask` with area ≥ `min_area_px`; every component is a **Weed** prediction with confidence 1.0; no crop predictions.
  - With `--baseline`, `evaluate_split` uses `baseline_prediction` as the predictor and `--run` names an output folder (e.g. `experiments/baseline_exg`) that holds a `config.yaml` written by the script (crop classes `[Crop]`, margin 20, `match_iou` 0.5).
- Values and their reasons (state on the site): `min_area_px = 64` (8×8 px) removes noise specks while keeping the smallest labelled plants (~16 px wide ≈ 256 px²); 3×3 opening; parameters fixed a priori, not tuned.

- [ ] **Step 1: Write the failing tests:** `test_exg_mask_finds_green_on_soil` (brown 100×100 image with a green 20×20 square → mask covers the square, IoU > 0.9); `test_baseline_drops_small_specks` (a 5×5 green speck and a 20×20 square → one prediction); `test_baseline_on_bare_soil_predicts_nothing` (uniform brown → zero predictions, no exception).
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_baseline.py -v` — FAIL.
- [ ] **Step 3: Implement** `exg_mask`, `baseline_prediction`, and the `--baseline` flag.
- [ ] **Step 4: Run** tests — PASS; full suite passes. Local check: `--baseline --run experiments/baseline_exg --split val --prepared-dir data/processed/_smoke/cropandweed --max-images 2` writes a summary; delete it afterwards.
- [ ] **Step 5: Commit** `git add src/baseline.py scripts/evaluate_split.py tests/test_baseline.py && git commit -m "Add excess-green vegetation baseline"`

### Task 5: Release weights, model selection, Kaggle validation guide

**Files:**
- Create: `models/released/<run>/best.pt` and `models/released/<run>/config.yaml` for `yolo11s_seg_cropandweed`, `yolo11s_seg_cropandweed_1280`, `yolo11s_seg_cropandweed_1280_aug`, `yolo11s_seg` (licence allows sharing trained models; ~20 MB each), `models/released/README.md` (what each is, licence note, citation)
- Create: `scripts/select_model.py`, `docs/kaggle_validation.md`
- Modify: `.gitignore` (allow `models/released/`), `docs/kaggle.md` (link)
- Test: `tests/test_select_model.py`

**Interfaces:**
- Consumes: `summary.json` files from Task 3.
- Produces: `select_final(summaries: dict[str, dict]) -> dict | None` in `scripts/select_model.py` — over candidates whose `optimal` is not `None`, the highest `optimal.weed_hit_rate`; ties → lower `inference_ms_per_image`; returns `{"run", "threshold", "val": optimal_row}`; CLI writes `results/final_selection.json` and prints the run and threshold. The baseline is never a candidate.
- `evaluate_split` must accept `--run models/released/<run>` (reads `config.yaml` and `best.pt` from that folder when `train/weights/best.pt` is absent) and write outputs to `--out experiments/<run>` (add `--out`, default = run dir).

- [ ] **Step 1: Write the failing tests:** `test_select_final_picks_highest_hit_rate_within_limit` (three summaries: optimal hit 0.80, 0.85, `None` → the 0.85 run); `test_select_final_none_when_no_candidate_qualifies` (all `None` → `None`).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `select_final`, the CLI, the `--out`/released-folder support, copy the weights + config snapshots, write `models/released/README.md`.
- [ ] **Step 4: Run** tests — PASS. Write `docs/kaggle_validation.md`: cells = clone/pull + pip; convert/download to `/tmp`; `prepare_data.py` (same overrides as before); three `evaluate_split.py --split val` calls (one per released CropAndWeed run, `--out experiments/<run>`) + baseline on val; `select_model.py`; `evaluate_split.py --split test` for the selected run and the baseline; zip `experiments/*/eval_val`, `experiments/*/eval_test`, `results/final_selection.json`. Expected time ~3.5 h; add `timeout 10h` to the eval cell. State that only the selected run is evaluated on test.
- [ ] **Step 5: Commit and push** (the Kaggle notebook clones the repo) `git add models/released scripts/select_model.py tests/test_select_model.py docs/kaggle_validation.md docs/kaggle.md .gitignore src/split_eval.py scripts/evaluate_split.py && git commit -m "Release trained weights; add model selection and Kaggle validation guide" && git push`
- [ ] **Step 6: Hand-off to the user:** ask them to run `docs/kaggle_validation.md` once exp 6 session 1 has freed the GPU, and to put the downloaded zip's contents into `experiments/` and `results/`.

### Task 6: Site data export (`scripts/site/export_site_data.py`)

**Files:**
- Create: `src/site_data.py`, `scripts/site/export_site_data.py`
- Test: `tests/test_site_data.py`

**Interfaces:**
- Consumes: `shots.jsonl` / `summary.json` (Task 3), `results/final_selection.json` (Task 5), `experiments/<run>/train/results.csv`, `data/raw/cropandweed_src/bboxes/*.csv` (annotations, local), `split_images` (`src/data/prepare.py`) with seed 42 and group pattern `^[a-z]+-\d{4}` to rebuild the split, `map_label` (`src/data/cropandweed.py`).
- Produces (all in `site/data/`, small JSON):
  - `dataset.json` — per split: images, sessions, crop / weed instances, ignored plants; plant-size histogram (sqrt box area, bins 0–16–24–32–48–64–128–∞) per class; top crop species counts; `ave`/`vwg` image counts.
  - `curves.json` — validation curve per candidate + baseline point, the 2 % rule's optimum per candidate, the selected run.
  - `comparison.json` — per candidate + baseline: validation weed_hit_rate / shot_precision / crop_shot_share / soil_shot_share at its optimum, val mask mAP50-95 at the best epoch (from `results.csv`), inference ms/image, and a one-line "what changed" note (1024 px; 1280 px; 1280 px + flips/rotation).
  - `test.json` — final run and baseline on test at the chosen threshold, with `bootstrap_ci`.
  - `errors.json` — final run on test: hit rate by plant-size bin, by set (`ave` real fields vs `vwg` experimental plots), crop shots by crop species (species from the source annotations, joined by image and box order), top sessions by crop shots.
  - `training.json` — epoch, train/val seg and cls loss, val mask mAP for the selected run.
  - Functions to test: `size_bins(sizes: list[float]) -> dict`, `hit_rate_by(records, threshold, key: Callable[[dict, dict], str]) -> dict`.

- [ ] **Step 1: Write failing tests** for `size_bins` (values 10, 20, 100 → one each in "0–16", "16–24", "64–128") and `hit_rate_by` (two images from sessions `ave-0001` / `vwg-0002`, keyed by set → correct per-set rates).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** functions + export script. **Step 4: Run** tests — PASS; run `.venv/Scripts/python scripts/site/export_site_data.py` → the six JSON files exist and `test.json` names the run in `results/final_selection.json`.
- [ ] **Step 5: Commit** `git add src/site_data.py scripts/site/export_site_data.py tests/test_site_data.py site/data/*.json && git commit -m "Export site data from validation and test evaluations"`

### Task 7: Gallery images and data

**Files:**
- Create: `src/gallery.py`, `scripts/site/make_gallery.py`, `site/gallery/*.jpg`, `site/data/gallery.json`
- Test: `tests/test_gallery.py`

**Interfaces:**
- Consumes: Task 2 `shot_records`/`fired`; `predict_instances`; final model from `results/final_selection.json` (`models/released/<run>`); cotton model `models/released/yolo11s_seg`; prepared CottonWeedDet12 test split (`data/processed/cottonweeddet12_seg`).
- Produces:
  - `shot_outcome(point, gt_weed_masks, gt_crop_masks) -> str` — `"weed"` | `"crop"` | `"soil"`.
  - `render_output(image_rgb, weed_masks, crop_masks, shots: list[dict]) -> np.ndarray` — predicted weed masks as light translucent grey fill with dark outline, predicted crop masks as dashed/dark grey outline, laser crosshairs green (weed) / red (crop) / grey (soil), vetoed shots as hollow grey circles. No other colours.
  - `render_gt(image_rgb, gt_weed_masks, gt_crop_masks) -> np.ndarray` — GT outlines (weeds solid dark, crops dashed).
  - `gallery.json`: list of `{"id", "dataset", "licence", "attribution_url", "note", "raw", "output", "gt", "weeds", "weeds_hit", "crop_shots", "soil_shots"}`; images resized to 1000 px wide JPEG (quality 85).
- Sugar Beets 2016: download only an annotated sample of ≥ 5 images (RGB + label images) into `data/raw/sugarbeets2016_sample/` (gitignored). Map its labels: sugar beet → crop, every weed class → weed. If no sample can be fetched without downloading tens of GB, **stop and ask the user** before using anything else.
- Selection: seed 0, `random.sample` of 5 from the available annotated Sugar Beets images and 5 from the CottonWeedDet12 test images; threshold = the selected threshold (CottonWeedDet12 uses its own `config.yaml` `eval.conf`, stated in the note).

- [ ] **Step 1: Write failing tests:** `test_shot_outcome_classifies_weed_crop_soil`; `test_render_output_uses_only_allowed_colours` (render on a mid-grey image with one shot of each outcome → every pixel that differs from the input and from greys is either pure green family `(46,125,50)` or red family `(198,40,40)`; define these two constants in `src/gallery.py`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** + `make_gallery.py`. **Step 4: Run** tests — PASS; run the script → 10 entries in `gallery.json`, 30 JPEGs in `site/gallery/`; open two of them and check visually.
- [ ] **Step 5: Commit** `git add src/gallery.py scripts/site/make_gallery.py tests/test_gallery.py site/gallery site/data/gallery.json && git commit -m "Add gallery of Sugar Beets 2016 and CottonWeedDet12 examples"`

### Task 8: Quarto scaffold, asset guard, and the first four sections

**Files:**
- Create: `site/_quarto.yml`, `site/index.qmd`, `site/styles.scss`, `site/references.bib`
- Modify: `.gitignore` (`site/_site/`, `site/.quarto/`)
- Test: `tests/test_site_assets.py`

**Interfaces:**
- Consumes: `site/data/dataset.json` (Task 6; until then Step 3 uses the real file once exported — do not invent numbers).
- Produces: one page `index.qmd` with `toc: true`, `toc-location: left`, `toc-depth: 2`, sections (H2, in order): Question · Data · Preprocessing · Baseline and methods · Evaluation · Results · Errors · Limitations · Conclusion · Reproduce · My contribution and AI use · References. Theme: `cosmo` + `styles.scss` (serif headings e.g. "Source Serif 4", body sans, greys; CSS variables `--hit: #2e7d32`, `--crop: #c62828`, `--soil: #9e9e9e`).

- [ ] **Step 1: Write the failing test** `test_site_images_are_only_permitted_gallery_files`: every `*.jpg|*.jpeg|*.png|*.webp|*.gif` under `site/` (excluding `_site/`, `.quarto/`) must appear in `site/data/gallery.json` (`raw`/`output`/`gt`) and its entry's `dataset` ∈ {"Sugar Beets 2016", "CottonWeedDet12"}; no file name may match `^(ave|vwg)-\d{4}-\d{4}`.
- [ ] **Step 2: Run** with a dummy `site/x.png` present — FAIL; remove it.
- [ ] **Step 3: Write** `_quarto.yml` (`project: type: website`, `output-dir: _site`, site title, repo link in navbar, `page-footer` with licence note), `styles.scss`, and the content of **Question** (spec's question, unit, prediction time, available info, target, limits), **Data** (CropAndWeed source, licence, size, classes, sessions, quality issues: unmapped species → ignore regions, Vegetation label, overlapping boxes → SAM, small plants; CottonWeedDet12 as the earlier weeds-only experiment; Observable Plot charts from `dataset.json`), **Preprocessing** (CropOrWeed2 mapping, ignore regions, semantic → instance masks, SAM only for ambiguous boxes, session-grouped 65/20/15 split with seed 42, no fitted preprocessing statistics, train-only augmentation → no leakage), **Baseline and methods** (excess-green baseline with its fixed parameters and why; YOLO11s-seg and why instance masks; target-point methods; 20 px crop gate; objective = segmentation loss, assumptions).
- [ ] **Step 4: Run** `cd site && quarto render` — Expected: renders without errors; `pytest tests/test_site_assets.py` PASS; open `site/_site/index.html` and check the sidebar lists all sections.
- [ ] **Step 5: Commit** `git add site .gitignore tests/test_site_assets.py && git commit -m "Scaffold Quarto site with question, data, preprocessing and methods"`

### Task 9: Evaluation section with interaction A

**Files:** Modify: `site/index.qmd` (Evaluation section)

**Interfaces:** Consumes `site/data/curves.json`, `site/data/test.json`.

- [ ] **Step 1: Write** the protocol text first (split design and why sessions are grouped; validation for all choices; test once; the 2 % rule stated before any number; metrics defined: weed hit rate, shot precision, crop-shot share, soil-shot share, mask mAP50-95; the honest note that early analysis looked at test and whether the redo changed the choices; validation also used by ultralytics for `best.pt`).
- [ ] **Step 2: Build interaction A** in OJS: `Inputs.range([0.05, 0.95], {step: 0.01, value: <optimum>, label: "Fire when confidence ≥"})`; three numbers (weeds destroyed, shots on soil, shots on crops; coloured per the palette); Observable Plot line of weed_hit_rate (y) vs 1 − shot_precision (x) for the selected run on validation, a moving dot at the slider value, a marked optimum ("optimal: most weeds with ≤ 2 % crop shots"), and the baseline point; beside it, the test result at the chosen threshold with its bootstrap interval.
- [ ] **Step 3: Verify:** `quarto render`; open the page; dragging the slider updates numbers and dot; values at the optimum equal `curves.json`. `pytest -q` passes.
- [ ] **Step 4: Commit** `git commit -am "Add evaluation section with confidence trade-off interaction"`

### Task 10: Results (interaction D + gallery), Errors, Limitations, Conclusion

**Files:** Modify: `site/index.qmd`

**Interfaces:** Consumes `comparison.json`, `test.json`, `errors.json`, `training.json`, `gallery.json`.

- [ ] **Step 1: Interaction D:** OJS `Inputs.select` over metrics (weeds destroyed, shots on crops, shots on soil, mask mAP50-95, inference ms) → horizontal Plot bar chart of baseline + exp 3/4/5 on validation, selected run in green, others grey; `title` tooltips with the "what changed" note. Headline test table (final vs baseline with intervals) and the training-curve chart (overfitting discussion).
- [ ] **Step 2: Gallery viewer:** thumbnail grid (10) → side-by-side raw | output with a "show ground truth" checkbox swapping the output for the GT render, the summary line, dataset/licence/note; keyboard-accessible buttons; stacks on narrow screens.
- [ ] **Step 3: Errors** (from `errors.json`: small weeds, real fields vs plots, crop shots by species — sunflower, soybean — and session concentration, gate cannot protect undetected crops, soil shots), **Limitations** (one dataset/camera, offline, label noise on tiny plants, exp 6 unfinished, test inspected during development, no field trial, prediction ≠ causation), **Conclusion** (answer to the question with the numbers and their uncertainty).
- [ ] **Step 4: Verify:** `quarto render`; dropdown switches bars; every thumbnail opens; GT toggle works; `pytest -q` passes.
- [ ] **Step 5: Commit** `git commit -am "Add results, gallery, errors, limitations and conclusion"`

### Task 11: Reproduce, AI disclosure, references, README

**Files:** Modify: `site/index.qmd`, `README.md`, `site/references.bib`; Create: `docs/viva_notes.md`

- [ ] **Step 1: Site sections:** **Reproduce** (environment, data download commands, split rule + seed, training configs, Kaggle guides, `evaluate_split.py` / `select_model.py` / export / gallery / `quarto render` commands, where results live), **My contribution and AI use** (solo; Claude Code only, for coding, debugging, analysis and writing; all results checked by me), **References** (CropAndWeed WACV 2023; CottonWeedDet12; Sugar Beets 2016 IJRR 2017; ultralytics YOLO11; SAM 2; Quarto; Observable Plot).
- [ ] **Step 2: README:** project summary + live site link placeholder → real URL in Task 12; setup; data access per dataset (no redistribution of CropAndWeed); split details; rebuild results (Kaggle guides + `evaluate_split.py` + `select_model.py`); rebuild site (`export_site_data.py`, `make_gallery.py`, `quarto render`, `quarto publish gh-pages`); code map (`src/`, `scripts/`, `configs/`, `site/`, `notebooks/`); released weights; AI-use disclosure; citations.
- [ ] **Step 3:** `docs/viva_notes.md`: key choices and why, evaluation design, a limitation, and three "small change" answers (e.g. crop limit 1 % instead of 2 %, 20 px → 10 px margin, imgsz 1024 instead of 1280) with what the curves show.
- [ ] **Step 4: Verify:** `quarto render` clean; every README command exists (`ls` the scripts); `pytest -q` passes.
- [ ] **Step 5: Commit** `git commit -am "Add reproduction steps, AI disclosure, references and README"`

### Task 12: Publish and final rubric check

- [ ] **Step 1:** `cd site && quarto publish gh-pages --no-prompt`; in GitHub repo settings, Pages source = `gh-pages` branch. Expected: live at `https://yuvraj-rathod-1202.github.io/weed-detection/`.
- [ ] **Step 2:** Open the live URL: every section, the slider, the comparison dropdown and all 10 gallery items work; no console errors; check on a narrow window.
- [ ] **Step 3:** Put the live URL in `README.md`; commit and push.
- [ ] **Step 4:** Rubric checklist against the spec's rubric section, item by item, noting where each is evidenced (section / file); fix gaps; final `pytest -q`.
- [ ] **Step 5:** Report to the user: live URL, repo URL, and the checklist.
