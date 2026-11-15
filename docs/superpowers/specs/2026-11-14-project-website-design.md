# Project website and submission package: design

Date: 2026-11-14. Solo project (ES 335). Deadline: 2026-11-16 (two days). GPU budget: 20 Kaggle GPU-hours this week.

## Intent

Package the weed-laser project for grading: a static Quarto website on GitHub Pages, a source repo
with a README (setup, rebuild, code locations, data access), and an AI-use disclosure with citations.
The target is full marks on the ES 335 rubric (80 shared ML-reasoning points + 20 deliverable points),
so every rubric criterion must have visible evidence on the site and in the repo.

The one thing a grader should leave knowing after two minutes: **the model can find weeds and avoid
crops** — with the evidence, the trade-off behind it, and its limits.

## The question the site answers

> Given one top-down RGB photo of a crop field, can a model place one laser point on each weed so that
> most weeds are hit and almost no crop plants are?

- Unit / row: one plant instance (weed or crop) in an image; for laser metrics, one shot.
- Prediction time: on the machine, per camera frame. Available: the RGB image only (no species, stems,
  previous frames).
- Target: per predicted weed, a mask and one (x, y) aim point; judged by whether the point lands inside
  a real weed and outside every crop.
- Limits: one main dataset (CropAndWeed, Europe), one camera setup, offline test images, no field
  trial, no laser-physics claim. The old "Indian vs non-Indian" hypothesis in `problem-statement.md`
  is dropped (no Indian data); the site must not claim it.

## Goals

1. A Quarto website with one section per rubric item: question, data, preprocessing, baseline &
   methods, evaluation, results, errors, limitations, conclusion, reproduction, plus AI disclosure &
   references and a short "about / my contribution" (solo).
2. Three interactive pieces, all from precomputed data:
   - **A. Confidence slider** (Evaluation section).
   - **D. Experiment comparison** (Results section).
   - **Gallery** of 10 images (Results/Errors).
3. A fair evaluation protocol: all model and threshold choices made on the **validation** split; the
   **test** split used once, for the final chosen setup (and the baseline).
4. A meaningful non-learned **baseline**.
5. README with setup, data access, split details, rebuild steps for results and site, code map,
   AI disclosure, citations.

## Non-goals

- New model training to improve accuracy (exp 6 at 1920 px is running on Kaggle; it is optional — see
  "Exp 6" below).
- A server, live model inference in the browser, or user image upload.
- Interaction C (error explorer) as an interactive piece — errors are a written section with static
  charts.

## Anti-goals (failure even if it "works")

- Any CropAndWeed image (raw or with overlays) on the public site or in the repo. Its licence forbids
  distributing the dataset "or modified versions"; only abstract derivatives (models, statistics) may be
  shared. Aggregated statistics and model outputs without image pixels are fine.
- Test data influencing any choice in the final protocol.
- Claims beyond the evidence (Indian fields, field-ready, causal statements).
- Cherry-picked gallery images.
- A site that is colourful / busy; colour beyond green/red-with-meaning.

## Constraints

- Static site only (GitHub Pages). Quarto 1.9.38 is installed locally; Ruby is not.
- 2 days; one Kaggle session (~4 GPU-hours) for the validation re-evaluation and baseline, after exp 6 session 1 frees the GPU.
- Repo is public (`github.com/yuvraj-rathod-1202/weed-detection`). `experiments/` and `data/raw/` are
  gitignored; only small derived JSON/PNGs for the site are committed.
- Kaggle has the full CropAndWeed data (downloaded per session into `/tmp`, ~1.25 h incl. masks);
  locally only ~1/4 of the images (all from the `ave` set) plus all annotations (`data/raw/cropandweed_src`).

## Approaches considered (publishing)

1. **Render locally, `quarto publish gh-pages` (chosen).** Export scripts write JSON/PNGs into `site/`,
   committed; one command renders and pushes HTML to a `gh-pages` branch. Fastest to a working site;
   reproducibility via README steps.
2. GitHub Action renders on every push. Always in sync with `main`, but CI setup/debugging risk with two
   days left and no rubric gain.
3. Render into a folder on `main` served by Pages. Pages serves only `/` or `/docs`, and `docs/` already
   holds project docs.

## Decisions (made with the user)

### Site structure and look
- Quarto website, **one long page with a left sidebar** (table of contents of the sections).
- **Clean paper** style: white background, serif headings, greys; **green only for "hit a weed",
  red only for "hit a crop"**, grey for soil/neutral. Minimal colour otherwise.
- Headline numbers at the top come from the final validation-selected model on test.

### Interaction A — confidence slider + trade-off curve
- Text: "the laser fires only when the model is at least X sure it is a weed."
- A slider over the weed-confidence threshold; three numbers update (weeds destroyed, shots on soil,
  shots on crops) and a dot moves along a curve of **weeds destroyed (y) vs share of shots not on a
  weed (x)** for all thresholds.
- Marked on the plot: the **optimal threshold** and the **baseline** operating point.
- Data: **validation** predictions of the selected model. Test numbers for the chosen threshold are
  shown beside it, once.
- **Optimal threshold rule (fixed before looking at validation numbers):** the threshold with the
  highest weed hit rate subject to **at most 2 % of shots landing on crops**.

### Interaction D — experiment comparison
- Bar chart: baseline, exp 3 (1024), exp 4 (1280), exp 5 (1280 + augmentation),
  each at its own optimal (2 %-rule) threshold on **validation**.
- Dropdown switches the metric: weeds destroyed, shots on crops, shots on soil, mask mAP50-95 (val),
  inference ms/image. Hover/tooltip: what changed in that experiment and why.
- The chosen model is highlighted (green).

### Gallery (user's idea)
- 10 thumbnails: **5 Sugar Beets 2016** images (Univ. Bonn, CC BY-SA 4.0) with the final CropAndWeed
  model, and **5 CottonWeedDet12** test images with the CottonWeedDet12-trained model
  (`experiments/yolo11s_seg`). Random pick with a fixed seed, stated on the page.
- Click → **side-by-side** viewer: raw photo | model output (masks + laser points; green = hit weed,
  red = hit crop, grey = soil; vetoed shots marked), a "show ground truth" toggle, and a summary line
  (weeds hit of N, crop shots, soil shots), dataset + licence + "farm/camera never seen in training"
  (Sugar Beets) or "this dataset labels no crops" (CottonWeedDet12).

### Final evaluation protocol (redo on validation)
- Candidates: exp 3, exp 4, exp 5, plus the
  baseline.
- For each candidate, run inference once on the **validation** split with a low confidence floor
  (0.05) and save, per predicted weed: confidence, target point (`center_weighted_dt`), whether it is
  ignored (ignore region rule as today), whether it is vetoed by the crop gate, which GT weed it hits
  (or none), whether it lands in a GT crop or on soil. Save per GT weed its size. This allows sweeping
  any threshold afterwards without re-running the model.
- Model selection: apply the 2 % rule to each candidate on validation; the candidate with the highest
  weed hit rate at its optimal threshold is the final model; its threshold is the final threshold.
- Then evaluate **only the final model at its threshold, and the baseline, on test** (same protocol).
- The site states openly: during development, test results were inspected; the final choices were
  redone on validation, and whether they changed.
- Note for the site: the validation split was also used by ultralytics to pick `best.pt` (early
  stopping). That is legitimate tuning use; test was not.

### Baseline
- Non-learned vegetation filter: excess-green index (2G − R − B) + Otsu threshold → morphological
  opening → connected components above a fixed minimum area → **every component is a weed**, aimed at
  with the same `center_weighted_dt` method. No crop gate (it cannot tell crops apart).
- Parameters fixed a priori (not tuned on test); evaluated on validation and test with the same scoring.
- It answers: is a trained model worth it over a simple "green = plant" filter? Expected: many crop
  shots, which is exactly the point.

### Exp 6 (1920 px, running on Kaggle)
- Session 2 is dropped (user). Session 1 ends part-trained (~50 of 80 epochs), so exp 6 is **not a
  candidate** (builder's call: a half-trained model would be an unfair comparison). The site mentions it
  as future work, with the out-of-memory lessons.
- The validation Kaggle session runs as soon as a GPU is free after exp 6 session 1.

### Crop gate during the threshold sweep (builder's call)
- One threshold for all predictions, as in every experiment so far: at slider value X, only crop
  predictions with confidence ≥ X feed the 20 px crop gate. To compute this exactly for every X, the
  validation export stores per shot the highest confidence of any crop prediction within the margin
  (`max_nearby_crop_conf`); a shot is vetoed at X when that value ≥ X.

### CropAndWeed images already in the repo
- `notebooks/03_results_analysis.ipynb` (commit `2a1be85`) embeds three `eval/visualizations` panels
  of CropAndWeed photos. **User decision: clear the notebook's outputs in a new commit; no history
  rewrite.** (Known remaining risk: the photos stay reachable in git history.) This happens before the
  site is published. Future notebook outputs must not embed CropAndWeed photos.

### Contribution / defence
- Solo project: a short "My contribution and AI use" section on the site and in the README.
- AI use: **only Claude Code** (Anthropic), used for coding, debugging, analysis and writing assistance.
  The disclosure states this, that all results were checked by the author, and cites external code and
  data (ultralytics, SAM 2.1, Quarto, CropAndWeed, CottonWeedDet12, Sugar Beets 2016).

## Rubric and required submission (from the course)

Shared ML reasoning (80): problem formulation & scope 10; data understanding 10; preprocessing &
statistical reasoning 10 (learned preprocessing must avoid test leakage); baseline & method selection
15; evaluation design & metrics 20 (tuning separate from final testing); results, errors & limitations
10 (representative errors, uncertainty, no overclaiming); reproducibility & technical integrity 5 (data
and credentials handled responsibly).
ES 335 deliverable (20): interactive website 10 (clear, navigable, works when published, explains
problem, evidence, conclusions); useful interaction 5 (helps inspect/understand a result); individual
understanding & defence 5 (viva: contribution, key choices, evaluation, a limitation, a small change).
Statistical tests optional (only if they answer a specific question).

Submit: live GitHub Pages URL; GitHub source-repo URL accessible to graders; README with setup/rebuild
steps, code location and data access; AI-use disclosure and citations. Site sections: question, data,
preprocessing, baseline & methods, evaluation, results, errors, limitations, conclusion, reproduction,
plus at least one useful interaction. Deadline: within 2 days (by 2026-11-16).

Uncertainty (rubric "results, errors & limitations"): report bootstrap confidence intervals over
**images/sessions** for the headline test metrics (weed hit rate, crop-shot share) — resampling
sessions respects the grouped split. This is the one statistical procedure; it answers "how much could
the headline move with a different sample of fields", not significance testing.

## Content sources for the sections (existing work)

- Data: CropAndWeed statistics from `data/raw/cropandweed_src` annotations (counts, class balance,
  sizes, sessions, ignore/Vegetation share, overlapping boxes needing SAM), CottonWeedDet12 as the
  earlier weeds-only experiment.
- Preprocessing: CropOrWeed2 label mapping, ignore regions, semantic → instance masks with SAM only for
  ambiguous boxes, session-grouped 65/20/15 split (seed 42), no fitted preprocessing statistics (YOLO
  normalises by /255; augmentation train-only) → no leakage.
- Methods: YOLO11s-seg (why instance segmentation, why masks beat boxes for aiming), point methods,
  crop safety gate (20 px margin).
- Errors: analyses already done (small weeds ≤ 24 px, young soybeans in session `ave-0543`, sunflower
  confused as weed, train-vs-test fit check) — recomputed or restated on validation/test as appropriate.
- Charts: Observable Plot inside Quarto (OJS) from committed JSON, so all charts share one style.

## Left to the builder

- Exact Quarto project layout (`site/` folder suggested), OJS vs small vanilla JS for the gallery viewer.
- Script names and JSON schemas for the exports; how the validation-eval script is structured (reuse
  `src/evaluation.py` pieces).
- How to fetch ~5 annotated Sugar Beets 2016 images (download size/feasibility); fallback if impractical
  must be raised with the user before switching.
- Baseline parameter values (minimum area etc.), stated with a reason on the site.
- Chart details, copy-editing, mobile behaviour of the sidebar (Quarto default).
- README restructuring.
