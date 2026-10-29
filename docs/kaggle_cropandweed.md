# Training on CropAndWeed in Kaggle

Runs `configs/experiments/yolo11s_seg_cropandweed.yaml` (YOLO11s-seg, classes Crop + Weed, crop
safety gate) on the **full** CropAndWeed dataset in a Kaggle GPU notebook. General Kaggle notes
(background runs, resuming, common errors) are in [kaggle.md](kaggle.md); this page has
everything specific to CropAndWeed.

## What is different from CottonWeedDet12

| | CottonWeedDet12 | CropAndWeed |
|---|---|---|
| Data on Kaggle | you upload it as a Kaggle Dataset | **downloaded in the notebook** from the official source. The licence forbids redistribution: never upload it, not even as a private Kaggle Dataset |
| Where the data lives | `/kaggle/input/...` | `/tmp/...` (big, not saved as notebook output) |
| GT masks | SAM for every box | the dataset's human-drawn masks; SAM only for overlapping boxes (~20 %) |
| Split | random | by recording session: no near-duplicate images across train/test |
| Image size | 640 | 1024 (1920 px images, many seedlings < 32 px) |

Data size: ~11.5 GB download, 8,034 images, ~112k plants. It does not fit in `/kaggle/working`
(20 GB limit, and everything there is saved as output), so the data goes to `/tmp` and only the
experiment results are kept in `/kaggle/working`.

## 1. Before the first run (on your PC)

1. Push the code: `git push` (the notebook clones the repo from GitHub).
2. Nothing else: no data upload.

## 2. Create the notebook

New Notebook, then in the right-hand panel:

- **Accelerator:** GPU T4 x2
- **Internet:** On (code, pip, weights and the dataset download)
- **Persistence:** off (the long run is a background commit, see step 4)
- If the repo is private: **Add-ons → Secrets** → `GITHUB_TOKEN` (a GitHub token with read access)

## 3. Notebook cells

**Cell 1: code and packages**

```python
import os
REPO = "/kaggle/working/weed-detection"
if not os.path.exists(REPO):
    !git clone https://github.com/yuvraj-rathod-1202/weed-detection.git {REPO}
    # private repo:
    # from kaggle_secrets import UserSecretsClient
    # token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    # !git clone https://{token}@github.com/yuvraj-rathod-1202/weed-detection.git {REPO}
%cd {REPO}
!git pull
!pip install -q -r requirements.txt
!df -h /tmp /kaggle/working    # /tmp needs ~15 GB free
```

**Cell 2: download + convert the dataset (~10–20 min, every session)**

Streams the 5 official archives straight into `/tmp` and converts them to the project's raw
layout. Re-running skips finished parts.

```python
!python scripts/convert_cropandweed.py --download --src /tmp/cnw_src --out /tmp/cropandweed
```

It should end with `8034 images in /tmp/cropandweed`.

**Cell 3: settings**

```python
CFG = "configs/experiments/yolo11s_seg_cropandweed.yaml"
# Keep OVR IDENTICAL in every session of this experiment (training settings can't change mid-run).
OVR = 'dataset.raw_dir=/tmp/cropandweed dataset.prepared_dir=/tmp/cropandweed_seg "train.device=[0,1]" train.batch=16'
```

- `prepared_dir` in `/tmp`: the prepared dataset links to the images instead of copying them
  (links only work on the same disk), so `/kaggle/working` stays small.
- `train.device=[0,1]` + `train.batch=16` trains on both T4s (8 images per GPU). For one GPU,
  drop both and the config's `batch: 8` is used.

**Cell 4: check the setup.** Must end with `READY`.

```python
!python scripts/verify_setup.py --config {CFG} --override {OVR}
```

**Cell 5: smoke test (optional, a few minutes).** 16 images, 1 epoch, separate output folder.

```python
!python scripts/smoke_test.py --config {CFG} --override {OVR}
```

**Cell 6: GT masks** (in its own process, so SAM's GPU memory is fully freed before training)

```python
!python scripts/prepare_data.py --dataset configs/datasets/cropandweed.yaml --override raw_dir=/tmp/cropandweed prepared_dir=/tmp/cropandweed_seg
```

This takes about 45–60 min on the full dataset. It is rebuilt in each new session because
`/tmp` is wiped, with the same split every time.

**Cell 7: the experiment** (train → evaluate; it finds the masks from cell 6 and skips preparation)

```python
!python scripts/run_experiment.py --config {CFG} --override {OVR}
```

**Cell 8: compare and pack the results**

```python
!python scripts/compare_experiments.py
!cd /kaggle/working && zip -qr results.zip weed-detection/experiments weed-detection/results
```

## 4. The long run

1. Run cells 1–6 interactively once. Then run cell 7 for 2–3 epochs and read the epoch time in
   `experiments/yolo11s_seg_cropandweed/run.log`, then stop it.
2. Check that the run fits in 12 h:
   `download (~15 min) + masks (~1 h) + 80 epochs × epoch time + evaluation (~15 min)`.
   If it does not fit, it still resumes in a second session (step 5). With 30 GPU h/week there is
   room for two sessions.
3. **Save Version → Save & Run All (Commit)**, then close the browser.
4. When it finishes: open the version → **Output** → download `results.zip`, and unzip it into
   this repo on your PC.

The stopped interactive test in step 1 leaves a half-trained run in `/kaggle/working`, but the
commit starts from a fresh copy of the notebook, so it doesn't interfere.

## 5. Resume in a new session

If the 12 h ran out before training finished:

1. **Add Input → Your Work** → this notebook (the stopped version's output).
2. Add this cell after cell 1. It copies only the experiment folder back; the data is
   re-downloaded and re-prepared by cells 2 and 6 as usual.

   ```python
   PREV = "/kaggle/input/<notebook-slug>/weed-detection"   # check with !ls /kaggle/input
   !mkdir -p experiments
   !cp -rn {PREV}/experiments/. experiments/
   ```

3. Run all cells again with the **same `OVR`**. Training continues from `last.pt`. The resumed
   session also spends ~1.25 h on download + masks before training continues.

## 6. Choosing the crop safety margin (after training)

`crop_margin_px: 20` ≈ 8 mm on the ground (about 0.4 mm/px at the 1.1 m camera height). To compare
margins, re-run only the evaluation. A `targeting.*` change keeps the trained model, but it
replaces the previous evaluation, so save each `results.json` first:

```python
for margin in [0, 10, 20, 30]:
    !python scripts/run_experiment.py --config {CFG} --override {OVR} targeting.crop_margin_px={margin}
    !cp experiments/yolo11s_seg_cropandweed/results.json results_margin{margin}.json
```

Compare `crop_hit_rate` (crop shots, want ≈ 0) and `weed_hit_rate` (weeds destroyed) under
`targeting.methods.<method>` across the files. Choose the smallest margin where `crop_hit_rate`
reaches about 0.

## 7. What to look at in the results

| Where | What |
|---|---|
| `results.json` → `segmentation.per_class_mask` | mask mAP for Crop and Weed separately |
| `results.json` → `targeting.methods.<method>` | `shot_precision`, `weed_hit_rate`, `crop_hit_rate`, `crop_hit_rate_ungated`, `veto_rate`, `clearance_px` |
| `eval/visualizations/` | GT panel (weeds red, crops blue, ignored plants grey) vs prediction panel (weeds green, crops blue, vetoed shots grey crosshairs) |
| `/tmp/cropandweed_seg/prepare_info.json` (gone after the session; `!cat` it in the notebook) | images per split, `sam_boxes` (boxes that needed SAM) |

## 8. Problems

| Problem | Fix |
|---|---|
| Download fails or is very slow | Internet is off, or the server is busy. Re-run cell 2; finished archives are skipped |
| `No space left on device` | check `!df -h /tmp`. The data needs ~12 GB in `/tmp` (converted images are links, not copies). Delete leftovers from earlier tries (`!rm -rf /tmp/cropandweed_seg`) and re-run |
| `CUDA out of memory` and the message says another process holds ~10 GiB (`Process N has 10.66 GiB memory in use`) | GT masks were made in the same process as training and SAM kept the GPU memory. `git pull` the fix and keep cell 6 as a separate cell |
| `CUDA out of memory` with nothing else on the GPU | lower the batch: `train.batch=8` with both GPUs. This is a training change, so also add a new name (`name=yolo11s_seg_cropandweed_b8`) |
| `Training config of '...' changed` | `OVR` differs from the first session. Use exactly the same one |
| `was prepared with different settings` | `/tmp/cropandweed_seg` came from another config. `!rm -rf /tmp/cropandweed_seg` and re-run |
| `crop classes ... not in` from verify_setup | `targeting.crop_classes` doesn't match `classes.txt` (`Crop`, `Weed`) |

## Licence and citation

CropAndWeed is for non-commercial use only, with no redistribution of images or annotations.
Trained weights may be shared. Cite Steininger et al., "The CropAndWeed Dataset: a Multi-Modal
Learning Approach for Efficient Crop and Weed Manipulation", WACV 2023.
