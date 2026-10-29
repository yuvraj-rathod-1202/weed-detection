# Running on Kaggle

The local machine has no GPU, so experiments run in a Kaggle GPU notebook. One notebook run
does the whole pipeline (SAM masks → training → evaluation) inside `/kaggle/working/`, and Kaggle
keeps that folder as the notebook's output.

For **CropAndWeed** (downloaded in the notebook, not uploaded) follow [kaggle_cropandweed.md](kaggle_cropandweed.md).

## 1. Upload the data and the code (once)

Kaggle can't see your PC, so both go up as private Kaggle **Datasets**
(kaggle.com → Datasets → New Dataset → drop the zip → Private → Create). Kaggle unzips them.

**Data:** upload only the raw box dataset. The SAM masks are generated on the Kaggle GPU.

```powershell
Compress-Archive -Path data/raw/cottonweeddet12 -DestinationPath cottonweeddet12.zip
```

**Code:** pick one option.

| Option | How | Updating code later |
|---|---|---|
| A. GitHub (recommended) | push this repo to GitHub, clone it in the notebook (cell 1) | `git push`, then re-run the notebook |
| B. Dataset | zip the code (below) and upload it as dataset `weed-detection-code` | upload a new zip with **New Version** on the dataset page |

```powershell
Compress-Archive -Path configs,scripts,src,tests,notebooks,docs,data/label_fixes,requirements.txt,pyproject.toml,README.md -DestinationPath weed-detection-code.zip
```

If you use option A with a private repo, create a GitHub token with read access and add it in the
notebook under **Add-ons → Secrets** as `GITHUB_TOKEN`.

## 2. Create the notebook

New Notebook. In the right-hand panel:

- **Accelerator:** GPU T4 x2
- **Internet:** On. This is needed for `pip` and for downloading the YOLO/SAM weights, and it
  requires a phone-verified account.
- **Add Input:** your `cottonweeddet12` dataset (plus `weed-detection-code` for option B)

## 3. Notebook cells

**Cell 1: get the code and install**

```python
import os
REPO = "/kaggle/working/weed-detection"

if not os.path.exists(REPO):
    # Option A: public repo
    !git clone https://github.com/<user>/weed-detection.git {REPO}
    # Option A: private repo
    # from kaggle_secrets import UserSecretsClient
    # token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    # !git clone https://{token}@github.com/<user>/weed-detection.git {REPO}
    # Option B: code dataset (check the folder with !ls /kaggle/input/weed-detection-code)
    # !cp -r /kaggle/input/weed-detection-code {REPO}

%cd {REPO}
!pip install -q -r requirements.txt
```

**Cell 2: settings**

```python
import glob
RAW = os.path.dirname(glob.glob("/kaggle/input/**/classes.txt", recursive=True)[0])  # folder with images/, labels/, classes.txt
CFG = "configs/experiments/baseline_yolo11n_seg.yaml"
OVR = f"dataset.raw_dir={RAW}"   # keep this IDENTICAL in every session of the same experiment
print(RAW)
```

**Cell 3: check the setup.** It should end with `READY`. WARN lines saying weights are "not
downloaded yet" are fine.

```python
!python scripts/verify_setup.py --config {CFG} --override {OVR}
```

**Cell 4: smoke test (optional).** This runs 16 images for 1 epoch and takes a few minutes. It
catches problems before the long run.

```python
!python scripts/smoke_test.py --config {CFG} --override {OVR}
```

**Cell 4b: prepare the data on its own (optional).** Cell 5 does this anyway. Running it
separately lets you check the SAM masks before training. This script reads the *dataset*
config, so the override is `raw_dir=...` rather than `dataset.raw_dir=...`. It writes to the
same folder cell 5 reads, so cell 5 then skips this step.

```python
!python scripts/prepare_data.py --dataset configs/datasets/cottonweeddet12.yaml --override raw_dir={RAW}
```

**Cell 5: run the experiment**

```python
!python scripts/run_experiment.py --config {CFG} --override {OVR}
```

**Cell 6: compare and pack the results**

```python
!python scripts/compare_experiments.py
!cd /kaggle/working && zip -qr results.zip weed-detection/experiments weed-detection/results
```

## 4. Do the long run in the background

1. Run cells 1–4 interactively to make sure everything works.
2. Click **Save Version → Save & Run All (Commit)**. Kaggle runs the whole notebook on its servers
   for up to **12 h**, and you can close the browser.
3. When it's done, open that version → **Output**, and download `results.zip` or single files.
   Unzip it into this repo locally, and the notebooks and `compare_experiments.py` will work on it.

Don't do the long run in an interactive session. `/kaggle/working/` is wiped when the session
stops (unless **Persistence** is turned on in the notebook settings), and idle sessions get shut
down.

**Will it fit in 12 h?** `run.log` shows the time per epoch. You need
epochs × epoch time + SAM preparation + evaluation to be under 12 h. A nano model on this dataset
fits easily. If a larger model doesn't fit, lower `train.epochs` in a **new** experiment config.
Be careful with ultralytics' `train.time: N`: it is not just a safety stop. It re-computes the
number of epochs so that training fills N hours, which overrides `train.epochs`.

## 5. Resume a run that stopped

Every stage saves as it goes. To continue a run that stopped part-way (SAM masks, epochs, or
evaluated images), you need the stopped version's output, if Kaggle saved it.

1. In the notebook, go to **Add Input → Your Work** and choose this notebook (that version's output).
2. Add this cell after cell 1. It copies the old outputs back to the **same paths**:

   ```python
   PREV = "/kaggle/input/<notebook-slug>/weed-detection"   # check with !ls /kaggle/input
   !mkdir -p experiments data/processed
   !cp -rn {PREV}/experiments/. experiments/
   !cp -rn {PREV}/data/processed/. data/processed/
   ```

3. Run the same cells with the same `OVR`. Training resumes from `last.pt`, and finished stages
   are skipped.

The copy is required because `/kaggle/input/` is read-only, and the pipeline writes into both
folders even when resuming.

## 6. Options and fixes

| Situation | What to do |
|---|---|
| Use both T4s | `OVR = f'dataset.raw_dir={RAW} "train.device=[0,1]"'`. Only training uses both; a resumed run keeps the GPUs it started with |
| `CUDA out of memory` | lower the batch size in a new experiment, e.g. add `train.batch=8` to `OVR` and change `name` |
| `Training config of '...' changed` | `OVR` differs from the first session. Use exactly the same overrides, or a new `name` |
| `was prepared with different settings` | the copied `data/processed` was made with another split/SAM config. Don't copy it, so it is rebuilt |
| `verify_setup` FAIL on raw dataset | `RAW` is wrong. `!find /kaggle/input -name classes.txt` shows where the data is |
| `pip` or weight download fails | Internet is off in the notebook settings |
| Several experiments | add one `run_experiment.py` line per config in cell 5. Each `name` gets its own folder, and cell 6 compares them all |
