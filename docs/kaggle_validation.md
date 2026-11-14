# Kaggle: validation re-evaluation, model selection, final test

One Kaggle session (~3–4 GPU-hours) that produces every number the website reports:

1. Each candidate model (exp 3, 4, 5, from `models/released/`) and the ExG–ExR baseline are evaluated
   on the **validation** split. Every predicted weed down to confidence 0.05 is saved, so any threshold
   can be swept offline (`src/targeting/sweep.py`).
2. `select_model.py` applies the rule fixed in advance — the threshold with the most weeds hit while at
   most 2 % of shots land on crops — to every candidate, and picks the model whose optimum hits the most
   weeds.
3. **Only the selected model and the baseline** are evaluated on the **test** split.

No training happens here. General notebook setup (GPU, Internet, background commits) is in
[kaggle_cropandweed.md](kaggle_cropandweed.md); the CropAndWeed licence forbids uploading the data, so it
is downloaded in the notebook.

## Notebook cells

Settings: **Accelerator GPU T4** (one is enough), **Internet on**. Run with **Save Version → Save & Run
All (Commit)**.

**Cell 1: code and packages**

```python
import os
REPO = "/kaggle/working/weed-detection"
if not os.path.exists(REPO):
    !git clone https://github.com/yuvraj-rathod-1202/weed-detection.git {REPO}
%cd {REPO}
!git pull
!pip install -q -r requirements.txt
```

**Cell 2: download + convert CropAndWeed (~15 min)**

```python
!python scripts/convert_cropandweed.py --download --src /tmp/cnw_src --out /tmp/cropandweed
```

Ends with `8034 images in /tmp/cropandweed`.

**Cell 3: GT masks (~45–60 min).** Same split as training (session-grouped, seed 42).

```python
!python scripts/prepare_data.py --dataset configs/datasets/cropandweed.yaml --override raw_dir=/tmp/cropandweed prepared_dir=/tmp/cropandweed_seg
```

**Cell 4: validation, all candidates + baseline (~1–1.5 h)**

```python
P = "--prepared-dir /tmp/cropandweed_seg"
RUNS = ["yolo11s_seg_cropandweed", "yolo11s_seg_cropandweed_1280", "yolo11s_seg_cropandweed_1280_aug"]
for run in RUNS:
    !timeout 3h python scripts/evaluate_split.py --run models/released/{run} --split val {P} --out experiments/{run}
!timeout 2h python scripts/evaluate_split.py --baseline --run experiments/baseline_exg --split val {P}
```

Each call prints its optimal threshold. If a call stops (timeout, crash), re-running it resumes.

**Cell 5: select the final model (validation only)**

```python
CANDIDATES = " ".join("experiments/" + r for r in RUNS)
!python scripts/select_model.py {CANDIDATES}
import json
FINAL_RUN = json.load(open("results/final_selection.json"))["run"]
print(FINAL_RUN)
```

**Cell 6: test, selected model + baseline only (~30 min)**

```python
!python scripts/evaluate_split.py --run models/released/{FINAL_RUN} --split test {P} --out experiments/{FINAL_RUN}
!python scripts/evaluate_split.py --baseline --run experiments/baseline_exg --split test {P}
```

The test summary also contains a sweep curve; the website reports test results **only at the threshold
chosen on validation** (`results/final_selection.json`).

**Cell 7: pack the results**

```python
!cd /kaggle/working/weed-detection && zip -qr /kaggle/working/validation_results.zip experiments/*/eval_val experiments/*/eval_test experiments/baseline_exg/config.yaml results/final_selection.json
!ls -la /kaggle/working/validation_results.zip
```

## After the session

Download `validation_results.zip` from the version's **Output** tab and unzip it in the repo root on
your PC. It adds `experiments/<run>/eval_val/`, `experiments/<run>/eval_test/`,
`experiments/baseline_exg/` and `results/final_selection.json`; nothing existing is overwritten.

Size: the shots files are a few MB per run; nothing from the CropAndWeed images is included.
