# Released trained weights

Trained YOLO11s-seg weights (`best.pt`, ~20 MB each) with the config snapshot of the run that produced
them. They let anyone reproduce the evaluation, the gallery and the website without retraining.

| Folder | Dataset | Classes | Image size | Notes |
|---|---|---|---|---|
| `yolo11s_seg` | CottonWeedDet12 | 12 weed species | 640 | earlier weeds-only experiment (no crop class) |
| `yolo11s_seg_cropandweed` | CropAndWeed (CropOrWeed2) | Crop, Weed | 1024 | exp 3 |
| `yolo11s_seg_cropandweed_1280` | CropAndWeed (CropOrWeed2) | Crop, Weed | 1280 | exp 4 |
| `yolo11s_seg_cropandweed_1280_aug` | CropAndWeed (CropOrWeed2) | Crop, Weed | 1280 | exp 5: + vertical flips, ±15° rotation |

Evaluate one on a split (writes `eval_<split>/shots.jsonl` and `summary.json` under `--out`):

```bash
python scripts/evaluate_split.py --run models/released/yolo11s_seg_cropandweed_1280_aug --split val \
    --prepared-dir <prepared CropAndWeed dataset> --out experiments/yolo11s_seg_cropandweed_1280_aug
```

## Licences

- CropAndWeed models: the dataset licence (AIT Austrian Institute of Technology) allows distributing
  derivative works that are abstract representations, such as trained models, for **non-commercial**
  use only. Cite: D. Steininger et al., "The CropAndWeed Dataset: a Multi-Modal Learning Approach for
  Efficient Crop and Weed Manipulation", WACV 2023.
- CottonWeedDet12 model: cite F. Dang et al., "YOLOWeeds: A novel benchmark of YOLO object detectors for
  multi-class weed detection in cotton production systems", Computers and Electronics in Agriculture,
  2023 (dataset: Zenodo record 7535814).
- Base model: Ultralytics YOLO11 (AGPL-3.0).
