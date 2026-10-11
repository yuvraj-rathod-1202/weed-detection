# Models

`models/weights/` (not version-controlled) holds pretrained checkpoints. Official ultralytics
checkpoints named in a config are downloaded there automatically on first use:

- `yolo11n-seg.pt`: baseline model (COCO-pretrained), `model.weights` in the experiment config
- `sam2.1_b.pt`: SAM used to create GT masks from boxes, `sam_model` in the dataset config

Trained weights are written to `experiments/<name>/train/weights/best.pt`.

`baseline_yolov5su.pt` and `strong_yolov8x.pt` are box detectors from the previous detection
pipeline and are no longer used.
