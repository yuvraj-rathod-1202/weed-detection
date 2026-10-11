#!/usr/bin/env python
"""Check everything an experiment needs before running it: packages, config, data, weights,
device, output folder and resume state. Exits with code 1 if anything is missing.

Usage:
    python scripts/verify_setup.py --config configs/experiments/baseline_yolo11n_seg.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = ("torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas", "omegaconf", "skimage", "yaml", "matplotlib", "tqdm", "psutil")

failures = 0


def report(status: str, message: str) -> None:
    global failures
    failures += status == "FAIL"
    print(f"[{status:4}] {message}")


def check_packages() -> None:
    for name in PACKAGES:
        try:
            __import__(name)
        except ImportError:
            report("FAIL", f"missing package '{name}' (pip install -r requirements.txt)")
    report("OK", "python packages")


def check_raw_data(ds_cfg, required: bool) -> None:
    from src.data.labels import list_images, read_class_names, read_yolo_boxes

    status = "FAIL" if required else "WARN"
    raw_dir = Path(ds_cfg.raw_dir)
    try:
        images = list_images(raw_dir / "images")
        class_names = read_class_names(raw_dir / "classes.txt")
    except FileNotFoundError as e:
        report(status, f"raw dataset: {e} (see data/README.md)")
        return
    label_files = list((raw_dir / "labels").glob("*.txt"))
    try:
        for label_file in label_files[:200]:
            for class_id, *_ in read_yolo_boxes(label_file):
                if not 0 <= class_id < len(class_names):
                    raise ValueError(f"{label_file}: class id {class_id} not in classes.txt")
    except ValueError as e:
        report(status, f"raw labels: {e}")
        return
    report("OK", f"raw dataset {raw_dir}: {len(images)} images, {len(label_files)} box label files, {len(class_names)} classes")


def check_prepared_data(ds_cfg) -> bool:
    prepared_dir = Path(ds_cfg.prepared_dir)
    if (prepared_dir / "prepare_info.json").exists():
        from src.utils.io import load_json

        splits = load_json(prepared_dir / "prepare_info.json")["splits"]
        report("OK", f"prepared dataset {prepared_dir}: " + ", ".join(f"{s}={v['num_images']}" for s, v in splits.items()))
        return True
    state = "partially prepared, will resume" if (prepared_dir / "splits.json").exists() else "not prepared yet"
    report("INFO", f"prepared dataset {prepared_dir}: {state}; SAM masks will be generated with {ds_cfg.sam_model}")
    return False


def check_weights(weights: str, what: str) -> None:
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    path = Path(weights)
    if path.exists():
        report("OK", f"{what} {path}")
    elif path.name in GITHUB_ASSETS_NAMES:
        report("WARN", f"{what} {path} not downloaded yet: ultralytics will fetch it (needs internet)")
    else:
        report("FAIL", f"{what} not found: {path}")


def check_device(train_device) -> None:
    import torch

    if torch.cuda.is_available():
        report("OK", f"CUDA: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    elif str(train_device).lower() == "cpu":
        report("OK", "device: cpu (as configured)")
    else:
        report("WARN", "no CUDA GPU found: training will run on CPU and be very slow")


def check_run_dir(cfg) -> None:
    from src.experiment import EVAL_SECTIONS, _changed_keys, run_dir_for
    from src.utils.config import to_container
    from src.utils.io import load_yaml_dict

    run_dir = run_dir_for(cfg)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_root).free / 1e9
    report("OK" if free_gb > 5 else "WARN", f"output folder {output_root}: {free_gb:.1f} GB free")

    if (run_dir / "config.yaml").exists():
        changed = _changed_keys(load_yaml_dict(run_dir / "config.yaml"), to_container(cfg))
        train_changes = [k for k in changed if k.split(".")[0] not in EVAL_SECTIONS]
        if train_changes:
            report("FAIL", f"{run_dir} was started with a different training config: {train_changes}")
        elif changed:
            report("WARN", f"eval config changed {changed}: evaluation will be recomputed")

    last_pt = run_dir / "train" / "weights" / "last.pt"
    if (run_dir / "results.json").exists():
        state = "complete (re-running only rebuilds results.json)"
    elif (run_dir / "train_summary.json").exists():
        state = "trained; evaluation will run/resume"
    elif last_pt.exists():
        state = f"interrupted; training resumes from {last_pt}"
    else:
        state = "new run"
    report("INFO", f"run {run_dir}: {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--override", nargs="*", default=[], help="Same overrides you pass to run_experiment.py")
    args = parser.parse_args()

    check_packages()
    if failures:
        sys.exit(1)

    from src.targeting.points import POINT_METHODS
    from src.utils.config import load_experiment_config

    try:
        cfg = load_experiment_config(args.config, args.override)
    except Exception as e:  # noqa: BLE001 - report any config problem
        report("FAIL", f"config: {e}")
        sys.exit(1)
    unknown = [m for m in cfg.targeting.methods if m not in POINT_METHODS]
    report("FAIL" if unknown else "OK", f"config {args.config}" + (f": unknown targeting methods {unknown}" if unknown else ""))

    prepared = check_prepared_data(cfg.dataset)
    check_raw_data(cfg.dataset, required=not prepared)
    if not prepared:
        check_weights(cfg.dataset.sam_model, "SAM weights")
    check_weights(cfg.model.weights, "model weights")
    check_device(cfg.train.get("device"))
    check_run_dir(cfg)

    print("\nREADY" if failures == 0 else f"\nNOT READY: {failures} problem(s) above")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
