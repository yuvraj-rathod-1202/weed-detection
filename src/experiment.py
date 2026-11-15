"""Run one experiment end to end: prepare data -> train -> evaluate -> results.json.

Everything lives in <output_root>/<name>/ and every stage is resumable, so re-running the
same command after a crash or a Kaggle timeout continues where it stopped:

    config.yaml               config snapshot (a changed training config is refused)
    label_fixes.json          label-fix hashes the model was trained with and evaluated with
    environment.json          software/hardware of the latest session
    run.log                   log of every session
    train/                    ultralytics output; weights/last.pt is saved every epoch
    train_summary.json        written when training is done (per-epoch history inside)
    eval/segmentation_metrics.json
    eval/predictions.jsonl    one line per test image, appended as it is evaluated
    eval/visualizations/      [ground truth | prediction] panels with laser points
    results.json              final structured results (input for comparison + plots)
    results_<split>-labels-<hash>.json   results from before the eval labels were last changed
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from src.data.label_fixes import split_hashes
from src.data.prepare import prepare_dataset
from src.evaluation import segmentation_metrics, targeting_metrics
from src.training import train_model
from src.utils.config import to_container
from src.utils.io import load_json, load_yaml_dict, save_json, save_yaml
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed
from src.utils.system_info import file_size_mb, get_environment_info

RESULTS_SCHEMA_VERSION = 1

# Config keys that may differ between sessions without invalidating anything (machine-specific settings).
# dataset.label_fixes: the file's content is tracked per split instead (check_label_fixes).
IGNORED_KEYS = {
    "output_root", "config_file", "description", "dataset.raw_dir", "dataset.prepared_dir",
    "dataset.label_fixes", "train.device",
}
EVAL_SECTIONS = ("eval", "targeting")
TRAINING_SPLITS = ("train", "val")


def run_dir_for(cfg) -> Path:
    return Path(cfg.output_root) / cfg.name


def run_experiment(cfg) -> dict:
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"experiment.{cfg.name}", run_dir / "run.log")
    logger.info("=== Experiment '%s' -> %s", cfg.name, run_dir)

    check_config_snapshot(cfg, run_dir, logger)
    save_json(get_environment_info(), run_dir / "environment.json")
    set_seed(cfg.seed)

    logger.info("[1/3] Dataset: %s", cfg.dataset.name)
    data_info = prepare_dataset(cfg.dataset, logger)
    data_yaml = Path(cfg.dataset.prepared_dir) / "data.yaml"
    labels_used = check_label_fixes(cfg, run_dir, data_info, logger)

    logger.info("[2/3] Training: %s", cfg.model.weights)
    train_summary = train_model(cfg, run_dir, data_yaml, logger)
    best_weights = run_dir / "train" / "weights" / "best.pt"

    logger.info("[3/3] Evaluation on '%s' split", cfg.eval.split)
    segmentation = segmentation_metrics(cfg, best_weights, data_yaml, run_dir, logger)
    targeting = targeting_metrics(cfg, best_weights, run_dir, data_info["class_names"], logger)

    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "experiment": {
            "name": cfg.name,
            "description": cfg.get("description"),
            "config_file": cfg.config_file,
            "run_dir": str(run_dir),
            "seed": cfg.seed,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
        "dataset": {
            "name": cfg.dataset.name,
            "region": cfg.dataset.get("region"),
            "mask_source": cfg.dataset.sam_model,
            "sam_fallback_boxes": data_info["sam_fallback_boxes"],
            "label_fixes": data_info.get("label_fixes"),
            "label_fixes_used": labels_used,  # label-fix hash per split this model was trained / evaluated with
            "splits": data_info["splits"],
        },
        "model": {"init_weights": cfg.model.weights, "best_weights": str(best_weights), **model_info(best_weights, cfg.eval.imgsz)},
        "training": train_summary,
        "segmentation": segmentation,
        "targeting": targeting,
        "environment": get_environment_info(),
    }
    save_json(results, run_dir / "results.json")

    primary = targeting["methods"][targeting["primary_method"]]
    logger.info(
        "Done. mask mAP50-95=%.3f | %s: shot_precision=%s weed_hit_rate=%s crop_hit_rate=%s | %.1f ms/img",
        segmentation["mask"]["map50_95"], targeting["primary_method"],
        _fmt(primary["shot_precision"]), _fmt(primary["weed_hit_rate"]), _fmt(primary.get("crop_hit_rate")),
        targeting["end_to_end_ms_per_image"] or 0,
    )
    return results


def model_info(weights: Path, imgsz: int) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(str(weights)).model
    return {
        "num_parameters": int(get_num_params(model)),
        "gflops": round(float(get_flops(model, imgsz)), 2),
        "weights_size_mb": round(file_size_mb(weights), 2),
    }


def check_config_snapshot(cfg, run_dir: Path, logger) -> None:
    """Protect a run from silently mixing configs.

    - training-related change (seed/dataset/model/train): refuse; use a new `name` instead.
    - eval/targeting-only change: keep the trained model, recompute the evaluation.
    """
    snapshot_path = run_dir / "config.yaml"
    new = to_container(cfg)
    if snapshot_path.exists():
        changed = _changed_keys(load_yaml_dict(snapshot_path), new)
        train_changes = [k for k in changed if k.split(".")[0] not in EVAL_SECTIONS]
        if train_changes:
            raise ValueError(
                f"Training config of '{cfg.name}' changed since the run started: {train_changes}.\n"
                f"Give the experiment a new `name`, or delete {run_dir} to start over."
            )
        if changed:
            logger.warning("Eval config changed %s: discarding old evaluation outputs.", changed)
            shutil.rmtree(run_dir / "eval", ignore_errors=True)
            (run_dir / "results.json").unlink(missing_ok=True)
    save_yaml(new, snapshot_path)


def check_label_fixes(cfg, run_dir: Path, data_info: dict, logger) -> dict:
    """Label fixes change the GT without changing the config. Records which labels the run used.

    - train/val labels changed while training is unfinished: refuse (resuming would mix labels).
    - train/val labels changed after training finished: the model is evaluated as it is; the
      labels it was trained with stay recorded. Train a new experiment to use the new labels.
    - eval split labels changed: keep the trained model, recompute the evaluation.
    Returns {"trained_with": {split: hash}, "evaluated_with": hash} (also in results.json).
    """
    state_path = run_dir / "label_fixes.json"
    split = cfg.eval.split
    current = split_hashes(data_info.get("label_fixes"), list(data_info["splits"]))
    started = (run_dir / "train").exists()
    finished = (run_dir / "train_summary.json").exists()
    if state_path.exists():
        state = load_json(state_path)
        if "trained_with" not in state:  # first version stored {split: hash}
            state = {"trained_with": {s: state.get(s) for s in TRAINING_SPLITS}, "evaluated_with": state.get(split)}
    elif started:  # a run from before label fixes existed was built without any
        none = split_hashes(None, current)
        state = {"trained_with": {s: none[s] for s in TRAINING_SPLITS}, "evaluated_with": none[split]}
    else:
        state = {"trained_with": {}, "evaluated_with": current[split]}

    if not started:  # training (re)starts from scratch on the current labels
        state["trained_with"] = {s: current[s] for s in TRAINING_SPLITS}
    changed = [s for s in TRAINING_SPLITS if state["trained_with"].get(s) != current[s]]
    if changed and not finished:
        raise ValueError(
            f"Label fixes for {changed} changed while '{cfg.name}' was training (resuming would mix labels).\n"
            f"Give the experiment a new `name`, or delete {run_dir} to start over."
        )
    if changed:
        logger.warning("'%s' was trained before the label fixes for %s changed: evaluating the trained model as it is "
                       "(train a new experiment to use the new labels).", cfg.name, changed)
    if state["evaluated_with"] != current[split]:
        archive = run_dir / f"results_{split}-labels-{state['evaluated_with']}.json"
        logger.warning("Label fixes for '%s' changed: re-evaluating (previous results kept as %s).", split, archive.name)
        shutil.rmtree(run_dir / "eval", ignore_errors=True)
        if (run_dir / "results.json").exists():
            os.replace(run_dir / "results.json", archive)
        state["evaluated_with"] = current[split]
    save_json(state, state_path)
    return state


def _changed_keys(old: dict, new: dict) -> list[str]:
    old_flat, new_flat = _flatten(old), _flatten(new)
    keys = sorted(set(old_flat) | set(new_flat))
    return [k for k in keys if k not in IGNORED_KEYS and old_flat.get(k) != new_flat.get(k)]


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"
