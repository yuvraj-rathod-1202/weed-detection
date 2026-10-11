"""Training stage: fine-tune an ultralytics segmentation model, resuming automatically.

Auto-save: ultralytics writes train/weights/last.pt (and best.pt) after every epoch.
If a run is interrupted, calling `train_model` again resumes from last.pt.
`train_summary.json` is written when training finishes and marks the stage as done.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.utils.config import to_container
from src.utils.io import load_json, save_json


def training_finished(last_pt: Path) -> bool:
    """ultralytics sets epoch = -1 in last.pt once training (or early stopping) completes."""
    import torch

    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", 0) == -1


def train_model(cfg, run_dir: Path, data_yaml: Path, logger) -> dict:
    """Train (or resume) and return the training summary. Skips if already done."""
    from ultralytics import YOLO

    summary_path = run_dir / "train_summary.json"
    if summary_path.exists():
        logger.info("Training already done: %s", summary_path)
        return load_json(summary_path)

    train_dir = run_dir / "train"
    last_pt = train_dir / "weights" / "last.pt"
    start = time.time()

    if last_pt.exists() and not training_finished(last_pt):
        logger.info("Resuming training from %s", last_pt)
        YOLO(str(last_pt)).train(resume=True, data=str(data_yaml))
    elif last_pt.exists():
        logger.info("Training finished earlier but summary is missing; rebuilding summary.")
    else:
        train_args = {k: v for k, v in to_container(cfg.train).items() if v is not None}
        train_args.setdefault("seed", cfg.seed)
        train_args.setdefault("deterministic", True)
        logger.info("Training %s with %s", cfg.model.weights, train_args)
        YOLO(cfg.model.weights).train(
            data=str(data_yaml),
            project=str(run_dir.resolve()),
            name="train",
            exist_ok=True,
            **train_args,
        )

    summary = summarize_training(train_dir)
    summary["wall_time_this_session_s"] = round(time.time() - start, 1)
    save_json(summary, summary_path)
    return summary


def summarize_training(train_dir: Path) -> dict:
    """Per-epoch history from ultralytics' results.csv + best epoch (by mask mAP50-95)."""
    history = pd.read_csv(train_dir / "results.csv")
    history.columns = [c.strip() for c in history.columns]
    best_col = "metrics/mAP50-95(M)"
    best_row = history.loc[history[best_col].idxmax()] if best_col in history else history.iloc[-1]
    # The "time" column restarts from 0 after a resume: add up the last value of every session.
    t = history["time"].tolist() if "time" in history else []
    train_time = sum(t[i] for i in range(len(t)) if i == len(t) - 1 or t[i + 1] < t[i])
    return {
        "epochs_completed": int(history["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "train_time_s": round(float(train_time), 1) if t else None,
        "best_weights": str(train_dir / "weights" / "best.pt"),
        "last_weights": str(train_dir / "weights" / "last.pt"),
        "history": history.to_dict(orient="records"),
    }
