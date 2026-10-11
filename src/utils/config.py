"""Experiment config loading (OmegaConf).

An experiment is fully described by one YAML file under configs/experiments/
plus optional `key.path=value` CLI overrides. Its `dataset` field is a path to
a dataset config, which is loaded and inlined here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

REQUIRED_KEYS = ("name", "seed", "output_root", "dataset", "model", "train", "eval", "targeting")
REQUIRED_DATASET_KEYS = ("name", "raw_dir", "prepared_dir", "split", "split_seed", "sam_model")


def load_yaml(path: str | Path) -> DictConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def load_experiment_config(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """Load an experiment config, inline its dataset config, apply dotlist overrides."""
    cfg = load_yaml(path)
    if isinstance(cfg.get("dataset"), str):
        cfg.dataset = load_yaml(cfg.dataset)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    missing += [f"dataset.{k}" for k in REQUIRED_DATASET_KEYS if k not in cfg.dataset]
    if missing:
        raise KeyError(f"{path}: missing required config keys: {missing}")
    if not cfg.targeting.methods:
        raise ValueError(f"{path}: targeting.methods must list at least one method")

    cfg.config_file = str(path)
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Plain-dict view of a config, for JSON/YAML serialization."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
