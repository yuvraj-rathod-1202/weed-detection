from pathlib import Path

import pytest

from src.experiment import _changed_keys
from src.targeting.points import POINT_METHODS
from src.utils.config import load_experiment_config

EXPERIMENT_CONFIGS = sorted(Path("configs/experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_every_experiment_config_loads(path):
    cfg = load_experiment_config(path)
    assert cfg.dataset.name
    assert all(m in POINT_METHODS for m in cfg.targeting.methods)


def test_overrides_apply():
    cfg = load_experiment_config(EXPERIMENT_CONFIGS[0], ["train.epochs=3", "dataset.max_images=5"])
    assert cfg.train.epochs == 3
    assert cfg.dataset.max_images == 5


def test_changed_keys_ignores_machine_paths():
    old = {"train": {"epochs": 10}, "dataset": {"raw_dir": "a"}, "output_root": "x"}
    new = {"train": {"epochs": 20}, "dataset": {"raw_dir": "b"}, "output_root": "y"}
    assert _changed_keys(old, new) == ["train.epochs"]
