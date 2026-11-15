"""Deterministic seeding across python/numpy/torch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all RNGs we know about so runs are reproducible.

    Note: full bit-for-bit determinism on GPU also depends on cuDNN
    algorithm selection, which we make deterministic here at a small
    performance cost. Some ultralytics ops may still be nondeterministic
    on GPU; this gives the best reasonably-achievable reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
