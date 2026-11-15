"""Device selection and resource-usage helpers (GPU/CPU memory, model size)."""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device string ("auto" | "cpu" | "cuda" | "cuda:0" | ...)."""
    if preference != "auto":
        return preference

    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def count_parameters(model) -> int:
    """Total parameter count for a torch.nn.Module."""
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def reset_gpu_peak_memory(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except ImportError:
        pass


def gpu_peak_memory_mb(device: str) -> float | None:
    """Peak allocated CUDA memory (MB) since the last reset, or None if N/A."""
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except ImportError:
        pass
    return None


def cpu_memory_mb() -> float:
    """Resident set size of the current process, in MB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_environment_info() -> dict:
    """Snapshot of the runtime environment, saved alongside each run's
    config so results can be traced back to the exact software stack used.
    """
    info: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False

    try:
        import ultralytics

        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = None

    return info
