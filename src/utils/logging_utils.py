"""Logger factory: consistent console + per-experiment file logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """Return a logger that writes to stdout and, optionally, a log file.

    Safe to call repeatedly with the same `name` (e.g. once per module) —
    handlers are only attached once per logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
