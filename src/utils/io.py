"""Small JSON / JSONL / YAML read-write helpers shared across the pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write JSON atomically (tmp file + rename) so an interrupted run never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_default)
    os.replace(tmp_path, path)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(record: dict, path: str | Path) -> None:
    """Append one record as a line and flush it to disk immediately (incremental auto-save)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file; a truncated last line (crash mid-write) is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml_dict(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_default(obj: Any) -> Any:
    # numpy scalars / arrays, Paths, etc.
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
