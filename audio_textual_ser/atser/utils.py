"""Small helpers shared by every module: logging, timing, seeding, hashing, json io."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def log(message: str) -> None:
    """Print a timestamped progress line and flush, so it shows up live in notebooks."""
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


@contextmanager
def timed(label: str):
    """Log how long a block took: ``with timed("fold 1"): ...``."""
    start = time.time()
    yield
    log(f"{label} — {(time.time() - start) / 60:.1f} دقیقه")


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy and torch exactly the way the author's notebooks do (cell "seed = ...")."""
    import torch
    from transformers import set_seed

    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def stable_hash(obj: Any, length: int = 8) -> str:
    """Short, deterministic hash of any json-able object (used to name cache directories)."""
    blob = json.dumps(to_jsonable(obj), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def slugify(name: str) -> str:
    """'HooshvareLab/bert-fa-base-uncased' -> 'HooshvareLab__bert-fa-base-uncased'."""
    name = name.rstrip("/").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", name)


def short_model_name(checkpoint: str) -> str:
    """'HooshvareLab/bert-fa-base-uncased' -> 'bert-fa-base-uncased'."""
    return checkpoint.rstrip("/").split("/")[-1]


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def dir_size(path: str | os.PathLike) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
