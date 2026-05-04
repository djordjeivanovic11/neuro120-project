"""Save and load cache files under ``results/cache/``.

Arrays go to compressed ``.npz`` via :func:`save_arrays` (used by
:mod:`pipeline` for lightweight checkpoints). Python objects go to compressed
``joblib`` via :func:`save_object` / :func:`load_object` / :func:`has_object`
(used for ``pipe_*.joblib`` pipeline memoisation). Prefer numpy arrays and plain
dicts so saves stay portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config import CACHE_DIR, ensure_dirs


def save_arrays(stem: str, **arrays: np.ndarray) -> Path:
    """Write ``results/cache/{stem}.npz`` (compressed)."""
    ensure_dirs()
    path = CACHE_DIR / f"{stem}.npz"
    np.savez_compressed(path, **arrays)
    return path


def save_object(name: str, obj: Any, *, compress: int = 3) -> Path:
    """Pickle-like cache using joblib (good for numpy-rich Python objects)."""
    ensure_dirs()
    path = CACHE_DIR / f"{_safe_name(name)}.joblib"
    joblib.dump(obj, path, compress=compress)
    return path


def load_object(name: str) -> Any:
    path = CACHE_DIR / f"{_safe_name(name)}.joblib"
    return joblib.load(path)


def has_object(name: str) -> bool:
    """Return True if ``results/cache/{safe(name)}.joblib`` exists."""
    return (CACHE_DIR / f"{_safe_name(name)}.joblib").is_file()


def delete_object(name: str) -> None:
    """Remove ``results/cache/{safe(name)}.joblib`` if it exists (ignore errors)."""
    path = CACHE_DIR / f"{_safe_name(name)}.joblib"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _safe_name(name: str) -> str:
    """Sanitise a cache key so it is safe as a single filename component."""
    return name.replace(" ", "_").replace("/", "_")
