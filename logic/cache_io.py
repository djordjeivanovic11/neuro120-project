"""Lightweight on-disk cache under ``results/cache/`` (see :data:`config.CACHE_DIR`).

* **NumPy-only** results: use :func:`save_arrays` / :func:`load_arrays` (same
  format as :func:`pipeline._save_cache`).
* **Arbitrary objects** (nested dicts, DataFrames, lists of arrays): use
  :func:`save_object` / :func:`load_object` (``joblib``, compressed) or
  :func:`cached_run` to skip recomputation in one line.

Tensors, live TF graphs, or open file handles may not serialize; in that
case cache only the arrays you need (``npz``) or a reduced summary dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

import joblib
import numpy as np

from config import CACHE_DIR, ensure_dirs

T = TypeVar("T")


def save_arrays(stem: str, **arrays: np.ndarray) -> Path:
    """Write ``results/cache/{stem}.npz`` (compressed). Same as ``pipeline._save_cache``."""
    ensure_dirs()
    path = CACHE_DIR / f"{stem}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_arrays(stem: str) -> dict[str, np.ndarray]:
    """Load ``.npz`` as a str -> array dict."""
    path = CACHE_DIR / f"{stem}.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def has_arrays(stem: str) -> bool:
    return (CACHE_DIR / f"{stem}.npz").is_file()


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
    return (CACHE_DIR / f"{_safe_name(name)}.joblib").is_file()


def cached_run(
    name: str,
    fn: Callable[[], T],
    *,
    force: bool = False,
) -> T:
    """Return ``fn()`` once, then reload from ``results/cache/{name}.joblib`` on later runs.

    Example::

        from cache_io import cached_run
        import pipeline

        out = cached_run("bellier_part2_run1", lambda: pipeline.run_bellier_vocal_component_model(
            supergrid, vocal_present, K=5, ...
        ))
    """
    if not force and has_object(name):
        return load_object(name)
    out = fn()
    save_object(name, out)
    return out


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")
