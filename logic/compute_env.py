"""Process-wide compute defaults (import before NumPy when you can).

``apply_blas_thread_env`` sets OpenMP / MKL / OpenBLAS thread caps so matrix
work uses all cores unless the user already exported limits.

``apply_torch_thread_env`` caps PyTorch CPU threads (helps when a GPU is used
so the machine does not oversubscribe). Safe to call once after ``import torch``.
"""
from __future__ import annotations

import os


def apply_blas_thread_env() -> int:
    """If unset, set BLAS/OpenMP thread env vars from ``NEURO120_NUM_THREADS``.

    ``NEURO120_NUM_THREADS``: empty or ``auto`` → ``os.cpu_count()``; else a
    positive integer. Uses ``os.environ.setdefault`` so explicit user exports
    win. Returns the numeric cap applied.
    """
    raw = os.environ.get("NEURO120_NUM_THREADS", "").strip().lower()
    if raw in ("", "auto"):
        n = max(1, os.cpu_count() or 1)
    else:
        try:
            n = max(1, int(raw))
        except ValueError:
            n = max(1, os.cpu_count() or 1)
    val = str(n)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, val)
    return n


def apply_torch_thread_env() -> None:
    """Tune PyTorch CPU thread pools (call after ``import torch``).

    ``NEURO120_TORCH_NUM_THREADS``: if set, that many intra-op threads; else
    about half of ``os.cpu_count()`` (minimum 1). Inter-op threads stay small.
    """
    import torch

    raw = os.environ.get("NEURO120_TORCH_NUM_THREADS", "").strip()
    if raw.isdigit():
        n = max(1, int(raw))
    else:
        n = max(1, (os.cpu_count() or 8) // 2)
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    inter = max(1, min(4, n // 4))
    try:
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        pass
