#!/usr/bin/env python3
"""Assemble the three write-up composite PDFs from existing pipeline work.

This script does **not** re-run the full analysis from scratch on purpose: it
calls ``pipeline.run_figure_composites``, which pulls each panel’s numbers from
the same ``run_*`` drivers as ``python logic/pipeline.py``. When pipeline
caching is on (default), those steps reload from ``results/cache/pipe_*.joblib``
if the cache key matches—so after a full ``run_all`` (or notebook) has filled
the cache, this is a fast way to refresh ``fig1``–``fig3`` under
``results/figures/``. If a cache entry is missing, the underlying ``run_*``
recomputes that piece.

Run from the repo root::

    python scripts/generate_composite_figures.py
    python scripts/generate_composite_figures.py --force
    python scripts/generate_composite_figures.py --only 3

Use ``--help`` for all flags. If loading ``results/cache/pipe_*.joblib`` fails
(e.g. pandas version mismatch), ``--force`` recomputes panels and overwrites
those entries; ``--no-cache`` skips reading and writing joblib cache entirely
(slower, but nothing pickle-related can break).

Figure 3 (Bellier) is omitted automatically if Bellier inputs are missing on
disk (same behaviour as inside ``run_figure_composites``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_logic_on_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    logic = root / "logic"
    if not logic.is_dir():
        raise SystemExit(f"Expected logic/ at {logic}")
    sys.path.insert(0, str(logic))
    return root


def _parse_which(s: str) -> set[str]:
    parts = {p.strip() for p in s.replace(" ", "").split(",") if p.strip()}
    bad = parts - {"1", "2", "3"}
    if bad:
        raise SystemExit(f"--only must use 1, 2, and/or 3 only; got {bad!r}")
    return parts


def main() -> None:
    repo_root = _ensure_logic_on_path()

    ap = argparse.ArgumentParser(
        description="Build fig1–fig3 composite PDFs/PNGs (same path as the notebooks)."
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute each panel and overwrite pipe_*.joblib cache (fixes stale/broken pickles)",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write results/cache/pipe_*.joblib for this run",
    )
    ap.add_argument(
        "--no-bellier",
        action="store_true",
        help="omit figure 3 even if Bellier data are present",
    )
    ap.add_argument(
        "--only",
        metavar="PANELS",
        help='comma-separated subset of {1,2,3}, e.g. "3" for Bellier composite only',
    )
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: config.RANDOM_STATE)")
    args = ap.parse_args()

    try:
        from config import FIG_DIR, RANDOM_STATE  # noqa: PLC0415
        from data_utils import build_dataset
        from pipeline import run_figure_composites
    except ImportError as exc:
        req = repo_root / "requirements.txt"
        mod = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            f"Missing Python dependency ({mod}). From the repo root:\n"
            f"  {sys.executable} -m pip install -r {req}"
        ) from exc

    seed = RANDOM_STATE if args.seed is None else args.seed
    which = _parse_which(args.only) if args.only else None

    ds = build_dataset()
    out = run_figure_composites(
        ds,
        seed=seed,
        use_cache=False if args.no_cache else None,
        force=args.force,
        include_bellier=not args.no_bellier,
        which=which,
    )

    print(f"{repo_root}\n{FIG_DIR}\n{sorted(out.keys())}")
    for key in ("figure1", "figure2", "figure3"):
        if key in out and isinstance(out[key], dict):
            print(f"  {key}: {out[key].get('pdf', out[key])}")
    for err_key in ("figure2_error", "figure3_error"):
        if err_key in out:
            print(f"  skipped {err_key}: {out[err_key]}", file=sys.stderr)


if __name__ == "__main__":
    main()
