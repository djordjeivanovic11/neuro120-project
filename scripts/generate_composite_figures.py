#!/usr/bin/env python3
"""Generate Georgia-style composite paper figures (PDF + PNG) into results/figures/."""

#
# From the repo root (the folder that contains data/, logic/, scripts/):
#   python scripts/generate_composite_figures.py
#   python scripts/generate_composite_figures.py --only norman
#   python scripts/generate_composite_figures.py --only 3
#
# Or with an absolute path (no cd needed):
#   python /Users/you/.../neuro120-project/scripts/generate_composite_figures.py
#
# Do not use a literal "cd /path/to/neuro120-project" unless that path exists.
#
# Outputs: fig1_main_composite, fig2_mechanisms_composite, fig3_bellier_composite
# If a long run was interrupted during Bellier/CNN, rerun with --only 3 to finish Figure 3 only.
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_logic_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    logic_dir = repo_root / "logic"
    if not logic_dir.is_dir():
        raise SystemExit(f"Expected logic/ at {logic_dir}")
    sys.path.insert(0, str(logic_dir))
    return repo_root


def main() -> None:
    repo_root = _ensure_logic_on_path()

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the comment block at the top of this file for paths and outputs.",
    )
    p.add_argument(
        "--only",
        choices=("all", "norman", "1", "2", "3"),
        default="all",
        help="all: figures 1–3 (3 omitted with --no-bellier). norman: 1+2 only. "
        "1/2/3: single composite (use --only 3 after an interrupted full run).",
    )
    p.add_argument(
        "--no-bellier",
        action="store_true",
        help="skip Figure 3 when --only all (ignored for --only 3)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="do not load pipeline joblib cache (always recompute upstream steps)",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="overwrite pipeline cache entries when cache is enabled",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from config import FIG_DIR  # noqa: WPS433 — after path setup
    from data_utils import build_dataset
    from pipeline import run_figure_composites

    if args.only == "3" and args.no_bellier:
        print(
            "Note: --only 3 requires Bellier; ignoring --no-bellier.", file=sys.stderr
        )

    print(f"Repo root: {repo_root}")
    print(f"Figures dir: {FIG_DIR}")

    which = None
    include_bellier = not args.no_bellier
    if args.only == "norman":
        which = {"1", "2"}
        include_bellier = False
    elif args.only == "1":
        which = {"1"}
        include_bellier = False
    elif args.only == "2":
        which = {"2"}
        include_bellier = False
    elif args.only == "3":
        which = {"3"}
        include_bellier = True

    ds = build_dataset()
    out = run_figure_composites(
        ds,
        seed=args.seed,
        use_cache=False if args.no_cache else None,
        force=args.force_recompute,
        include_bellier=include_bellier,
        which=which,
    )

    print("Done. Artifact keys:", sorted(out.keys()))
    for key in ("figure1", "figure2", "figure3"):
        if key in out and isinstance(out[key], dict):
            print(f"  {key}: {out[key].get('pdf', out[key])}")
    for err_key in ("figure2_error", "figure3_error"):
        if err_key in out:
            print(f"  WARNING {err_key}: {out[err_key]}")
    if args.only == "all" and include_bellier and "figure3" not in out:
        print(
            "  Hint: Figure 3 missing (skipped, failed, or interrupted). To build only Figure 3:\n"
            "    python scripts/generate_composite_figures.py --only 3",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
