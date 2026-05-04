"""Paths, seeds, and hyperparameters for the whole project.

Change numbers here instead of copying them into other files. One shared
``RANDOM_STATE`` keeps CV and permutations reproducible.

Speed knobs: ``NEURO120_NUM_THREADS`` (BLAS/OpenMP), ``NEURO120_SKLEARN_JOBS``
(sklearn/joblib), ``NEURO120_DEVICE`` / ``NEURO120_TORCH_NUM_THREADS`` (PyTorch;
see :func:`get_torch_device`).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from compute_env import apply_blas_thread_env

    apply_blas_thread_env()
except ImportError:  # pragma: no cover
    pass

# When True, :func:`pipeline` ``run_*`` restorers return values from
# ``results/cache/pipe_*.joblib`` when a matching key exists (faster
# re-runs). Set env ``NEURO120_CACHE_PIPELINE=0`` to always recompute.
USE_COMPUTATION_CACHE = os.environ.get("NEURO120_CACHE_PIPELINE", "1") not in (
    "0",
    "false",
    "False",
)


# reproducibility seed shared by every resampling and cv step
RANDOM_STATE = 42

# time window in seconds relative to stimulus onset
TIME_MIN = 0.0
TIME_MAX = 2.0

# sliding window used by time resolved decoders and divergence curves
WINDOW_SEC = 0.15
STEP_SEC = 0.05

# outer cross validation folds and repeats
N_SPLITS = 5
N_REPEATS = 20

# resampling budgets for bootstrap, permutation, and random subset draws
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05
PERM_N = 1000
RANDOM_SUBSETS_N = 1000

# size for the matched random subset control, set from the song only
# electrode count (7), callers may override via arguments
SUBSET_SIZE = 7

# early window used for summary statistics on the time resolved curves
EARLY_WINDOW = (0.2, 0.6)

# ---------------------------------------------------------------------------
# Norman--Haignere stimulus labels (fine -> coarse collapse in ``data_utils``)
# ---------------------------------------------------------------------------
SPEECH_FINE = {"EngSpeech", "ForSpeech"}
SONG_FINE = {"Song"}
MUSIC_FINE = {"Music"}
MAJOR_CLASSES = ("song", "speech", "music")

# config lives at <project_root>/logic/config.py so two parent hops
# land on the project root that holds data/ and results/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
NORMAN_HAIGNERE_DIR = DATA_ROOT / "norman_haignere_2022"
BELLIER_DIR = DATA_ROOT / "bellier_2023"

# primary dataset used for the main analyses
DATA_DIR = NORMAN_HAIGNERE_DIR / "individual_electrodes"

# Acoustic regressors for ridge partialling: ``acoustic_features.mat`` and
# ``ecog_component_responses.mat`` (canonical 165-stimulus order; see upstream MATLAB).
NORMAN_ACOUSTIC_MAT = NORMAN_HAIGNERE_DIR / "acoustic_features.mat"
NORMAN_COMPONENT_RESP_MAT = NORMAN_HAIGNERE_DIR / "ecog_component_responses.mat"

# ---------------------------------------------------------------------------
# Output directories (created by :func:`ensure_dirs`)
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"

# ---------------------------------------------------------------------------
# Bellier 2023 extension (continuous stimulus, 100 Hz HFA)
# ---------------------------------------------------------------------------
BELLIER_HFA_DIR = BELLIER_DIR / "hfa"
BELLIER_STIM_DIR = BELLIER_DIR / "audio"
BELLIER_AUDIO_PATH = BELLIER_STIM_DIR / "thewall1.wav"      # bellier provided wav
BELLIER_VOCAL_CSV = BELLIER_STIM_DIR / "vocal_segments.csv" # user supplied annotation

BELLIER_FS = 100                # hfa sampling rate in hz
BELLIER_T = 19072               # total samples, roughly 190,72 seconds
BELLIER_CV_FOLDS = 5            # blocked time folds
BELLIER_WIN_SEC = 0.5           # decoder feature window
BELLIER_WIN_STEP_S = 0.1        # step between windows
BELLIER_BOOT_N = 500            # bootstrap replicates for confidence intervals

# TinyTemporalCNN (PyTorch): only trained when logreg bacc clears chance by margin below
CNN_EPOCHS = 40
CNN_BATCH = 64
CNN_LR = 1e-3
CNN_MIN_BACC_OVER_CHANCE = 0.05

# event locked profile window around each onset
PROFILE_PRE_SEC = 0.5
PROFILE_POST_SEC = 1.5


def get_torch_device():
    """Return a ``torch.device`` for optional GPU acceleration (Bellier CNN).

    Controlled by ``NEURO120_DEVICE`` (case-insensitive):

    * ``auto`` (default) — CUDA if available, else Apple MPS if available, else CPU.
    * ``cuda`` / ``gpu`` — CUDA when available, otherwise CPU.
    * ``mps`` — Apple Silicon GPU when available, otherwise CPU.
    * ``cpu`` / ``0`` / ``no`` / ``none`` — always CPU (closest to bit-reproducible training).

    The main Norman--Haignere analyses use NumPy/sklearn only (CPU). Logistic
    regression on Bellier is also CPU; this hook only affects the small PyTorch CNN.
    """
    import torch

    from compute_env import apply_torch_thread_env

    raw = os.environ.get("NEURO120_DEVICE", "auto").strip().lower()
    if raw in ("cpu", "none", "no", "0"):
        dev = torch.device("cpu")
    elif raw in ("cuda", "gpu"):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif raw == "mps":
        mps = getattr(torch.backends, "mps", None)
        dev = (
            torch.device("mps")
            if mps is not None and mps.is_available()
            else torch.device("cpu")
        )
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    apply_torch_thread_env()
    return dev


def sklearn_n_jobs() -> int:
    """Parallel workers for sklearn / joblib (``-1`` = all cores).

    ``NEURO120_SKLEARN_JOBS``: empty, ``auto``, or ``-1`` → ``-1``; ``1`` forces
    single-threaded fits (debug / notebooks that must stay deterministic step-by-step).
    """
    raw = os.environ.get("NEURO120_SKLEARN_JOBS", "").strip().lower()
    if raw in ("", "auto", "-1", "all"):
        return -1
    try:
        j = int(raw)
        return j if j != 0 else 1
    except ValueError:
        return -1


def ensure_dirs() -> None:
    """Create the results subdirectories if they do not exist.

    Idempotent: safe to call at the start of any pipeline entry point.
    Creates ``results/``, ``results/figures/``, ``results/tables/``, and
    ``results/cache/`` under the project root.
    """
    for d in (RESULTS_DIR, FIG_DIR, TAB_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
