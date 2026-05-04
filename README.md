# NEURO 120 Final Project — Song vs Speech vs Instrumental Music in Human ECoG

A re-analysis of two human electrocorticography (ECoG) datasets, asking whether
the brain regions that respond most strongly to song also carry the most
information a decoder could use to tell song apart from instrumental music.

## The two papers

If you are new to this literature, here is enough context to read the writeup.

**Norman-Haignere et al. (2022) — "A neural population selective for song in
human auditory cortex"** is the paper that kicked the question off. They played
165 natural sounds (songs, speech, instrumental music, and everyday sounds) to
patients with ECoG electrodes on the surface of their auditory cortex and
recorded the neural response to each sound. Using a hypothesis-free statistical
decomposition, they found that some electrodes respond much more strongly to
song than to any other category. Seven electrodes were flagged as
"song-selective" on that basis. Their main claim is that a distinct neural
population tracks song specifically, separate from speech and instrumental
music.

**Bellier et al. (2023) — "Music can be reconstructed from human auditory
cortex activity using nonlinear decoding models"** runs a different experiment.
Twenty-nine ECoG patients listened to a continuous 191-second excerpt of Pink
Floyd's *Another Brick in the Wall, Part 1*, and the authors showed that the
song's acoustics can be reconstructed from the neural recordings. They also
report that vocal content drives stronger responses in right STG than left STG.
We include Bellier as a second setting: patients heard one long uninterrupted
excerpt of a real song, whereas Norman–Haignere used short repeated sound clips,
so we can see whether patterns generalize across stimulus style and duration.

**Our re-analysis** asks one question across both datasets: are the
song-selective electrodes actually more *informative* for a decoder than their
neighbours, or do they just fire more loudly? See `writeup.pdf` for the full
answer. The short version is: informative, yes; privileged, no.

## Fork this repo — what you need to reproduce

`requirements.txt` lists **Python libraries only**. When you run `pip install -r
requirements.txt`, those packages go into **whatever environment is currently
active** (for example a `python -m venv .venv` you activated, or a conda env
after `conda activate myenv`). Pip does **not** download the ECoG datasets,
install LaTeX, or pick a notebook kernel for you—the table below separates pip
from those manual steps.

| Installed by pip (`requirements.txt`) | You install or place separately |
|---------------------------------------|----------------------------------|
| NumPy, SciPy, pandas, scikit-learn, joblib, matplotlib, seaborn, tqdm | **Norman–Haignere** and **Bellier** public releases under `data/` per **`data/README.md`** |
| PyTorch (CPU wheel by default; optional CUDA build from [pytorch.org](https://pytorch.org)) | **`data/bellier_2023/audio/vocal_segments.csv`** — hand-built vocal timing file; if missing, `run_all` still runs Norman-only work and **skips** Bellier with a logged `FileNotFoundError`. |
| Jupyter, nbformat, ipykernel | **LaTeX** with `latexmk` (e.g. MacTeX / TeX Live) to rebuild `writeup.pdf` from `writeup.tex`. |

**Python version:** use **3.10, 3.11, or 3.12**

**Notebooks — pick the right kernel:** Open a `.ipynb` in VS Code or Jupyter.
Use the **kernel / interpreter picker** (often **top-right** of the notebook)
and select the **same conda environment or venv** where you ran `pip install -r
requirements.txt`. That way the notebook sees numpy, torch, and the rest.
If your env does not show up, activate it in a terminal and run once:
`python -m ipykernel install --user --name=neuro120`, then choose **neuro120**
from the picker (or select **Enter interpreter path** and point at that env’s
`python`).

**Optional script:** `python scripts/generate_composite_figures.py` rebuilds the write-up composite PDFs under `results/figures/`. Run `python scripts/generate_composite_figures.py --help` for flags. If joblib cache load errors (e.g. after a Python or pandas upgrade), use **`--force`** to recompute and refresh `results/cache/pipe_*.joblib`, or **`--no-cache`** to skip cache I/O entirely for that run.

## How to run it

**1. Install dependencies** (one-time; Python **3.10–3.12** recommended — see table above):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Optional: speed (CPU + GPU).** Importing `logic.config` (or running `python logic/pipeline.py`, which loads `compute_env` first) sets BLAS/OpenMP thread caps from `NEURO120_NUM_THREADS` (`auto` = all logical cores) so NumPy/sklearn use the machine well. Heavy loops (random-subset null, RDM bootstrap/permutation, LOO) also use **joblib** across cores (`NEURO120_SKLEARN_JOBS`, default `-1` = all).

```bash
export NEURO120_NUM_THREADS=auto     # BLAS / OpenMP (default: all cores)
export NEURO120_SKLEARN_JOBS=-1     # joblib + LogisticRegressionCV inner CV (default: all)
export NEURO120_TORCH_NUM_THREADS=8 # PyTorch CPU ops when a GPU is used (default: ~half of cores)
```

**Bellier CNN (PyTorch GPU).** The optional TinyTemporalCNN runs only when Bellier log-reg balanced accuracy clears chance by `CNN_MIN_BACC_OVER_CHANCE` in `logic/config.py` (default +0.05 over 0.5); otherwise the decoder prints `cnn skipped`. Device selection: CUDA, then Apple MPS, then CPU. Override with:

```bash
export NEURO120_DEVICE=auto   # default: cuda → mps → cpu
export NEURO120_DEVICE=cuda   # NVIDIA GPU if available
export NEURO120_DEVICE=mps    # Apple Silicon GPU if available
export NEURO120_DEVICE=cpu    # always CPU (most reproducible)
```

Install a CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org) if `torch.cuda.is_available()` is false but you have a GPU.

**Notebooks:** for the strongest BLAS effect, run a first cell *before* `import numpy` that does `import compute_env; compute_env.apply_blas_thread_env()` (or `import config`), then import the rest.

**2. Put the data in `data/`** following `data/README.md`. Both datasets are
publicly released with their respective papers. The code expects them in the
layout `data/norman_haignere_2022/` and `data/bellier_2023/`.

**3. Run the full analysis.** Pick either option; both produce the same
figures and tables.

Option A — notebook (recommended if you want to see the intermediate
results):

```bash
jupyter lab neuro120_main_reproduction.ipynb
```

For the Bellier extension and supplementary plots (matched random-subset nulls,
exploratory histograms), open `neuro120_bellier_supplement.ipynb`.

Then select the Python kernel for your virtual environment and run all cells.

Option B — from the command line (one command, headless):

```bash
python logic/pipeline.py
python logic/pipeline.py --help          # --skip, --no-bellier, --no-cache, --force-recompute
python logic/pipeline.py --force-recompute   # after env upgrade or broken joblib pickles
```

**Composites only** (faster than a full `run_all` when you only need `fig1_main_composite` etc.):

```bash
python scripts/generate_composite_figures.py
python scripts/generate_composite_figures.py --force
python scripts/generate_composite_figures.py --only 3
```

**4. Look at the output.** Everything lands in `results/`:

- `results/figures/` — every figure in `writeup.pdf` as a PDF.
- `results/tables/` — every table as a CSV.
- `results/cache/` — intermediate artifacts (including the Bellier supergrid,
  which is large and takes the longest to build on a cold run).

**5. Rebuild the writeup** if you edit `writeup.tex` (requires a LaTeX install with `latexmk`, e.g. MacTeX or TeX Live):

```bash
latexmk -pdf writeup.tex
```

## Reproducing every number in the writeup

Every numeric claim in `writeup.pdf` (abstract, results, tables) is read
from `results/tables/*.csv` produced by `python logic/pipeline.py`
(random seed `42`, defined in `logic/config.py`). To regenerate the full
packet from a clean checkout:

```bash
python logic/pipeline.py        # populates results/tables and results/figures
latexmk -pdf writeup.tex        # recompiles writeup.pdf against the fresh outputs
```

The single source of truth for the headline random-subset numbers is
`results/tables/random_subset_control_summary.csv`; for acoustic
partialling, `results/tables/acoustic_partition_divergence.csv`; for the
Bellier extension, `results/tables/bellier_decoder_summary.csv`.

## Troubleshooting

- **`FileNotFoundError` under `data/`**: the analysis expects the two
  upstream releases laid out as `data/norman_haignere_2022/` and
  `data/bellier_2023/`. See `data/README.md` for the exact files and
  formats. The pipeline does not download data automatically.
- **Bellier supergrid is slow on first run**: the cross-patient
  concatenation is cached under `results/cache/`. Subsequent runs
  reuse the cache; delete it to force a rebuild.
- **`latexmk` reports undefined references on the first pass**: this is
  expected; `latexmk` runs `pdflatex` enough times to resolve them. Run
  it once more if you see leftover `??` markers.