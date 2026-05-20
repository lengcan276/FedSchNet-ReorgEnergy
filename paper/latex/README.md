# PC²-FedReorg LaTeX project — Overleaf-ready

This directory contains an Overleaf-ready LaTeX rendering of the manuscript and the Supporting Information. The Markdown sources in the parent directory (`paper/manuscript.md`, `paper/SI.md`) remain the authoritative content; this `paper/latex/` tree is a derived view for journal submission.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Main manuscript (translated from `paper/manuscript.md` on 2026-05-12) |
| `SI.tex` | Supporting Information (translated from `paper/SI.md` on 2026-05-12) |
| `references.bib` | Bibliography (copy of `paper/references.bib` — 17 cited entries with `note = {TODO}` preserved verbatim) |
| `README.md` | This file |

Figures are loaded from `../figures/Fig1..Fig6_*.pdf` using the project-level relative path `../figures/`.  No figure files are duplicated into `paper/latex/`.

## Compilation

### Local (TeX Live ≥ 2020)

```
cd paper/latex
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex

pdflatex SI.tex
bibtex   SI
pdflatex SI.tex
pdflatex SI.tex
```

Or, on systems with `latexmk` installed:

```
cd paper/latex
latexmk -pdf -bibtex main.tex
latexmk -pdf -bibtex SI.tex
```

### Overleaf upload (recommended)

Two acceptable workflows:

**Option A — upload the entire `paper/` folder.**

1. Tar/zip the entire `paper/` directory: `cd <repo-root>/FedSchNet-ReorgEnergy && tar -czf paper.tgz paper/`
2. Create a new Overleaf project and upload `paper.tgz`. Overleaf will preserve the directory structure.
3. In the Overleaf project settings set the main document to `paper/latex/main.tex`.
4. The relative paths `../figures/FigN_*.pdf` and `../tables/...` resolve correctly because Overleaf compiles each project with the main `.tex` file's directory as the working directory.

**Option B — upload only the LaTeX project with figures co-located.**

1. Inside the project root locally, create a flat upload directory:
   ```
   mkdir -p overleaf_upload
   cp paper/latex/*.tex paper/latex/references.bib paper/latex/README.md overleaf_upload/
   mkdir -p overleaf_upload/figures
   cp paper/figures/Fig*.pdf overleaf_upload/figures/
   ```
2. Edit `main.tex` and `SI.tex` to change `../figures/` to `figures/` (one global find-and-replace).
3. Upload `overleaf_upload/` as a zip to Overleaf and set the main document to `main.tex`.

Option A keeps the `paper/` tree intact and is easier to keep in sync with the Markdown sources; Option B produces a self-contained Overleaf project at the cost of a one-time path edit.

## TODO items preserved verbatim

The following items are marked `\TODO{...}` in red in `main.tex` and `SI.tex`. They appear in the same locations as in the Markdown source and must be cleared before journal submission.

| File | Where | What to supply |
|---|---|---|
| `main.tex` | `\author{}` line | Author list, affiliations, corresponding-author email |
| `main.tex` | Data and Code Availability | Repository URL |
| `main.tex` | Author Contributions | Per-author contribution statements |
| `main.tex` | Acknowledgements | Compute resources, funding, data-providing laboratories |
| `SI.tex` | `\author{}` line | Authors as in main.tex |
| `SI.tex` | Top paragraph | Repository URL |

References with outstanding DOI / metadata TODOs are listed in `paper/reference_todo_audit.md`. These were intentionally **not** filled in: do not invent DOIs. The five entries (Sasabe2011_OLED, AtahanEvrenk2019_RE, Nelsen1987, EfficientKAN2024, RDKit2023) keep their `note = {TODO: ...}` text exactly as in `paper/references.bib`.

## Notes on the LaTeX class and packages

- `\documentclass[11pt,a4paper]{article}` — journal-neutral; switch to `achemso`, `iopart`, `RSC`, `elsarticle`, `wlscirep` or the target journal's class on submission day.
- Citation style: `\usepackage[numbers,sort&compress]{natbib}` with `\bibliographystyle{unsrtnat}` — numeric, in citation order. The Markdown source uses `[1]`–`[18]`; bibkeys are listed in the same order in `references.bib`.
- Fonts: `lmodern` Latin Modern (TNR-compatible). The figure files themselves were rendered with Liberation Serif; if the final submission requires Times New Roman, re-run `paper/make_figures.py` on a machine that has Times New Roman registered with matplotlib's font manager (see `paper/submission_checklist.md` §5).
- Math: `amsmath`, `amssymb`. Units handled by `siunitx`.
- Hyperlinks: `hyperref` loaded last; coloured blue.

## Numerical fidelity

All numerical values in `main.tex` (MAE, RMSE, $R^2$, $p$-values, CI bounds, win/loss counts, T-matrix entries, label statistics) were transcribed from:

- `results/pc2_phase5_stats.csv`
- `results/preverify/T_transferability.json`
- `results/preverify/V2_classical_baselines.csv` and `V2_summary.md`
- `paper/tables/table[1-4]_*.csv`

without rounding beyond the precision used in the Markdown source. If any result file is regenerated, re-derive the corresponding LaTeX tables (or, more efficiently, switch to `\input{}`-style table inclusion from a CSV-to-LaTeX preprocessor).

## What this build does NOT do

- It does **not** include figures rendered with Times New Roman — see §5 of `paper/submission_checklist.md`.
- It does **not** include a cover letter or compile-on-Overleaf preflight check.
- It does **not** auto-resolve `\TODO{...}` items; those remain in red in the compiled PDF as visual reminders.
- It is **not** a journal-specific template — switch document class on submission day.

## Provenance

LaTeX project generated by translating `paper/manuscript.md` (commit on 2026-05-12) and `paper/SI.md` into TeX. All scientific statements, numerical values and TODO notes are preserved verbatim. No DOIs were invented during this translation. No code in `src/`, `experiments/`, or `results/` was modified.
