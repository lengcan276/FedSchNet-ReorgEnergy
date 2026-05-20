# Overleaf upload package — Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction

This zip is the Overleaf-ready bundle for the post-E71 version of the manuscript
(title above). It compiles cleanly with TeXLive 2020 (Overleaf default toolchain)
and contains everything needed to produce `main.pdf` (26 pages) and `SI.pdf`
(16 pages).

## 1. Compile order

The main manuscript and the SI share a single bibliography (`references.bib`).
SI text does not cross-reference main.tex via `xr`/`xr-hyper` (see §5 below);
this means **main.tex must be compiled before SI.tex** only because Overleaf
caches both auxiliary files, but each `.tex` is otherwise self-contained.

### main.tex
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Expected output: `main.pdf`, 26 pages, ~1.0 MB. No errors. Two cosmetic
`Overfull \hbox` warnings in two table rows (pre-existing, do not affect
visible layout).

### SI.tex
```
pdflatex SI.tex
bibtex SI
pdflatex SI.tex
pdflatex SI.tex
```
Expected output: `SI.pdf`, 16 pages, ~0.58 MB. No errors, no undefined
references. Three cosmetic `'h' → 'ht'` float-specifier adjustments
(automatic LaTeX behavior).

### One-shot (Overleaf "Recompile" button is sufficient)

Overleaf's standard `latexmk -pdf` runs the above passes automatically when the
project is opened. The "main document" dropdown should be set to `main.tex`
for the front-page compilation; switch to `SI.tex` to generate `SI.pdf`.

## 2. Files in this bundle

```
main.tex                   — Manuscript LaTeX source (post-E71 rewrite, title updated)
SI.tex                     — Supporting Information LaTeX source (post-E71 rewrite, §S7.6 E71 control)
references.bib             — Shared bibliography (5 entries carry TODO DOIs; see §3)
figures/
  Fig1_task_client_molecules.pdf
  Fig2_pc2_method.pdf
  Fig3_transferability_matrices.pdf
  Fig4_main_performance.pdf
  Fig5_ablation_forest.pdf  (3-panel calibration-mechanism figure, post-E71 rewrite)
  FigS1_per_molecule_cases.pdf  (demoted from main-text Figure 6 to SI Figure S1)
README_Overleaf.md         — This file
final_pdf_audit.md         — Detailed audit log of the post-E71 PDF freeze
```

`tables/` (markdown + CSV versions of Tables 1–4) is **not** included in the
zip because all tables are formatted inline inside `main.tex` floats; the
external markdown/CSV files are author-side working data, not LaTeX dependencies.

Excluded from this zip (cleaned via the `.gitignore`-style filter used during
packaging): `__pycache__/`, `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`,
`*.toc`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, any CPU box scratch build
artifacts.

## 3. Remaining TODO blockers (author-side; not pre-compile issues)

These do not block compilation but must be cleared before journal submission.
They are also listed in `paper/submission_checklist.md` §0.

### B2 — Author metadata / affiliations / corresponding-author email
The `\TODO{...}` placeholder strings remain in:
- `main.tex` line 74: `\author{\TODO{Author list, affiliations, and corresponding-author email -- to be supplied}}`
- `SI.tex` line 71: `\author{\TODO{Authors as in main.tex}}`

Replace with the actual author block. The `\TODO{}` command renders the
placeholder in red so it is visually obvious in proofs.

### B3 — Five references with unverified DOIs
`references.bib` contains 5 entries marked `note = {TODO: verify DOI ...}`:
1. `Sasabe2011_OLED`
2. `AtahanEvrenk2019_RE`
3. `Nelsen1987`
4. `EfficientKAN2024`
5. `RDKit2023`

Each should be replaced with the canonical DOI verified against Crossref or
the journal's own DOI registry. A working log is in `paper/reference_todo_audit.md`.

### B4 — Data and Code Availability URL
Both `main.tex` and `SI.tex` Data-and-Code-Availability sections contain
`\TODO{Repository URL -- to be supplied}` placeholders. Replace with the
canonical public URL (Zenodo, GitHub, or institutional). `final_pdf_audit.md`
documents the expected scope of the released code/data.

### Final-export Times New Roman (if required by target journal)
The current LaTeX uses Latin Modern (TeXLive's default `lmodern` package). If
the target journal (JCIM, Digital Discovery, etc.) requires Times New Roman:

1. Replace `\usepackage{lmodern}` in both `main.tex` and `SI.tex` with
   `\usepackage{newtxtext,newtxmath}` (Latin Modern → Times-derived).
2. Recompile both documents (no other source changes required; figure files
   are vector PDFs and re-render automatically against the new text font).
3. Verify no overfull-hbox regression in the table floats on `main.tex` pages
   23–25 (pre-existing cosmetic warnings may shift slightly under a different
   font metric set).

The current Latin Modern build is acceptable for supervisor review and for
preprint posting (arXiv, ChemRxiv). Switch fonts only if the target journal
explicitly demands it.

## 4. Headline scientific findings encoded in this LaTeX

For reviewer convenience, the key results in the PDF that distinguish this
package from earlier drafts:

1. **Title** has been updated from *Protocol-Calibrated Personalized FL* to
   **Calibration-Aware Personalized FL** to reflect the E71-identified
   mechanism. The acronym **PC²-FedReorg** is retained as the framework's
   brand name.

2. **Abstract** carries the E71 control result: FedPer + calibration only is
   not statistically distinguishable from PC²-FedReorg on C-triplet (paired
   Wilcoxon p = 0.929), while improving over FedPer without calibration
   (p = 0.028) and over PC² without calibration (p = 0.021). The Abstract
   explicitly says no equivalence margin was pre-registered and the
   chemistry/protocol gate is positioned as auditable compatibility metadata
   rather than a performance lever.

3. **§1 contributions** are reordered: (1) task-specific calibration under
   personalized heads, (2) task-client modeling, (3) calibration-aware
   personalized FL baseline (E71), (4) auditable compatibility metadata,
   (5) small-sample evaluation protocol.

4. **§4.4** is retitled *Calibration under private heads explains the
   C-triplet gain* and leads with the three E71 contrasts before the four
   PC² component ablations.

5. **Discussion §5** is rewritten as six independent points; the chemistry/
   protocol gate is described as a deterministic, inspectable record rather
   than a compliance/contamination-prevention claim.

6. **Limitations §6** items 9 and 10 add the LOOCV paired-error independence
   caveat and the two-estimand consistency note (Table 2 = 5-seed mean MAE;
   Table 3 = per-molecule median-aggregated MAE; differ by ≤ 0.003 eV).

7. **Figure 5** is redrawn as a three-panel calibration-mechanism figure:
   (a) C-triplet/C-hole bar plot for E63/E70/E66/E71; (b) forest plot of three
   E71 contrasts plus four PC² ablations; (c) interpretation box.

8. **Figure 6** has been demoted to **Figure S1** (per-molecule error cases,
   qualitative aid only) in the Supporting Information.

9. **Tables 2/3/4** carry E71 rows (4/6/2 new rows respectively).

10. **SI §S7.6** ("E71 — FedPer with calibration only (positive control)") is
    a new section in this round containing the full reproducibility spec,
    across-seed metrics, paired comparisons on both C targets, and the
    runnable command.

11. **Data-provenance audit (2026-05-20).** The `paper/audit/` directory (not
    shipped in this zip; see project repository) records an independent
    recompute of every numerical claim in the manuscript from the raw
    prediction JSONs. Two estimand-mix wordings were corrected in this
    round of main.tex:
    - line 306 (was "the entire CI sits below zero" for E66 vs FedProx /
      FedPer): softened to "CI upper bound sits just below zero" with an
      explicit caveat that the upper bounds are within ~0.01 eV of zero
      and shift across the zero line under a different bootstrap RNG seed
      (independent reseed values disclosed in-line). The Wilcoxon p-values
      remain the primary inferential anchor.
    - line 320 (§4.4 paragraph lead): was "0.654 ± 0.025 eV (mean ± seed-SD),
      versus 0.660 ± 0.022 eV", which mixed per-molecule-median central
      values with seed-SD spread. Rewritten as "0.654 eV versus 0.660 eV
      (per-molecule median-aggregated MAE) with 5-seed mean MAE ± seed-SD
      = 0.653 ± 0.025 vs. 0.671 ± 0.022 eV" and pointers to which
      table/figure-panel each estimand matches.

## 5. Notes for Overleaf-side maintainers

- **No xr-hyper / xr cross-document references.** SI.tex used to reference
  `main.tex`'s Table 1 via `\ref{tab:task-clients}`, but both `xr-hyper` and
  plain `xr` conflict with `natbib`'s `\citation{...}` entries in `main.aux`
  (produces `! Argument of \@citex has an extra }.` errors). The single
  cross-reference is hardcoded as `Table~1` in `SI.tex` line 128, with a
  documentation comment near the package-import block (`SI.tex` lines
  46–51) explaining the conflict. If you add more cross-doc references,
  the recommended fix is to migrate the bibliography to BibLaTeX, which
  uses a `.bcf` workflow that does not conflict with xr.

- **Figure files are vector PDFs (matplotlib + RDKit output).** Re-rendering
  is not necessary for Overleaf compilation; the embedded fonts in each
  figure PDF are Type-1 and ship at 600 dpi.

- **Two cosmetic Overfull `\hbox` warnings in `main.tex`** appear at table
  floats around lines 471–482 and 525–548 (in the pre-audit numbering;
  line numbers shifted by ~10 after the 2026-05-20 estimand-mix and
  CI-caveat edits). They are caused by long column headers in Table 3
  (paired-stats table) and do not affect the visible layout. Pre-existing.

- **§S7 numbering offset between SI.md and SI.tex.** The working markdown
  source `paper/SI.md` (not shipped in this zip) has a §S7.1 *Ablation
  matrix* intro subsection that was never transferred to `SI.tex`. As a
  result, the LaTeX-rendered SI uses §S7.1 ... §S7.6 while the markdown
  uses §S7.2 ... §S7.7. Conceptual content matches; only the leading digit
  differs. Decide whether to add the *Ablation matrix* subsection to
  `SI.tex` (advances all §S7 numbers by 1) or to remove it from `SI.md`
  (preserves the LaTeX numbering). Either resolution is acceptable.

- **All conclusions are traceable** to `results/*` files in the project
  repository. The `final_pdf_audit.md` shipped in this bundle records the
  exact `results/e71_*` / `results/pc2_*` artifacts that back each Figure
  and Table.

## 6. Single-line audit summary

> Compiles clean (TeXLive 2020); 26 + 16 pages; no errors, no undefined
> references; E71 mechanism rewrite fully integrated; three author-side
> `\TODO{}` blockers remain (B2 author metadata, B3 five DOIs, B4 repo URL).
