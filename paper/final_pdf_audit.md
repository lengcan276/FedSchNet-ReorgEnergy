# Final PDF audit — pre-submission freeze (2026-05-18)

This document records the pre-submission-freeze audit of `paper/latex/main.pdf`
and `paper/latex/SI.pdf` after the E71 control experiment and the resulting
mechanism rewrite. It is read-only of the scientific conclusions: the only
modifications made during the audit were the synchronization of one section
(SI §S7.7 / SI.tex §S7.6) that had been added to `paper/SI.md` but not
mirrored to `paper/latex/SI.tex`.

---

## 0. Headline status

**Status: PDF and source are consistent on every E71-related content axis after two fixes this round (SI §S7.7 content sync, plus SI cross-doc reference rendering).** Two pre-existing non-content issues remain (one §S7 numbering offset between SI.md and SI.tex, three placeholder `\TODO{}` blocks in author/repo metadata). None of these are scientific defects; all of them are explicitly delegated to the author / corresponding-author side per `paper/submission_checklist.md` §0.

| File | Pages | File size | Last compile | Source match |
|---|---|---|---|---|
| `paper/latex/main.pdf` | **26** | 1.00 MB | 2026-05-18 16:14 | ✅ matches `manuscript.md` |
| `paper/latex/SI.pdf`   | **16** | 0.58 MB | 2026-05-18 16:30 (post-S7.7-sync + cross-ref fix) | ✅ matches `SI.md` (one numbering offset noted below) |

Both PDFs compiled on the CPU box (10.98.137.100; TeXLive 2020, `pdflatex` + `bibtex`) via rsync round-trip, with the standard 3-pass + BibTeX workflow. No `latexmk`, `xelatex`, or `tectonic` was needed.

---

## 1. Title check

| Location | Title |
|---|---|
| `paper/manuscript.md` line 1 | ✅ Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction |
| `paper/SI.md` line 3          | ✅ Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction |
| `paper/latex/main.tex` line 72  | ✅ Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction |
| `paper/latex/SI.tex` line 62 | ✅ Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction |
| `paper/latex/main.pdf` page 1 | ✅ "Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction" |
| `paper/latex/SI.pdf` page 1  | ✅ "Supporting Information for *Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction*" |

Old title "Protocol-Calibrated Personalized Federated Learning…" purged from all `^# `, `\title{`, and `\emph{` positions. (Note: the **PC²** acronym still expands to "Protocol-Calibrated Personalized Federated Learning" inline once, which is the historical brand name of the framework. The Abstract reframes the framework as "calibration-aware personalized FL made auditable" without renaming the acronym.)

---

## 2. Abstract check

| E71 conclusion required | In `manuscript.md` | In `main.tex` | Visible in `main.pdf` |
|---|---|---|---|
| FedPer + calibration ≈ PC²-FedReorg on C-triplet | ✅ | ✅ | ✅ ("not statistically distinguishable from PC²-FedReorg's at our sample size") |
| MAE 0.654 vs. 0.660 eV (paired-stats estimand) | ✅ | ✅ | ✅ |
| Paired Wilcoxon p = 0.929 vs. E66 | ✅ | ✅ | ✅ |
| Paired Wilcoxon p = 0.028 vs. FedPer | ✅ | ✅ | ✅ |
| Paired Wilcoxon p = 0.021 vs. no-calibration | ✅ | ✅ | ✅ |
| 5-seed mean MAE 0.653 ± 0.011 vs. 0.671 ± 0.010 (Table 2 estimand) | ✅ | ✅ | ✅ |
| Bootstrap 95% CI straddles zero | ✅ | ✅ | ✅ |
| "calibration under personalized heads, not gate weighting" framing | ✅ | ✅ | ✅ |
| No-equivalence-margin caveat | ✅ | ✅ | ✅ |
| Transferability matrix retained as deterministic/inspectable record | ✅ | ✅ | ✅ |

Keywords (`personalized federated learning; molecular reorganization energy; task-specific calibration; label-scale heterogeneity; quantum chemistry; organic semiconductors; TADF emitters; small-sample molecular learning`) match across all sources.

---

## 3. Introduction §1 contributions order check

Both `manuscript.md` lines 31–37 and `main.tex` lines 110–116 list contributions in this exact order:

1. ✅ **Task-specific calibration under personalized heads as the only ablation-detectable component.**
2. ✅ **Task-client modeling for heterogeneous cation/hole/triplet λ tasks.**
3. ✅ **A calibration-aware personalized federated baseline that reproduces PC² gains.**
4. ✅ **Deterministic, inspectable compatibility metadata for protocol/quantity constraints.**
5. ✅ **A reliable small-sample evaluation protocol.**

The §1 narrative paragraph ends with: "Section 4.4 reports a targeted control (E71) showing that, in our federation, the operative element is component (ii) under personalized heads; components (iii) and (iv) serve as auditability scaffolding rather than additional performance drivers". Verified in both `manuscript.md` line 28 and `main.tex` line 107.

---

## 4. §4.4 E71 contrasts check

| Required contrast | `manuscript.md` | `main.tex` |
|---|---|---|
| `E71 vs. PC²-FedReorg (E66)` paragraph | ✅ 1 hit (bullet list in §4.4) | ✅ 2 hits (LaTeX itemize + caption ref) |
| `E71 vs. FedPer (E63)` paragraph | ✅ 1 hit | ✅ 2 hits |
| `E71 vs. PC²-FedReorg without calibration (E70)` paragraph | ✅ 1 hit | ✅ 2 hits |

§4.4 title in both sources: **"Calibration under private heads explains the C-triplet gain"** (`manuscript.md` line 196; `main.tex` line 316, `\label{sec:ablation}`).

§4.4 bold summary statement appears in both: descriptive form "*Together, E71 and the four PC² ablations are consistent with task-specific calibration under personalized heads carrying the only ablation-detectable signal in our federation. E71 reaches a C-triplet MAE that is not statistically distinguishable from PC²-FedReorg's at our sample size; we did not pre-register an equivalence margin…*"

---

## 5. Limitations old-language purge

Forbidden phrases in Limitations:

| Phrase | Hit count across all files |
|---|---|
| "E71 deferred" | ✅ 0 |
| "highest-priority pre-submission addition" | ✅ 0 |
| "deferred work in §6" | ✅ 0 |

Replacement (Limitations item 7) confirmed verbatim in `manuscript.md` line 262 and `main.tex` line 387: *"Component-isolation ablation settled by E71; further isolation deferred. […] The E71 control settles this question in the present federation: FedPer + calibration is statistically indistinguishable from PC²-FedReorg on C-triplet (paired Wilcoxon p = 0.929)…"*

Two new Limitations items added (both present in `manuscript.md` and `main.tex`):
- Item 9: Paired-error independence assumption (LOOCV fold overlap ≈ 96–98%, seed aggregation does not eliminate seed-by-method interaction).
- Item 10: Estimand consistency between Table 2 (5-seed mean MAE) and Table 3 (per-molecule median-aggregated MAE) — differ by ≤ 0.003 eV.

The "deferred to follow-up work" wording still appears in legitimate non-E71 contexts (Limitations item 6 about *learnable affine calibration*, and SI §S10 about MOON/ChemProp baselines). Those are not the forbidden E71 phrases and are documented as remaining open extensions.

---

## 6. Figure 5 check

`paper/figures/Fig5_ablation_forest.pdf` regenerated 2026-05-18 14:57 by `paper/make_figures.py:fig5_ablation_forest` (rewritten as a 3-panel calibration-mechanism figure):

| Required element | Status |
|---|---|
| Panel (a) C-triplet + C-hole MAE bar plot for E63 / E70 / E66 / E71 | ✅ visible |
| Panel (b) E71 vs E66 forest row | ✅ visible (top of forest) |
| Panel (b) E71 vs E63 forest row | ✅ visible |
| Panel (b) E71 vs E70 forest row | ✅ visible |
| Panel (b) 4 PC² ablation rows (E70 / E69 / E68 / E65) | ✅ visible |
| Panel (c) interpretation box | ✅ visible with bullet "Gate is not required for the observed C-triplet gain" (in-figure text) |
| Caption title (figure_captions.md) | ✅ "Calibration under personalized heads explains the C-triplet gain" |
| Estimand-difference disclosure in caption | ✅ "Panel (a) shows the 5-seed mean of per-method MAE_mean (matching Table 2); paired comparisons in panel (b) use the per-molecule median-across-seeds absolute error vector required for paired Wilcoxon testing (matching Table 3)" |
| PDF Figure 5 embedded correctly in `main.pdf` | ✅ page 18 (Results section) |

---

## 7. Figure 6 → Figure S1 demotion

| Check | Status |
|---|---|
| `paper/figures/Fig6_*.{pdf,svg,png}` does not exist | ✅ (only FigS1_per_molecule_cases.{pdf,svg,png}) |
| `main.tex` includes no `Fig6_` image | ✅ 0 hits |
| `main.tex` no "Figure 6" float | ✅ removed |
| `SI.tex` line 456 includes `FigS1_per_molecule_cases.pdf` | ✅ |
| `manuscript.md` §4.5 references "Figure S1 in the Supporting Information" | ✅ |
| `figure_captions.md` "Figure S1" caption present | ✅ |

---

## 8. Tables 2 / 3 / 4 — E71 rows

| Table | CSV E71 hits | MD E71 hits |
|---|---|---|
| `table2_main_performance` | 4 (A 5-fold, B 5-fold, C-hole LOOCV, C-triplet LOOCV) | 4 |
| `table3_paired_tests` | 6 (E71 vs E66/E63/E70 × {C-hole, C-triplet}) | 6 |
| `table4_ablation` | 2 (C-hole + C-triplet positive control) | 2 |

LaTeX Tables 2, 3, 4 inside `main.tex` carry the same rows (verified by `pdftotext -layout main.pdf | grep "E71"` returning 6+ hits in the table-of-results section, page 23). The phrasing "E71 FedPer + calibration only" (Table 2/4) and "E71" reference column (Table 3) are consistent across CSV / MD / PDF.

---

## 9. Table 1 (task-client dataset summary) — A/B protocol wording

| Check | Status |
|---|---|
| A row functional/basis/charge/geometry = `— (same-source)` (not "unknown") | ✅ |
| B row functional/basis/charge/geometry = `— (same-source)` | ✅ |
| `protocol_note` column carries "Protocol note: same public cation-reorg source protocol. Head-sharing note: A↔B compatible by same-source shortcut (Section 2.4)." for both A and B | ✅ |
| Legitimate "unknown" entries remain only for C-hole / C-triplet `basis_in_csv` (genuinely missing in source CSV) and D `charge_state` / `geometry` (genuinely not stated in Atahan-Evrenk 2019 metadata) | ✅ |
| `paper/make_figures.py:table1_task_clients` writer carries the wording so future regeneration preserves it | ✅ |

---

## 10. Forbidden-phrase scan (full repo, case-insensitive)

All scans run across `paper/manuscript.md`, `paper/SI.md`, `paper/latex/main.tex`, `paper/latex/SI.tex`, `paper/figure_captions.md`:

| Phrase | Hits | Notes |
|---|---|---|
| `state-of-the-art` | ✅ 0 | |
| `state of the art` | ✅ 0 | |
| `gate drives performance` | ✅ 0 | |
| `gate is a performance` | ✅ 0 | |
| `PC² routing outperforms` | ✅ 0 | |
| `C-hole significantly improves` | ✅ 0 | |
| `C-hole improves` | ✅ 0 | |
| `D→C causes negative transfer` | ✅ 0 | |
| `D causes negative transfer` | ✅ 0 | |
| `produces negative transfer` | ✅ 2 (negated context — explicit disclaimer "**Not demonstrated by ablation evidence:** a claim that naive D → C supervised pretrain produces negative transfer (E65)"; these hits *implement* the required disclaimer rather than violating it) | |
| `E71 deferred` | ✅ 0 | |
| `highest-priority pre-submission addition` | ✅ 0 | |
| `highest-priority pre-submission` | ✅ 0 | |

---

## 11. PDF compile log review

### main.pdf — clean
- ✅ 26 pages, 1.0 MB
- ✅ 0 missing citations
- ✅ 0 undefined references
- ⚠️ 2 cosmetic Overfull `\hbox` warnings (lines 471–482, 525–548 of `main.tex`); both are inside table rows with long content and do not affect the visual layout in the produced PDF. Pre-existing. Not introduced by E71 work.

### SI.pdf — clean after cross-ref fix
- ✅ 16 pages, 0.58 MB
- ✅ 0 missing citations in SI bib pass
- ✅ 0 undefined references (the previously-pre-existing `tab:task-clients` warning has been resolved this round — see §16 below)
- ⚠️ 3 "h" → "ht" float-specifier adjustments — automatic LaTeX behavior; cosmetic.

---

## 12. §S7 numbering offset between SI.md and SI.tex (pre-existing, post-S7.7 sync)

After the audit-time addition of §S7.7 to `SI.tex` to mirror `SI.md`, the section numbering between the two sources differs by 1 because `SI.md` has a §S7.1 "Ablation matrix" intro that was not transferred to `SI.tex` when the LaTeX version of §S7 was originally written:

| Topic | `SI.md` | `SI.tex` autonumbered | PDF page |
|---|---|---|---|
| Ablation matrix overview | §S7.1 | (not present) | — |
| E70 — no calibration | §S7.2 | §S7.1 | p. 11 |
| E68 — no C-cross | §S7.3 | §S7.2 | p. 11 |
| E69 — uniform gate | §S7.4 | §S7.3 | p. 12 |
| E65 — D→C pretrain control | §S7.5 | §S7.4 | p. 12 |
| Synthesis | §S7.6 | §S7.5 | p. 12 |
| E71 — FedPer + calibration only | §S7.7 | §S7.6 | p. 13–14 |

The conceptual content is identical and the cross-references within the LaTeX (`\ref{si:abl-e70}`, `\ref{si:abl-e65}`, `\ref{si:abl-e71}`, `\ref{si:v2}`) resolve correctly. This is the only structural mismatch and is documented here so the author can decide whether to add a `§S7.1 Ablation matrix` intro to `SI.tex` (cosmetic) or to remove it from `SI.md` (also cosmetic). Either resolution is acceptable.

---

## 13. Single sync change made during the audit (the only edit to scientific source this round)

`paper/latex/SI.tex` was missing the §S7.7 ("E71 — FedPer with calibration only (positive control)") section that had been added to `paper/SI.md` during the E71 rewrite. The audit detected this PDF-vs-SI inconsistency, ported the section from `SI.md` to `SI.tex` (inserting two tables for the across-seed metrics and the C-triplet / C-hole paired comparisons in LaTeX), and updated the §S7.6 synthesis bullet list to reference the new §S7.6 (LaTeX-numbered) positive-control subsection. SI was then recompiled (3 passes + BibTeX) and confirmed to contain the new content. No scientific conclusions were modified during this synchronization — the content was transcribed verbatim from the post-MCP-review `SI.md`.

---

## 14. Remaining blockers (not for this audit to clear; consistent with `paper/submission_checklist.md` §0)

| # | Blocker | Owner | Acceptance check |
|---|---|---|---|
| **B2** | Author metadata / affiliations / corresponding-author email | Lead author + corresponding author | `grep -n "TODO" paper/latex/main.tex paper/latex/SI.tex` returns zero hits in author/affiliation blocks |
| **B3** | 5 unverified DOIs in `references.bib` (`Sasabe2011_OLED`, `AtahanEvrenk2019_RE`, `Nelsen1987`, `EfficientKAN2024`, `RDKit2023`) | Lead author | `grep -n "TODO: verify DOI" paper/latex/references.bib` returns zero hits |
| **B4** | Data and Code Availability URL not assigned | Lead author + institution | `\TODO{Repository URL}` placeholders replaced in `main.tex` and `SI.tex` |

One pre-existing structural note (not a blocker, can ship as-is):
- The SI §S7 numbering between MD and TeX is offset by 1, as documented in §12 above.

The earlier-flagged SI cross-document reference issue (`tab:task-clients`) was resolved this round — see §15 below.

---

## 15. SI cross-document reference fix (2026-05-18, follow-up to §11 SI.pdf warnings)


**Problem.** The SI references the main-text Table 1 via `\ref{tab:task-clients}` on `SI.tex` line 121 (`the per-field entries in main-text Table~\ref{tab:task-clients}`). `SI.tex` did not load any cross-document-reference package, so the citation rendered as `main-text Table~??` in the compiled SI.pdf and produced a `LaTeX Warning: Reference 'tab:task-clients' on page 2 undefined` message during every SI compile.

**What was tried and what failed.**
1. **`\usepackage{xr-hyper}\externaldocument{main}` (preferred clickable cross-refs):** failed. `xr-hyper` parses `main.aux` to resolve labels, but `main.aux` contains `\citation{...}` and `\bibcite{...}` entries written by `natbib`. The brace structure of these entries breaks xr-hyper's tokenizer, producing 5+ `! Argument of \@citex has an extra }.` errors per pdflatex pass and eventually a fatal `no output PDF` failure on the third pass.
2. **`\usepackage{xr}\externaldocument{main}` (non-clickable plain cross-ref):** same root cause, same `\@citex` errors, same fatal failure. The plain `xr` package shares xr-hyper's `main.aux` parser.

**What worked (final fix).** Replaced the single `\ref{tab:task-clients}` callout with the hardcoded literal `Table~1`, which is the stable position of that table in `main.tex` (it is the only `\begin{table*}` in §2.1 and is unambiguously the first table number). Added a documentation comment in `SI.tex` (lines 46–51) explaining the natbib/xr conflict and noting that the cross-reference is hardcoded as plain text rather than imported, so that any future author who tries to re-introduce `xr-hyper` is warned upfront. Only one cross-doc `\ref` existed in `SI.tex` (all other `\ref{si:*}` callouts are intra-SI labels), so the hardcoding affects exactly one line of the SI body.

**Verification.** After the patch, the SI compile is clean:
- `pdflatex` 3-pass + `bibtex` produce SI.pdf 16 pages, 0.58 MB, exit code 0.
- `grep '^! ' SI.log` → empty (no errors).
- `grep 'LaTeX Warning.*[Uu]ndefined' SI.log` → empty (no undefined refs).
- `pdftotext SI.pdf | grep 'main-text Table'` confirms the body now reads "the per-field entries in main-text Table 1 are accordingly shown as '— (same-source)'" instead of "main-text Table ??".

**Trade-off accepted.** The hardcoded "Table 1" is not a clickable hyperlink in the PDF, whereas xr-hyper would have made it clickable. For a single cross-doc callout this is an acceptable trade-off; if more cross-doc refs are added later, the cleaner solution is to upgrade natbib to a version that writes xr-compatible aux entries, or to switch the SI bibliography to BibLaTeX (which avoids the natbib aux conflict entirely).

**Scope confirmation.** This fix is a presentation-layer correction (cross-reference rendering), not a scientific-content change. No claim was added, removed, or modified — the SI text says exactly the same thing it said before, just with the table number rendered as `1` instead of `??`.

---

## 16. What did NOT change this round

- ❌ No edits to `src/`, `experiments/E60-E70`, or pre-existing `results/` files (audit was read-only of scientific content; the only edit was the SI.tex §S7.6 sync, which is text transcription not a new conclusion).
- ❌ No retraining of any experiment.
- ❌ No `.docx` regeneration.
- ❌ No `git commit`.
- ❌ No changes to forbidden-claim wording (the rewrite from prior rounds already satisfies all 10 forbidden phrases).
