# PC²-FedReorg — Review Package for Supervisor / Collaborators

**Document purpose:** a single-stop briefing for supervisors and co-authors to scan the paper draft state, the supported claims, the explicitly forbidden claims, the outstanding TODOs, and the recommended target venues — without having to read the full manuscript and SI end-to-end.

**What this document is not:** it is not a substitute for reading `paper/manuscript.md` and `paper/SI.md`.  It is a navigation and risk-management layer on top of them.

---

## 1. Current draft state

### Files under `paper/`

| File | Size | Purpose | Status |
|---|---|---|---|
| `manuscript.md` | 37 KB / 310 lines | Main text, 7 numbered sections + abstract + references list | Draft ready for supervisor review |
| `SI.md` | 40 KB / 556 lines | Supporting Information, S1–S10 | Draft ready; S2.2 populated from real V2 data |
| `SI_outline.md` | 20 KB | Earlier outline kept as version history | Frozen reference only |
| `figure_captions.md` | 12 KB | Submission-style captions for Fig 1–6 and Table 1–4 | Draft ready |
| `references.bib` | 12 KB | Cited entries [1]–[18] (17 unique keys) | 5 entries have `note = {TODO}`; see §5 below |
| `references_unused.bib` | 7 KB | Uncited entries archived verbatim | Archive only |
| `submission_checklist.md` | 16 KB | Claim-to-evidence map, overclaim sweep, reviewer-risk list | Updated; binary checklist tracks remaining tasks |
| `reference_todo_audit.md` | 16 KB | Per-entry audit of the five `note = {TODO}` references | Read-only; no patches applied |

### Figures and tables

All six main-text figures exist in three formats (PDF, SVG, PNG@600dpi) under `paper/figures/`; all four main-text tables exist in CSV + Markdown under `paper/tables/`.  No regeneration required.  Font: Liberation Serif (TNR metric-compatible; substitution authorised on 2026-05-11 because Times New Roman is not installed on either compute node).

### Code, results, and tests not touched

The PC²-FedReorg algorithm code (`src/`), the experiment registry (`experiments/`), the result artefacts (`results/`), and the 93 unit tests are unchanged from the freeze recorded in `logs/pc2_batch1/FROZEN_INFO.txt`.

---

## 2. Core innovations — three sentences

1. **Task-client modelling.** PC²-FedReorg separates *physical clients* from *task-clients*, treating distinct reorganization-energy quantities (hole-λ vs. triplet-λ) measured at the same laboratory as distinct federation nodes throughout aggregation, regression heads, and label calibration.

2. **Task-specific calibration with universal aggregation exclusion.** Each task-client carries a private, fold-isolated `(μ, σ)` calibration buffer computed strictly from its own training fold; training loss is computed in z-space and metrics in eV-space, with calibration buffers excluded from every federated aggregation operator by construction.

3. **Layer-wise personalization with auditable compatibility metadata.** Encoder and an optional residual adapter are shared under a per-key routing scheme driven by a deterministic chemistry/protocol transferability matrix, while heads and calibration remain fully private; the transferability matrix is positioned as *algorithmic transparency* rather than as a performance lever.

---

## 3. Strongest empirical evidence

### 3.1 PC²-FedReorg significantly outperforms FedAvg/FedProx/FedPer on C-triplet

Source: `results/pc2_phase5_stats.csv`, rows with `target = loocv_c_triplet` and `reference = E66`.

| Comparator | MAE PC² (eV) | MAE other (eV) | ΔMAE (eV) | 95 % CI | Wilcoxon p | Wins / loses |
|---|---|---|---|---|---|---|
| FedAvg | 0.660 | 0.763 | −0.104 | (−0.220, +0.004) | **0.045** | 31 / 18 |
| FedProx | 0.660 | 0.784 | −0.124 | (−0.254, −0.007) | **0.045** | 30 / 19 |
| FedPer | 0.660 | 0.762 | −0.103 | (−0.205, −0.002) | **0.042** | 32 / 17 |

Relative reduction approximately **13–16 %**; all three comparisons cross the 0.05 significance threshold; per-molecule wins ≥ 30 out of 49 in each case.

Manuscript locations: §4.3 main results; Table 3; Figure 4 right panel.

### 3.2 No-calibration ablation produces significant degradation on C-triplet

Source: same CSV, row with `target = loocv_c_triplet` and `compared_to = E70`.

| Quantity | Value |
|---|---|
| MAE PC² | 0.660 eV |
| MAE no-calibration variant | 0.779 eV |
| ΔMAE | −0.119 eV |
| Bootstrap 95 % CI | (−0.225, −0.015) |
| Wilcoxon p | **0.043** |
| Per-molecule wins (PC² better / worse) | 35 / 14 |

The CI sits entirely below zero.  Task-specific calibration is the **only** component-level ablation that produces a statistically significant change on either target, and the effect is restricted to C-triplet.

Manuscript locations: §4.4 first paragraph; Discussion §5(i); Figure 5 first row; SI §S6.3, §S7.2.

### 3.3 C-hole non-significance — reported honestly

Source: `results/pc2_phase5_stats.csv`, rows with `target = loocv_c_hole`.

| Property | Value |
|---|---|
| MAE band across all 9 evaluated methods (E60–E70) | 0.368–0.414 eV |
| Lowest paired Wilcoxon p observed | 0.198 (PC² vs. FedPer) |
| All 8 paired comparisons | p ≥ 0.20 after rounding |
| Bootstrap CIs | every CI straddles zero |
| Best classical baseline R² (SI §S2.2) | −0.101 (KNN) |

We explicitly do not claim a hole-task improvement; we report that all methods sit at a small-sample data ceiling that is independently visible from the classical baselines.

Manuscript locations: §4.3 second paragraph; Figure 4 left panel; SI §S2.2 (V2 ceiling table) and §S6.2.

---

## 4. Claims we are *not* making (forbidden phrasing list)

The manuscript and SI have been audited multiple times to ensure none of the following statements appear, except where explicitly *denied*.

| Forbidden statement | Why it is not supported by our data |
|---|---|
| "PC²-FedReorg improves all reorganization-energy tasks." | False — C-hole is not significantly improved; the manuscript explicitly states the C-hole non-significance and the data-ceiling diagnosis. |
| "PC²-FedReorg achieves comprehensive improvement / outperforms all baselines." | Same as above. |
| "The chemistry/protocol gate drives PC²'s performance gain." | False — ablations E68 (no C-cross encoder sharing) and E69 (uniform FedAvg weighting) leave performance statistically unchanged (p = 0.07 and p = 0.18 respectively).  The chemistry gate is positioned as auditable metadata, not a performance lever. |
| "The chemistry/protocol gate is the active ingredient." | False — see above.  Calibration is the only ablation-supported active ingredient (E70, p = 0.043). |
| "Naive D → C supervised transfer produces negative transfer." | Not demonstrated — the E65 D→C-pretrain control is not statistically distinguishable from PC² on either C target (p = 0.85, p = 0.76).  The V1 protocol divergence (KS p ≈ 10⁻¹¹⁶) motivates the conservative T_head clause, but the empirical comparison does not establish a negative-transfer claim. |
| "PC² is state-of-the-art on reorganization-energy prediction." | Not claimed; the manuscript does not contain the phrase "state-of-the-art". |
| "PC² uses SchNet for the encoder." | False — all reported experiments use the 5-layer GIN encoder; SchNet is documented only as a codebase capability not exercised in this study. |
| "C-hole is significantly improved." | False — see §3.3 above. |

Reviewers may raise some of these as questions — Section 7 of `paper/submission_checklist.md` lists the prepared responses.

---

## 5. Items requiring supervisor / co-author confirmation before submission

### 5.1 Author list and affiliations

`paper/manuscript.md` line 3 currently reads:
```
*[Author list, affiliations, and corresponding-author email — to be supplied]*
```

Action required: lead author to supply the ordered author list, affiliations, ORCID identifiers (if available), and corresponding-author email.

### 5.2 Acknowledgements

`paper/manuscript.md` Section "Acknowledgements" currently reads:
```
*[Compute resources, funding sources, data-providing laboratories — to be supplied.]*
```

Action required: lead author to supply funding-source grant numbers, compute-cluster acknowledgements, and acknowledgement of the data-providing laboratories.

### 5.3 Data and Code Availability

`paper/manuscript.md` Section "Data and Code Availability" currently reads:
```
Source code, ... are released at `[repository URL — to be supplied]`. The in-house
C-hole / C-triplet labels are made available upon request and within the constraints
of the originating laboratory's data-sharing agreement; public Clients A, B and D
use third-party datasets and are linked in `data/README.md`.
```

Action required:
1. Supply the repository URL (GitHub, Zenodo, or institutional repository).
2. Have the in-house laboratory PI confirm the data-availability statement.
3. Add explicit data-source links for A (QM9 split), B (QM9 split), D (Atahan-Evrenk 2019).

### 5.4 Five references with `note = {TODO}`

Full per-entry audit is in `paper/reference_todo_audit.md`.  Summary:

| # | Bib key | Action | Estimated time |
|---|---|---|---|
| [3] | Sasabe2011_OLED | Confirm DOI `10.1021/cm1024309` via ACS *Chem. Mater.* page | ~2 min |
| [5]/[12] | AtahanEvrenk2019_RE | Resolve DOI conflict (`9b02733` vs `9b00873`) via ACS *J. Phys. Chem. A* page; highest priority because the entry covers both [5] methodology and [12] dataset | ~5 min |
| [14] | Nelsen1987 | Confirm both the DOI and the attribution of "four-point Nelsen method" to this 1987 paper vs. a later (1996) Nelsen paper — chemistry-side judgment required | requires expert input |
| [17] | EfficientKAN2024 | Confirm whether a Zenodo DOI exists; otherwise use the recommended software-citation form documented in `reference_todo_audit.md` (installed version verified locally as 0.1.0) | ~2 min |
| [18] | RDKit2023 | Bib key year mismatches installed version; resolve to `RDKit2024` (installed version 2024.09.6) and pin the Zenodo DOI for `Release_2024_09_6` | ~5 min |

### 5.5 Figure 6 chemistry naming

Figure 6 (per-molecule error analysis, C-triplet) currently uses the low-risk phrasing "*a halogenated polycyclic conjugated system*" for the top-improved molecule (SMILES `ClC1=C(Cl)C(=C2C(c3ccccc3)=C2c2ccccc2)C(Cl)=C1Cl`, true λ_T = 1.13 eV, |error| 3.06 → 1.70 eV).

The actual structural class is a *1,2,3,4-tetrachlorocyclopenta-2,4-dien-5-ylidene linked through an exocyclic C=C to a diphenylcyclopropene* (a tetrachloro-pentafulvene / diphenyl-cyclopropene hybrid).

Action requested: chemistry-side co-author to decide whether to:

- (a) keep the low-risk phrasing as is, or
- (b) replace with a more specific structural-class name (e.g. "diphenyl-cyclopropylidene–tetrachloropentafulvene hybrid").

Either choice is acceptable; option (a) is safer for reviewer optics, option (b) demonstrates chemistry-domain awareness.

---

## 6. Recommended target venues

Ranked from best-fit to acceptable, based on the manuscript's content profile (federated learning + cheminformatics + small-sample reorganization-energy prediction + emphasis on auditable methodology).

| Venue | Fit | Reasoning |
|---|---|---|
| **Journal of Chemical Information and Modeling (JCIM)** | **Strong fit** | ACS journal; explicitly accepts federated-learning + cheminformatics work; recent precedent for FL papers on molecular property prediction; readership includes both ML-method and chemistry-domain reviewers. |
| **Journal of Cheminformatics (BMC / Springer)** | Strong fit | Open-access; cheminformatics + ML focus; software-citation–friendly policy; receptive to methodological federation papers with full code release. |
| **Digital Discovery (RSC)** | Strong fit | RSC's newer flagship for ML-driven chemistry; explicitly emphasises reproducibility and method transparency, which matches our auditable-transferability-matrix angle; small-sample paired statistics are well-received. |
| **Journal of Chemical Theory and Computation (JCTC)** | Medium fit | If the manuscript emphasises the *DFT-protocol heterogeneity* and the *reorganization-energy theory* side more strongly, JCTC is a credible target.  Likely needs slightly heavier theoretical framing (Marcus theory, Nelsen four-point convention). |
| **npj Computational Materials** | Medium fit | If the framing is broadened toward general organic-semiconductor design and the federation is presented as a materials-discovery enabler. Editorial bar is high; would benefit from MOON / ChemProp baselines added (currently deferred). |
| **J. Phys. Chem. A** | Medium fit | The reorganization-energy domain matches; less ideal for federated-learning methodology framing. |
| **Patterns (Cell Press)** | Lower fit | Has hosted FedChem-family papers, but generally prefers broader impact narratives. |

Recommendation: **submit to JCIM first**.  Backup target: **Digital Discovery** (especially if a reviewer at JCIM asks for additional baselines such as MOON / ChemProp — Digital Discovery's faster-turnaround editorial process is helpful for a revision-heavy paper).

---

## 7. What is *not* in this review package

- No docx export (out of scope; awaiting explicit instruction).
- No cover letter (out of scope; awaiting explicit instruction).
- No git commit (out of scope; never performed without explicit instruction).
- No changes to `manuscript.md`, `SI.md`, `references.bib`, `figure_captions.md`, `submission_checklist.md`, or any figure / table.
- No new experiments, no training, no re-rendering of figures.

The current package is a *read-and-decide* artefact for supervisor review.  Patches to the manuscript and references should be issued in a follow-up turn after the items in §5 above are resolved.

---

*End of review package.*
