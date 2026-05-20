# PC²-FedReorg — Pre-Submission Checklist

> Intended for the lead author and supervisors to run through before any external submission.  Each entry is either ✅ (done in this writing pass) or ⚠️ (still to be completed by the human authors).  Where a TODO depends on chemistry-side knowledge, the responsible person is named.

---

## 0. Pre-submission status banner

**STATUS as of 2026-05-18: NOT submission-ready (three blockers remain; one cleared this round).**

The manuscript can be submitted to a SCI journal once the three remaining blockers below are cleared. The previously top-priority blocker B1 (E71 control experiment) is now resolved: the control was executed, the manuscript was restructured around the resulting mechanism finding (Outcome A: calibration explains the C-triplet gain), and the title was updated to *Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction*.

| # | Blocker | Why it blocks | Acceptance check | Owner | Evidence pointer when cleared |
|---|---|---|---|---|---|
| ~~B1~~ | ~~E71 control experiment (FedPer + calibration-only) not yet run.~~ | ~~The §4.4 calibration claim rests on E70 (no-calibration → degradation). The complementary control — yes-calibration, no-rest-of-PC² — is missing.~~ | ✅ **CLEARED 2026-05-17/18.** Outcome A: `results/e71_summary.csv` (5 seeds), `results/e71_predictions.json`, `results/e71_vs_e66_stats.{csv,md}`, `results/e71_vs_e63_stats.md`, `results/e71_vs_e70_stats.md` all written. C-triplet: E71 vs E66 MAE = 0.654 vs 0.660 eV, paired Wilcoxon p = 0.929 (n.s.); E71 vs E63 nominal p = 0.028; E71 vs E70 nominal p = 0.021. Abstract, §1 contributions, §4.3/§4.4, Discussion (6 points), Limitations item 7, Conclusions, Figure 5 (redrawn as calibration-mechanism figure with 3 panels), Tables 2/3/4, SI §S7.7 all updated. | — | `results/e71_*`; `paper/figures/Fig5_ablation_forest.{pdf,svg,png}`; `paper/SI.md` §S7.7 |
| **B2** | **Author metadata / affiliations / corresponding-author email not filled.** | LaTeX `\TODO{...}` placeholders remain in `paper/latex/main.tex` and `paper/latex/SI.tex` author blocks. | (i) `grep -n "TODO" paper/latex/main.tex paper/latex/SI.tex` returns zero hits in author/affiliation blocks; (ii) `references.bib` has zero entries with `note = {TODO: verify DOI ...}` *or* every such entry has been confirmed by the corresponding author with a real DOI. | Lead author + corresponding author | `paper/latex/PC2-FedReorg-overleaf.zip:UPLOAD.md` §"Remaining `\TODO{...}` placeholders preserved in the LaTeX" |
| **B3** | **5 references have unverified DOIs** (`Sasabe2011_OLED`, `AtahanEvrenk2019_RE`, `Nelsen1987`, `EfficientKAN2024`, `RDKit2023`). | JCIM / Digital Discovery require verifiable DOIs for every cited reference. | (i) `grep -n "TODO: verify DOI" paper/latex/references.bib` returns zero hits; (ii) each entry replaced with the canonical DOI verified against Crossref. | Lead author | `paper/latex/references.bib`; `paper/reference_todo_audit.md` |
| **B4** | **Data and Code Availability URL not assigned.** | `paper/latex/main.tex` and `paper/latex/SI.tex` Data and Code Availability sections contain `\TODO{Repository URL}`. | (i) Public-facing repository URL chosen (Zenodo, GitHub, or institutional); (ii) `\TODO` placeholders replaced with the canonical URL in both `main.tex` and `SI.tex`. | Lead author + institution | `paper/latex/main.tex` (Data and Code Availability section); the new repository |

**Cleared in the 2026-05-17/18 round (E71 mechanism rewrite):**

- ✅ **E71 = FedPer + calibration-only executed (5 seeds, dual-A30, ~2 h 27 min).** Outcome A on C-triplet (E71 ≈ E66; calibration explains the gain). Raw outputs in `results/e71_*`; statistical pipeline at `experiments/run_e71_multiseed.py` and `experiments/compute_e71_stats.py`; registration at `experiments/run_all.py` (E71 entry, additive only).
- ✅ **Title updated** to *Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction* in `paper/manuscript.md`, `paper/SI.md`, `paper/latex/main.tex`, `paper/latex/SI.tex`.
- ✅ **Abstract rewritten** with the E71 reproduction result + final sentence positioning calibration (not gate weighting) as the empirical source.
- ✅ **§1 contributions reordered** to (1) calibration-mechanism, (2) task-client modeling, (3) calibration-aware FL baseline, (4) auditable compatibility metadata, (5) evaluation protocol; PC² routing demoted to auditability scaffolding.
- ✅ **§4.3 + §4.4 rewritten.** §4.3 gained an E71-pointer paragraph; §4.4 retitled to "Calibration under private heads explains the C-triplet gain" and now leads with three paired E71 contrasts (vs E66, E63, E70) before the four PC² ablations.
- ✅ **Discussion rewritten as six points** answering: (i) calibration is the active component; (ii) why the mechanism is more than ordinary normalization (six properties); (iii) gate is auditability not lever; (iv) E71 changes the interpretation; (v) C-hole at data ceiling; (vi) p-values nominal, not over-claimed.
- ✅ **Limitations item 7 updated** from "E71 deferred" to "E71 settles this question; remaining adapter-isolation deferred". Other items (MOON/ChemProp, seed budget, Times New Roman, learnable cal) preserved.
- ✅ **Figure 5 redrawn as 3-panel calibration-mechanism figure**: (a) C-triplet/C-hole MAE bar plot for E63/E70/E66/E71; (b) forest plot of three E71 contrasts plus four PC² ablations; (c) interpretation box. New title "Calibration under personalized heads explains the C-triplet gain". `paper/figures/Fig5_ablation_forest.{pdf,svg,png}` at 600 dpi.
- ✅ **Tables 2/3/4 extended with E71 rows**; `paper/tables/table{2,3,4}_*.{csv,md}` carry the E71 entries.
- ✅ **SI §S7.7 added**: full E71 spec (hyperparameters, wall-clock, released files), across-seed metrics, paired comparisons on both targets, reproducibility command. SI §S7.6 synthesis updated. §S8.1 lists `results/e71_predictions.json`.
- ✅ **Figure 5 caption rewritten** in `paper/figure_captions.md` to describe the new 3-panel layout. Figure 4 caption gained an E71 cross-pointer.
- ✅ **Keywords updated** to: personalized federated learning; molecular reorganization energy; task-specific calibration; label-scale heterogeneity; quantum chemistry; organic semiconductors; TADF emitters; small-sample molecular learning.

**Explicitly NOT changed in this round (per user constraint):**
- ❌ No `src/`, `experiments/E60–E70`, or pre-existing `results/` files modified (only additive: new `results/e71_*`, new `experiments/run_e71_multiseed.py`, new `experiments/compute_e71_stats.py`, additive E71 entry in `experiments/run_all.py`).
- ❌ No retraining of any other experiment.
- ❌ No `.docx` files regenerated.
- ❌ No git commits.

**Explicitly NOT changed in this round (per user constraint):**
- ❌ No `src/`, `experiments/`, or `results/` modifications.
- ❌ No training runs initiated.
- ❌ No `.docx` files generated.
- ❌ No git commits.
- ❌ No existing experimental numbers changed.

---

## 1. Claim-to-evidence map

Every claim in the manuscript must be traceable to a Figure, Table, or released result file.

| Manuscript claim | Section | Supporting evidence | Status |
|---|---|---|---|
| Five task-client federation; per-node n and label statistics | §2.1, §3.1, §4.1 | Figure 1, Table 1; `results/preverify/T_transferability.json` `n_kept` / `label_means` / `label_stds` | ✅ |
| Task-client abstraction motivated by quantity, protocol, label-scale heterogeneity | §1, §2.1, §4.1 | Figure 1; Table 1; `protocol_meta` in T_transferability.json | ✅ |
| Transferability matrices are auditable and deterministic | §2.4, §4.2 | Figure 3; `results/preverify/T_transferability.json` | ✅ |
| T_head[D, C-hole] = T_head[D, C-triplet] = T_head[C-hole, C-triplet] = T_head[C-triplet, C-hole] = 0 | §2.4, §4.2 | Figure 3 highlighted cells; T_transferability.json; tests/test_transferability.py | ✅ |
| PC²-FedReorg beats FedAvg on C-triplet (p = 0.045) | §4.3 | Figure 4 right; Table 3; `results/pc2_phase5_stats.csv` row (loocv_c_triplet, E66, E61) | ✅ |
| PC²-FedReorg beats FedProx on C-triplet (p = 0.045) | §4.3 | same as above (E62) | ✅ |
| PC²-FedReorg beats FedPer on C-triplet (p = 0.042) | §4.3 | same as above (E63) | ✅ |
| C-hole all paired tests p ≥ 0.20 | §4.3 | Table 3; `results/pc2_phase5_stats.csv` rows with target = loocv_c_hole | ✅ |
| Calibration ablation E70 produces significant degradation on C-triplet (p = 0.043) | §4.4, §5(i) | Figure 5; Table 3 row (loocv_c_triplet, E66, E70) | ✅ |
| No-C-cross ablation E68 not significant | §4.4, §5(ii) | Figure 5; Table 3 row (loocv_c_triplet, E66, E68); p = 0.070 | ✅ |
| Uniform-gate ablation E69 not significant | §4.4, §5(ii) | Figure 5; Table 3 (loocv_c_triplet, E66, E69); p = 0.181 | ✅ |
| D → C pretrain control E65 not significant vs. PC² | §4.4, §5(iii) | Figure 5; Table 3 (loocv_c_triplet, E66, E65); p = 0.851 | ✅ |
| Best classical baseline R² = −0.10 on C-hole | §2.5, §4.3 | SI Section S2.2; `results/preverify/V2_classical_baselines.csv`; `results/preverify/V2_summary.md` | ✅ Filled in SI S2.2 from real V2 files on 2026-05-11. Best classical R² = −0.101 (KNN) on C-hole and +0.084 (RF) on C-triplet, both as means across the same 5 seeds used by the federated runs. No V2 re-run required. |
| All 93 unit tests pass on the freeze commit | §2.3, S3.7, S9.6 | `logs/pc2_batch1/FROZEN_INFO.txt`; `tests/` directory | ⚠️ Lead author to confirm the commit hash recorded in FROZEN_INFO.txt matches the current `main` HEAD before submission. |

---

## 2. Overclaiming scan (final)

Grep for forbidden phrases across `paper/manuscript.md` and `paper/SI.md`:

| Phrase | Acceptable context | Status |
|---|---|---|
| "improves all tasks" / "outperforms all" / "outperforms across" | only in defensive denial | ✅ not present |
| "comprehensive improvement" / "全面提升" | never | ✅ not present |
| "state-of-the-art" | never used as a self-claim | ✅ not present |
| "chemistry gate drives performance" / "gate is the active ingredient" | only in defensive denial | ✅ never used as positive claim |
| "negative transfer" demonstrated | only in defensive denial | ✅ defensive-only |
| "SchNet results" (in main results) | only as codebase-support statement | ✅ codebase-only mention |
| "C-hole significantly improves" / "hole-task improvement" | never claimed positively | ✅ not present |

Re-run command before submission:

```bash
for p in "improves all" "全面提升" "drives performance" "state-of-the-art" "gate is the active" \
         "significantly improves the hole" "comprehensive improvement" "outperforms all" "outperforms across"; do
  echo "=== $p ==="; grep -ni "$p" paper/manuscript.md paper/SI.md || echo "(not found)"
done
```

---

## 3. Numerical consistency check

Re-verify before submission that every number in the manuscript matches the corresponding result file.

| Claim in manuscript | Source file | Status |
|---|---|---|
| n = 6,020 / 9,190 / 53 / 49 / 5,876 | `results/preverify/T_transferability.json:n_kept` | ✅ |
| Label means 0.821 / 0.658 / 1.215 / 2.638 / 0.258 | same JSON `label_means` | ✅ |
| C-triplet MAE: PC² 0.660; baselines 0.762–0.784 | `results/pc2_phase5_stats.csv` `mae_ref` / `mae_other` | ✅ |
| C-triplet relative improvement 13–16 % | computed from the above MAE values | ✅ |
| C-hole MAE band 0.368–0.414 | `results/pc2_phase5_stats.csv` C-hole rows | ✅ |
| Paired Wilcoxon p-values for C-triplet vs. {E61, E62, E63}: 0.045 / 0.045 / 0.042 | `results/pc2_phase5_stats.csv` `wilcoxon_p` | ✅ |
| E70 ΔMAE = −0.119 eV; p = 0.043; 35 / 14 wins | same | ✅ |
| Mean ratio C-triplet / C-hole = 2.17 | 2.638 / 1.215 = 2.171 | ✅ |
| T_repr[D, C-\*] ≈ 0.05 | T_transferability.json | ✅ |
| T_repr[C-hole, C-triplet] = 0.98–1.00 (asymmetric) | T_transferability.json | ✅ |
| T_head[A, B] = 0.33 | T_transferability.json | ✅ |
| GPU-hours ≈ 20 | logs/pc2_multiseed runtime sum | ✅ |

If any of the result files are regenerated, re-run the audit script described in §F of the audit report.

---

## 4. Figure and table availability

| Asset | File | Size | Status |
|---|---|---|---|
| Figure 1 | `paper/figures/Fig1_task_client_molecules.{pdf,svg,png}` | 359 K / 3.4 M / 1.5 M | ✅ |
| Figure 2 | `paper/figures/Fig2_overall_pc2_federated_architecture.{pdf,svg,png}` | 174 K / 816 K / 865 K | ✅ |
| Figure 3 | `paper/figures/Fig3_transferability_matrices.{pdf,svg,png}` | 23 K / 55 K / 527 K | ✅ |
| Figure 4 | `paper/figures/Fig4_main_performance.{pdf,svg,png}` | 17 K / 23 K / 465 K | ✅ |
| Figure 5 | `paper/figures/Fig5_ablation_forest.{pdf,svg,png}` | 24 K / 15 K / 493 K | ✅ |
| Figure 6 | `paper/figures/Fig6_molecule_error_cases.{pdf,svg,png}` | 50 K / 62 K / 480 K | ✅ |
| Table 1 | `paper/tables/table1_task_clients.{csv,md}` | 670 B / 925 B | ✅ |
| Table 2 | `paper/tables/table2_main_performance.{csv,md}` | 1.3 K / 1.8 K | ✅ |
| Table 3 | `paper/tables/table3_paired_tests.{csv,md}` | 1.4 K / 2.0 K | ✅ |
| Table 4 | `paper/tables/table4_ablation.{csv,md}` | 613 B / 815 B | ✅ |

Optional: SI figures (S-R²-distributions, S-gate-dynamics) are flagged in `paper/SI.md` as future additions; they are not required for the present submission.

---

## 5. Font status

- ⚠️ **Times New Roman is not installed** on either of this project's compute nodes (verified by `fc-match` and `font_manager.fontManager.ttflist` on both the GPU box and the CPU box on 2026-05-11).
- ✅ Current draft figures were rendered with Liberation Serif (TNR metric-compatible) under explicit user authorisation; the hard-stop fall-back guard in `paper/make_figures.py` was satisfied.
- ⚠️ Before camera-ready PDF submission, the figure-generation script must be re-run on a machine that has Times New Roman registered with matplotlib's font manager. The script's font setup line (`mpl.rcParams["font.family"] = "serif"`) should be changed to `["Times New Roman"]` and the hard-stop reactivated.
- Alternative: many SCI journals accept Liberation Serif or Nimbus Roman because they are metric-compatible substitutes for Times New Roman. Confirm the target journal's font policy before re-rendering.

---

## 6. References sanity check

`paper/references.bib` now contains only entries cited inline in `paper/manuscript.md` (numeric refs [1]–[18]).  All other entries from earlier drafts have been moved verbatim to `paper/references_unused.bib` to preserve the bibliography record without inflating the active reference list.

**Active bib (`paper/references.bib`)** — 17 unique entries covering 18 numeric citations (entry [5] = [12] = Atahan-Evrenk 2019, used for both the methodology and the Client D dataset role):

| Numeric ref | Key | DOI / arXiv | Status |
|---|---|---|---|
| [1] | Marcus1993 | 10.1103/RevModPhys.65.599 | ✅ verified |
| [2] | Coropceanu2007 | 10.1021/cr050140x | ✅ verified |
| [3] | Sasabe2011_OLED | 10.1021/cm1024309 | ⚠️ TODO — confirm DOI against publisher metadata |
| [4] | Uoyama2012_TADF | 10.1038/nature11687 | ✅ verified |
| [5], [12] | AtahanEvrenk2019_RE | 10.1021/acs.jpca.9b02733 | ⚠️ TODO — confirm DOI against publisher metadata |
| [6] | Schutt2017_SchNet_NeurIPS | arXiv:1706.08566 | ✅ verified arXiv id |
| [7] | McMahan2017_FedAvg | arXiv:1602.05629 | ✅ verified arXiv id |
| [8] | Li2020_FedProx | arXiv:1812.06127 | ✅ verified arXiv id |
| [9] | Arivazhagan2019_FedPer | arXiv:1912.00818 | ✅ verified arXiv id |
| [10] | Li2021_FedBN | arXiv:2102.07623 | ✅ verified arXiv id |
| [11] | Li2021_MOON | arXiv:2103.16257 | ✅ verified arXiv id |
| [13] | Yang2019_ChemProp | 10.1021/acs.jcim.9b00237 | ✅ verified DOI |
| [14] | Nelsen1987 | 10.1021/ja00237a007 | ⚠️ TODO — verify DOI; the four-point Nelsen reference |
| [15] | Xu2019_GIN | arXiv:1810.00826 | ✅ verified arXiv id |
| [16] | KAN2024 | arXiv:2404.19756 | ✅ verified arXiv id |
| [17] | EfficientKAN2024 | github.com/Blealtan/efficient-kan | ⚠️ TODO — pin to a tagged release / Zenodo DOI |
| [18] | RDKit2023 | www.rdkit.org | ⚠️ TODO — pin to a specific release with Zenodo DOI |

**Remaining DOI/metadata TODOs to clear before submission:** [3] Sasabe2011_OLED, [5/12] AtahanEvrenk2019_RE, [14] Nelsen1987, [17] EfficientKAN2024, [18] RDKit2023.

**Inactive bib (`paper/references_unused.bib`)** — 14 entries retained but not cited in the current draft: Li2023_SchNet, FedChem2022, Heyndrickx2023 (MELLODDY), FLAP2023, Schutt2018_SchNet (JCP version of [6]), Gilmer2017_MPNN, Gasteiger2020_DimeNet, Schutt2021_PaiNN, You2020_GraphCL, Hu2020_SSL, KAGNN2025, FedLG2025, Wu2018_MoleculeNet, Coley2020_GCN_review.  Reinstate any of these by pasting it back into `paper/references.bib` and adding a corresponding inline `[N]` in the main text.

---

## 7. Reviewer-risk list and response strategy

| # | Anticipated reviewer concern | Manuscript section that pre-empts it | Recommended response if asked |
|---|---|---|---|
| 1 | "C-hole is not improved — why include it?" | §4.3 final paragraph; SI S2.2 | Explain that all evaluated methods sit at a small-sample data ceiling identified by classical baselines (R² = −0.10). C-hole is reported transparently as part of the federation, *not* as a positive result. We do not claim hole-task improvement. |
| 2 | "E68 / E69 show the chemistry gate is not the source of performance — why keep it?" | §4.2, §4.4, §5(ii); SI S4.4, S7.6 | The gate is positioned as auditable algorithmic transparency, not as a performance lever. The invariants T_head[D, C-\*] = 0 and T_head[C-hole, C-triplet] = 0 follow from V1/V3 evidence and can be inspected from `T_transferability.json` without any model state. We explicitly do not claim performance gains from the gate. |
| 3 | "E65 is not significant — D → C is not necessarily negative transfer." | §4.4, §5(iii); SI S7.5 | Agreed. We avoid the stronger claim that D → C is necessarily harmful. PC² and naive D → C pretrain are statistically indistinguishable on either C target. Likely reasons (low gate-induced encoder weight; small-n test power) are discussed in §5(iii). |
| 4 | "No MOON baseline." | §6 Limitations item 3; SI S10.1 | Acknowledged. MOON is flagged as a deferred baseline; PC² covers personalization without contrastive regularisation. We commit to running MOON in a revision if the reviewer prefers. |
| 5 | "No ChemProp / D-MPNN baseline." | §6 Limitations item 3; SI S10.1 | Acknowledged. ChemProp would be a strong message-passing baseline; deferred for GPU-hour reasons. PC² focuses on the federated-learning family. |
| 6 | "Only 5 seeds — small seed budget." | §6 Limitations item 4; SI S9.1 | Each federated run takes ≈ 30 minutes on dual A30; 5 seeds × 9 methods × 2 LOOCV targets = 22 GPU-hours. Bootstrap 95 % CIs partially compensate for limited seed repeats. |
| 7 | "Chemistry-gate not tested under diverse-chemistry pairs." | §6 Limitations item 2; SI S10.4 | Agreed. C-hole and C-triplet share most molecules, so the gate is not stress-tested at moderate Tanimoto-K. Future work with intermediate-similarity task-clients is recommended. |
| 8 | "Calibration is too simple — only fixed (μ, σ)." | §6 Limitations item 6; SI S10.2 | A learnable affine variant (E72) is planned. We chose fixed (μ, σ) to avoid extra parameters at n = 49–53. |
| 9 | "Why no public-private-mixed evaluation under DP / secure-aggregation?" | not yet addressed | We can address this in a revision. The architecture supports adding DP-SGD or secure-aggregation since per-key routing does not constrain the underlying aggregation primitive. |
| 10 | "R² is negative — model is bad." | §2.5, §4.3, SI S2.2, SI S5.5 | We report R² as auxiliary only. Classical baselines also produce negative R² on the same n = 53 LOOCV; the issue is small-sample regression variance, not model quality. |

---

## 8. Final pre-submission checklist (binary)

Items the lead author still needs to clear before camera-ready submission.

- [ ] **References** — verify DOIs marked TODO in `paper/references.bib` (currently five entries: Sasabe2011_OLED, AtahanEvrenk2019_RE, Nelsen1987, EfficientKAN2024, RDKit2023)
- [ ] **Figures** — regenerate with Times New Roman (or the journal's preferred font) once a machine with the font installed is available; current draft uses Liberation Serif (TNR-compatible) under explicit user authorisation
- [ ] **Author list, affiliations and corresponding-author email** — supplied by the lead author
- [ ] **Author Contributions section** — supplied by the lead author with input from co-authors
- [ ] **Acknowledgements** — supplied by the lead author (compute resources, funding sources, data-providing laboratories)
- [ ] **Data & Code Availability** — repository URL inserted; in-house C-hole / C-triplet data-availability statement reviewed by the laboratory PI
- [ ] **Manuscript export** — converted to journal-preferred format (docx / LaTeX); not yet performed
- [ ] **Cover letter** — drafted separately; not yet performed
- [ ] **Conflict-of-interest disclosure** — prepared by the corresponding author
- [ ] **Final overclaim sweep on the formatted PDF** — re-run the patterns in Section 2 after conversion to journal format

Items already cleared in the pre-submission TODO cleanup pass (2026-05-11):

- [x] **V2 classical baselines** — SI Section S2.2 populated from real `results/preverify/V2_*.csv|md` files
- [x] **References tidied** — only 17 cited entries (covering [1]–[18]) remain in `paper/references.bib`; the other 14 entries archived to `paper/references_unused.bib`
- [x] **Overclaim sweep** — no forbidden phrases in either `paper/manuscript.md` or `paper/SI.md`; defensive denials of "negative transfer" / "performance lever" / "drives performance" remain
- [x] **Numerical consistency** — all manuscript numbers reconciled with `results/pc2_phase5_stats.csv`, `results/preverify/T_transferability.json`, and `results/preverify/V2_classical_baselines.csv`
- [x] **Inline citations [14]–[18] added** for Nelsen, Xu (GIN), Liu (KAN), efficient-KAN, RDKit methodology references
- [x] **Fig 6 chemistry phrasing** — uses low-risk description "halogenated polycyclic conjugated system"; chemistry-side co-author may refine before submission if desired

---

---

## 9. Major reviewer risks and how the revised manuscript addresses them

Added in the 2026-05-12 Major-Revision pass (`Phase 1` audit + `Phase 2`–`Phase 4` patches in this directory). Each risk is paired with the concrete edit that addresses it.

### Risk 1 — Quantity inconsistency for A/B/D

**Reviewer attack:** "You write A, B and D 'all measure hole-related reorganisation energies' in §4.1, but A/B are clearly cation-λ public data while D is a different hole-λ source. This is a factual error."

**Action taken:**
- `manuscript.md` §4.1 paragraph rewritten to make A/B = cation-λ, C-hole/D = hole-λ (but at different functionals and chemical spaces), C-triplet = triplet-λ; explicit statement that all five are "reorganization-energy related but not interchangeable".
- `manuscript.md` §2.1 Table updated.
- `paper/tables/table1_task_clients.{csv,md}` updated.
- `SI.md` §S1.1 paragraph updated.
- `figure_captions.md` Fig 1 caption now emphasises the quantity-class distinction.

### Risk 2 — A/B protocol metadata inconsistency with T_head[A, B] = 0.33

**Reviewer attack:** "Your §2.4 says any unknown protocol field → T_head = 0. But A/B have all protocol fields = unknown in the released JSON, and T_head[A, B] = 0.33. Which is it?"

**Root cause (read from `src/transferability.py:137`):** `t_head_compatibility` has a **same-source shortcut** (`if meta_src['source_id'] == meta_dst['source_id']: return True, None`) that bypasses the per-field unknown check for clients derived from a single source. A and B share `source_id = "qm9_public_reorg_15210"`, so the shortcut fires for A ↔ B only. The behaviour is deliberate, not a bug.

**Action taken:**
- `manuscript.md` §2.4 expanded to document the same-source shortcut as a numbered step (between the V1 hard rule and the per-field unknown clause), matching the actual `t_head_compatibility` evaluation order.
- `manuscript.md` §3.1 explicitly states that A ↔ B head sharing is permitted via the same-source shortcut and that auxiliary protocol fields are not separately curated for the public QM9 split.
- `paper/tables/table1_task_clients.{csv,md}` A/B rows changed from "unknown" to "source-consistent public protocol" with an explicit `protocol_note`.
- `SI.md` §S4 expanded with the same shortcut documentation.
- `SI.md` §S4.3 invariant list updated to list T_head[A, B] = 0.33 alongside the four zero-invariants, with the firing-clause explanation.
- `figure_captions.md` Fig 3 caption now explains that A ↔ B is the only non-zero T_head off-diagonal and **why**.

**Resolution status:** ✅ no remaining inconsistency.

### Risk 3 — Chemistry/protocol gate is not a performance driver, and E69 (uniform gate) is numerically slightly better than PC²

**Reviewer attack:** "Your Figure 5 shows E69 (uniform gate) has lower MAE than PC² on C-triplet by 0.043 eV. Your method's chemistry gate is therefore not adding value — and might be slightly hurting performance."

**Action taken:**
- Manuscript §4.4 paragraph on E69 rewritten to explicitly disclose the numerical direction (E69 is 0.043 eV lower), report the bootstrap CI (–0.001, +0.092) and the wins/losses (21/28 in favor of E69), and state that we **do not interpret** this as evidence either against or in favor of the gate at the current seed budget.
- Manuscript Abstract, §1 contribution list, §4.3, §5(ii), §6 Limitations item 2 (with the "future federations with chemically diverse but protocol-compatible clients" pointer), and Conclusions all re-framed: the gate is **auditable compatibility metadata**, never a performance lever.
- Conclusions and Abstract reorder the contributions: calibration first, task-client modelling second, layer-wise personalization third, transferability metadata fourth (and explicitly marked as not-a-performance-lever).
- `figure_captions.md` Fig 5 caption now spells out the E69 numerical direction and the absence of a performance claim for the gate.
- `next_experiments_plan.md` E73 entry clarifies that the current E69 ablation already provides the gate-ablation-with-calibration control.

### Risk 4 — Calibration framed as "just standardisation"

**Reviewer attack:** "Per-client standardisation is decades-old. Your 'task-specific calibration' is not novel."

**Action taken:**
- Manuscript §5 Discussion (i) explicitly notes that per-client standardisation has been used in single-task federations; the novelty claim is restricted to (a) **fold-level isolation under LOOCV**, (b) **task-client granularity under heterogeneous quantum-chemical FL**, and (c) **universal aggregation exclusion** via the shared `_get_exclude_keys` filter.
- `next_experiments_plan.md` E71 (FedPer + calibration-only) and E72 (Local-only + calibration-only) are the proposed empirical strengtheners. Strongly recommended pre-submission.

### Risk 5 — Borderline p-values

**Reviewer attack:** "Your significant comparisons sit at p = 0.042–0.045, unadjusted. With three comparisons against PC², the family-wise error rate is closer to 0.13. This is borderline at best."

**Action taken:**
- All three occurrences of "significantly improves" in the manuscript (Abstract, §4.3 header, Conclusions) softened to "achieves nominally significant MAE reductions" or "paired-Wilcoxon-significant" with explicit p-value reporting.
- Manuscript §4.3 paragraph now states: "p-values are unadjusted and sit close to the conventional 0.05 threshold"; bootstrap CIs are reported with explicit "touches zero at upper bound" disclosure for the PC²-vs-FedAvg pair.
- Figure 4 caption now explicitly labels the comparisons as **nominally significant** and notes that p-values are unadjusted (no multiple-comparison correction).

### Risk 6 — Missing MOON / ChemProp baselines

**Reviewer attack:** "How does PC²-FedReorg compare to MOON [Li 2021] or D-MPNN/ChemProp [Yang 2019]?"

**Action taken:**
- Manuscript §6 Limitations items 3 already flag both as recommended additions.
- `next_experiments_plan.md` E74 (MOON) and E75 (ChemProp) entries propose the runs with full reviewer-attack matrix and paper-claim adjustment branches. Marked Optional / Major-revision-tier.

### Risk 7 — Figure quality / placement

**Reviewer attack:** "Figure 6 (per-molecule cases) is qualitative and clutters the main text; some main figures are too small for double-column layout."

**Action taken:**
- Figure 6 already labelled as SI candidate in main-text Figure & Table Index and in the Fig 6 caption.
- `paper/latex/main.tex` typography passes (CPU-box TeX Live 2020) produce only 2 sub-8pt overfull boxes; SI.tex compiles with 0 overfull. No further figure-quality patches required pre-submission. Times-New-Roman font swap remains a documented TODO (Section 5 of this checklist).

---

## 10. Open issues remaining after the major revision pass

| ID | Issue | Owner | Required for submission? |
|---|---|---|---|
| 1 | E71 (FedPer + calibration-only) run | Lead author | Strongly recommended |
| 2 | E72 (Local-only + calibration-only) run | Lead author | Strongly recommended |
| 3 | Five `note = {TODO}` references in `references.bib` (Sasabe2011_OLED, AtahanEvrenk2019_RE, Nelsen1987, EfficientKAN2024, RDKit2023) | Lead author | Yes |
| 4 | Times New Roman font swap on a machine with TNR registered | Submitting author / typesetter | Yes if journal requires |
| 5 | Author list, affiliations, Acknowledgements, Data & Code Availability URL | Lead author | Yes |
| 6 | MOON / ChemProp baselines | Lead author | No (revision-tier) |
| 7 | Chemistry-co-author confirmation of Fig 6 structural class naming | Chemistry co-author | No (current phrasing is low-risk) |

---

---

## 11. Claim hierarchy after final polish

### Strongly supported
- Removing the per-task-client calibration buffer (E70) degrades C-triplet MAE: paired Wilcoxon p = 0.043, bootstrap 95 % CI (−0.225, −0.015) entirely below zero, 35 / 14 per-molecule wins for PC²-FedReorg.
- The transferability matrix is deterministic and auditable from `T_transferability.json` before any aggregation round.

### Nominally supported (p < 0.05 unadjusted; would not survive Bonferroni α/3 ≈ 0.017)
- PC²-FedReorg has lower MAE than FedAvg, FedProx and FedPer on C-triplet (p = 0.045, 0.045, 0.042; relative reduction 13–16 %; bootstrap CI upper bound touches zero for FedAvg).

### Explicitly NOT claimed
- C-hole improvement (all paired tests p ≥ 0.20; band 0.368–0.414 eV across nine methods).
- Chemistry/protocol gate drives performance — E69 (uniform-gate) is numerically slightly *better* than PC²-FedReorg on C-triplet, statistically indistinguishable (p = 0.18).
- Naive D → C transfer is necessarily harmful — E65 (D → C pretrain) is statistically indistinguishable from PC²-FedReorg.
- Calibration alone explains the C-triplet gain — a FedPer + calibration control is missing and listed as the highest-priority pre-submission addition.
- State-of-the-art status, comprehensive improvement, or robust significance across tasks.

### Main vs. SI allocation
- **Main text:** Figures 1–5; Tables 1–3.
- **SI:** Figure 6 (per-molecule cases — qualitative only) recommended for demotion to Figure S1; full ablation rows beyond E70 in SI Table S4; per-molecule predictions, V1/V2/V3 pre-verification, per-seed performance, and the 93-unit-test reproducibility checklist in SI Sections S1–S10.

---

## 12. Reviewer attack / response table

| # | Anticipated reviewer attack | Response in the manuscript | Section |
|---|---|---|---|
| 1 | "Is this just per-client standardization with a new name?" | Standardization in single-task federations is acknowledged in §5(i); the novelty claim is restricted to fold-level isolation under LOOCV, task-client granularity under heterogeneous quantum-chemical FL, and universal aggregation exclusion via the shared filter. | §5(i) |
| 2 | "Why keep the gate if uniform-gate (E69) is numerically better?" | Auditability before training, deterministic invariants encoded from V1/V3 evidence, bootstrap CI on uniform-vs-gate crossing zero — dedicated paragraph in §5(ii). | §5(ii), §6 |
| 3 | "Where are MOON / ChemProp / D-MPNN baselines?" | Explicitly deferred in §6 item 3. No coverage claimed in the present manuscript. | §6 |
| 4 | "Why is C-hole not improved?" | C-hole reported transparently as non-significant; data ceiling verified by V2 classical baselines (best R² = −0.10). | §4.3, SI §S2.2, §6 item 1 |
| 5 | "Are p-values robust after Bonferroni correction?" | No — the three C-triplet baseline comparisons would not cross the Bonferroni-corrected threshold (α/3 ≈ 0.017). Explicit in §4.3 and §6 item 8. | §4.3, §6 item 8 |
| 6 | "Is A/B head-sharing artificially permitted despite unknown protocol fields?" | A ↔ B share `source_id`; the same-source shortcut in `t_head_compatibility` (§2.4 step 3) fires before the per-field unknown check. Documented in §2.4 and Figure 3 caption. | §2.4, Fig 3 |
| 7 | "Why not use centralized training as a control?" | Centralized training requires raw-data sharing across the in-house TADF lab and the public sources, which the federation setup is explicitly designed to avoid. E60 local-only is the no-federation lower bound. | §3.2 |
| 8 | "Is KAN necessary, or would MLP suffice?" | The codebase supports both via `head_type`; reported runs use KAN. A head-architecture comparison is out of scope. | §2.2, SI §S3.2 |
| 9 | "Is the effect size chemically meaningful?" | The C-triplet ΔMAE of −0.10 to −0.12 eV sits at the edge of the conformer-averaging noise of ωB97X-D reorganization energies (≈ ±0.1 eV). Flagged in §6 item 8. | §6 item 8 |
| 10 | "Bootstrap resampling unit may overstate confidence." | Bootstrap resamples molecule indices (the paired-comparison unit), 5,000 resamples. Documented in §2.5 and SI §S6.1. | §2.5, SI §S6.1 |

---

## 13. MCP reviewer findings during final polish

### MCP review_plan (Phase 0)

| GPT verdict | Decision |
|---|---|
| BLOCKING: calibration claim too strong without FedPer+cal control | **Partially adopted** — narrowed claim to "among the four ablations evaluated"; added §6 item 7 deferring FedPer+cal as highest-priority pre-submission. Rejected the demand to run E71 (user forbids new experiments). |
| BLOCKING: Table 1 inconsistency | **Rejected as blocking** — already unified in Major Revision; no SHA-256 hash needed. |
| BLOCKING: E71 required for submission | **Rejected** — user forbids new experiments. Listed in `next_experiments_plan.md` as strongly recommended pre-submission. |
| RECOMMEND: E69 transparency | **Adopted** — new §5(ii) paragraph dedicated to "why retain the gate when uniform-gate is numerically slightly better". |
| RECOMMEND: LOOCV power analysis | **Adopted** — expanded §6 item 8 with bootstrap-CI-first interpretation. |
| RECOMMEND: visibility of C-hole and D → C non-claims | **Adopted** — tightened §5(iii). |
| NICE-TO-HAVE: figure style | Adopted as planned. |

### MCP adversarial_review (Phase 3)

| GPT verdict | Decision |
|---|---|
| BLOCKING: multiple-comparison correction missing | **Adopted** — added Bonferroni note (α/3 ≈ 0.017) to §4.3 and §6 item 8; recommended bootstrap CIs as the primary evidence. |
| BLOCKING: component-isolation gap | **Already addressed** in contribution 1 and §6 item 7 (deferred FedPer+cal). No further action. |
| BLOCKING: gate-retention logic circular | **Rejected** — §5(iii) already says "we cannot empirically prove that the gate decision was correct, only that it was deterministic and auditable". |
| RECOMMEND: rename "contributions" to less-assertive header | **Adopted** — rewrote header to "five design components ... in decreasing order of empirical support in this study, with explicit boundaries on what the present ablations can and cannot conclude". |
| RECOMMEND: drop C-hole ceiling and D → C discussions | **Rejected** — V2 baselines verify the ceiling; D → C is an explicit non-claim that strengthens transparency. |
| NICE-TO-HAVE: define effect-size threshold | **Adopted** — added ±0.1 eV conformer-averaging-noise context to §6 item 8. |
| NICE-TO-HAVE: document bootstrap unit | **Already in §2.5** (molecule indices, 5,000 resamples). No further action. |

---

*End of submission checklist.*
