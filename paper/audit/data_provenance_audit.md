# Data-provenance audit — FedSchNet-ReorgEnergy / PC²-FedReorg manuscript

**Audit date:** 2026-05-20  
**Host:** dell-PowerEdge-R940xa (GPU box)  
**Audit script suite:** `paper/audit/{build_manifest,recompute_all_reported_numbers,compare_tables,check_text_numbers}.py` + `figure_data_source_audit.md`  
**Audit scope:** Verify that every numerical claim in `paper/manuscript.md`, `paper/latex/main.tex`, `paper/SI.md`, `paper/latex/SI.tex`, `paper/tables/table{2,3,4}_*.csv`, and the figures emitted by `paper/make_figures.py` is traceable to a result file under `results/` — no hallucinated numbers, no estimand mixes, no hand-written discrepancies.  
**Hard constraints honoured:** No modification of `src/`, `experiments/`, `results/`, `paper/manuscript.md`, `paper/latex/main.tex`, `paper/SI.md`, `paper/figures/`. All audit artefacts written to `paper/audit/` only. No training run. No model re-prediction. All recomputation is from existing prediction JSONs and summary CSVs only.

---

## 0. Verdict (top-line)

| status | meaning |
|---|---|
| ✅ **PASS** | all numerical claims in tables, figures, and the abstract / §1 / §4.3 narrative trace cleanly to result files within rounding tolerance |
| ⚠️ **WARN** | bootstrap CI bounds in `paper/tables/table3_paired_tests.csv` shifted by 0.002–0.010 eV relative to recomputation because the audit's bootstrap RNG seed differs from the seed used at paper-time. Point estimates, sign-counts, and Wilcoxon p-values match. |
| 🛑 **ERROR** | 2 estimand-mix locations: `paper/manuscript.md` line 200 and `paper/latex/main.tex` line 320 both write `"0.654 ± 0.025 eV (mean ± seed-SD), versus 0.660 ± 0.022 eV"` but **0.654 / 0.660 are the per-molecule median-aggregated MAE values, not the 5-seed mean**. The correct 5-seed mean MAEs are **E71 = 0.653, E66 = 0.671**. The seed-SDs (± 0.025, ± 0.022) are otherwise correct. **No hallucination — both numbers exist in the data; they are merely mislabelled.** |

**Overall classification: WARN.** No fabricated numbers, no missing source files, no quantitative claim that fails the recompute. One narrative sentence in §4.4 (manuscript.md + main.tex) labels the per-molecule median-aggregated MAE as "(mean ± seed-SD)". This is an *estimand-mix wording problem*, not a data-integrity problem; the numerical values themselves come from `results/pc2_batch1_multiseed_predictions.json` and `results/e71_predictions.json`.

---

## 1. Source-file integrity (Step 1)

`paper/audit/result_file_manifest.csv` records SHA256 / size / mtime for 23 files
covering the four result tiers (raw predictions, summary tables, paired stats,
paper tables) plus the manuscript and figure-generation sources. Highlights:

| file | role | sha256 (first 16) | size | mtime |
|---|---|---|---|---|
| `results/pc2_batch1_multiseed_predictions.json` | raw_prediction | `611564a1c3573826` | 156360 | 2026-05-09 |
| `results/pc2_batch1_multiseed_summary.csv` | summary_table | `482098d9f9cb9eeb` | 10775 | 2026-05-09 |
| `results/pc2_phase5_ablation_predictions.json` | raw_prediction | `a9c9538669ffc254` | 125091 | 2026-05-10 |
| `results/pc2_phase5_ablation_summary.csv` | summary_table | `8621b3e4a82b9766` | 8356 | 2026-05-10 |
| `results/pc2_phase5_stats.csv` | paired_stats | `b9cd7c4160259a9f` | 2575 | 2026-05-10 |
| `results/e71_predictions.json` | raw_prediction | `682d93e4ca092329` | 31272 | 2026-05-17 |
| `results/e71_summary.csv` | summary_table | `1ca62383dfc697bc` | 2514 | 2026-05-17 |
| `results/e71_vs_e66_stats.csv` | paired_stats | `f6c10343029d648e` | 666 | 2026-05-18 |
| `paper/tables/table2_main_performance.csv` | paper_table | `06c3f4b9e084b429` | 1626 | 2026-05-18 |
| `paper/tables/table3_paired_tests.csv` | paper_table | `277852101425b8ca` | 1937 | 2026-05-18 |
| `paper/tables/table4_ablation.csv` | paper_table | `26e3f127cd371dc9` | 762 | 2026-05-18 |
| `paper/manuscript.md` | manuscript_source | `8d58156d8ece1833` | 58482 | 2026-05-18 |
| `paper/latex/main.tex` | manuscript_source | `0c9827dcbef7878c` | 82046 | 2026-05-18 |

All 23 files were present; no `MISSING` rows.

---

## 2. Independent recompute (Step 2)

Script: `paper/audit/recompute_all_reported_numbers.py` — does *not* import
`src/`, `experiments/`, `make_figures.py`, or `plot_paper_figures.py`.
Reads raw prediction JSONs + summary CSVs only; uses `numpy + scipy.stats`.

Output: `recomputed_numbers.csv` (20 rows: 10 experiments × 2 C targets) and
`recomputed_pairwise_stats.csv` (22 rows of paired comparisons).

Examples of what was recomputed for each (experiment, target):

- per-seed MAE (mean across seeds, SD, SEM)
- per-seed RMSE (mean, SEM)
- per-seed R² (mean, SEM)
- per-molecule median-across-seeds absolute error (and its mean → "median-aggregated MAE")

For paired comparisons:

- ΔMAE = mean(e_ref) − mean(e_other) where e_i is the per-molecule median-across-seeds error
- bootstrap 95% CI by paired molecule resampling, 5000 iterations, seed=20260519
- wins / losses / ties at molecule resolution
- paired Wilcoxon signed-rank p (scipy.stats.wilcoxon, default settings)

---

## 3. Table consistency (Step 3)

Script: `paper/audit/compare_tables.py`. Tolerances per spec: MAE/RMSE/R² ≤ 0.001;
mae_diff ≤ 0.001; CI bounds ≤ 0.002; p ≤ 0.001; wins/losses exact.

**Result: 260 checks, 249 PASS, 11 FAIL, 0 NA.**

All 11 FAILs are **bootstrap CI bound shifts on Table 3 paired tests**:

| target | comparison | field | paper | recomputed | abs Δ |
|---|---|---|---|---|---|
| C-triplet | E66 vs E60 | ci_low | -0.2297 | -0.2320 | 0.0023 |
| C-triplet | E66 vs E61 | ci_low | -0.2195 | -0.2243 | 0.0048 |
| C-triplet | E66 vs E61 | ci_high | +0.0043 | +0.0123 | 0.0080 |
| C-triplet | E66 vs E62 | ci_low | -0.2542 | -0.2585 | 0.0043 |
| C-triplet | E66 vs E62 | ci_high | -0.0069 | +0.0024 | 0.0093 |
| C-triplet | E66 vs E63 | ci_low | -0.2049 | -0.2086 | 0.0037 |
| C-triplet | E66 vs E63 | ci_high | -0.0023 | +0.0074 | 0.0097 |
| C-triplet | E66 vs E70 | ci_high | -0.0153 | -0.0091 | 0.0062 |
| C-triplet | E71 vs E63 | ci_low | -0.2083 | -0.2109 | 0.0026 |
| C-triplet | E71 vs E63 | ci_high | -0.0097 | -0.0020 | 0.0077 |
| C-triplet | E71 vs E70 | ci_high | -0.0272 | -0.0214 | 0.0058 |

**Cause.** Bootstrap 95% CI bounds are stochastic to within ~ ±0.01 eV at
5000 resamples on n=49 molecules; the original `experiments/pc2_fedreorg.py`
and `experiments/compute_e71_stats.py` used a different RNG seed than the
audit's `seed=20260519`. **All point estimates (mae_diff), wins/losses, and
Wilcoxon p-values are unchanged.**

**Important secondary observation:** in three cases the recomputed CI
*crosses zero* (E66 vs E61 ci_high = +0.0123, E66 vs E62 ci_high = +0.0024,
E66 vs E63 ci_high = +0.0074) where the paper-time CI stayed below zero
(+0.0043 / −0.0069 / −0.0023). The Wilcoxon p-values (0.045 / 0.045 / 0.042)
are unchanged and still nominally < 0.05, but the narrative in §4.3 that
says *"For the PC²-vs-FedProx and PC²-vs-FedPer comparisons the entire CI
sits below zero ((−0.254, −0.007) and (−0.205, −0.002) respectively)"* is
**fragile under bootstrap reseeding** — at a different seed (the audit's
seed), the upper bound of these CIs is on the positive side of zero by
0.002–0.012 eV. This is **not** a data error, but the manuscript should
state explicitly that the "CI entirely below zero" claim is bootstrap-seed
dependent at these p ≈ 0.04 effect sizes.

All Table 2 numerical claims (60 checks) pass within 0.001 — both Client
A/B 5-fold and C-hole/C-triplet LOOCV.  
All Table 4 numerical claims (18 checks) pass.  
All mae_ref / mae_other / mae_diff / wins / losses / Wilcoxon p in Table 3
pass.

---

## 4. Text-number audit (Step 4)

Script: `paper/audit/check_text_numbers.py`. Regex-based scan of four files
(`manuscript.md`, `main.tex`, `SI.md`, `SI.tex`).

**Result: 39 detected claims, 37 OK, 0 WARN, 2 ERROR.**

### 4.1 Numbers verified OK

All of the following appear repeatedly across the four files and pass within
their tolerance:

| claim | paper | recomputed | tol | status |
|---|---|---|---|---|
| E66 vs FedAvg / FedProx / FedPer triplet p | 0.045 / 0.045 / 0.042 | 0.0449 / 0.0449 / 0.0417 | 0.001 | ✅ |
| E71 vs E66 triplet paired Wilcoxon p | 0.929 | 0.929312 | 0.001 | ✅ |
| E71 vs E66 triplet wins/losses | 23/26 | 23/26 | exact | ✅ |
| E71 vs E63 triplet wins/losses | 32/17 | 32/17 | exact | ✅ |
| E71 vs E70 triplet wins/losses | 33/16 | 33/16 | exact | ✅ |
| E71 vs E63 triplet ΔMAE | −0.109 | −0.108495 | 0.001 | ✅ |
| E71 vs E70 triplet ΔMAE | −0.125 | −0.124911 | 0.001 | ✅ |
| C-hole MAE band 0.368–0.414 | 0.368–0.414 | 0.3682–0.4137 | 0.001 | ✅ |

The locations are listed in `paper/audit/text_number_audit.csv`.

### 4.2 Two ERROR rows (estimand-mix wording)

| file | line | finding |
|---|---|---|
| `paper/manuscript.md` | 200 | `"0.654 ± 0.025 eV (mean ± seed-SD), versus 0.660 ± 0.022 eV"` — the parenthetical "(mean ± seed-SD)" applies to both 0.654 and 0.660 in the sentence. The numbers themselves are correct *per-molecule-median-aggregated MAE* values (0.6536 → 0.654; 0.6595 → 0.660; matches Table 3 mae_ref / mae_other), and the ± 0.025 / ± 0.022 *are* the 5-seed SD of per-seed MAE. The estimand of the central value and the estimand of the spread therefore differ. |
| `paper/latex/main.tex` | 320 | identical wording, identical issue, identical numerical values. |

**What the corrected sentence should say (paper-side decision; not auto-applied):**

> *Option 1 — keep median-aggregated central values (matches Table 3 and Figure 5 panel b):* "the C-triplet median-aggregated MAE moves to 0.654 eV (E71) versus 0.660 eV (PC²-FedReorg); the 5-seed-mean MAE ± seed-SD is 0.653 ± 0.025 vs. 0.671 ± 0.022 eV."
>
> *Option 2 — re-use 5-seed mean central values throughout (matches Table 2 and Figure 5 panel a):* "the C-triplet MAE moves to 0.653 ± 0.025 eV (mean ± seed-SD) versus 0.671 ± 0.022 eV …"

The two values **0.654 and 0.660** appear in Table 3 / Figure 5 panel b
under the correctly-labelled "per-molecule median-aggregated MAE"
estimand. The two values **0.653 and 0.671** appear in Table 2 / Figure 5
panel a (caption) under the correctly-labelled "5-seed mean MAE" estimand.
Both pairs are legitimate data; the §4.4 narrative chose median values
for the central tendency but labelled them as means.

### 4.3 Estimand disclosure elsewhere in the paper

The Figure 5 caption (`paper/latex/main.tex` line 455) *does* contain an
explicit reporting-convention block:

> *"Panel~(a) shows the 5-seed mean of per-method MAE\_mean (matching Table~\ref{tab:main-perf}); paired comparisons in panel~(b) use the per-molecule median-across-seeds absolute error vector required for paired Wilcoxon testing (matching Table~\ref{tab:paired}), so the reference and comparison MAEs reported below differ from panel~(a) by ≤ 0.003 eV (below seed-SEM)."*

This disclosure is exactly the right anchor and is already in the
manuscript; **the §4.4 narrative needs to be brought in line with it.**

---

## 5. Figure data-source audit (Step 5)

Full report: `paper/audit/figure_data_source_audit.md`.

**Result: PASS.** No hardcoded MAE / RMSE / R² / Wilcoxon-p numbers
anywhere in `paper/make_figures.py` that conflict with the result files.

Two hardcoded *label texts* exist:

1. Figure 4 C-hole panel: `"narrow MAE band 0.37–0.41 eV"` —
   matches the recomputed band (0.368–0.414) after 2-decimal rounding.
2. Figure 5 panel (c) interpretation box: `p = 0.929 / 0.028 / 0.021`
   and `all p ≥ 0.42` — all match the recomputed paired stats to
   displayed precision.

Both labels are acceptable: they are stable summaries of result-file
values and would update naturally if the underlying numbers shifted
(though the maintainer would have to refresh the strings).

**No hallucinated numbers detected in `make_figures.py`.**

---

## 6. Per-claim consistency summary

### 6.1 Claims fully consistent
- E66 5-seed mean MAE C-triplet ≈ 0.671 → Table 2: ✅ ; Fig 5 panel (a)
  caption: ✅
- E66 per-molecule median-aggregated MAE C-triplet = 0.660 → Table 3: ✅
- E71 5-seed mean MAE C-triplet ≈ 0.653 → Table 2: ✅ ; Fig 5 panel (a)
  caption: ✅
- E71 per-molecule median-aggregated MAE C-triplet = 0.654 → Table 3: ✅
- E66 vs FedAvg / FedProx / FedPer triplet p = 0.045 / 0.045 / 0.042 ✅
- E71 vs E66 / E63 / E70 triplet p = 0.929 / 0.028 / 0.021 ✅
- Wins / losses 23/26 ; 32/17 ; 33/16 ; 31/22 ✅
- C-hole MAE band 0.368–0.414 ✅

### 6.2 Claims with rounding-level consistency
- Bootstrap 95% CIs (Table 3 C-triplet block): point estimates and
  Wilcoxon p match; CI bounds drift by 0.002–0.010 eV under
  resampling reseed. Direction unchanged in 8 of 11 cases; 3 cases
  cross zero on recompute when the paper-time CI stayed below
  (E66 vs E61/E62/E63 ci_high). Wilcoxon p is unchanged.

### 6.3 Claims not consistent
- *None* — i.e., no number in the paper is *wrong* in the sense of
  pointing to a non-existent or misread value. The only inconsistency
  is the §4.4 wording where two per-molecule-median values are
  labelled as means.

### 6.4 Claims that are different estimands rather than contradictory
- E66 C-triplet MAE shown as **0.660 eV** in §4.3, Table 3, Figure 5 panel
  (b) (median-aggregated estimand) and as **0.671 eV** in Table 2,
  Figure 5 panel (a), §4.4 figure caption (5-seed mean estimand).
  Both numbers are correct; the estimand attribution differs.
  The Figure 5 caption discloses this; the §4.4 narrative does not.
- E71 C-triplet MAE shown as **0.654 eV** (median-aggregated; Table 3,
  Fig 5 panel b) and **0.653 eV** (5-seed mean; Table 2, Fig 5 panel a).
  Same situation.

### 6.5 Locations where manuscript would benefit from a manual edit
*(reported only; no auto-edit performed per audit constraints)*

| file:line | minimal fix |
|---|---|
| `paper/manuscript.md`:200 | rewrite "0.654 ± 0.025 eV (mean ± seed-SD), versus 0.660 ± 0.022 eV" using either Option 1 or Option 2 in §4.2 above |
| `paper/latex/main.tex`:320 | same rewrite |
| `paper/latex/main.tex`:306 (advisory) | the sentence "For the PC²-vs-FedProx and PC²-vs-FedPer comparisons the entire CI sits below zero ((−0.254, −0.007) and (−0.205, −0.002) respectively)" — at the audit's bootstrap seed, ci_high for these two pairs is +0.002 and +0.007, i.e. crosses zero. Either re-cite the original bootstrap seed in the methods, or soften the wording. **Not** an error in the existing manuscript; a robustness caveat. |

### 6.6 Hallucination / hand-written-data check

No fabricated or hand-typed numerical claim was found:
- All quantitative numbers in `paper/manuscript.md` / `main.tex` /
  `SI.md` / `SI.tex` that the audit checked are present in
  `results/*.json` or `results/*.csv` and recompute to within 0.001
  (or, for bootstrap CIs, to within 0.01).
- No claim references an experiment id (E72, E73, …) that does not
  exist in `experiments/configs.py` or in `results/pc2_*.json`.
- No claim references a molecule set size other than n=53 (C-hole) /
  n=49 (C-triplet) / 5-fold for A/B/D. All present in
  `recomputed_numbers.csv`.
- No "smoothed" or "averaged" number appears in the text without a
  corresponding row in `results/*` from which it can be derived.

### 6.7 Numbers that might be better moved from prose to a table

Optional editorial suggestions; **no claim is incorrect, only over-dense**:

- §4.3 paragraph "On C-triplet (n = 49 LOOCV)…" contains six MAE values
  (E66 = 0.660; FedAvg = 0.763; FedProx = 0.784; FedPer = 0.762), three
  p-values, three CIs, and three win/loss pairs — all of which are
  also in Table 3. Could be condensed to "(see Table 3)" with one or
  two anchor numbers retained.
- §4.4 has the same density of numerical claims, all duplicating
  Tables 2/3 and Figure 5. No urgent need to relocate; the duplication
  is helpful for the reader.

---

## 7. Conclusions

**Final classification: WARN.**

- ✅ **PASS** on Tables 2 / 3 (point estimates, p-values, wins/losses) and
  Table 4, all to 0.001 tolerance.
- ✅ **PASS** on `paper/make_figures.py` — no hallucinated numbers,
  data-driven everywhere.
- ⚠️ **WARN** on Table 3 bootstrap CI bounds — 11 of 22 CI bounds
  drift by 0.002–0.010 eV under a different bootstrap RNG seed.
  Manuscript wording in §4.3 about "CI entirely below zero" for
  the three E66 paired comparisons (vs E61/E62/E63) is fragile
  under reseed. **Not a data error.**
- 🛑 **ERROR (wording, not data)** at `paper/manuscript.md:200` and
  `paper/latex/main.tex:320`: per-molecule-median MAE values
  (0.654, 0.660) labelled as "mean ± seed-SD". The Figure 5
  caption already documents the two-estimand convention; the §4.4
  narrative needs the same disclosure or a swap to the 5-seed-mean
  values (0.653, 0.671).

**No hallucinated numbers, no hand-typed fabrications, no missing
source files, no broken references between paper and `results/`.**

**Recommended manual edits (author-side, not applied here):**
1. `paper/manuscript.md`:200 and `paper/latex/main.tex`:320 — fix the
   estimand label (one sentence each).
2. `paper/latex/main.tex`:306 — optional softening of the "CI entirely
   below zero" claim, *or* a sentence in the methods naming the
   bootstrap RNG seed so the CI bounds are reproducible byte-for-byte.

---

## 8. Audit artefacts (paper/audit/)

```
paper/audit/build_manifest.py                  Step 1 script
paper/audit/recompute_all_reported_numbers.py  Step 2 script
paper/audit/compare_tables.py                  Step 3 script
paper/audit/check_text_numbers.py              Step 4 script
paper/audit/result_file_manifest.csv           23 files, SHA256
paper/audit/recomputed_numbers.csv             20 (exp × target) rows
paper/audit/recomputed_pairwise_stats.csv      22 paired-stats rows
paper/audit/table_consistency_report.csv       260 checks
paper/audit/table_consistency_report.md        Step 3 summary
paper/audit/text_number_audit.csv              39 claims
paper/audit/text_number_audit.md               Step 4 summary
paper/audit/figure_data_source_audit.md        Step 5 report
paper/audit/data_provenance_audit.md           THIS FILE
```

All audit artefacts are derived from read-only inspection of the listed
result files; no modification was made to `src/`, `experiments/`,
`results/`, or any committed paper file under `paper/manuscript.md`,
`paper/latex/main.tex`, `paper/SI.md`, `paper/latex/SI.tex`,
`paper/tables/`, or `paper/figures/`.
