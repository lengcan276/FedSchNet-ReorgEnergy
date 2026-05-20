# Figure data-source audit

**Scope.** `paper/make_figures.py` is the only script that emits the
shipped figures (`paper/figures/Fig{1..5}*.{pdf,svg,png}` and
`FigS1*`). This audit determines, per figure, whether the data
points and statistical labels are read from `results/*` /
`paper/tables/*` files or whether they are hardcoded. The script
itself is read-only (line 1099 `sys.exit(main())`); the audit also
verifies that all path constants in the script point at the canonical
result files.

**Path constants (`paper/make_figures.py:79-90`).** All point to
authoritative sources:

```
AB_CSV         = data/client_a_b/public_reorg_energy_15210.csv
C_HOLE_CSV     = ../logs/hole_reorg_molecular.csv
C_TRIPLET_CSV  = ../logs/reorganization_energy_summary.csv
D_CSV          = ../logs/atahan_reorg_5876.csv
T_TRANSFER     = results/preverify/T_transferability.json
T_CHEMISTRY    = results/preverify/T_chemistry.json
PC2_MAIN_CSV   = results/pc2_batch1_multiseed_summary.csv
PC2_MAIN_JSON  = results/pc2_batch1_multiseed_predictions.json
PC2_ABL_CSV    = results/pc2_phase5_ablation_summary.csv
PC2_ABL_JSON   = results/pc2_phase5_ablation_predictions.json
PC2_STATS_CSV  = results/pc2_phase5_stats.csv
```

The result-file SHA256 hashes captured by `paper/audit/build_manifest.py`
are listed in `paper/audit/result_file_manifest.csv`.

---

## Figure 1 — `fig1_task_clients` (lines 256+ in `make_figures.py`)

**Reads:** `AB_CSV` (line 179), `C_HOLE_CSV` (192), `C_TRIPLET_CSV` (196),
`D_CSV` (205), `T_TRANSFER` (line 1066 in `main()`).

**Per-node statistics:** mean / std / min / max of `label` column,
recomputed in `node_metadata()` (lines 222-238) from the raw CSV.
**No hardcoded numbers.**

**Hardcoded text labels:**
- `descr` dict (lines 215-221) hardcodes the source-name strings
  ("QM9 public", "private TADF", "Atahan 2019") and the quantity strings
  ("cation reorg λ", "hole reorg λ", "triplet reorg λ"). These are
  conceptual labels, not numerical claims. **Acceptable.**

**Verdict: PASS.** Fig 1 data is fully data-driven from the raw CSVs.

---

## Figure 2 — `fig2_architecture` (lines 330+)

Architecture / method-diagram schematic. No data-derived numbers in the
figure. **N/A.**

---

## Figure 3 — `fig3_transferability` (line 466+)

**Reads:** `T_TRANSFER` JSON (`results/preverify/T_transferability.json`)
at line 468. The 5×5 transferability matrix entries are read directly
from the JSON. **No hardcoded numbers.**

**Verdict: PASS.**

---

## Figure 4 — `fig4_main_performance` (line 517+)

**Reads:**
- `PC2_MAIN_CSV` = `results/pc2_batch1_multiseed_summary.csv` (line 519)
- `PC2_STATS_CSV` = `results/pc2_phase5_stats.csv` (line 520)

**Bars:** `means[]` and `sems[]` (lines 530-536) are computed from the
loaded summary CSV (5-seed mean of per-seed MAE; SEM from seed SD/√n).
**No hardcoded MAE values.**

**Significance brackets:** `p` (line 559) is read from
`pc2_phase5_stats.csv` `wilcoxon_p` column. Significance text
constructed as f-string. **Not hardcoded.**

**Hardcoded annotation strings** (lines 571-585):
- C-hole panel: `"All paired tests vs PC² are n.s.\n(narrow MAE band 0.37–0.41 eV; small-n data ceiling)"`
  - "0.37–0.41 eV" is a rounded annotation. Recomputed C-hole band
    (`recomputed_numbers.csv`) per-mol-median: **0.3682–0.4137**. After
    rounding to 2 decimals, **0.37–0.41**. Source is stable; rounding
    matches data. **Acceptable annotation.**
- C-triplet panel: hardcoded `"PC² > FedAvg / FedProx / FedPer\n(paired Wilcoxon nominal p < 0.05; unadjusted, would not survive Bonferroni — §4.3)"`.
  - This is a qualitative direction statement, not a number. Recomputed
    triplet p-values are 0.0449 / 0.0449 / 0.0417, all < 0.05.
    **Acceptable.**

**Verdict: PASS.** All bar heights and statistical brackets are
data-driven. Two text-only annotations are hardcoded but verified
against the recomputed data.

---

## Figure 5 — `fig5_ablation_forest` (line 594+)

**Reads (lines 628-634):**
- `results/e71_summary.csv`
- `results/e71_vs_e66_stats.csv`
- `PC2_MAIN_CSV` / `PC2_ABL_CSV` / `PC2_STATS_CSV`

**Panel (a) bar plot:** `_across_seed_mae(df, exp, target)` (lines 619-626)
recomputes per-method 5-seed mean MAE ± SEM from the summary CSVs.
**No hardcoded MAE values.** The four configurations and their data
sources:
- E63 → from `PC2_MAIN_CSV`
- E70 → from `PC2_ABL_CSV`
- E66 → from `PC2_MAIN_CSV`
- E71 → from `e71_summary.csv`

**Panel (b) forest plot:**
- E71 contrasts (`_e71_vs_arbitrary`, lines 676-722): per-molecule median-
  across-seeds absolute error from `results/e71_predictions.json` and the
  comparison's prediction JSON (`pc2_batch1_multiseed_predictions.json` or
  `pc2_phase5_ablation_predictions.json`). ΔMAE, bootstrap 95% CI
  (5000 resamples, seed=0 inside `_e71_vs_arbitrary`), Wilcoxon
  p — all recomputed live, not hardcoded.
- PC² ablations (`_pc2_row`, lines 666-672): read from
  `pc2_phase5_stats.csv`. **Not hardcoded.**

**Panel (c) interpretation box (lines 790-812):** This text **does**
contain hardcoded p-values:
| line | hardcoded value | data source | match? |
|---|---|---|---|
| 791 | `p = 0.929` (E71 vs E66 triplet) | `e71_vs_e66_stats.csv` wilcoxon_p = 0.9293 | ✅ rounds to 0.929 |
| 795 | `p = 0.028` (E71 vs E63 triplet) | `e71_vs_e63_stats.md` p = 0.0277 | ✅ rounds to 0.028 |
| 799 | `p = 0.021` (E71 vs E70 triplet) | `e71_vs_e70_stats.md` p = 0.0206 | ✅ rounds to 0.021 |
| 811 | `all p ≥ 0.42` (C-hole) | min p in C-hole E71 contrasts = 0.4230 | ✅ matches |

These four hardcoded labels are stable summaries of the recomputed
paired stats and match the data within rounding. **Acceptable annotation
(label-only; underlying point estimates and CIs in the forest plot
itself are recomputed live).**

**Verdict: PASS** — figure data and statistical brackets are recomputed
from the result files at draw time; the interpretation-box text is
the only hardcoded numeric block and it matches the data to the
displayed precision.

---

## Figure 6 / S1 — `fig6_molecule_cases` (line 832+)

**Reads:** `PC2_MAIN_JSON` (`pc2_batch1_multiseed_predictions.json`) at
line 834. Per-molecule absolute errors recomputed from raw predictions.
SMILES list taken from `node_dfs["C-triplet"]` (Fig-1 data flow).
**No hardcoded numbers.**

**Verdict: PASS.**

---

## Cross-cutting findings

1. `make_figures.py` does not hardcode any MAE / RMSE / R² / Wilcoxon-p
   number that is also reported in the paper tables. Every quantitative
   element is read from a result-file path captured in the manifest.
2. Two figures contain hardcoded *label text* that summarises the
   computed numbers:
   - Fig 4 C-hole panel: `"narrow MAE band 0.37–0.41 eV"` — matches
     recomputed C-hole band (0.368–0.414) after 2-decimal rounding.
   - Fig 5 panel (c): `p = 0.929 / 0.028 / 0.021` and `all p ≥ 0.42` —
     all match the recomputed paired stats to displayed precision.
3. No hallucinated numbers detected in `make_figures.py`. No "fake
   data" or placeholder constants found. No `np.array([...])` blocks
   that would suggest hand-pasted numerical content.
4. Reproducibility risk: if any of the source result files
   (`pc2_batch1_multiseed_*`, `pc2_phase5_*`, `e71_*`) is regenerated,
   Fig 4 / Fig 5 bar heights and forest-plot points will update
   automatically, but the hardcoded labels in the two annotations
   above will not. Recommend (advisory only, no auto-edit) that future
   maintenance of these labels track the changes — or, ideally,
   derive them in-line from `recomputed_numbers.csv` to make them
   fully reactive.

**Overall verdict on `make_figures.py`: PASS.** All paper-reported
numerical claims that appear in figures are traceable to result files
in this repository; the only hardcoded label text matches the
recomputed values to the displayed rounding precision.
