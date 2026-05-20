"""E71 paired-stats analysis: compare E71 (FedPer + calibration only)
against E66 (PC²-FedReorg), E63 (FedPer legacy), E70 (PC² no-calibration).

Inputs (read-only):
  - results/e71_predictions.json           (this script's input; from run_e71_multiseed)
  - results/pc2_batch1_multiseed_predictions.json (E63, E66)
  - results/pc2_phase5_ablation_predictions.json  (E70)

Outputs (e71-namespaced; does not touch existing files):
  - results/e71_vs_e66_stats.csv
  - results/e71_vs_e66_stats.md
  - results/e71_vs_e63_stats.md
  - results/e71_vs_e70_stats.md

Statistical method (matches run_phase5_step5_stats.py):
  - Paired Wilcoxon signed-rank on per-molecule absolute errors;
    per-molecule error = median across 5 seeds.
  - Bootstrap 95% CI on (MAE_E71 - MAE_other), 5000 resamples.
  - Sign-counted wins/losses/ties.

Convention: E71 is the REFERENCE. mae_diff = MAE(E71) - MAE(other).
  - Negative mae_diff  => E71 has lower MAE (E71 better).
  - Positive mae_diff  => E71 has higher MAE (other better).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


E71_PRED = PROJECT_ROOT / 'results' / 'e71_predictions.json'
BATCH1_PRED = PROJECT_ROOT / 'results' / 'pc2_batch1_multiseed_predictions.json'
PHASE5_PRED = PROJECT_ROOT / 'results' / 'pc2_phase5_ablation_predictions.json'

OUT_CSV_VS66 = PROJECT_ROOT / 'results' / 'e71_vs_e66_stats.csv'
OUT_MD_VS66 = PROJECT_ROOT / 'results' / 'e71_vs_e66_stats.md'
OUT_MD_VS63 = PROJECT_ROOT / 'results' / 'e71_vs_e63_stats.md'
OUT_MD_VS70 = PROJECT_ROOT / 'results' / 'e71_vs_e70_stats.md'

REFERENCE_EXP = 'E71'
TARGETS = ['loocv_c_hole', 'loocv_c_triplet']
N_BOOTSTRAP = 5000


def _load_predictions():
    out = {}
    for path in (E71_PRED, BATCH1_PRED, PHASE5_PRED):
        if not path.exists():
            print(f"WARN: missing {path}; skipping")
            continue
        d = json.loads(path.read_text())
        for exp_id, seed_map in d.items():
            out.setdefault(exp_id, {}).update(seed_map)
    return out


def _per_molecule_abs_errors(predictions, exp_id, target):
    seed_map = predictions.get(exp_id)
    if not seed_map:
        return None, None, None
    seeds = sorted(seed_map.keys())
    err_per_seed = []
    pred_per_seed = []
    y_true_ref = None
    for seed in seeds:
        blob = seed_map[seed].get(target)
        if not blob:
            continue
        yt = np.array(blob['y_true'], dtype=float)
        yp = np.array(blob['y_pred'], dtype=float)
        if y_true_ref is None:
            y_true_ref = yt
        err_per_seed.append(np.abs(yp - yt))
        pred_per_seed.append(yp)
    if not err_per_seed:
        return None, None, None
    err_stacked = np.vstack(err_per_seed)  # (n_seeds, n_mols)
    pred_stacked = np.vstack(pred_per_seed)
    median_err = np.median(err_stacked, axis=0)
    median_pred = np.median(pred_stacked, axis=0)
    return median_err, y_true_ref, median_pred


def _paired_test(err_ref, err_other):
    diff = err_ref - err_other
    n_eff = int((diff != 0).sum())
    out = {
        'n_total': int(len(diff)),
        'n_nonzero': n_eff,
        'mean_diff': float(diff.mean()),
        'median_diff': float(np.median(diff)),
        'ref_wins': int((diff < 0).sum()),    # ref has lower error (E71 better)
        'ref_loses': int((diff > 0).sum()),
        'ties': int((diff == 0).sum()),
    }
    if HAS_SCIPY and n_eff > 0:
        try:
            w_stat, p_val = wilcoxon(err_ref, err_other, zero_method='wilcox',
                                     alternative='two-sided')
            out['wilcoxon_stat'] = float(w_stat)
            out['wilcoxon_p'] = float(p_val)
        except ValueError as e:
            out['wilcoxon_stat'] = float('nan')
            out['wilcoxon_p'] = float('nan')
            out['wilcoxon_error'] = str(e)
    else:
        out['wilcoxon_stat'] = float('nan')
        out['wilcoxon_p'] = float('nan')
        if not HAS_SCIPY:
            out['wilcoxon_error'] = 'scipy not installed'
    return out


def _bootstrap_mae_diff(err_ref, err_other, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    n = len(err_ref)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = err_ref[idx].mean() - err_other[idx].mean()
    return {
        'mae_ref': float(err_ref.mean()),
        'mae_other': float(err_other.mean()),
        'mae_diff': float(err_ref.mean() - err_other.mean()),
        'mae_diff_ci_low': float(np.percentile(diffs, 2.5)),
        'mae_diff_ci_high': float(np.percentile(diffs, 97.5)),
        'n_bootstrap': n_boot,
    }


def _descriptive_stats(err, y_true, y_pred):
    """Compute MAE, RMSE, R² from median-across-seeds errors / predictions."""
    mae = float(err.mean())
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    ss_res = float(((y_pred - y_true) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return mae, rmse, r2


def _format_row_md(ref, other, bt, wt, m_ref, m_oth):
    sig = ''
    p = wt.get('wilcoxon_p', float('nan'))
    if not np.isnan(p):
        if p < 0.001: sig = '***'
        elif p < 0.01: sig = '**'
        elif p < 0.05: sig = '*'
    return (f"| {ref} vs {other} | "
            f"{m_ref['mae']:.4f} | {m_oth['mae']:.4f} | "
            f"{m_ref['rmse']:.4f} | {m_oth['rmse']:.4f} | "
            f"{m_ref['r2']:+.3f} | {m_oth['r2']:+.3f} | "
            f"{bt['mae_diff']:+.4f} ({bt['mae_diff_ci_low']:+.4f}, {bt['mae_diff_ci_high']:+.4f}) | "
            f"{wt['ref_wins']}/{wt['ref_loses']} | "
            f"{p:.4f} | {sig} |")


def _table_header():
    return [
        '| Comparison | MAE_ref | MAE_other | RMSE_ref | RMSE_other | '
        'R²_ref | R²_other | mae_diff (95% CI) | wins/loses | Wilcoxon p | sig |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]


def main():
    preds = _load_predictions()
    if REFERENCE_EXP not in preds:
        print(f"ERROR: {REFERENCE_EXP} predictions not found.")
        print(f"  Expected: {E71_PRED}")
        return 1
    for ex in ('E63', 'E66'):
        if ex not in preds:
            print(f"WARN: {ex} predictions missing; vs-{ex} comparison will be skipped.")

    csv_rows = []
    # CSV: store all three comparisons in one file (only --vs E66 is the primary)
    summary_lines = ['# E71 vs PC²-FedReorg / FedPer / no-calibration ablation\n']
    summary_lines.append('Reference: **E71** (FedPer + calibration only). '
                        'Per-molecule abs error = median across 5 seeds. ')
    summary_lines.append('Negative `mae_diff` => E71 has lower MAE than the comparison '
                        '(E71 better).')
    summary_lines.append('Wilcoxon signed-rank: two-sided, paired on molecule index. '
                        f'Bootstrap CI: {N_BOOTSTRAP} resamples.\n')

    for compare_against, out_md_path in [
        ('E66', OUT_MD_VS66),
        ('E63', OUT_MD_VS63),
        ('E70', OUT_MD_VS70),
    ]:
        if compare_against not in preds:
            print(f"Skipping E71 vs {compare_against}: predictions missing.")
            continue
        md = [f'# E71 vs {compare_against}\n']
        contrast = {
            'E66': 'PC²-FedReorg (full architecture)',
            'E63': 'FedPer legacy baseline (no calibration)',
            'E70': 'PC² with adapter + T-gate but WITHOUT calibration',
        }[compare_against]
        md.append(f'Reference: **E71** = FedPer + calibration only. Compared to: '
                  f'**{compare_against}** = {contrast}.\n')
        md.append('| Target | n | MAE_E71 | MAE_other | RMSE_E71 | RMSE_other | '
                  'R²_E71 | R²_other | ΔMAE (E71-other; 95% CI) | wins/loses | '
                  'Wilcoxon p | sig |')
        md.append('|---|---|---|---|---|---|---|---|---|---|---|---|')

        for target in TARGETS:
            err_ref, yt_ref, yp_ref = _per_molecule_abs_errors(preds, REFERENCE_EXP, target)
            err_other, yt_other, yp_other = _per_molecule_abs_errors(preds, compare_against, target)
            if err_ref is None or err_other is None:
                continue
            if len(err_ref) != len(err_other):
                print(f"WARN: length mismatch for {target}: {len(err_ref)} vs {len(err_other)}")
                continue

            wt = _paired_test(err_ref, err_other)
            bt = _bootstrap_mae_diff(err_ref, err_other)
            mae_r, rmse_r, r2_r = _descriptive_stats(err_ref, yt_ref, yp_ref)
            mae_o, rmse_o, r2_o = _descriptive_stats(err_other, yt_other, yp_other)
            n_mols = len(err_ref)

            sig = ''
            p = wt.get('wilcoxon_p', float('nan'))
            if not np.isnan(p):
                if p < 0.001: sig = '***'
                elif p < 0.01: sig = '**'
                elif p < 0.05: sig = '*'
            md.append(
                f"| {target} | {n_mols} | "
                f"{mae_r:.4f} | {mae_o:.4f} | "
                f"{rmse_r:.4f} | {rmse_o:.4f} | "
                f"{r2_r:+.3f} | {r2_o:+.3f} | "
                f"{bt['mae_diff']:+.4f} ({bt['mae_diff_ci_low']:+.4f}, {bt['mae_diff_ci_high']:+.4f}) | "
                f"{wt['ref_wins']}/{wt['ref_loses']} | "
                f"{p:.4f} | {sig} |"
            )

            if compare_against == 'E66':
                csv_rows.append({
                    'target': target,
                    'reference': REFERENCE_EXP,
                    'compared_to': compare_against,
                    'n_mols': n_mols,
                    'mae_ref': mae_r, 'mae_other': mae_o,
                    'rmse_ref': rmse_r, 'rmse_other': rmse_o,
                    'r2_ref': r2_r, 'r2_other': r2_o,
                    'mae_diff': bt['mae_diff'],
                    'mae_diff_ci_low': bt['mae_diff_ci_low'],
                    'mae_diff_ci_high': bt['mae_diff_ci_high'],
                    'wilcoxon_stat': wt.get('wilcoxon_stat', float('nan')),
                    'wilcoxon_p': p,
                    'ref_wins': wt['ref_wins'], 'ref_loses': wt['ref_loses'],
                    'ties': wt['ties'],
                })

        md.append('')
        md.append('## Notes\n')
        md.append('- `wins`/`loses`: number of molecules where E71 has '
                  'lower / higher abs error than the comparison.')
        md.append('- significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 '
                  '(unadjusted nominal). Bonferroni threshold for 3 comparisons '
                  'would be α/3 ≈ 0.017.')
        md.append('- All comparisons are PAIRED on the same molecule set.')
        md.append('- Per-molecule errors aggregated as the MEDIAN across the 5 seeds.')

        out_md_path.write_text('\n'.join(md))
        print(f"Wrote {out_md_path}")

    if csv_rows:
        fields = list(csv_rows[0].keys())
        with OUT_CSV_VS66.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"Wrote {OUT_CSV_VS66} ({len(csv_rows)} rows)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
