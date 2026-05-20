"""Phase 5 Step 5: paired statistical tests on multiseed LOOCV outputs.

Reads:
  - results/pc2_batch1_multiseed_predictions.json (E60-E63, E66)
  - results/pc2_phase5_ablation_predictions.json  (E65, E68-E70)

For each LOOCV target (C-hole / C-triplet):
  - Aggregates per-molecule absolute errors across seeds (median across seeds).
  - Runs paired Wilcoxon signed-rank: E66 vs each baseline (E60/E61/E62/E63 +
    E65 negative-control + E68/E69/E70 ablations).
  - Bootstrap-based 95% CI on the MAE difference (E66 - baseline), resampling
    molecule indices.

Writes:
  - results/pc2_phase5_stats.csv
  - results/pc2_phase5_stats.md (human-readable summary)

Reports MAE / RMSE / R² as descriptive stats; the inferential tests use MAE
(per-molecule abs error) and median-bootstrap. R² is auxiliary (paper-claim
already noted near-zero R² is V2 'data ceiling').
"""
from __future__ import annotations

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


PRED_FILES = [
    PROJECT_ROOT / 'results' / 'pc2_batch1_multiseed_predictions.json',
    PROJECT_ROOT / 'results' / 'pc2_phase5_ablation_predictions.json',
]
OUT_CSV = PROJECT_ROOT / 'results' / 'pc2_phase5_stats.csv'
OUT_MD = PROJECT_ROOT / 'results' / 'pc2_phase5_stats.md'

REFERENCE_EXP = 'E66'
TARGETS = ['loocv_c_hole', 'loocv_c_triplet']
N_BOOTSTRAP = 5000


def _load_all_predictions() -> dict:
    """Merge per-experiment per-seed predictions from both JSON files."""
    out = {}
    for path in PRED_FILES:
        if not path.exists():
            print(f"WARN: missing {path}; skipping")
            continue
        d = json.loads(path.read_text())
        for exp_id, seed_map in d.items():
            out.setdefault(exp_id, {}).update(seed_map)
    return out


def _per_molecule_abs_errors(predictions: dict, exp_id: str, target: str) -> np.ndarray | None:
    """Return shape (n_mols,) of median-across-seeds absolute errors for the
    given (exp_id, target). y_true is identical across seeds (LOOCV is
    deterministic), so we use the first seed's y_true."""
    seed_map = predictions.get(exp_id)
    if not seed_map:
        return None
    seeds = sorted(seed_map.keys())
    err_per_seed = []
    y_true_ref = None
    for seed in seeds:
        blob = seed_map[seed].get(target)
        if not blob:
            continue
        yt = np.array(blob['y_true'], dtype=float)
        yp = np.array(blob['y_pred'], dtype=float)
        if y_true_ref is None:
            y_true_ref = yt
        elif not np.allclose(yt, y_true_ref, atol=1e-6):
            print(f"WARN: y_true differs between seeds for {exp_id}/{target}; "
                  "this should not happen with deterministic LOOCV ordering.")
        err_per_seed.append(np.abs(yp - yt))
    if not err_per_seed:
        return None
    err_stacked = np.vstack(err_per_seed)  # (n_seeds, n_mols)
    return np.median(err_stacked, axis=0)


def _paired_test(err_ref: np.ndarray, err_other: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank: H0 = no difference; H1 = difference.
    Reports also the sign-counted win/lose/tie of ref vs other."""
    diff = err_ref - err_other
    n_eff = int((diff != 0).sum())
    out = {
        'n_total': int(len(diff)),
        'n_nonzero': n_eff,
        'mean_diff': float(diff.mean()),
        'median_diff': float(np.median(diff)),
        'ref_wins': int((diff < 0).sum()),     # ref better (lower error)
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


def _bootstrap_mae_diff(err_ref: np.ndarray, err_other: np.ndarray,
                        n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Bootstrap CI on MAE(ref) - MAE(other). Negative => ref better (lower MAE)."""
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


def main():
    print(f"Reference: {REFERENCE_EXP}")
    print(f"Loading predictions from:")
    for p in PRED_FILES:
        print(f"  - {p} ({'exists' if p.exists() else 'MISSING'})")
    predictions = _load_all_predictions()
    if not predictions:
        print("\nNo predictions available. Run multiseed batches first.")
        return 1

    if REFERENCE_EXP not in predictions:
        print(f"\nReference experiment {REFERENCE_EXP} missing.")
        return 1

    rows = []
    md = ['# PC²-FedReorg Phase 5 statistical comparison\n']
    md.append(f'Reference: **{REFERENCE_EXP}** (PC²-FedReorg)\n')
    md.append('Per-molecule abs-error aggregated as median across seeds. ')
    md.append('Negative `mae_diff` (= MAE_ref − MAE_other) means PC² has lower MAE.')
    md.append('')
    md.append(f'Wilcoxon signed-rank: paired, two-sided. Bootstrap CI on MAE diff '
              f'with N={N_BOOTSTRAP} resamples.')
    md.append('')

    for target in TARGETS:
        err_ref = _per_molecule_abs_errors(predictions, REFERENCE_EXP, target)
        if err_ref is None:
            print(f"WARN: no errors for {REFERENCE_EXP}/{target}")
            continue
        n_mols = len(err_ref)
        md.append(f'## Target: `{target}` (n={n_mols} molecules)\n')
        md.append('| Comparison | MAE_ref | MAE_other | mae_diff (95% CI) | '
                  'wins/loses | Wilcoxon p | sig? |')
        md.append('|---|---|---|---|---|---|---|')

        for other_exp in sorted(predictions.keys()):
            if other_exp == REFERENCE_EXP:
                continue
            err_other = _per_molecule_abs_errors(predictions, other_exp, target)
            if err_other is None:
                continue
            if len(err_other) != n_mols:
                print(f"WARN: {other_exp}/{target} has {len(err_other)} mols, "
                      f"expected {n_mols}; skipping")
                continue

            wt = _paired_test(err_ref, err_other)
            bt = _bootstrap_mae_diff(err_ref, err_other)

            row = {
                'target': target, 'reference': REFERENCE_EXP, 'compared_to': other_exp,
                'n_mols': n_mols,
                'mae_ref': bt['mae_ref'], 'mae_other': bt['mae_other'],
                'mae_diff': bt['mae_diff'],
                'mae_diff_ci_low': bt['mae_diff_ci_low'],
                'mae_diff_ci_high': bt['mae_diff_ci_high'],
                'ref_wins': wt['ref_wins'], 'ref_loses': wt['ref_loses'],
                'wilcoxon_p': wt.get('wilcoxon_p', float('nan')),
            }
            rows.append(row)

            sig = ''
            p = wt.get('wilcoxon_p', float('nan'))
            if not np.isnan(p):
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
            md.append(
                f"| {REFERENCE_EXP} vs {other_exp} | "
                f"{bt['mae_ref']:.4f} | {bt['mae_other']:.4f} | "
                f"{bt['mae_diff']:+.4f} ({bt['mae_diff_ci_low']:+.4f}, {bt['mae_diff_ci_high']:+.4f}) | "
                f"{wt['ref_wins']}/{wt['ref_loses']} | "
                f"{p:.4f} | {sig} |"
            )
        md.append('')

    md.append('## Notes\n')
    md.append('- `wins/loses`: number of molecules where reference has '
              'lower / higher abs error than the comparison.')
    md.append('- significance: `*` p<0.05, `**` p<0.01, `***` p<0.001.')
    md.append('- All comparisons are PAIRED on the same molecule set.')
    md.append('- Per-molecule errors are the MEDIAN across the 5 seeds '
              '(robust to seed outliers).')

    OUT_MD.write_text('\n'.join(md))

    fields = ['target', 'reference', 'compared_to', 'n_mols',
              'mae_ref', 'mae_other', 'mae_diff',
              'mae_diff_ci_low', 'mae_diff_ci_high',
              'ref_wins', 'ref_loses', 'wilcoxon_p']
    import csv
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
