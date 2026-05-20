#!/usr/bin/env python3
"""Step 2 of the data-provenance audit.

Independently recomputes every numerical claim in the paper from the raw
prediction JSONs and summary CSVs. Does NOT import from src/, experiments/,
plot_paper_figures.py, or paper/make_figures.py. Uses only numpy + scipy.

Outputs
-------
paper/audit/recomputed_numbers.csv          per (experiment, target) MAE/RMSE/R2,
                                            5-seed mean & SEM & SD,
                                            per-mol median-across-seeds MAE.
paper/audit/recomputed_pairwise_stats.csv   per (target, ref, other) paired stats:
                                            ΔMAE, bootstrap 95% CI (5000 resamples),
                                            wins/losses/ties (median estimand),
                                            paired Wilcoxon p-value.

Definitions used (consistent with the *.md notes shipped with results/):
- Per-seed metrics (LOOCV C): MAE = mean(|y_pred - y_true|), RMSE = sqrt(MSE),
  R2 computed against the seed's own residuals.
- 5-seed mean MAE: mean over seeds of per-seed MAE (matches summary_table).
- Per-molecule median-across-seeds absolute error e_i = median_s |y_pred_{s,i} - y_true_{s,i}|;
  this is the *estimand used for paired tests* (the stat .md files say so).
- Paired tests are between TWO methods on the SAME molecule set, using e_i values.
- Bootstrap CI: paired resampling of molecule indices (n=mols) with replacement,
  5000 resamples, percentile [2.5%, 97.5%] interval on mean(e_ref - e_other).
- Wilcoxon: scipy.stats.wilcoxon on (e_ref - e_other) with zero_method='wilcox'
  (default), two-sided.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sps

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "paper" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_SOURCES: list[pathlib.Path] = [
    RESULTS / "pc2_batch1_multiseed_predictions.json",   # E60, E61, E62, E63, E66
    RESULTS / "pc2_phase5_ablation_predictions.json",    # E65, E68, E69, E70
    RESULTS / "e71_predictions.json",                    # E71
]

LOOCV_TARGETS = ("loocv_c_hole", "loocv_c_triplet")

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260519  # fixed for reproducibility of the audit


# ---------------------------- IO ----------------------------

def load_all_predictions() -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Returns predictions[exp][seed][target] = {'y_true': np.ndarray,
    'y_pred': np.ndarray}.
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
    for p in PRED_SOURCES:
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open() as fh:
            d = json.load(fh)
        for exp, seeds in d.items():
            out.setdefault(exp, {})
            for seed, targets in seeds.items():
                out[exp].setdefault(seed, {})
                for tgt, arrs in targets.items():
                    if not isinstance(arrs, dict) or "y_true" not in arrs:
                        continue
                    out[exp][seed][tgt] = {
                        "y_true": np.asarray(arrs["y_true"], dtype=float),
                        "y_pred": np.asarray(arrs["y_pred"], dtype=float),
                    }
    return out


# -------------------------- metrics --------------------------

def _mae(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(yp - yt)))


def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# -------------------- aggregation helpers --------------------

def per_mol_median_abs_err(
    seed_preds: Dict[str, Dict[str, np.ndarray]],
    target: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-seed (|y_pred - y_true|) into (n_seeds, n_mols),
    take median across seeds → (n_mols,). Also returns the (common) y_true."""
    seeds_sorted = sorted(seed_preds.keys(), key=lambda s: int(s))
    abs_errs = []
    y_true_ref = None
    for s in seeds_sorted:
        if target not in seed_preds[s]:
            continue
        yt = seed_preds[s][target]["y_true"]
        yp = seed_preds[s][target]["y_pred"]
        if y_true_ref is None:
            y_true_ref = yt
        else:
            if not np.allclose(yt, y_true_ref, atol=1e-6):
                raise ValueError(
                    f"y_true mismatch across seeds for target {target}; "
                    "molecule ordering must be stable for paired tests."
                )
        abs_errs.append(np.abs(yp - yt))
    arr = np.vstack(abs_errs)  # (n_seeds, n_mols)
    return np.median(arr, axis=0), y_true_ref


def bootstrap_paired_ci(
    e_ref: np.ndarray,
    e_other: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Returns (mean_diff, ci_low, ci_high) for (e_ref - e_other) by paired
    bootstrap over molecule indices."""
    diff = e_ref - e_other
    n = diff.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------- Step 2a: scalars ----------------------

def write_recomputed_numbers(preds, out_path: pathlib.Path) -> None:
    rows: list[dict] = []
    for exp in sorted(preds.keys()):
        for tgt in LOOCV_TARGETS:
            seeds_with_tgt = [s for s in preds[exp]
                              if tgt in preds[exp][s]]
            if not seeds_with_tgt:
                continue
            mae_per_seed = []
            rmse_per_seed = []
            r2_per_seed = []
            for s in sorted(seeds_with_tgt, key=lambda x: int(x)):
                yt = preds[exp][s][tgt]["y_true"]
                yp = preds[exp][s][tgt]["y_pred"]
                mae_per_seed.append(_mae(yt, yp))
                rmse_per_seed.append(_rmse(yt, yp))
                r2_per_seed.append(_r2(yt, yp))
            mae_arr = np.asarray(mae_per_seed)
            rmse_arr = np.asarray(rmse_per_seed)
            r2_arr = np.asarray(r2_per_seed)
            # median-aggregated estimand
            e_med, _ = per_mol_median_abs_err(preds[exp], tgt)
            rows.append({
                "experiment": exp,
                "target": tgt,
                "n_seeds": len(mae_arr),
                "n_mols": int(preds[exp][sorted(seeds_with_tgt)[0]][tgt]["y_true"].size),
                "MAE_seed_mean": f"{mae_arr.mean():.6f}",
                "MAE_seed_sd": f"{mae_arr.std(ddof=1):.6f}" if len(mae_arr) > 1 else "nan",
                "MAE_seed_sem": f"{mae_arr.std(ddof=1) / np.sqrt(len(mae_arr)):.6f}" if len(mae_arr) > 1 else "nan",
                "RMSE_seed_mean": f"{rmse_arr.mean():.6f}",
                "RMSE_seed_sem": f"{rmse_arr.std(ddof=1) / np.sqrt(len(rmse_arr)):.6f}" if len(rmse_arr) > 1 else "nan",
                "R2_seed_mean": f"{r2_arr.mean():.6f}",
                "R2_seed_sem": f"{r2_arr.std(ddof=1) / np.sqrt(len(r2_arr)):.6f}" if len(r2_arr) > 1 else "nan",
                "MAE_perMolMedian": f"{e_med.mean():.6f}",  # equivalent to mean over molecules of median-across-seeds |err|
            })
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({len(rows)} rows)")


# ---------------------- Step 2b: paired stats ----------------------

# Pairs to recompute. ref vs other.
PAIRS: List[Tuple[str, str]] = [
    # E66 vs baselines (Table 3 main block)
    ("E66", "E60"), ("E66", "E61"), ("E66", "E62"), ("E66", "E63"),
    ("E66", "E65"), ("E66", "E68"), ("E66", "E69"), ("E66", "E70"),
    # E71 paired triad (E71 ref)
    ("E71", "E66"), ("E71", "E63"), ("E71", "E70"),
]


def write_pairwise_stats(preds, out_path: pathlib.Path) -> None:
    rows: list[dict] = []
    for tgt in LOOCV_TARGETS:
        for ref, other in PAIRS:
            if ref not in preds or other not in preds:
                continue
            if not any(tgt in preds[ref][s] for s in preds[ref]):
                continue
            if not any(tgt in preds[other][s] for s in preds[other]):
                continue
            e_ref, y_true_ref = per_mol_median_abs_err(preds[ref], tgt)
            e_oth, y_true_oth = per_mol_median_abs_err(preds[other], tgt)
            if e_ref.size != e_oth.size:
                raise ValueError(
                    f"size mismatch on {tgt} {ref} vs {other}: "
                    f"{e_ref.size} vs {e_oth.size}"
                )
            if not np.allclose(y_true_ref, y_true_oth, atol=1e-6):
                raise ValueError(
                    f"y_true mismatch between {ref} and {other} on {tgt}; "
                    "cannot perform paired test."
                )
            mae_ref = float(e_ref.mean())
            mae_oth = float(e_oth.mean())
            mae_diff, ci_lo, ci_hi = bootstrap_paired_ci(e_ref, e_oth)
            wins_ref = int(np.sum(e_ref < e_oth))
            loses_ref = int(np.sum(e_ref > e_oth))
            ties = int(np.sum(e_ref == e_oth))
            diff = e_ref - e_oth
            # scipy.stats.wilcoxon: skip the all-zero edge case
            if np.allclose(diff, 0):
                w_stat, p_val = float("nan"), float("nan")
            else:
                w_stat, p_val = sps.wilcoxon(e_ref, e_oth)
                w_stat = float(w_stat)
                p_val = float(p_val)
            rows.append({
                "target": tgt,
                "reference": ref,
                "compared_to": other,
                "n_mols": int(e_ref.size),
                "mae_ref_perMolMedian": f"{mae_ref:.6f}",
                "mae_other_perMolMedian": f"{mae_oth:.6f}",
                "mae_diff": f"{mae_diff:.6f}",
                "ci_low": f"{ci_lo:.6f}",
                "ci_high": f"{ci_hi:.6f}",
                "ref_wins": wins_ref,
                "ref_loses": loses_ref,
                "ties": ties,
                "wilcoxon_stat": f"{w_stat:.4f}" if not np.isnan(w_stat) else "nan",
                "wilcoxon_p": f"{p_val:.6f}" if not np.isnan(p_val) else "nan",
            })
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({len(rows)} rows)")


def main() -> int:
    preds = load_all_predictions()
    print(f"loaded predictions for experiments: {sorted(preds.keys())}")
    write_recomputed_numbers(preds, OUT_DIR / "recomputed_numbers.csv")
    write_pairwise_stats(preds, OUT_DIR / "recomputed_pairwise_stats.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
