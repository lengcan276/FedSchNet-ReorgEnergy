"""Compute paired statistics for Phase 1 and Phase 2.

Independent of `paper/audit/recompute_all_reported_numbers.py` and of the
main-paper stats scripts. Reads per-method prediction JSONs written by the
two phase drivers and emits a pairwise-stats CSV per phase.

Paired estimand:
  per-target absolute error vector aggregated across seeds (median across
  seeds; matches the main-paper convention in
  `paper/audit/data_provenance_audit.md`).

For each comparison pair (ref, other):
  - ΔMAE  = mean(e_ref - e_other) (eV)
  - bootstrap 95% CI on ΔMAE (5000 resamples, fixed seed 20260520)
  - wins / losses / ties (paired molecule resolution)
  - paired Wilcoxon signed-rank p (scipy default)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import sys
from typing import Iterable

import numpy as np
from scipy import stats as sps

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PHASE1_DIR = PROJECT_ROOT / "results" / "calibration_generalization" / "label_scale_stress"
PHASE2_DIR = PROJECT_ROOT / "results" / "calibration_generalization" / "pseudo_task_validation"

BOOTSTRAP_SEED = 20260520
BOOTSTRAP_RESAMPLES = 5000


def _load_predictions_phase1(method: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stack per-seed predictions for one method on the single Phase-1 target.

    Returns (y_true_per_seed, abs_err_per_seed, seeds_present).
    Per-seed arrays are length n_target (= 100 in full run).
    """
    files = sorted(glob.glob(str(PHASE1_DIR / "predictions" / f"{method}_seed*.json")))
    if not files:
        return None, None, []
    seeds = []
    y_true_stack = []
    y_pred_stack = []
    for path in files:
        with open(path) as fh:
            d = json.load(fh)
        seeds.append(d["seed"])
        y_true_stack.append(np.asarray(d["y_true"], dtype=float))
        y_pred_stack.append(np.asarray(d["y_pred"], dtype=float))
    # Predictions for the same seed may differ in fold ordering across methods,
    # but each method's stacked-across-folds vector covers the same target
    # molecules in some order. For paired tests we use the *per-method* abs
    # error vector aligned by element index (same fold splits are produced
    # because the RNG seed is identical and the fold permutation is
    # generated in `run_kfold_on_target` deterministically from `seed`).
    y_true_arr = np.vstack(y_true_stack)          # (n_seeds, n_target)
    y_pred_arr = np.vstack(y_pred_stack)
    abs_err = np.abs(y_pred_arr - y_true_arr)     # (n_seeds, n_target)
    return y_true_arr, abs_err, seeds


def _paired_test(e_ref: np.ndarray, e_oth: np.ndarray, label: str) -> dict:
    """Paired stats on two equal-length abs-error vectors."""
    assert e_ref.shape == e_oth.shape
    diff = e_ref - e_oth
    mae_ref = float(e_ref.mean())
    mae_oth = float(e_oth.mean())
    mae_diff = float(diff.mean())
    # bootstrap CI
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = diff.size
    boots = np.empty(BOOTSTRAP_RESAMPLES)
    for b in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        boots[b] = diff[idx].mean()
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    # Wilcoxon
    if np.allclose(diff, 0):
        p = float("nan"); stat = float("nan")
    else:
        stat, p = sps.wilcoxon(e_ref, e_oth)
        stat = float(stat); p = float(p)
    return {
        "comparison": label,
        "n_pairs": int(n),
        "mae_ref": mae_ref,
        "mae_other": mae_oth,
        "mae_diff": mae_diff,
        "ci_low": ci_lo, "ci_high": ci_hi,
        "ref_wins": int(np.sum(e_ref < e_oth)),
        "ref_loses": int(np.sum(e_ref > e_oth)),
        "ties": int(np.sum(e_ref == e_oth)),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
    }


# ---------------------- Phase 1 ----------------------

def compute_phase1():
    methods = ["local", "fedavg_no_cal", "fedper", "fedper_cal"]
    method_e = {}
    for m in methods:
        _, abs_err, seeds = _load_predictions_phase1(m)
        if abs_err is None:
            print(f"[phase1] no predictions for {m}; skipping")
            continue
        # median across seeds → per-target per-mol abs error
        e_med = np.median(abs_err, axis=0)
        method_e[m] = e_med
        print(f"[phase1] {m}: median-aggregated MAE = {e_med.mean():.4f}, n_mol={e_med.size}, seeds={seeds}")
    if "fedper" not in method_e or "fedper_cal" not in method_e:
        print("[phase1] missing required methods; aborting")
        return

    pairs = [
        ("fedper_cal", "fedper"),
        ("fedper_cal", "fedavg_no_cal"),
        ("fedper_cal", "local"),
        ("fedper",     "fedavg_no_cal"),
        ("fedper",     "local"),
    ]
    rows = []
    for ref, oth in pairs:
        if ref not in method_e or oth not in method_e:
            continue
        r = _paired_test(method_e[ref], method_e[oth], f"{ref} vs {oth}")
        r["reference"] = ref; r["compared_to"] = oth
        rows.append(r)
    out_path = PHASE1_DIR / "pairwise_stats.csv"
    if rows:
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {out_path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")


# ---------------------- Phase 2 ----------------------

def compute_phase2():
    pred_files = sorted(glob.glob(str(PHASE2_DIR / "predictions" / "*.json")))
    if not pred_files:
        print("[phase2] no predictions directory; skipping")
        return

    # group by (task_id, method); each group has multiple seeds.
    grouped: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for path in pred_files:
        with open(path) as fh:
            d = json.load(fh)
        key = (d["task_id"], d["method"])
        grouped.setdefault(key, {})
        yt = np.asarray(d["y_true"]); yp = np.asarray(d["y_pred"])
        grouped[key][d["seed"]] = np.abs(yp - yt)

    # For each task, aggregate per-method abs-err across seeds → e_med per mol
    methods = ["local", "fedavg_no_cal", "fedper", "fedper_cal"]
    task_ids = sorted({k[0] for k in grouped})
    per_task_e: dict[str, dict[str, np.ndarray]] = {}
    for task in task_ids:
        per_task_e[task] = {}
        for m in methods:
            if (task, m) not in grouped: continue
            per_seed = grouped[(task, m)]
            stack = np.vstack([per_seed[s] for s in sorted(per_seed)])
            per_task_e[task][m] = np.median(stack, axis=0)

    # Comparison set per task
    pairs = [
        ("fedper_cal", "fedper"),
        ("fedper_cal", "fedavg_no_cal"),
        ("fedper_cal", "local"),
        ("fedper",     "fedavg_no_cal"),
    ]
    rows = []
    for task in task_ids:
        e_dict = per_task_e[task]
        for ref, oth in pairs:
            if ref not in e_dict or oth not in e_dict: continue
            r = _paired_test(e_dict[ref], e_dict[oth],
                              f"{ref} vs {oth}  [{task}]")
            r["task_id"] = task
            r["reference"] = ref; r["compared_to"] = oth
            rows.append(r)
    # write per-task pairwise stats
    out_path = PHASE2_DIR / "pairwise_stats.csv"
    if rows:
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {out_path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")

    # Aggregate: improvement rate of fedper_cal vs fedper
    cal_vs_fp = [r for r in rows if r["reference"] == "fedper_cal" and r["compared_to"] == "fedper"]
    if cal_vs_fp:
        improved = sum(1 for r in cal_vs_fp if r["mae_diff"] < 0)
        n = len(cal_vs_fp)
        median_d = float(np.median([r["mae_diff"] for r in cal_vs_fp]))
        q25 = float(np.percentile([r["mae_diff"] for r in cal_vs_fp], 25))
        q75 = float(np.percentile([r["mae_diff"] for r in cal_vs_fp], 75))
        print(f"[phase2] calibration vs FedPer: improved on {improved}/{n} tasks "
              f"(median ΔMAE = {median_d:+.4f} eV; IQR [{q25:+.4f}, {q75:+.4f}])")
        agg_path = PHASE2_DIR / "improvement_summary.csv"
        with agg_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["improved_tasks", "total_tasks",
                                               "median_delta_MAE", "q25", "q75"])
            w.writeheader()
            w.writerow({"improved_tasks": improved, "total_tasks": n,
                        "median_delta_MAE": median_d, "q25": q25, "q75": q75})
        print(f"wrote {agg_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "phase2", "both"),
                        default="both")
    args = parser.parse_args()
    if args.phase in ("phase1", "both"):
        compute_phase1()
    if args.phase in ("phase2", "both"):
        compute_phase2()


if __name__ == "__main__":
    main()
