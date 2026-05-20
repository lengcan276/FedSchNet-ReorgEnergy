"""Multi-seed batch runner for PC²-FedReorg Batch 1.

Sequentially runs each (exp_id, seed) combination through run_single_experiment,
sets torch / numpy / cuda seeds at the start of each run for reproducibility,
and saves:
  - results/pc2_batch1_multiseed_summary.csv  -- one row per (exp, seed, target)
  - results/pc2_batch1_multiseed_predictions.json -- per-LOOCV-molecule preds
                                                     for paired tests

Defaults: seeds = [42, 123, 456, 789, 1000], exps = [E60, E61, E62, E63, E66].
Override with --seeds / --exps if needed.

Does NOT modify any PC² algorithm code (frozen as of 2026-05-09).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_all import run_single_experiment, setup_devices  # noqa: E402
from src.data_utils import prepare_all_clients  # noqa: E402


DEFAULT_SEEDS = [42, 123, 456, 789, 1000]
DEFAULT_EXPS = ['E60', 'E61', 'E62', 'E63', 'E66']

CSV_OUT = PROJECT_ROOT / 'results' / 'pc2_batch1_multiseed_summary.csv'
JSON_OUT = PROJECT_ROOT / 'results' / 'pc2_batch1_multiseed_predictions.json'


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)


def _f(v):
    """Best-effort float coercion (json may have stored numpy as str)."""
    if v is None:
        return float('nan')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _list_or_scalar(v):
    """Return (mean, std) tuple if v is a [mean, std] list, else (scalar, nan)."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return _f(v[0]), _f(v[1])
    return _f(v), float('nan')


def extract_summary_rows(exp_id: str, seed: int, result: dict) -> list:
    """Return a list of CSV rows for this (exp, seed)."""
    rows = []
    cv_a = result.get('cv_a') or {}
    cv_b = result.get('cv_b') or {}
    ch = result.get('loocv_c_hole') or {}
    ct = result.get('loocv_c_triplet') or {}

    for target, blob in (('A_5fold', cv_a), ('B_5fold', cv_b)):
        if not blob:
            continue
        mae_m, mae_s = _list_or_scalar(blob.get('MAE'))
        rmse_m, rmse_s = _list_or_scalar(blob.get('RMSE'))
        r2_m, r2_s = _list_or_scalar(blob.get('R2'))
        rows.append(dict(
            exp_id=exp_id, seed=seed, target=target,
            MAE_mean=mae_m, MAE_std=mae_s,
            RMSE_mean=rmse_m, RMSE_std=rmse_s,
            R2_mean=r2_m, R2_std=r2_s, n_units='5_folds',
        ))

    for target, blob in (('C-hole', ch), ('C-triplet', ct)):
        if not blob:
            continue
        # LOOCV: scalar MAE/RMSE/R²; the "std" slot is NaN (no fold std for LOOCV).
        rows.append(dict(
            exp_id=exp_id, seed=seed, target=target,
            MAE_mean=_f(blob.get('MAE')), MAE_std=float('nan'),
            RMSE_mean=_f(blob.get('RMSE')), RMSE_std=float('nan'),
            R2_mean=_f(blob.get('R2')), R2_std=float('nan'),
            n_units=str(len(blob.get('y_true', []))) + '_LOOCV',
        ))
    return rows


def extract_predictions(result: dict) -> dict:
    """Per-molecule (y_true, y_pred) for LOOCV targets only."""
    out = {}
    for target_key in ('loocv_c_hole', 'loocv_c_triplet'):
        blob = result.get(target_key) or {}
        if 'y_true' not in blob or 'y_pred' not in blob:
            continue
        yt = blob['y_true']
        yp = blob['y_pred']
        # numpy arrays serialize as list-of-floats
        out[target_key] = {
            'y_true': [float(x) for x in (yt.tolist() if hasattr(yt, 'tolist') else list(yt))],
            'y_pred': [float(x) for x in (yp.tolist() if hasattr(yp, 'tolist') else list(yp))],
        }
    return out


def write_csv_atomic(rows: list, path: Path):
    """Write CSV atomically (tmp + rename)."""
    if not rows:
        return
    tmp = path.with_suffix('.tmp')
    fields = ['exp_id', 'seed', 'target',
              'MAE_mean', 'MAE_std', 'RMSE_mean', 'RMSE_std',
              'R2_mean', 'R2_std', 'n_units']
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    tmp.replace(path)


def write_json_atomic(payload: dict, path: Path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--exps', type=str, default=','.join(DEFAULT_EXPS))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    exps = [e.strip() for e in args.exps.split(',') if e.strip()]

    print(f"PC²-FedReorg multiseed run: {len(seeds)} seeds x {len(exps)} exps "
          f"= {len(seeds) * len(exps)} runs")
    print(f"Seeds: {seeds}")
    print(f"Exps:  {exps}")
    print()

    device_a, device_b = setup_devices()
    print(f"device_a={device_a}, device_b={device_b}")

    print('\n[Loading data once]')
    t0 = time.time()
    data_hole = prepare_all_clients(target_type='hole', include_d=True)
    data_triplet = prepare_all_clients(target_type='triplet', include_d=True)
    print(f"Data loaded in {time.time() - t0:.1f}s\n")

    all_rows = []
    all_preds = {}
    if CSV_OUT.exists():
        print(f"  (note: overwriting existing {CSV_OUT.name})")
    if JSON_OUT.exists():
        print(f"  (note: overwriting existing {JSON_OUT.name})")

    n_total = len(seeds) * len(exps)
    n_done = 0
    n_failed = 0
    overall_start = time.time()

    for seed in seeds:
        for exp_id in exps:
            n_done += 1
            elapsed_total = (time.time() - overall_start) / 60.0
            print(f"\n{'#' * 70}")
            print(f"# [{n_done}/{n_total}] seed={seed}  exp={exp_id}  "
                  f"(elapsed: {elapsed_total:.1f} min)")
            print(f"{'#' * 70}")

            set_all_seeds(seed)
            t_run = time.time()
            try:
                result = run_single_experiment(
                    exp_id, data_hole, data_triplet,
                    device_a, device_b, verbose=args.verbose,
                )
                rows = extract_summary_rows(exp_id, seed, result)
                preds = extract_predictions(result)
                all_rows.extend(rows)
                all_preds.setdefault(exp_id, {})[str(seed)] = preds
            except Exception:
                n_failed += 1
                print(f"!!! EXCEPTION in {exp_id} seed={seed}:")
                traceback.print_exc()
                # Continue to next run; don't kill the whole batch.

            print(f"  elapsed: {(time.time() - t_run) / 60.0:.1f} min")

            # Incremental save after each run so partial progress survives crashes.
            write_csv_atomic(all_rows, CSV_OUT)
            write_json_atomic(all_preds, JSON_OUT)

    total_min = (time.time() - overall_start) / 60.0
    print(f"\n{'=' * 70}")
    print(f"DONE: {n_done - n_failed}/{n_total} succeeded, {n_failed} failed.")
    print(f"Total time: {total_min:.1f} min")
    print(f"CSV:  {CSV_OUT}")
    print(f"JSON: {JSON_OUT}")
    print(f"{'=' * 70}")
    return 0 if n_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
