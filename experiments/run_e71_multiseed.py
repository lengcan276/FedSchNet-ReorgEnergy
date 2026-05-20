"""Multi-seed runner for E71 (FedPer + calibration-only, PC² ablation).

Isolated from existing PC²-FedReorg result files: writes results/e71_summary.csv
and results/e71_predictions.json. Does NOT modify any other E* outputs.

Default seeds: [42, 123, 456, 789, 1000] -- matches E60-E70 multiseed protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_all import run_single_experiment, setup_devices  # noqa: E402
from experiments.run_pc2_multiseed import (  # noqa: E402
    extract_summary_rows, extract_predictions,
    write_csv_atomic, write_json_atomic, set_all_seeds,
)
from src.data_utils import prepare_all_clients  # noqa: E402


DEFAULT_SEEDS = [42, 123, 456, 789, 1000]
EXP_ID = 'E71'

CSV_OUT = PROJECT_ROOT / 'results' / 'e71_summary.csv'
JSON_OUT = PROJECT_ROOT / 'results' / 'e71_predictions.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    print(f"E71 multiseed run: {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    print(f"CSV  -> {CSV_OUT}")
    print(f"JSON -> {JSON_OUT}")
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
    n_total = len(seeds)
    n_done = 0
    n_failed = 0
    overall_start = time.time()

    for seed in seeds:
        n_done += 1
        elapsed_total = (time.time() - overall_start) / 60.0
        print(f"\n{'#' * 70}")
        print(f"# [{n_done}/{n_total}] seed={seed}  exp={EXP_ID}  "
              f"(elapsed: {elapsed_total:.1f} min)")
        print(f"{'#' * 70}")

        set_all_seeds(seed)
        t_run = time.time()
        try:
            result = run_single_experiment(
                EXP_ID, data_hole, data_triplet,
                device_a, device_b, verbose=args.verbose,
            )
            rows = extract_summary_rows(EXP_ID, seed, result)
            preds = extract_predictions(result)
            all_rows.extend(rows)
            all_preds.setdefault(EXP_ID, {})[str(seed)] = preds
        except Exception:
            n_failed += 1
            print(f"!!! EXCEPTION in {EXP_ID} seed={seed}:")
            traceback.print_exc()

        print(f"  seed elapsed: {(time.time() - t_run) / 60.0:.1f} min")

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
