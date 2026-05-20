"""Phase 5 Step 3 (E65) + Step 4 (E68/E69/E70) multi-seed driver.

E65 = D supervised pretrain -> C finetune (negative-control baseline).
E68 = PC² ablation: T_repr cross between C-hole and C-triplet zeroed (uses
      a pre-generated T_transferability_no_ccross.json).
E69 = PC² ablation: adapter + calibration ON but uniform FedAvg (no T-gating).
E70 = PC² ablation: pc2_fed + adapter ON, calibration OFF.

E68/E69/E70 dispatch through run_single_experiment using newly registered
EXPERIMENTS entries (run_all.py).
E65 has a custom D-pretrain -> C-finetune pipeline below.

Output (compatible with run_pc2_multiseed):
  results/pc2_phase5_ablation_summary.csv
  results/pc2_phase5_ablation_predictions.json

Does NOT modify any frozen PC² algorithm code.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs import (  # noqa: E402
    BATCH_SIZE_AB, KAN_GRID, LR_FINETUNE, N_EPOCHS_LOCAL, N_FOLDS_AB,
)
from experiments.run_all import run_single_experiment, setup_devices  # noqa: E402
from experiments.run_pc2_multiseed import (  # noqa: E402
    extract_predictions, extract_summary_rows,
    set_all_seeds, write_csv_atomic, write_json_atomic,
)
from src.data_utils import NODE_FEAT_DIM, prepare_all_clients  # noqa: E402
from src.models import ReorgEnergyModel  # noqa: E402
from src.train_eval import compute_metrics, evaluate, train_one_epoch  # noqa: E402


DEFAULT_SEEDS = [42, 123, 456, 789, 1000]
DEFAULT_EXPS = ['E65', 'E68', 'E69', 'E70']

CSV_OUT = PROJECT_ROOT / 'results' / 'pc2_phase5_ablation_summary.csv'
JSON_OUT = PROJECT_ROOT / 'results' / 'pc2_phase5_ablation_predictions.json'


# ============ E65 custom pipeline ============

def _train_d_pretrain(d_data, device, n_epochs=50, lr=1e-3, batch_size=512,
                       kan_grid=5, verbose=False) -> dict:
    """Stage 1 of E65: train ReorgEnergyModel on Client D's supervised labels.

    Returns the full state_dict (encoder + head) so it can be loaded as a
    starting point for Stage 2 fine-tuning. Calibration NOT used here -- this
    is the deliberate "naive supervised transfer" negative control.
    """
    model = ReorgEnergyModel(
        head_type='kan', in_dim=NODE_FEAT_DIM, kan_grid=kan_grid,
        encoder_out_dim=256, use_global_desc=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    loader = DataLoader(d_data, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(n_epochs):
        loss = train_one_epoch(model, loader, optimizer, device, criterion)
        scheduler.step()
        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == n_epochs - 1):
            print(f"    [D pretrain] epoch {ep+1}/{n_epochs}: loss={loss:.4f}")

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _loocv_finetune_from_state(c_data, pretrain_state, device, n_epochs=200,
                                lr=1e-4, kan_grid=5, verbose=False) -> dict:
    """Stage 2 of E65: per-fold LOOCV, loading D's pretrained state_dict
    (encoder + head) and fine-tuning on c_train, then predicting c_test."""
    n_c = len(c_data)
    y_true_all, y_pred_all = [], []
    criterion = nn.MSELoss()

    for fold_i in range(n_c):
        c_train = [c_data[j] for j in range(n_c) if j != fold_i]
        c_test = [c_data[fold_i]]

        model = ReorgEnergyModel(
            head_type='kan', in_dim=NODE_FEAT_DIM, kan_grid=kan_grid,
            encoder_out_dim=256, use_global_desc=True,
        ).to(device)
        # strict=False so we tolerate any minor key drift (none expected).
        model.load_state_dict({k: v.to(device) for k, v in pretrain_state.items()}, strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        c_bs = min(len(c_train), 64)
        train_loader = DataLoader(c_train, batch_size=c_bs, shuffle=True, drop_last=False)
        test_loader = DataLoader(c_test, batch_size=1, shuffle=False)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        for ep in range(n_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
            scheduler.step()
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 30:
                break
        if best_state is not None:
            model.load_state_dict(best_state)

        yt, yp = evaluate(model, test_loader, device)
        y_true_all.append(yt[0]); y_pred_all.append(yp[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['y_true'] = y_true_all
    metrics['y_pred'] = y_pred_all
    return metrics


def run_e65_for_seed(seed, data_hole, data_triplet, device_a, device_b, verbose=False):
    """E65 negative-control: D supervised pretrain -> C finetune.

    Returns a result dict with keys:
        loocv_c_hole, loocv_c_triplet, total_time
    """
    t_start = time.time()
    print(f"  [E65 seed={seed}] Stage 1: pretrain on Client D ({len(data_hole['client_d'])} mols)")
    set_all_seeds(seed)
    pretrain_state = _train_d_pretrain(
        data_hole['client_d'], device_a,
        n_epochs=50, lr=1e-3, batch_size=BATCH_SIZE_AB, kan_grid=KAN_GRID,
        verbose=verbose,
    )

    print(f"  [E65 seed={seed}] Stage 2a: C-hole LOOCV finetune")
    set_all_seeds(seed)
    res_hole = _loocv_finetune_from_state(
        data_hole['client_c'], pretrain_state, device_b,
        n_epochs=200, lr=LR_FINETUNE, kan_grid=KAN_GRID, verbose=verbose,
    )
    print(f"    C-hole MAE={res_hole['MAE']:.4f}, R²={res_hole['R2']:+.4f}")

    print(f"  [E65 seed={seed}] Stage 2b: C-triplet LOOCV finetune")
    set_all_seeds(seed)
    res_triplet = _loocv_finetune_from_state(
        data_triplet['client_c'], pretrain_state, device_b,
        n_epochs=200, lr=LR_FINETUNE, kan_grid=KAN_GRID, verbose=verbose,
    )
    print(f"    C-triplet MAE={res_triplet['MAE']:.4f}, R²={res_triplet['R2']:+.4f}")

    return {
        'exp_id': 'E65',
        'desc': 'D->C supervised pretrain (negative control)',
        'loocv_c_hole': res_hole,
        'loocv_c_triplet': res_triplet,
        'total_time': time.time() - t_start,
    }


# ============ Main driver ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default=','.join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument('--exps', type=str, default=','.join(DEFAULT_EXPS))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    exps = [e.strip() for e in args.exps.split(',') if e.strip()]

    print(f"Phase 5 Step 3+4: {len(seeds)} seeds x {len(exps)} exps "
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

            t_run = time.time()
            try:
                if exp_id == 'E65':
                    result = run_e65_for_seed(seed, data_hole, data_triplet,
                                               device_a, device_b,
                                               verbose=args.verbose)
                else:
                    set_all_seeds(seed)
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

            print(f"  elapsed: {(time.time() - t_run) / 60.0:.1f} min")
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
