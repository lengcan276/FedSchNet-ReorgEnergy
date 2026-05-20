"""Minimal federated trainer for the calibration-generalization study.

Implements four methods on the pseudo-federation:

  * ``local``           — train the target client alone (no federation).
  * ``fedper``          — FedPer (encoder shared, head + bn private), labels native.
  * ``fedper_cal``      — FedPer + per-client (mu, sigma) calibration buffer.
  * ``fedavg_no_cal``   — FedAvg, no calibration (control: no scale absorption).

All methods reuse ``src.models.ReorgEnergyModel``, ``src.federated.*``,
``src.train_eval.train_one_epoch / evaluate / install_calibration_from_train``
WITHOUT modification. This file is the only training-side code path created
for the new experiments.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data_utils import NODE_FEAT_DIM
from src.federated import (
    fedavg_aggregate,
    fedper_aggregate,
    fedavg_strategy,
    distribute_params,
)
from src.models import ReorgEnergyModel
from src.train_eval import (
    train_one_epoch,
    evaluate,
    install_calibration_from_train,
    compute_metrics,
)

from experiments.calibration_generalization.pseudo_federation import PseudoClient


# ----------------------- seeding -----------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------- model factory -----------------------

def _new_model(use_calibration: bool, device: torch.device,
               kan_grid: int = 5) -> ReorgEnergyModel:
    """Fresh ReorgEnergyModel matching the main-paper encoder config.

    11-dim node features (same as Client A/B path in src/data_utils.py),
    256-dim encoder output, KAN head with grid_size=5 (the main-paper default
    for E63 / E66 / E70 / E71). Calibration toggled by argument.
    """
    m = ReorgEnergyModel(
        head_type='kan',
        in_dim=NODE_FEAT_DIM,
        kan_grid=kan_grid,
        encoder_out_dim=256,
        use_global_desc=True,
        use_physics_feat=False,
        phys_feats_dim=0,
        use_adapter=False,
        use_calibration=use_calibration,
    ).to(device)
    return m


# ----------------------- single-fold runner -----------------------

@dataclass
class FoldResult:
    mae: float
    rmse: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray
    elapsed_sec: float


def _train_and_eval_fold(
    *,
    clients: list[PseudoClient],
    target_idx: int,
    train_idx_in_target: np.ndarray,
    test_idx_in_target: np.ndarray,
    method: str,
    n_rounds: int,
    n_local_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> FoldResult:
    """Train a single federation under a single fold and return target-client
    test metrics."""
    t_start = time.time()
    _seed_everything(seed)

    target = clients[target_idx]
    target_train = [target.data_list[i] for i in train_idx_in_target]
    target_test = [target.data_list[i] for i in test_idx_in_target]

    if method == 'local':
        # No federation: train target on its own training fold.
        model = _new_model(use_calibration=False, device=device)
        if hasattr(model, 'calibration'):
            pass  # nothing to install; use_calibration=False
        loader_tr = DataLoader(target_train, batch_size=batch_size, shuffle=True)
        loader_te = DataLoader(target_test, batch_size=batch_size, shuffle=False)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = torch.nn.MSELoss()
        for _ in range(n_rounds * n_local_epochs):
            train_one_epoch(model, loader_tr, opt, device, crit, fedprox_reg=None)
        y_true, y_pred = evaluate(model, loader_te, device)
        m = compute_metrics(y_true, y_pred)
        return FoldResult(mae=m['MAE'], rmse=m['RMSE'], r2=m['R2'],
                          y_true=y_true, y_pred=y_pred,
                          elapsed_sec=time.time() - t_start)

    # --- federated methods below ---
    # Build a model per client. The target uses its training fold; non-target
    # clients use all of their data (analogous to A/B in the main paper).
    client_train_data: list[list] = []
    for i, c in enumerate(clients):
        if i == target_idx:
            client_train_data.append(target_train)
        else:
            client_train_data.append(c.data_list)

    use_cal = (method == 'fedper_cal')
    models = []
    for i, c in enumerate(clients):
        m = _new_model(use_calibration=use_cal, device=device)
        if use_cal:
            install_calibration_from_train(m, client_train_data[i])
        models.append(m)

    loaders = [DataLoader(client_train_data[i], batch_size=batch_size,
                          shuffle=True, drop_last=False) for i in range(len(clients))]
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    criterion = torch.nn.MSELoss()

    # Aggregation function selection.
    if method == 'fedper' or method == 'fedper_cal':
        aggregate_fn = fedper_aggregate
    elif method == 'fedavg_no_cal':
        aggregate_fn = fedavg_strategy   # universal calibration-key exclusion only
    else:
        raise ValueError(f"unknown method={method!r}")

    # Aggregation weights ∝ n_train per client.
    n_per_client = np.array([len(d) for d in client_train_data], dtype=np.float64)
    weights = (n_per_client / n_per_client.sum()).tolist()

    for round_idx in range(n_rounds):
        # Local epochs
        for client_i, (m, ld, opt) in enumerate(zip(models, loaders, optimizers)):
            for _ in range(n_local_epochs):
                train_one_epoch(m, ld, opt, device, criterion, fedprox_reg=None)
        # Aggregate + redistribute
        avg = aggregate_fn(models, weights)
        distribute_params(models, avg)

    # Evaluate on target's test fold (target's own model).
    target_model = models[target_idx]
    loader_te = DataLoader(target_test, batch_size=batch_size, shuffle=False)
    y_true, y_pred = evaluate(target_model, loader_te, device)
    m = compute_metrics(y_true, y_pred)
    return FoldResult(mae=m['MAE'], rmse=m['RMSE'], r2=m['R2'],
                      y_true=y_true, y_pred=y_pred,
                      elapsed_sec=time.time() - t_start)


# ----------------------- multi-fold runner -----------------------

def run_kfold_on_target(
    *,
    clients: list[PseudoClient],
    target_idx: int,
    method: str,
    n_folds: int = 5,
    n_rounds: int = 10,
    n_local_epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> dict:
    """K-fold CV on the target pseudo-client. Returns per-fold metrics
    and aggregated mean/std."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n = len(clients[target_idx])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_bounds = np.linspace(0, n, n_folds + 1, dtype=int)

    fold_results: list[FoldResult] = []
    for fk in range(n_folds):
        test_idx = perm[fold_bounds[fk]: fold_bounds[fk + 1]]
        train_idx = np.array([i for i in range(n) if i not in set(test_idx.tolist())])
        # Seed the fold deterministically (seed + fold index).
        fold_seed = seed * 100 + fk
        res = _train_and_eval_fold(
            clients=clients,
            target_idx=target_idx,
            train_idx_in_target=train_idx,
            test_idx_in_target=test_idx,
            method=method,
            n_rounds=n_rounds,
            n_local_epochs=n_local_epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            seed=fold_seed,
        )
        fold_results.append(res)

    y_true_all = np.concatenate([r.y_true for r in fold_results])
    y_pred_all = np.concatenate([r.y_pred for r in fold_results])

    per_fold = [{"fold": i, "mae": r.mae, "rmse": r.rmse, "r2": r.r2,
                 "n_test": len(r.y_true), "elapsed_sec": r.elapsed_sec}
                for i, r in enumerate(fold_results)]

    return {
        "method": method,
        "target": clients[target_idx].name,
        "seed": seed,
        "n_folds": n_folds,
        "per_fold": per_fold,
        "mae_mean": float(np.mean([r.mae for r in fold_results])),
        "mae_std": float(np.std([r.mae for r in fold_results], ddof=1)),
        "rmse_mean": float(np.mean([r.rmse for r in fold_results])),
        "rmse_std": float(np.std([r.rmse for r in fold_results], ddof=1)),
        "r2_mean": float(np.mean([r.r2 for r in fold_results])),
        "r2_std": float(np.std([r.r2 for r in fold_results], ddof=1)),
        "y_true_all": y_true_all.tolist(),
        "y_pred_all": y_pred_all.tolist(),
        "total_elapsed_sec": float(sum(r.elapsed_sec for r in fold_results)),
    }
