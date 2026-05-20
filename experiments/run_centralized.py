"""
E16: Centralized SchNet+KAN Upper Bound Experiment

Pools all 4 clients' data, performs LOOCV on Client C (TADF),
establishing the upper bound for comparison with federated E51/E52.

Usage:
    python experiments/run_centralized.py
    python experiments/run_centralized.py --target hole
    python experiments/run_centralized.py --target triplet
    python experiments/run_centralized.py --target both
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import prepare_all_clients_3d
from src.models import SchNetKANModel
from src.train_eval import compute_metrics, train_one_epoch, evaluate

RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_centralized_loocv(target_type: str = 'hole', device=None):
    """Pool all clients, LOOCV on Client C (TADF).

    Args:
        target_type: 'hole' or 'triplet'
        device: torch device

    Returns:
        dict with MAE, RMSE, R2, y_true, y_pred
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Centralized Upper Bound — LOOCV on Client C ({target_type})")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # 1. Load 3D data — NO normalization (centralized sees all data)
    data = prepare_all_clients_3d(target_type=target_type, include_d=True, normalize_y=False)

    # Code naming: client_c = paper Client D (TADF), client_d = paper Client C (Atahan)
    pool_abd = data['client_a'] + data['client_b'] + data['client_d']  # paper A+B+C
    tadf_data = data['client_c']  # paper Client D (TADF)

    n = len(tadf_data)
    print(f"\n  Pool (A+B+D): {len(pool_abd)} molecules")
    print(f"  TADF (C):     {n} molecules")
    print(f"  Total train per fold: ~{len(pool_abd) + n - 1} molecules")

    # Hyperparameters (same as E51/E52)
    hidden_channels = 256
    num_filters = 128
    num_interactions = 6
    cutoff = 7.5
    kan_grid = 5
    n_epochs = 100
    lr = 1e-3
    patience = 15
    batch_size = 64

    # 2. LOOCV
    y_true_all = []
    y_pred_all = []

    for i in tqdm(range(n), desc=f'  Centralized LOOCV ({target_type})'):
        test_data = [tadf_data[i]]
        train_data = pool_abd + [d for j, d in enumerate(tadf_data) if j != i]

        # Fresh model each fold
        model = SchNetKANModel(
            hidden_channels=hidden_channels, num_filters=num_filters,
            num_interactions=num_interactions, cutoff=cutoff,
            head_type='kan', kan_grid=kan_grid,
        ).to(device)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.MSELoss()

        best_train_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
            scheduler.step()

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_true, y_pred = evaluate(model, test_loader, device)
        y_true_all.append(y_true[0])
        y_pred_all.append(y_pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['y_true'] = y_true_all
    metrics['y_pred'] = y_pred_all

    print(f"\n  Centralized LOOCV ({target_type}): "
          f"MAE={metrics['MAE']:.4f} eV, R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Centralized SchNet+KAN Upper Bound')
    parser.add_argument('--target', choices=['hole', 'triplet', 'both'], default='both',
                        help='Target type for LOOCV (default: both)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    targets = ['hole', 'triplet'] if args.target == 'both' else [args.target]
    results = {}
    csv_rows = []

    for target in targets:
        t0 = time.time()
        metrics = run_centralized_loocv(target_type=target, device=device)
        elapsed = time.time() - t0
        results[target] = metrics
        csv_rows.append({
            'target': target,
            'MAE': f"{metrics['MAE']:.4f}",
            'R2': f"{metrics['R2']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}",
            'n_molecules': len(metrics['y_true']),
            'time_s': f"{elapsed:.1f}",
        })

    # Save CSV
    csv_path = RESULTS_DIR / 'centralized_upper_bound.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target', 'MAE', 'R2', 'RMSE', 'n_molecules', 'time_s'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved to: {csv_path}")

    # Comparison print
    print(f"\n{'='*60}")
    print(f"=== Centralized Upper Bound ===")
    print(f"{'='*60}")
    # Reference values from all_experiments_summary.csv
    print(f"  E50  Local SchNet D (hole):     R²=-0.543, MAE=0.389 eV")
    print(f"  E51  FedAvg SchNet 4-Cl (hole):  R²=+0.181, MAE=0.304 eV")
    print(f"  E52  FedPer SchNet 4-Cl (hole):  R²=+0.139, MAE=0.288 eV")
    if 'hole' in results:
        m = results['hole']
        print(f"  E16  Centralized SchNet (hole):  R²={m['R2']:+.3f}, MAE={m['MAE']:.3f} eV")
        print(f"  Gap (FedAvg vs Central): ΔR² = {m['R2'] - 0.181:+.3f}")
    if 'triplet' in results:
        m = results['triplet']
        print(f"\n  E50  Local SchNet D (triplet):     R²=-0.258, MAE=0.666 eV")
        print(f"  E51  FedAvg SchNet 4-Cl (triplet):  R²=-1.026, MAE=0.848 eV")
        print(f"  E52  FedPer SchNet 4-Cl (triplet):  R²=-0.439, MAE=0.681 eV")
        print(f"  E16  Centralized SchNet (triplet):  R²={m['R2']:+.3f}, MAE={m['MAE']:.3f} eV")
        print(f"  Gap (FedPer vs Central): ΔR² = {m['R2'] - (-0.439):+.3f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
