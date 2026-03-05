"""
Ablation studies.
- A1: KAN grid size ablation (3, 5, 8, 10)
- A2: SSL pretraining rounds ablation (0, 5, 10, 20, 40)
- A3: Client C data amount ablation (20%, 40%, 60%, 80%, 100%)
- A4: Communication rounds ablation (5, 10, 20, 30, 50, 80)
"""

import argparse
import copy
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs import (
    N_FED_ROUNDS, N_LOCAL_EPOCHS, N_SSL_ROUNDS,
    LR_FINETUNE, BATCH_SIZE_AB, PATIENCE, KAN_GRID,
)
from src.data_utils import prepare_all_clients, NODE_FEAT_DIM
from src.train_eval import federated_loocv_c_fast
from src.ssl_pretrain import federated_ssl_pretrain

RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_ablation_csv(results, filename):
    """Save ablation results to CSV."""
    import csv
    filepath = RESULTS_DIR / filename
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Ablation results saved: {filepath}")


def setup_devices():
    """Detect GPUs and assign devices."""
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device_a = torch.device('cuda:0')
        device_b = torch.device(f'cuda:{min(1, n_gpu - 1)}')
        print(f"GPU: {n_gpu}x {torch.cuda.get_device_name(0)}")
    else:
        device_a = device_b = torch.device('cpu')
        print("CPU mode")
    return device_a, device_b


def run_ablation_kan_grid(client_data_hole, client_data_triplet, device_a, device_b):
    """A1: KAN grid ablation (3, 5, 8, 10)"""
    print(f"\n{'#'*60}")
    print(f"# A1: KAN Grid Ablation")
    print(f"{'#'*60}")

    grids = [3, 5, 8, 10]
    results = []

    for g in grids:
        print(f"\n  --- grid={g} ---")
        r = federated_loocv_c_fast(
            client_data_hole, head_type='kan', fed_strategy='fedper',
            n_rounds=N_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            kan_grid=g, verbose=True,
        )
        results.append({'grid': g, 'MAE': r['MAE'], 'R2': r['R2'], 'RMSE': r['RMSE']})
        print(f"  grid={g}: MAE={r['MAE']:.4f}, R2={r['R2']:.4f}")

    return results


def run_ablation_ssl_rounds(client_data_hole, client_data_triplet, device_a, device_b):
    """A2: SSL pretraining rounds ablation (0, 5, 10, 20, 40)"""
    print(f"\n{'#'*60}")
    print(f"# A2: SSL Pretraining Rounds Ablation")
    print(f"{'#'*60}")

    ssl_rounds_list = [0, 5, 10, 20, 40]
    results = []

    for sr in ssl_rounds_list:
        print(f"\n  --- ssl_rounds={sr} ---")
        pretrained_encoder = None

        if sr > 0:
            encoder_state, _ = federated_ssl_pretrain(
                client_data_hole, ssl_method='atom_mask',
                n_rounds=sr, n_local_epochs=5,
                lr=1e-3, batch_size_ab=BATCH_SIZE_AB,
                device_a=device_a, device_b=device_b,
            )
            pretrained_encoder = encoder_state

        r = federated_loocv_c_fast(
            client_data_hole, head_type='kan', fed_strategy='fedper',
            n_rounds=N_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=pretrained_encoder,
            kan_grid=KAN_GRID, verbose=True,
        )
        results.append({'ssl_rounds': sr, 'MAE': r['MAE'], 'R2': r['R2'], 'RMSE': r['RMSE']})
        print(f"  ssl_rounds={sr}: MAE={r['MAE']:.4f}, R2={r['R2']:.4f}")

    return results


def run_ablation_data_amount(client_data_hole, client_data_triplet, device_a, device_b):
    """A3: Client C data amount ablation (20%, 40%, 60%, 80%, 100%)"""
    print(f"\n{'#'*60}")
    print(f"# A3: Client C Data Amount Ablation")
    print(f"{'#'*60}")

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    c_data = client_data_hole['client_c']
    n_total = len(c_data)
    results = []

    for ratio in ratios:
        n_use = max(3, int(n_total * ratio))
        print(f"\n  --- ratio={ratio:.0%} ({n_use}/{n_total} molecules) ---")

        torch.manual_seed(42)
        perm = torch.randperm(n_total)[:n_use].tolist()
        subset = [c_data[i] for i in perm]

        data_subset = copy.deepcopy(client_data_hole)
        data_subset['client_c'] = subset

        r = federated_loocv_c_fast(
            data_subset, head_type='kan', fed_strategy='fedper',
            n_rounds=N_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            kan_grid=KAN_GRID, verbose=True,
        )
        results.append({
            'ratio': ratio, 'n_molecules': n_use,
            'MAE': r['MAE'], 'R2': r['R2'], 'RMSE': r['RMSE'],
        })
        print(f"  ratio={ratio:.0%}: MAE={r['MAE']:.4f}, R2={r['R2']:.4f}")

    return results


def run_ablation_comm_rounds(client_data_hole, client_data_triplet, device_a, device_b):
    """A4: Communication rounds vs performance"""
    print(f"\n{'#'*60}")
    print(f"# A4: Communication Rounds Ablation")
    print(f"{'#'*60}")

    round_list = [5, 10, 20, 30, 50, 80]
    results = []

    for nr in round_list:
        print(f"\n  --- n_rounds={nr} ---")
        r = federated_loocv_c_fast(
            client_data_hole, head_type='kan', fed_strategy='fedper',
            n_rounds=nr, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            kan_grid=KAN_GRID, verbose=True,
        )
        results.append({'n_rounds': nr, 'MAE': r['MAE'], 'R2': r['R2'], 'RMSE': r['RMSE']})
        print(f"  n_rounds={nr}: MAE={r['MAE']:.4f}, R2={r['R2']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation Studies')
    parser.add_argument('--kan_grid', action='store_true', help='A1: KAN grid size ablation')
    parser.add_argument('--ssl_rounds', action='store_true', help='A2: SSL pretraining rounds ablation')
    parser.add_argument('--data_size', action='store_true', help='A3: Client C data amount ablation')
    parser.add_argument('--comm_rounds', action='store_true', help='A4: Communication rounds ablation')
    parser.add_argument('--all', action='store_true', help='Run all ablation studies')
    args = parser.parse_args()

    if not any([args.kan_grid, args.ssl_rounds, args.data_size, args.comm_rounds, args.all]):
        parser.print_help()
        print("\nAblation studies:")
        print("  A1: KAN grid size ablation (3, 5, 8, 10)")
        print("  A2: SSL pretraining rounds ablation (0, 5, 10, 20, 40)")
        print("  A3: Client C data amount ablation (20%, 40%, 60%, 80%, 100%)")
        print("  A4: Communication rounds ablation (5, 10, 20, 30, 50, 80)")
        return

    device_a, device_b = setup_devices()

    print("\nLoading data...")
    data_hole = prepare_all_clients(target_type='hole')
    data_triplet = prepare_all_clients(target_type='triplet')

    if args.kan_grid or args.all:
        a1 = run_ablation_kan_grid(data_hole, data_triplet, device_a, device_b)
        save_ablation_csv(a1, 'ablation_kan_grid.csv')

    if args.ssl_rounds or args.all:
        a2 = run_ablation_ssl_rounds(data_hole, data_triplet, device_a, device_b)
        save_ablation_csv(a2, 'ablation_ssl_rounds.csv')

    if args.data_size or args.all:
        a3 = run_ablation_data_amount(data_hole, data_triplet, device_a, device_b)
        save_ablation_csv(a3, 'ablation_data_size.csv')

    if args.comm_rounds or args.all:
        a4 = run_ablation_comm_rounds(data_hole, data_triplet, device_a, device_b)
        save_ablation_csv(a4, 'ablation_comm_rounds.csv')


if __name__ == '__main__':
    main()
