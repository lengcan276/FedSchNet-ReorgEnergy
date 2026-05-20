"""Phase 5 smoke check for E60 (Local-only) and E66 (PC²-FedReorg).

Runs a minimum-viable invocation of each:
  - E60: local LOOCV on Client C-hole / C-triplet, 2 epochs, 5 folds.
  - E66: federated_loocv_c_fast with PC² flags (use_adapter / use_calibration /
         aggregation_strategy='pc2_fed'), 2 federated rounds, 1 local epoch,
         2 finetune epochs, Client C trimmed to 5 molecules so LOOCV is 5
         folds, A/B/D trimmed to 200 each so the federation runs in minutes.

The goal is to verify Phase 4 wiring works end-to-end without crashing. We
do NOT report training metrics here -- silence (exit 0) is success.

Usage:
    python experiments/smoke_pc2.py --exp E60
    python experiments/smoke_pc2.py --exp E66
    python experiments/smoke_pc2.py --exp both
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data_utils import prepare_all_clients  # noqa: E402
from src.train_eval import (  # noqa: E402
    federated_loocv_c_fast, loocv_evaluate,
)


T_PATH = PROJECT_ROOT / 'results' / 'preverify' / 'T_transferability.json'


def setup_devices():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return torch.device('cuda:0'), torch.device(f'cuda:{min(1, n - 1)}')
    return torch.device('cpu'), torch.device('cpu')


def smoke_e60(client_data_hole, client_data_triplet, device_a, device_b):
    """E60 = Local-only on Client C-hole / C-triplet.

    Smokes 5 LOOCV folds × 2 epochs each. ~10 sec on GPU.
    """
    print("\n" + "=" * 60)
    print("E60 smoke: Local-only on Client C")
    print("=" * 60)
    c_hole = client_data_hole['client_c'][:5]
    c_triplet = client_data_triplet['client_c'][:5]
    t0 = time.time()
    res_hole = loocv_evaluate(
        c_hole, head_type='kan', n_epochs=2, lr=1e-3,
        device=device_b, kan_grid=5, verbose=False,
    )
    res_triplet = loocv_evaluate(
        c_triplet, head_type='kan', n_epochs=2, lr=1e-3,
        device=device_b, kan_grid=5, verbose=False,
    )
    elapsed = time.time() - t0
    print(f"E60 smoke: hole MAE={res_hole['MAE']:.4f}, triplet MAE={res_triplet['MAE']:.4f} "
          f"({elapsed:.1f}s)")
    print("E60 smoke OK")


def smoke_e66(client_data_hole, device_a, device_b):
    """E66 = PC²-FedReorg with full flags but tiny scale.

    Smokes 2 federated rounds × 1 local epoch + 5-fold LOOCV × 2 epochs.
    Trimmed data: A/B/D to 200 mols, C to 5. Verifies pc2_fed_dispatch is
    invoked and runs without shape mismatch.
    """
    print("\n" + "=" * 60)
    print("E66 smoke: PC²-FedReorg")
    print("=" * 60)
    if not T_PATH.exists():
        print(f"FATAL: {T_PATH} missing. Run `python -m src.transferability` first.")
        sys.exit(2)

    cd = {
        'client_a': client_data_hole['client_a'][:200],
        'client_b': client_data_hole['client_b'][:200],
        'client_c': client_data_hole['client_c'][:5],
    }
    if 'client_d' in client_data_hole and client_data_hole['client_d']:
        cd['client_d'] = client_data_hole['client_d'][:200]

    print(f"Smoke clients: " + ", ".join(f"{k}({len(v)})" for k, v in cd.items()))

    t0 = time.time()
    res = federated_loocv_c_fast(
        cd, head_type='kan', fed_strategy='fedavg',
        n_rounds=2, n_local_epochs=1, n_finetune_epochs=2,
        lr=1e-3, batch_size_ab=64,
        device_a=device_a, device_b=device_b,
        kan_grid=5, verbose=True,
        use_adapter=True, adapter_type='mlp',
        use_calibration=True,
        aggregation_strategy='pc2_fed',
        transferability_path=str(T_PATH),
        target_type='hole',
    )
    elapsed = time.time() - t0
    print(f"E66 smoke: C-hole MAE={res['MAE']:.4f} ({elapsed:.1f}s)")
    print("E66 smoke OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['E60', 'E66', 'both'], default='both')
    args = ap.parse_args()

    print(f"Smoke start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    device_a, device_b = setup_devices()
    print(f"device_a={device_a}, device_b={device_b}")

    print("\n[Loading data]")
    t0 = time.time()
    client_data_hole = prepare_all_clients(target_type='hole', include_d=True)
    client_data_triplet = prepare_all_clients(target_type='triplet', include_d=True)
    print(f"Data loaded in {time.time() - t0:.1f}s")

    try:
        if args.exp in ('E60', 'both'):
            smoke_e60(client_data_hole, client_data_triplet, device_a, device_b)
        if args.exp in ('E66', 'both'):
            smoke_e66(client_data_hole, device_a, device_b)
    except Exception:
        print("\nSMOKE FAILED with exception:")
        traceback.print_exc()
        sys.exit(1)

    print("\nAll smoke checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
