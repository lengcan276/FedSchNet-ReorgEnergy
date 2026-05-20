"""
Experiment matrix entry point
- E1-E21: Main experiments
- Batch 6: 4-client federation
- Batch 7: 3D conformer features
- Batch 8: SchNet 3D encoder
- Results saved to results/tables/ directory
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.configs import (
    N_FED_ROUNDS, N_LOCAL_EPOCHS, N_SSL_ROUNDS,
    LR_FINETUNE, BATCH_SIZE_AB, PATIENCE, KAN_GRID,
    N_FOLDS_AB, FEDPROX_MU,
)
from src.data_utils import (prepare_all_clients, augment_nodes_with_physics, normalize_phys_feats,
                        normalize_client_labels, NODE_FEAT_DIM, PHYSICS_FEAT_DIM, PHYS_FEATS_DIM,
                        CONF3D_FEATS_DIM, compute_3d_features_for_all, attach_3d_features,
                        print_3d_feature_stats)
from src.train_eval import (
    run_experiment, cv_evaluate, loocv_evaluate,
    train_federated, federated_loocv_c, federated_loocv_c_fast,
    compute_metrics, evaluate, _adapt_encoder_state_dict,
)
from src.ssl_pretrain import federated_ssl_pretrain

RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============ Experiment definitions ============

EXPERIMENTS = {
    # --- Batch 1: Local baselines ---
    'E1':  {'desc': 'Local_A GIN+MLP',            'head': 'mlp', 'fed': None,      'ssl': None},
    'E2':  {'desc': 'Local_B GIN+MLP',            'head': 'mlp', 'fed': None,      'ssl': None},
    'E3':  {'desc': 'Local_C GIN+MLP',            'head': 'mlp', 'fed': None,      'ssl': None},
    'E7':  {'desc': 'Local_A GIN+KAN',            'head': 'kan', 'fed': None,      'ssl': None},
    'E9':  {'desc': 'Local_C GIN+KAN',            'head': 'kan', 'fed': None,      'ssl': None},

    # --- Batch 2: Core federated comparison ---
    'E10': {'desc': 'FedAvg GIN+MLP',             'head': 'mlp', 'fed': 'fedavg',  'ssl': None},
    'E12': {'desc': 'FedPer GIN+MLP',             'head': 'mlp', 'fed': 'fedper',  'ssl': None},
    'E14': {'desc': 'FedPer GIN+KAN',             'head': 'kan', 'fed': 'fedper',  'ssl': None},
    'E15': {'desc': 'FedPer+FedBN GIN+KAN',       'head': 'kan', 'fed': 'fedbn',   'ssl': None},

    # --- Batch 3: SSL enhancement ---
    'E16': {'desc': 'FedPer+AtomMask GIN+KAN',    'head': 'kan', 'fed': 'fedper',  'ssl': 'atom_mask'},
    'E17': {'desc': 'FedPer+EdgePred GIN+KAN',    'head': 'kan', 'fed': 'fedper',  'ssl': 'edge_pred'},
    'E18': {'desc': 'FedPer+GraphCL GIN+KAN',     'head': 'kan', 'fed': 'fedper',  'ssl': 'graph_cl'},

    # --- Batch 4: Physics enhancement ---
    'E20': {'desc': 'E16+Physics',                'head': 'kan', 'fed': 'fedper',  'ssl': 'atom_mask', 'physics': True},
    'E21': {'desc': 'E16+HOMO/LUMO_node',         'head': 'kan', 'fed': 'fedper',  'ssl': 'atom_mask', 'node_physics': True},

    # ----- Batch 5: PC²-FedReorg (Phase 4 wired; not yet executed) -----
    # Constraint 8 (user spec): legacy fedper / fedbn semantics MUST be
    # preserved for E63; do NOT silently swap to fedper_pc2.
    'E60': {'desc': 'PC2 baseline: Local-only GIN+KAN',
            'head': 'kan', 'fed': None,      'ssl': None,
            'pc2': {}},
    'E61': {'desc': 'PC2 baseline: FedAvg GIN+KAN',
            'head': 'kan', 'fed': 'fedavg',  'ssl': None,
            'pc2': {}},
    'E62': {'desc': 'PC2 baseline: FedProx GIN+KAN (mu=0.01)',
            'head': 'kan', 'fed': 'fedavg_prox', 'ssl': None,
            'pc2': {}},
    'E63': {'desc': 'PC2 baseline: FedPer (legacy semantics) GIN+KAN',
            'head': 'kan', 'fed': 'fedper',  'ssl': None,
            'pc2': {}},
    # E65 D->C supervised pretrain (negative control). The "pretrain on D,
    # finetune on C" pipeline is not implemented in train_eval today; this
    # entry is registered as a placeholder for the Batch-1 result table and
    # will be wired in a follow-up. Constraint 9: D's head must NEVER enter
    # C in PC²; that's enforced by T_head[D, C-*] = 0 in Phase 1, not here.
    'E65': {'desc': 'PC2 negative control: D->C supervised pretrain (placeholder)',
            'head': 'kan', 'fed': 'fedper',  'ssl': None,
            'pc2': {'placeholder': True, 'pretrain_source': 'client_d',
                    'finetune_target': 'client_c'}},
    # ★ E66: PC²-FedReorg main entry. All PC² flags ON.
    # 'fed': 'fedavg' is a routing scaffold so run_single_experiment enters
    # the federated branch; the actual aggregation is overridden by
    # pc2_kwargs['aggregation_strategy']='pc2_fed' inside train_federated
    # (verified by smoke run on 2026-05-09).
    'E66': {'desc': 'PC2-FedReorg (ours)',
            'head': 'kan', 'fed': 'fedavg', 'ssl': None,
            'pc2': {'use_adapter': True,
                    'adapter_type': 'mlp',
                    'use_calibration': True,
                    'aggregation_strategy': 'pc2_fed',
                    'transferability_path': 'results/preverify/T_transferability.json'}},

    # ----- Phase 5 ablations (Step 4): probe individual PC² components -----
    # E68: T_repr cross between C-hole and C-triplet forced to floor (1e-3).
    # Tests whether C-triplet improvement comes from cross-quantity representation
    # sharing. All other PC² flags identical to E66.
    'E68': {'desc': 'PC2 ablation: T_repr no C-cross (no C-hole<->C-triplet repr share)',
            'head': 'kan', 'fed': 'fedavg', 'ssl': None,
            'pc2': {'use_adapter': True,
                    'adapter_type': 'mlp',
                    'use_calibration': True,
                    'aggregation_strategy': 'pc2_fed',
                    'transferability_path': 'results/preverify/T_transferability_no_ccross.json'}},

    # E69: same architecture as E66 (adapter + calibration ON) but uniform
    # FedAvg aggregation -- no T-gating. Tests whether E66 gains come from the
    # gate or from extra parameters / calibration alone.
    'E69': {'desc': 'PC2 ablation: adapter + calibration with uniform FedAvg (no T-gate)',
            'head': 'kan', 'fed': 'fedavg', 'ssl': None,
            'pc2': {'use_adapter': True,
                    'adapter_type': 'mlp',
                    'use_calibration': True,
                    # NO aggregation_strategy -> falls back to fed='fedavg'
                    }},

    # E70: PC² gate + adapter ON, calibration OFF. Tests calibration's
    # contribution to label-scale handling.
    'E70': {'desc': 'PC2 ablation: pc2_fed + adapter, no calibration',
            'head': 'kan', 'fed': 'fedavg', 'ssl': None,
            'pc2': {'use_adapter': True,
                    'adapter_type': 'mlp',
                    'use_calibration': False,
                    'aggregation_strategy': 'pc2_fed',
                    'transferability_path': 'results/preverify/T_transferability.json'}},

    # E71: FedPer + per-task-client calibration buffer. No adapter, no T-gate.
    # Purpose: tests whether PC²'s C-triplet gain over FedPer is fully
    # explained by the calibration buffer alone. Differs from E63 only in
    # use_calibration=True; differs from E66 in use_adapter=False AND no
    # aggregation_strategy ('fedper' routing handles encoder/head split).
    'E71': {'desc': 'PC2 ablation: FedPer + calibration only (no adapter, no T-gate)',
            'head': 'kan', 'fed': 'fedper', 'ssl': None,
            'pc2': {'use_adapter': False,
                    'use_calibration': True,
                    # NO aggregation_strategy -> uses fed='fedper' aggregation
                    # NO transferability_path -> uniform mask
                    }},
}

BATCHES = {
    '1': ['E1', 'E2', 'E3', 'E7', 'E9'],
    '2': ['E10', 'E12', 'E14', 'E15'],
    '3': ['E16', 'E17', 'E18'],
    '4': ['E20', 'E21'],
    # Batch 5 is the PC²-FedReorg main table (Phase 5 will run these; Phase 4
    # only registers them).
    '5': ['E60', 'E61', 'E62', 'E63', 'E65', 'E66'],
}


def setup_devices():
    """Detect GPUs and assign devices"""
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device_a = torch.device('cuda:0')
        device_b = torch.device(f'cuda:{min(1, n_gpu - 1)}')
        print(f"GPU: {n_gpu}x {torch.cuda.get_device_name(0)}")
    else:
        device_a = device_b = torch.device('cpu')
        print("CPU mode")
    return device_a, device_b


# ============ Run single experiment ============

def _extract_pc2_kwargs(exp_cfg: dict, target_type: str) -> dict:
    """Extract PC²-FedReorg keyword arguments to forward into train_federated /
    federated_loocv_c_fast.

    Returns an empty dict for legacy experiments (which do NOT have a 'pc2'
    entry, or whose 'pc2' entry is empty) -- preserving E1-E59 behavior.

    Phase 4 contract: defaults are off; only entries that explicitly set
    use_adapter / use_calibration / aggregation_strategy can opt in. The
    'placeholder' marker (used by E65) skips wiring entirely so the caller
    can route to a custom pretrain-then-finetune pipeline (not implemented
    in Phase 4).
    """
    pc2 = exp_cfg.get('pc2')
    if not pc2 or pc2.get('placeholder'):
        return {}
    kwargs = {}
    for key in ('use_adapter', 'adapter_type', 'adapter_bottleneck',
                'use_calibration', 'aggregation_strategy', 'transferability_path'):
        if key in pc2:
            kwargs[key] = pc2[key]
    # target_type comes from the per-call context (hole-side vs triplet-side).
    if pc2.get('aggregation_strategy') == 'pc2_fed' or pc2.get('use_calibration'):
        kwargs['target_type'] = target_type
    return kwargs


def run_single_experiment(
    exp_id: str,
    client_data_hole: dict,
    client_data_triplet: dict,
    device_a, device_b,
    verbose: bool = True,
) -> dict:
    """Run a single experiment and return results"""
    exp_cfg = EXPERIMENTS[exp_id]
    head_type = exp_cfg['head']
    fed_strategy = exp_cfg['fed']
    ssl_method = exp_cfg['ssl']
    use_physics = exp_cfg.get('physics', False)
    use_node_physics = exp_cfg.get('node_physics', False)
    pc2_marker = exp_cfg.get('pc2', {}) or {}

    print(f"\n{'#'*60}")
    print(f"# {exp_id}: {exp_cfg['desc']}")
    print(f"#   head={head_type}, fed={fed_strategy}, ssl={ssl_method}, "
          f"physics={use_physics}, node_physics={use_node_physics}")
    if pc2_marker.get('placeholder'):
        print(f"#   [placeholder] pc2={pc2_marker} -- not implemented in Phase 4")
        return {'exp_id': exp_id, 'desc': exp_cfg['desc'],
                'placeholder': True, 'pc2': pc2_marker}
    if pc2_marker:
        print(f"#   PC² flags: {pc2_marker}")
    print(f"{'#'*60}")

    start_time = time.time()
    result = {'exp_id': exp_id, 'desc': exp_cfg['desc']}

    # For local experiments, only evaluate specific Client
    if fed_strategy is None:
        # Local training
        if exp_id in ('E1', 'E7'):
            # Client A only
            result['cv_a'] = _run_local_cv(
                client_data_hole['client_a'], head_type, device_a,
                'Client A', verbose, use_physics=False,
            )
        elif exp_id == 'E2':
            # Client B only
            result['cv_b'] = _run_local_cv(
                client_data_hole['client_b'], head_type, device_b,
                'Client B', verbose, use_physics=False,
            )
        elif exp_id in ('E3', 'E9', 'E60'):
            # Client C only (hole LOOCV + triplet LOOCV).
            # E60 = "PC2 baseline: Local-only GIN+KAN" -- mirrors E3/E9 path
            # (local LOOCV, no federation, no PC² flags). The pc2 dict is
            # empty, so use_calibration / use_adapter stay off.
            result['loocv_c_hole'] = _run_local_loocv(
                client_data_hole['client_c'], head_type, device_b,
                'Client C hole', verbose, use_physics=use_physics,
            )
            result['loocv_c_triplet'] = _run_local_loocv(
                client_data_triplet['client_c'], head_type, device_b,
                'Client C triplet', verbose, use_physics=use_physics,
            )
    else:
        # Federated training
        pretrained_encoder = None

        # SSL pretraining (using original data, in_dim=11)
        if ssl_method is not None:
            print(f"\n  [SSL pretraining] method={ssl_method}")
            encoder_state, _ = federated_ssl_pretrain(
                client_data_hole, ssl_method=ssl_method,
                n_rounds=N_SSL_ROUNDS, n_local_epochs=5,
                lr=1e-3, batch_size_ab=BATCH_SIZE_AB,
                device_a=device_a, device_b=device_b,
            )
            pretrained_encoder = encoder_state

        # Node-level physics feature augmentation (E21)
        if use_node_physics:
            in_dim = NODE_FEAT_DIM + PHYSICS_FEAT_DIM
            print(f"\n  [Node physics augmentation] in_dim: {NODE_FEAT_DIM} -> {in_dim}")
            # Adapt SSL pretrained encoder weights: 11-dim -> 15-dim
            if pretrained_encoder is not None:
                pretrained_encoder = _adapt_encoder_state_dict(
                    pretrained_encoder, NODE_FEAT_DIM, in_dim)
            # Augment all Client data
            data_hole_fed = {
                **client_data_hole,
                'client_a': augment_nodes_with_physics(client_data_hole['client_a'], pad_zeros=True),
                'client_b': augment_nodes_with_physics(client_data_hole['client_b'], pad_zeros=True),
                'client_c': augment_nodes_with_physics(client_data_hole['client_c'], pad_zeros=False),
            }
            data_triplet_fed = {
                **client_data_triplet,
                'client_a': augment_nodes_with_physics(client_data_triplet['client_a'], pad_zeros=True),
                'client_b': augment_nodes_with_physics(client_data_triplet['client_b'], pad_zeros=True),
                'client_c': augment_nodes_with_physics(client_data_triplet['client_c'], pad_zeros=False),
            }
        else:
            in_dim = NODE_FEAT_DIM
            data_hole_fed = client_data_hole
            data_triplet_fed = client_data_triplet

        # Federated training + Client A CV
        print(f"\n  [Federated training + Client A CV]")
        # PC² flags (constraint 7): empty for E1-E59; populated for E66 etc.
        pc2_kwargs_hole = _extract_pc2_kwargs(exp_cfg, target_type='hole')
        fed_result = train_federated(
            data_hole_fed, head_type=head_type, fed_strategy=fed_strategy,
            n_rounds=N_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB, patience=PATIENCE,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=pretrained_encoder,
            kan_grid=KAN_GRID, use_physics_c=use_physics, in_dim=in_dim,
            mu=FEDPROX_MU, verbose=verbose,
            **pc2_kwargs_hole,
        )

        # Use federated trained encoder for CV
        fed_encoder_a = copy.deepcopy(fed_result['model_a'].encoder.state_dict())
        result['cv_a'] = cv_evaluate(
            data_hole_fed['client_a'], head_type=head_type,
            n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
            batch_size=BATCH_SIZE_AB, device=device_a,
            pretrained_encoder=fed_encoder_a, kan_grid=KAN_GRID,
            in_dim=in_dim, verbose=verbose,
        )

        fed_encoder_b = copy.deepcopy(fed_result['model_b'].encoder.state_dict())
        result['cv_b'] = cv_evaluate(
            data_hole_fed['client_b'], head_type=head_type,
            n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
            batch_size=BATCH_SIZE_AB, device=device_b,
            pretrained_encoder=fed_encoder_b, kan_grid=KAN_GRID,
            in_dim=in_dim, verbose=verbose,
        )

        # Client C LOOCV (hole) -- fast: federated train once + LOOCV only trains head
        print(f"\n  [Client C LOOCV - hole (fast)]")
        result['loocv_c_hole'] = federated_loocv_c_fast(
            data_hole_fed, head_type=head_type, fed_strategy=fed_strategy,
            n_rounds=N_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=pretrained_encoder,
            kan_grid=KAN_GRID, use_physics_c=use_physics, in_dim=in_dim,
            mu=FEDPROX_MU, verbose=verbose,
            **_extract_pc2_kwargs(exp_cfg, target_type='hole'),
        )

        # Client C LOOCV (triplet) -- fast
        print(f"\n  [Client C LOOCV - triplet (fast)]")
        result['loocv_c_triplet'] = federated_loocv_c_fast(
            data_triplet_fed, head_type=head_type, fed_strategy=fed_strategy,
            n_rounds=N_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=pretrained_encoder,
            kan_grid=KAN_GRID, use_physics_c=use_physics, in_dim=in_dim,
            mu=FEDPROX_MU, verbose=verbose,
            **_extract_pc2_kwargs(exp_cfg, target_type='triplet'),
        )

        result['fed_history'] = fed_result['history']

    result['total_time'] = time.time() - start_time
    return result


def _run_local_cv(data_list, head_type, device, name, verbose, use_physics=False):
    """Local CV helper function"""
    print(f"\n  [{name} CV]")
    return cv_evaluate(
        data_list, head_type=head_type,
        n_folds=N_FOLDS_AB, n_epochs=200, lr=LR_FINETUNE,
        batch_size=BATCH_SIZE_AB, device=device,
        kan_grid=KAN_GRID, verbose=verbose, use_physics=use_physics,
    )


def _run_local_loocv(data_list, head_type, device, name, verbose, use_physics=False):
    """Local LOOCV helper function"""
    print(f"\n  [{name} LOOCV]")
    return loocv_evaluate(
        data_list, head_type=head_type,
        n_epochs=200, lr=LR_FINETUNE, device=device,
        kan_grid=KAN_GRID, verbose=verbose, use_physics=use_physics,
    )


# ============ Results saving ============

def extract_metrics_row(exp_id: str, result: dict) -> dict:
    """Extract a CSV row from experiment results"""
    row = {
        'exp_id': exp_id,
        'desc': result.get('desc', ''),
        'MAE_A': '', 'R2_A': '',
        'MAE_B': '', 'R2_B': '',
        'MAE_Ch': '', 'R2_Ch': '',
        'MAE_Ct': '', 'R2_Ct': '',
        'time_s': f"{result.get('total_time', 0):.1f}",
    }

    if 'cv_a' in result:
        m = result['cv_a']
        row['MAE_A'] = f"{m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}"
        row['R2_A'] = f"{m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}"

    if 'cv_b' in result:
        m = result['cv_b']
        row['MAE_B'] = f"{m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}"
        row['R2_B'] = f"{m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}"

    if 'loocv_c_hole' in result:
        m = result['loocv_c_hole']
        row['MAE_Ch'] = f"{m['MAE']:.4f}"
        row['R2_Ch'] = f"{m['R2']:.4f}"

    if 'loocv_c_triplet' in result:
        m = result['loocv_c_triplet']
        row['MAE_Ct'] = f"{m['MAE']:.4f}"
        row['R2_Ct'] = f"{m['R2']:.4f}"

    # Compatibility with run_experiment() old format
    if 'loocv_c' in result and 'loocv_c_hole' not in result:
        m = result['loocv_c']
        row['MAE_Ch'] = f"{m['MAE']:.4f}"
        row['R2_Ch'] = f"{m['R2']:.4f}"

    return row


def save_results_csv(rows: list, filename: str = 'experiment_results.csv'):
    """Save experiment results to CSV (incremental append, deduplicate by exp_id keeping latest)"""
    filepath = RESULTS_DIR / filename
    fieldnames = ['exp_id', 'desc', 'MAE_A', 'R2_A', 'MAE_B', 'R2_B',
                  'MAE_Ch', 'R2_Ch', 'MAE_Ct', 'R2_Ct', 'time_s']

    # Read existing results
    existing = {}
    if filepath.exists():
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['exp_id']] = row

    # Overwrite with new results for same experiment IDs
    for row in rows:
        existing[row['exp_id']] = row

    # Sort by exp_id and write
    def sort_key(eid):
        num = ''.join(c for c in eid if c.isdigit())
        prefix = ''.join(c for c in eid if not c.isdigit())
        return (prefix, int(num) if num else 0)

    sorted_rows = [existing[k] for k in sorted(existing.keys(), key=sort_key)]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
    print(f"\nResults saved: {filepath}")


def save_ablation_csv(results: list, filename: str):
    """Save ablation experiment results"""
    filepath = RESULTS_DIR / filename
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Ablation results saved: {filepath}")


def save_full_results(all_results: dict, filename: str = 'experiment_results_full.json'):
    """Save full experiment results (incremental merge, deduplicate by exp_id keeping latest)"""
    filepath = RESULTS_DIR / filename

    # Read existing results and merge
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}

    # Serialization handling
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            s = {}
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    s[kk] = vv.tolist()
                elif isinstance(vv, dict):
                    s[kk] = {kkk: vvv.tolist() if isinstance(vvv, np.ndarray)
                             else list(vvv) if isinstance(vvv, tuple)
                             else vvv
                             for kkk, vvv in vv.items()
                             if not kkk.startswith('folds')}
                else:
                    s[kk] = vv
            serializable[k] = s
        else:
            serializable[k] = v

    # Merge: existing results + new results (new overwrites same keys)
    existing.update(serializable)

    with open(filepath, 'w') as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"Full results saved: {filepath}")


# ============ Phase 2: Ultimate verification experiments ============

def run_phase2_ultimate(client_data_hole, client_data_triplet, device_a, device_b):
    """Phase 2: Ultimate verification experiments -- physics feature enhancement + optimal hyperparams

    E_best_phys: GIN(out=300) + 5D physics features + KAN(grid=10), FedPer+FedBN, AtomMask 40 rounds, 30 fed rounds
    E_best_phys_graphcl: same as above, GraphCL 40 rounds
    """
    print(f"\n{'#'*60}")
    print(f"# Phase 2: Ultimate verification experiments")
    print(f"# Physics feature enhancement + optimal hyperparams (Grid=10, SSL=40 rounds, FedBN, 30 fed rounds)")
    print(f"{'#'*60}")

    # Phase 2 hyperparams (best from ablation)
    P2_ENCODER_OUT_DIM = 300
    P2_KAN_GRID = 10
    P2_SSL_ROUNDS = 40
    P2_FED_ROUNDS = 30
    P2_FED_STRATEGY = 'fedbn'  # FedPer + FedBN
    P2_LR = LR_FINETUNE

    # Step 1: StandardScaler normalization for phys_feats
    print("\n[Step 1] StandardScaler normalization for phys_feats")
    normalize_phys_feats([
        client_data_hole['client_a'],
        client_data_hole['client_b'],
        client_data_hole['client_c'],
    ])
    normalize_phys_feats([
        client_data_triplet['client_a'],
        client_data_triplet['client_b'],
        client_data_triplet['client_c'],
    ])

    csv_rows = []
    all_results = {}

    for exp_id, ssl_method, desc in [
        ('E_best_phys', 'atom_mask',
         'GIN300+Phys5+KAN(g10), FedBN, AtomMask40'),
        ('E_best_phys_graphcl', 'graph_cl',
         'GIN300+Phys5+KAN(g10), FedBN, GraphCL40'),
    ]:
        print(f"\n{'#'*60}")
        print(f"# {exp_id}: {desc}")
        print(f"# encoder_out_dim={P2_ENCODER_OUT_DIM}, phys_feats_dim={PHYS_FEATS_DIM}, "
              f"grid={P2_KAN_GRID}")
        print(f"# ssl={ssl_method} {P2_SSL_ROUNDS} rounds, fed={P2_FED_STRATEGY} {P2_FED_ROUNDS} rounds")
        print(f"{'#'*60}")

        start_time = time.time()
        result = {'exp_id': exp_id, 'desc': desc}

        # Step 2: SSL pretraining (standard encoder, out_dim=300, 40 rounds)
        print(f"\n  [SSL pretraining] {ssl_method} {P2_SSL_ROUNDS} rounds, encoder_out_dim={P2_ENCODER_OUT_DIM}")
        encoder_state, _ = federated_ssl_pretrain(
            client_data_hole, ssl_method=ssl_method,
            n_rounds=P2_SSL_ROUNDS, n_local_epochs=5,
            lr=1e-3, batch_size_ab=BATCH_SIZE_AB,
            out_dim=P2_ENCODER_OUT_DIM,
            device_a=device_a, device_b=device_b,
        )

        # Step 3: Federated training + Client A/B CV
        print(f"\n  [Federated training] {P2_FED_STRATEGY} {P2_FED_ROUNDS} rounds")
        fed_result = train_federated(
            client_data_hole, head_type='kan', fed_strategy=P2_FED_STRATEGY,
            n_rounds=P2_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=P2_LR, batch_size_ab=BATCH_SIZE_AB, patience=PATIENCE,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=encoder_state,
            kan_grid=P2_KAN_GRID, in_dim=NODE_FEAT_DIM,
            phys_feats_dim=PHYS_FEATS_DIM, encoder_out_dim=P2_ENCODER_OUT_DIM,
            verbose=True,
        )

        # Client A CV
        print(f"\n  [Client A CV]")
        fed_encoder_a = copy.deepcopy(fed_result['model_a'].encoder.state_dict())
        result['cv_a'] = cv_evaluate(
            client_data_hole['client_a'], head_type='kan',
            n_folds=N_FOLDS_AB, n_epochs=100, lr=P2_LR,
            batch_size=BATCH_SIZE_AB, device=device_a,
            pretrained_encoder=fed_encoder_a, kan_grid=P2_KAN_GRID,
            in_dim=NODE_FEAT_DIM, verbose=True,
            phys_feats_dim=PHYS_FEATS_DIM, encoder_out_dim=P2_ENCODER_OUT_DIM,
        )

        # Client B CV
        print(f"\n  [Client B CV]")
        fed_encoder_b = copy.deepcopy(fed_result['model_b'].encoder.state_dict())
        result['cv_b'] = cv_evaluate(
            client_data_hole['client_b'], head_type='kan',
            n_folds=N_FOLDS_AB, n_epochs=100, lr=P2_LR,
            batch_size=BATCH_SIZE_AB, device=device_b,
            pretrained_encoder=fed_encoder_b, kan_grid=P2_KAN_GRID,
            in_dim=NODE_FEAT_DIM, verbose=True,
            phys_feats_dim=PHYS_FEATS_DIM, encoder_out_dim=P2_ENCODER_OUT_DIM,
        )

        # Step 4: Client C LOOCV (hole) -- fast
        print(f"\n  [Client C LOOCV - hole (fast)]")
        result['loocv_c_hole'] = federated_loocv_c_fast(
            client_data_hole, head_type='kan', fed_strategy=P2_FED_STRATEGY,
            n_rounds=P2_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=P2_LR, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=encoder_state,
            kan_grid=P2_KAN_GRID, in_dim=NODE_FEAT_DIM,
            phys_feats_dim=PHYS_FEATS_DIM, encoder_out_dim=P2_ENCODER_OUT_DIM,
            verbose=True,
        )

        # Client C LOOCV (triplet) -- fast
        print(f"\n  [Client C LOOCV - triplet (fast)]")
        result['loocv_c_triplet'] = federated_loocv_c_fast(
            client_data_triplet, head_type='kan', fed_strategy=P2_FED_STRATEGY,
            n_rounds=P2_FED_ROUNDS, n_local_epochs=N_LOCAL_EPOCHS,
            lr=P2_LR, batch_size_ab=BATCH_SIZE_AB,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=encoder_state,
            kan_grid=P2_KAN_GRID, in_dim=NODE_FEAT_DIM,
            phys_feats_dim=PHYS_FEATS_DIM, encoder_out_dim=P2_ENCODER_OUT_DIM,
            verbose=True,
        )

        result['fed_history'] = fed_result['history']
        result['total_time'] = time.time() - start_time

        # Highlight Client C results
        print(f"\n{'='*60}")
        print(f"*** {exp_id} Results ***")
        print(f"{'='*60}")
        if 'cv_a' in result:
            m = result['cv_a']
            print(f"  Client A: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_b' in result:
            m = result['cv_b']
            print(f"  Client B: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'loocv_c_hole' in result:
            m = result['loocv_c_hole']
            print(f"  *** Client C (hole lambda):    MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        if 'loocv_c_triplet' in result:
            m = result['loocv_c_triplet']
            print(f"  *** Client C (triplet lambda): MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        print(f"  Elapsed: {result['total_time']:.1f}s")

        all_results[exp_id] = result
        csv_rows.append(extract_metrics_row(exp_id, result))

    # Save Phase 2 results (do not overwrite Phase 1 files)
    save_ablation_csv(csv_rows, 'phase2_ultimate_best.csv')
    save_full_results(all_results, 'phase2_ultimate_full.json')

    print(f"\n{'='*60}")
    print(f"Phase 2 complete! Results saved to:")
    print(f"  {RESULTS_DIR / 'phase2_ultimate_best.csv'}")
    print(f"  {RESULTS_DIR / 'phase2_ultimate_full.json'}")
    print(f"{'='*60}")

    return all_results


# ============ Batch 6: 4-Client federation + Per-Client Label Normalization ============

BATCH6_EXPERIMENTS = {
    'E30_Local_D_KAN': {
        'desc': 'Local_D GIN+KAN (5876 mol)',
        'local_only': 'D', 'head': 'kan', 'kan_grid': 5,
    },
    'E31_FedAvg_4Client': {
        'desc': 'FedAvg 4-Client GIN+KAN + y-norm',
        'fed_strategy': 'fedavg', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30, 'include_d': True,
    },
    'E32_FedBN_4Client': {
        'desc': 'FedBN 4-Client GIN+KAN + y-norm',
        'fed_strategy': 'fedbn', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30, 'include_d': True,
    },
    'E33_FedBN_SSL_4Client': {
        'desc': 'FedBN 4-Client GIN+KAN + AtomMask + y-norm',
        'fed_strategy': 'fedbn', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30, 'include_d': True,
        'ssl_method': 'atom_mask', 'ssl_rounds': 20,
    },
}


def run_batch6_4client(device_a, device_b, verbose: bool = True):
    """Batch 6: 4-Client federated experiments (Client A/B/C/D) with per-client label normalization

    Client D: Atahan-Evrenk 2019, 5876 conjugated organic semiconductors, hole lambda
    Per-client y normalization addresses the y-domain shift problem.
    """
    print(f"\n{'#'*60}")
    print(f"# Batch 6: 4-Client federated experiments + Per-Client Label Normalization")
    print(f"{'#'*60}")

    # Load data with Client D included and y normalization
    print("\n[Loading data: include_d=True, normalize_y=True]")
    data_hole = prepare_all_clients(target_type='hole', include_d=True, normalize_y=True)
    data_triplet = prepare_all_clients(target_type='triplet', include_d=True, normalize_y=True)

    norm_stats_hole = data_hole.get('norm_stats')
    norm_stats_triplet = data_triplet.get('norm_stats')

    if norm_stats_hole:
        print("\n[Per-client y normalization stats (hole)]:")
        for k, (m, s) in norm_stats_hole.items():
            print(f"  {k}: mean={m:.4f}, std={s:.4f}")
    if norm_stats_triplet:
        print("\n[Per-client y normalization stats (triplet)]:")
        for k, (m, s) in norm_stats_triplet.items():
            print(f"  {k}: mean={m:.4f}, std={s:.4f}")

    csv_rows = []
    all_results = {}

    for exp_id, cfg in BATCH6_EXPERIMENTS.items():
        print(f"\n{'#'*60}")
        print(f"# {exp_id}: {cfg['desc']}")
        print(f"{'#'*60}")

        start_time = time.time()
        result = {'exp_id': exp_id, 'desc': cfg['desc']}

        head_type = cfg['head']
        kan_grid = cfg.get('kan_grid', KAN_GRID)

        if 'local_only' in cfg:
            # ===== E30: Local D only =====
            client_key = f"client_{cfg['local_only'].lower()}"
            dev = device_a if cfg['local_only'] in ('A', 'D') else device_b

            print(f"\n  [Local {cfg['local_only']} CV]")
            result['cv_d'] = cv_evaluate(
                data_hole[client_key], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=200, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=dev,
                kan_grid=kan_grid, verbose=verbose,
            )
            # Denormalize E30 Local D results
            if norm_stats_hole and client_key in norm_stats_hole:
                d_std = norm_stats_hole[client_key][1]
                result['cv_d']['MAE'] = (result['cv_d']['MAE'][0] * d_std, result['cv_d']['MAE'][1] * d_std)
                result['cv_d']['RMSE'] = (result['cv_d']['RMSE'][0] * d_std, result['cv_d']['RMSE'][1] * d_std)
                print(f"  [Denormalize] {cfg['local_only']}: MAE={result['cv_d']['MAE'][0]:.4f} eV")
        else:
            # ===== E31-E33: 4-Client Federated =====
            fed_strategy = cfg['fed_strategy']
            n_rounds = cfg.get('n_rounds', 30)
            ssl_method = cfg.get('ssl_method')
            ssl_rounds = cfg.get('ssl_rounds', 20)

            pretrained_encoder = None

            # SSL pretraining
            if ssl_method is not None:
                print(f"\n  [SSL pretraining] {ssl_method} {ssl_rounds} rounds (4 clients)")
                encoder_state, _ = federated_ssl_pretrain(
                    data_hole, ssl_method=ssl_method,
                    n_rounds=ssl_rounds, n_local_epochs=5,
                    lr=1e-3, batch_size_ab=BATCH_SIZE_AB,
                    device_a=device_a, device_b=device_b,
                )
                pretrained_encoder = encoder_state

            # Federated training
            print(f"\n  [4-Client federated training] {fed_strategy}, {n_rounds} rounds")
            fed_result = train_federated(
                data_hole, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB, patience=PATIENCE,
                device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=norm_stats_hole,
            )

            # Client A CV (results in normalized space, need denormalization for MAE/RMSE)
            print(f"\n  [Client A CV]")
            fed_encoder_a = copy.deepcopy(fed_result['model_a'].encoder.state_dict())
            result['cv_a'] = cv_evaluate(
                data_hole['client_a'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=device_a,
                pretrained_encoder=fed_encoder_a, kan_grid=kan_grid,
                verbose=verbose,
            )
            if norm_stats_hole and 'client_a' in norm_stats_hole:
                a_std = norm_stats_hole['client_a'][1]
                result['cv_a']['MAE'] = (result['cv_a']['MAE'][0] * a_std, result['cv_a']['MAE'][1] * a_std)
                result['cv_a']['RMSE'] = (result['cv_a']['RMSE'][0] * a_std, result['cv_a']['RMSE'][1] * a_std)
                print(f"  [Denormalize] Client A: MAE={result['cv_a']['MAE'][0]:.4f} eV")

            # Client B CV
            print(f"\n  [Client B CV]")
            fed_encoder_b = copy.deepcopy(fed_result['model_b'].encoder.state_dict())
            result['cv_b'] = cv_evaluate(
                data_hole['client_b'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=device_b,
                pretrained_encoder=fed_encoder_b, kan_grid=kan_grid,
                verbose=verbose,
            )
            if norm_stats_hole and 'client_b' in norm_stats_hole:
                b_std = norm_stats_hole['client_b'][1]
                result['cv_b']['MAE'] = (result['cv_b']['MAE'][0] * b_std, result['cv_b']['MAE'][1] * b_std)
                result['cv_b']['RMSE'] = (result['cv_b']['RMSE'][0] * b_std, result['cv_b']['RMSE'][1] * b_std)
                print(f"  [Denormalize] Client B: MAE={result['cv_b']['MAE'][0]:.4f} eV")

            # Client D CV
            print(f"\n  [Client D CV]")
            fed_encoder_d = copy.deepcopy(fed_result['model_d'].encoder.state_dict())
            result['cv_d'] = cv_evaluate(
                data_hole['client_d'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=device_a,
                pretrained_encoder=fed_encoder_d, kan_grid=kan_grid,
                verbose=verbose,
            )
            if norm_stats_hole and 'client_d' in norm_stats_hole:
                d_std = norm_stats_hole['client_d'][1]
                result['cv_d']['MAE'] = (result['cv_d']['MAE'][0] * d_std, result['cv_d']['MAE'][1] * d_std)
                result['cv_d']['RMSE'] = (result['cv_d']['RMSE'][0] * d_std, result['cv_d']['RMSE'][1] * d_std)
                print(f"  [Denormalize] Client D: MAE={result['cv_d']['MAE'][0]:.4f} eV")

            # Client C LOOCV (hole) -- fast
            # NOTE: Do not pass norm_stats to avoid double normalization (data already normalized by prepare_all_clients)
            # LOOCV results are in normalized space, manually denormalize to eV afterwards
            print(f"\n  [Client C LOOCV - hole (fast, 4-client)]")
            result['loocv_c_hole'] = federated_loocv_c_fast(
                data_hole, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
                device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=None,
            )
            # Denormalize LOOCV results to eV
            if norm_stats_hole and 'client_c' in norm_stats_hole:
                c_mean, c_std = norm_stats_hole['client_c']
                yt = result['loocv_c_hole']['y_true'] * c_std + c_mean
                yp = result['loocv_c_hole']['y_pred'] * c_std + c_mean
                result['loocv_c_hole']['MAE'] = float(np.mean(np.abs(yt - yp)))
                result['loocv_c_hole']['RMSE'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                result['loocv_c_hole']['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
                result['loocv_c_hole']['y_true'] = yt
                result['loocv_c_hole']['y_pred'] = yp
                print(f"  [Denormalize] Client C hole: MAE={result['loocv_c_hole']['MAE']:.4f} eV, "
                      f"R2={result['loocv_c_hole']['R2']:.4f}")

            # Client C LOOCV (triplet) -- fast
            print(f"\n  [Client C LOOCV - triplet (fast, 4-client)]")
            result['loocv_c_triplet'] = federated_loocv_c_fast(
                data_triplet, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
                device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=None,
            )
            # Denormalize LOOCV results to eV
            if norm_stats_triplet and 'client_c' in norm_stats_triplet:
                c_mean, c_std = norm_stats_triplet['client_c']
                yt = result['loocv_c_triplet']['y_true'] * c_std + c_mean
                yp = result['loocv_c_triplet']['y_pred'] * c_std + c_mean
                result['loocv_c_triplet']['MAE'] = float(np.mean(np.abs(yt - yp)))
                result['loocv_c_triplet']['RMSE'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                result['loocv_c_triplet']['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
                result['loocv_c_triplet']['y_true'] = yt
                result['loocv_c_triplet']['y_pred'] = yp
                print(f"  [Denormalize] Client C triplet: MAE={result['loocv_c_triplet']['MAE']:.4f} eV, "
                      f"R2={result['loocv_c_triplet']['R2']:.4f}")

            result['fed_history'] = fed_result['history']

        result['total_time'] = time.time() - start_time

        # Print results
        print(f"\n{'='*60}")
        print(f"*** {exp_id} Results ***")
        print(f"{'='*60}")
        if 'cv_a' in result:
            m = result['cv_a']
            print(f"  Client A: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_b' in result:
            m = result['cv_b']
            print(f"  Client B: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_d' in result:
            m = result['cv_d']
            print(f"  Client D: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'loocv_c_hole' in result:
            m = result['loocv_c_hole']
            print(f"  *** Client C (hole lambda):    MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        if 'loocv_c_triplet' in result:
            m = result['loocv_c_triplet']
            print(f"  *** Client C (triplet lambda): MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        print(f"  Elapsed: {result['total_time']:.1f}s")

        all_results[exp_id] = result
        csv_rows.append(extract_metrics_row_batch6(exp_id, result))

    # Save (do not overwrite existing files)
    save_ablation_csv(csv_rows, 'batch6_4client.csv')
    save_full_results(all_results, 'batch6_4client_full.json')

    print(f"\n{'='*60}")
    print(f"Batch 6 complete! Results saved to:")
    print(f"  {RESULTS_DIR / 'batch6_4client.csv'}")
    print(f"  {RESULTS_DIR / 'batch6_4client_full.json'}")
    print(f"{'='*60}")

    return all_results


def extract_metrics_row_batch6(exp_id: str, result: dict) -> dict:
    """Batch 6: Extract CSV row (includes Client D column)"""
    row = {
        'exp_id': exp_id,
        'desc': result.get('desc', ''),
        'MAE_A': '', 'R2_A': '',
        'MAE_B': '', 'R2_B': '',
        'MAE_D': '', 'R2_D': '',
        'MAE_Ch': '', 'R2_Ch': '',
        'MAE_Ct': '', 'R2_Ct': '',
        'time_s': f"{result.get('total_time', 0):.1f}",
    }

    if 'cv_a' in result:
        m = result['cv_a']
        row['MAE_A'] = f"{m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}"
        row['R2_A'] = f"{m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}"

    if 'cv_b' in result:
        m = result['cv_b']
        row['MAE_B'] = f"{m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}"
        row['R2_B'] = f"{m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}"

    if 'cv_d' in result:
        m = result['cv_d']
        row['MAE_D'] = f"{m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}"
        row['R2_D'] = f"{m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}"

    if 'loocv_c_hole' in result:
        m = result['loocv_c_hole']
        row['MAE_Ch'] = f"{m['MAE']:.4f}"
        row['R2_Ch'] = f"{m['R2']:.4f}"

    if 'loocv_c_triplet' in result:
        m = result['loocv_c_triplet']
        row['MAE_Ct'] = f"{m['MAE']:.4f}"
        row['R2_Ct'] = f"{m['R2']:.4f}"

    return row


# ============ Batch 7: 3D conformer feature experiments ============

BATCH7_EXPERIMENTS = {
    'E40_FedBN_4Client_3D': {
        'desc': 'FedBN 4-Client GIN+KAN + 11D phys (5D+6D 3D)',
        'fed_strategy': 'fedbn', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30, 'ssl_method': None,
    },
    'E41_FedBN_SSL_4Client_3D': {
        'desc': 'FedBN 4-Client GIN+KAN + AtomMask + 11D phys',
        'fed_strategy': 'fedbn', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30, 'ssl_method': 'atom_mask', 'ssl_rounds': 20,
    },
    'E42_Local_C_3D': {
        'desc': 'Local_C GIN+KAN + 11D phys (LOOCV, no federation)',
        'local_only': 'C', 'head': 'kan', 'kan_grid': 5,
    },
}


def run_batch7_3d_features(device_a, device_b, verbose: bool = True):
    """Batch 7: 3D conformer feature enhancement experiments

    On top of Batch 6's 4-Client FedBN, add 6D ETKDG 3D conformer features to existing 5D phys_feats,
    totaling 11D phys_feats.
    """
    from scipy.stats import pearsonr

    print(f"\n{'#'*60}")
    print(f"# Batch 7: 3D conformer feature enhancement experiments")
    print(f"# 5D existing phys + 6D 3D conformer = 11D phys_feats")
    print(f"{'#'*60}")

    # 1. Load data (4 clients, y-normalized)
    print("\n[Loading data: include_d=True, normalize_y=True]")
    data_hole = prepare_all_clients(target_type='hole', include_d=True, normalize_y=True)
    data_triplet = prepare_all_clients(target_type='triplet', include_d=True, normalize_y=True)

    norm_stats_hole = data_hole.get('norm_stats')
    norm_stats_triplet = data_triplet.get('norm_stats')

    # 2. Collect all SMILES and compute 3D features
    print("\n[Computing 3D conformer features]")
    all_smiles = []
    for key in ['client_a', 'client_b', 'client_c', 'client_d']:
        for dataset in [data_hole, data_triplet]:
            if key in dataset and dataset[key]:
                all_smiles.extend([d.smiles for d in dataset[key] if hasattr(d, 'smiles')])
    all_smiles = list(set(all_smiles))
    print(f"  Total {len(all_smiles)} unique SMILES")

    features_3d = compute_3d_features_for_all(all_smiles)

    # 3. Concatenate 3D features to all client data
    print("\n[Concatenating 3D features to phys_feats]")
    for key in ['client_a', 'client_b', 'client_c', 'client_d']:
        for dataset in [data_hole, data_triplet]:
            if key in dataset and dataset[key]:
                attach_3d_features(dataset[key], features_3d)

    # 4. Normalize phys_feats (11D)
    print("\n[Normalizing 11D phys_feats]")
    hole_lists = [data_hole[k] for k in ['client_a', 'client_b', 'client_c', 'client_d']
                  if k in data_hole and data_hole[k]]
    triplet_lists = [data_triplet[k] for k in ['client_a', 'client_b', 'client_c', 'client_d']
                     if k in data_triplet and data_triplet[k]]
    normalize_phys_feats(hole_lists)
    normalize_phys_feats(triplet_lists)

    # 5. Print 3D feature statistics
    print("\n[3D feature statistics (raw values already normalized)]")
    for key in ['client_a', 'client_b', 'client_c', 'client_d']:
        if key in data_hole and data_hole[key]:
            print_3d_feature_stats(data_hole[key], f"{key} (hole)")

    # 6. Print Pearson correlation (3D features vs lambda)
    print(f"\n{'='*60}")
    print(f"Pearson correlation: 3D features vs reorganization energy")
    print(f"{'='*60}")
    feat_names_3d = ['n_rotatable', 'rmsd_mean', 'rmsd_max',
                     'energy_spread', 'max_dihedral', 'asphericity']

    for key in ['client_a', 'client_b', 'client_c', 'client_d']:
        if key not in data_hole or not data_hole[key]:
            continue
        ys = np.array([d.y.item() for d in data_hole[key]])
        feats = np.stack([d.phys_feats.squeeze(0).numpy() for d in data_hole[key]])
        # Last 6 columns are 3D features
        n_feats = feats.shape[1]
        print(f"\n  {key} ({len(ys)} mol, phys_feats_dim={n_feats}):")
        for i, name in enumerate(feat_names_3d):
            col_idx = n_feats - 6 + i  # Last 6 columns
            if col_idx < n_feats:
                try:
                    r, p = pearsonr(feats[:, col_idx], ys)
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    print(f"    {name:20s}: r={r:+.4f}, p={p:.4f} {sig}")
                except Exception:
                    print(f"    {name:20s}: computation failed")

    phys_feats_dim_total = PHYS_FEATS_DIM + CONF3D_FEATS_DIM  # 5 + 6 = 11

    # 7. Run experiments
    csv_rows = []
    all_results = {}

    for exp_id, cfg in BATCH7_EXPERIMENTS.items():
        print(f"\n{'#'*60}")
        print(f"# {exp_id}: {cfg['desc']}")
        print(f"{'#'*60}")

        start_time = time.time()
        result = {'exp_id': exp_id, 'desc': cfg['desc']}

        head_type = cfg['head']
        kan_grid = cfg.get('kan_grid', KAN_GRID)

        if 'local_only' in cfg:
            # ===== E42: Local C only LOOCV =====
            print(f"\n  [Local C LOOCV with 11D phys_feats]")

            # hole
            result['loocv_c_hole'] = loocv_evaluate(
                data_hole['client_c'], head_type=head_type,
                n_epochs=200, lr=LR_FINETUNE, device=device_b,
                kan_grid=kan_grid, verbose=verbose,
                phys_feats_dim=phys_feats_dim_total,
            )
            # Denormalize
            if norm_stats_hole and 'client_c' in norm_stats_hole:
                c_mean, c_std = norm_stats_hole['client_c']
                yt = result['loocv_c_hole']['y_true'] * c_std + c_mean
                yp = result['loocv_c_hole']['y_pred'] * c_std + c_mean
                from src.train_eval import compute_metrics as _cm
                metrics_ev = _cm(yt, yp)
                result['loocv_c_hole']['MAE'] = metrics_ev['MAE']
                result['loocv_c_hole']['RMSE'] = metrics_ev['RMSE']
                result['loocv_c_hole']['R2'] = metrics_ev['R2']
                result['loocv_c_hole']['y_true'] = yt
                result['loocv_c_hole']['y_pred'] = yp
                print(f"  [Denormalize] Client C hole: MAE={metrics_ev['MAE']:.4f} eV, R2={metrics_ev['R2']:.4f}")

            # triplet
            result['loocv_c_triplet'] = loocv_evaluate(
                data_triplet['client_c'], head_type=head_type,
                n_epochs=200, lr=LR_FINETUNE, device=device_b,
                kan_grid=kan_grid, verbose=verbose,
                phys_feats_dim=phys_feats_dim_total,
            )
            if norm_stats_triplet and 'client_c' in norm_stats_triplet:
                c_mean, c_std = norm_stats_triplet['client_c']
                yt = result['loocv_c_triplet']['y_true'] * c_std + c_mean
                yp = result['loocv_c_triplet']['y_pred'] * c_std + c_mean
                metrics_ev = _cm(yt, yp)
                result['loocv_c_triplet']['MAE'] = metrics_ev['MAE']
                result['loocv_c_triplet']['RMSE'] = metrics_ev['RMSE']
                result['loocv_c_triplet']['R2'] = metrics_ev['R2']
                result['loocv_c_triplet']['y_true'] = yt
                result['loocv_c_triplet']['y_pred'] = yp
                print(f"  [Denormalize] Client C triplet: MAE={metrics_ev['MAE']:.4f} eV, R2={metrics_ev['R2']:.4f}")

        else:
            # ===== E40/E41: 4-Client Federated with 3D features =====
            fed_strategy = cfg['fed_strategy']
            n_rounds = cfg.get('n_rounds', 30)
            ssl_method = cfg.get('ssl_method')
            ssl_rounds = cfg.get('ssl_rounds', 20)

            pretrained_encoder = None
            if ssl_method is not None:
                print(f"\n  [SSL pretraining] {ssl_method} {ssl_rounds} rounds (4 clients)")
                encoder_state, _ = federated_ssl_pretrain(
                    data_hole, ssl_method=ssl_method,
                    n_rounds=ssl_rounds, n_local_epochs=5,
                    lr=1e-3, batch_size_ab=BATCH_SIZE_AB,
                    device_a=device_a, device_b=device_b,
                )
                pretrained_encoder = encoder_state

            # Federated training
            print(f"\n  [4-Client federated training] {fed_strategy}, {n_rounds} rounds, phys_feats_dim={phys_feats_dim_total}")
            fed_result = train_federated(
                data_hole, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB, patience=PATIENCE,
                device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=norm_stats_hole,
                phys_feats_dim=phys_feats_dim_total,
            )

            # Client A CV
            print(f"\n  [Client A CV]")
            fed_encoder_a = copy.deepcopy(fed_result['model_a'].encoder.state_dict())
            result['cv_a'] = cv_evaluate(
                data_hole['client_a'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=device_a,
                pretrained_encoder=fed_encoder_a, kan_grid=kan_grid,
                verbose=verbose, phys_feats_dim=phys_feats_dim_total,
            )
            if norm_stats_hole and 'client_a' in norm_stats_hole:
                a_std = norm_stats_hole['client_a'][1]
                result['cv_a']['MAE'] = (result['cv_a']['MAE'][0] * a_std, result['cv_a']['MAE'][1] * a_std)
                result['cv_a']['RMSE'] = (result['cv_a']['RMSE'][0] * a_std, result['cv_a']['RMSE'][1] * a_std)
                print(f"  [Denormalize] Client A: MAE={result['cv_a']['MAE'][0]:.4f} eV")

            # Client B CV
            print(f"\n  [Client B CV]")
            fed_encoder_b = copy.deepcopy(fed_result['model_b'].encoder.state_dict())
            result['cv_b'] = cv_evaluate(
                data_hole['client_b'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=device_b,
                pretrained_encoder=fed_encoder_b, kan_grid=kan_grid,
                verbose=verbose, phys_feats_dim=phys_feats_dim_total,
            )
            if norm_stats_hole and 'client_b' in norm_stats_hole:
                b_std = norm_stats_hole['client_b'][1]
                result['cv_b']['MAE'] = (result['cv_b']['MAE'][0] * b_std, result['cv_b']['MAE'][1] * b_std)
                result['cv_b']['RMSE'] = (result['cv_b']['RMSE'][0] * b_std, result['cv_b']['RMSE'][1] * b_std)
                print(f"  [Denormalize] Client B: MAE={result['cv_b']['MAE'][0]:.4f} eV")

            # Client D CV
            print(f"\n  [Client D CV]")
            fed_encoder_d = copy.deepcopy(fed_result['model_d'].encoder.state_dict())
            result['cv_d'] = cv_evaluate(
                data_hole['client_d'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=LR_FINETUNE,
                batch_size=BATCH_SIZE_AB, device=device_a,
                pretrained_encoder=fed_encoder_d, kan_grid=kan_grid,
                verbose=verbose, phys_feats_dim=phys_feats_dim_total,
            )
            if norm_stats_hole and 'client_d' in norm_stats_hole:
                d_std = norm_stats_hole['client_d'][1]
                result['cv_d']['MAE'] = (result['cv_d']['MAE'][0] * d_std, result['cv_d']['MAE'][1] * d_std)
                result['cv_d']['RMSE'] = (result['cv_d']['RMSE'][0] * d_std, result['cv_d']['RMSE'][1] * d_std)
                print(f"  [Denormalize] Client D: MAE={result['cv_d']['MAE'][0]:.4f} eV")

            # Client C LOOCV (hole) -- fast
            print(f"\n  [Client C LOOCV - hole (fast, 4-client)]")
            result['loocv_c_hole'] = federated_loocv_c_fast(
                data_hole, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
                device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=None,
                phys_feats_dim=phys_feats_dim_total,
            )
            if norm_stats_hole and 'client_c' in norm_stats_hole:
                c_mean, c_std = norm_stats_hole['client_c']
                yt = result['loocv_c_hole']['y_true'] * c_std + c_mean
                yp = result['loocv_c_hole']['y_pred'] * c_std + c_mean
                result['loocv_c_hole']['MAE'] = float(np.mean(np.abs(yt - yp)))
                result['loocv_c_hole']['RMSE'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                result['loocv_c_hole']['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
                result['loocv_c_hole']['y_true'] = yt
                result['loocv_c_hole']['y_pred'] = yp
                print(f"  [Denormalize] Client C hole: MAE={result['loocv_c_hole']['MAE']:.4f} eV, "
                      f"R2={result['loocv_c_hole']['R2']:.4f}")

            # Client C LOOCV (triplet) -- fast
            print(f"\n  [Client C LOOCV - triplet (fast, 4-client)]")
            result['loocv_c_triplet'] = federated_loocv_c_fast(
                data_triplet, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=LR_FINETUNE, batch_size_ab=BATCH_SIZE_AB,
                device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=None,
                phys_feats_dim=phys_feats_dim_total,
            )
            if norm_stats_triplet and 'client_c' in norm_stats_triplet:
                c_mean, c_std = norm_stats_triplet['client_c']
                yt = result['loocv_c_triplet']['y_true'] * c_std + c_mean
                yp = result['loocv_c_triplet']['y_pred'] * c_std + c_mean
                result['loocv_c_triplet']['MAE'] = float(np.mean(np.abs(yt - yp)))
                result['loocv_c_triplet']['RMSE'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                result['loocv_c_triplet']['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
                result['loocv_c_triplet']['y_true'] = yt
                result['loocv_c_triplet']['y_pred'] = yp
                print(f"  [Denormalize] Client C triplet: MAE={result['loocv_c_triplet']['MAE']:.4f} eV, "
                      f"R2={result['loocv_c_triplet']['R2']:.4f}")

            result['fed_history'] = fed_result['history']

        result['total_time'] = time.time() - start_time

        # Print results
        print(f"\n{'='*60}")
        print(f"*** {exp_id} Results ***")
        print(f"{'='*60}")
        if 'cv_a' in result:
            m = result['cv_a']
            print(f"  Client A: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_b' in result:
            m = result['cv_b']
            print(f"  Client B: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_d' in result:
            m = result['cv_d']
            print(f"  Client D: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'loocv_c_hole' in result:
            m = result['loocv_c_hole']
            print(f"  *** Client C (hole lambda):    MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        if 'loocv_c_triplet' in result:
            m = result['loocv_c_triplet']
            print(f"  *** Client C (triplet lambda): MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        print(f"  Elapsed: {result['total_time']:.1f}s")

        all_results[exp_id] = result
        csv_rows.append(extract_metrics_row_batch6(exp_id, result))

    # Save
    save_ablation_csv(csv_rows, 'batch7_3d_features.csv')
    save_full_results(all_results, 'batch7_3d_features_full.json')

    print(f"\n{'='*60}")
    print(f"Batch 7 complete! Results saved to:")
    print(f"  {RESULTS_DIR / 'batch7_3d_features.csv'}")
    print(f"  {RESULTS_DIR / 'batch7_3d_features_full.json'}")
    print(f"{'='*60}")

    return all_results


# ============ Batch 8: SchNet 3D Encoder ============

BATCH8_EXPERIMENTS = {
    'E50_Local_C_SchNet': {
        'desc': 'Local_C SchNet+KAN (LOOCV, no federation)',
        'local_only': 'C', 'head': 'kan', 'kan_grid': 5,
    },
    'E51_FedAvg_SchNet': {
        'desc': 'FedAvg 4-Client SchNet+KAN',
        'fed_strategy': 'fedavg', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30,
    },
    'E52_FedPer_SchNet': {
        'desc': 'FedPer 4-Client SchNet+KAN',
        'fed_strategy': 'fedper', 'head': 'kan', 'kan_grid': 5,
        'n_rounds': 30,
    },
}


def run_batch8_schnet(device_a, device_b, verbose: bool = True):
    """Batch 8: SchNet 3D encoder experiments

    Uses true 3D coordinates (z, pos) instead of 2D graph (x, edge_index).
    SchNet learns distance-property relationships for reorganization energy.
    """
    from src.models import SchNetKANModel
    from src.data_utils import prepare_all_clients_3d

    print(f"\n{'#'*60}")
    print(f"# Batch 8: SchNet 3D Encoder Experiments")
    print(f"# True 3D coordinates -> SchNet -> KAN head")
    print(f"{'#'*60}")

    # 1. Load 3D data (4 clients, y-normalized)
    print("\n[Loading 3D data: include_d=True, normalize_y=True]")
    data_hole = prepare_all_clients_3d(target_type='hole', include_d=True, normalize_y=True)
    data_triplet = prepare_all_clients_3d(target_type='triplet', include_d=True, normalize_y=True)

    norm_stats_hole = data_hole.get('norm_stats')
    norm_stats_triplet = data_triplet.get('norm_stats')

    # 2. Print 3D data statistics
    print(f"\n{'='*60}")
    print(f"3D data statistics")
    print(f"{'='*60}")
    for key in ['client_a', 'client_b', 'client_c', 'client_d']:
        if key in data_hole and data_hole[key]:
            graphs = data_hole[key]
            n_atoms = [d.z.size(0) for d in graphs]
            print(f"  {key} (hole): {len(graphs)} mol, atoms: "
                  f"mean={np.mean(n_atoms):.1f}, max={max(n_atoms)}, min={min(n_atoms)}")

    # 3. Run experiments
    csv_rows = []
    all_results = {}

    for exp_id, cfg in BATCH8_EXPERIMENTS.items():
        print(f"\n{'#'*60}")
        print(f"# {exp_id}: {cfg['desc']}")
        print(f"{'#'*60}")

        start_time = time.time()
        result = {'exp_id': exp_id, 'desc': cfg['desc']}

        head_type = cfg['head']
        kan_grid = cfg.get('kan_grid', KAN_GRID)

        def make_schnet_factory():
            return lambda: SchNetKANModel(
                hidden_channels=256, num_filters=128,
                num_interactions=6, cutoff=7.5,
                head_type=head_type, kan_grid=kan_grid,
            )

        model_factory = make_schnet_factory()

        if 'local_only' in cfg:
            # ===== E50: Local C only LOOCV =====
            print(f"\n  [Local C LOOCV with SchNet]")

            # hole
            result['loocv_c_hole'] = loocv_evaluate(
                data_hole['client_c'], head_type=head_type,
                n_epochs=200, lr=1e-3, device=device_b,
                kan_grid=kan_grid, verbose=verbose,
                model_factory=model_factory,
            )
            # Denormalize
            if norm_stats_hole and 'client_c' in norm_stats_hole:
                c_mean, c_std = norm_stats_hole['client_c']
                yt = result['loocv_c_hole']['y_true'] * c_std + c_mean
                yp = result['loocv_c_hole']['y_pred'] * c_std + c_mean
                from src.train_eval import compute_metrics as _cm
                metrics_ev = _cm(yt, yp)
                result['loocv_c_hole']['MAE'] = metrics_ev['MAE']
                result['loocv_c_hole']['RMSE'] = metrics_ev['RMSE']
                result['loocv_c_hole']['R2'] = metrics_ev['R2']
                result['loocv_c_hole']['y_true'] = yt
                result['loocv_c_hole']['y_pred'] = yp
                print(f"  [Denormalize] Client C hole: MAE={metrics_ev['MAE']:.4f} eV, R2={metrics_ev['R2']:.4f}")

            # triplet
            result['loocv_c_triplet'] = loocv_evaluate(
                data_triplet['client_c'], head_type=head_type,
                n_epochs=200, lr=1e-3, device=device_b,
                kan_grid=kan_grid, verbose=verbose,
                model_factory=model_factory,
            )
            if norm_stats_triplet and 'client_c' in norm_stats_triplet:
                c_mean, c_std = norm_stats_triplet['client_c']
                yt = result['loocv_c_triplet']['y_true'] * c_std + c_mean
                yp = result['loocv_c_triplet']['y_pred'] * c_std + c_mean
                metrics_ev = _cm(yt, yp)
                result['loocv_c_triplet']['MAE'] = metrics_ev['MAE']
                result['loocv_c_triplet']['RMSE'] = metrics_ev['RMSE']
                result['loocv_c_triplet']['R2'] = metrics_ev['R2']
                result['loocv_c_triplet']['y_true'] = yt
                result['loocv_c_triplet']['y_pred'] = yp
                print(f"  [Denormalize] Client C triplet: MAE={metrics_ev['MAE']:.4f} eV, R2={metrics_ev['R2']:.4f}")

        else:
            # ===== E51/E52: 4-Client Federated SchNet =====
            fed_strategy = cfg['fed_strategy']
            n_rounds = cfg.get('n_rounds', 30)

            # Federated training (smaller batch for 3D data -- atoms include H)
            print(f"\n  [4-Client federated training] SchNet, {fed_strategy}, {n_rounds} rounds")
            fed_result = train_federated(
                data_hole, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=1e-3, batch_size_ab=64, patience=PATIENCE,
                device_a=device_a, device_b=device_b,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=norm_stats_hole,
                model_factory=model_factory,
            )

            # Client A CV
            print(f"\n  [Client A CV]")
            fed_encoder_a = copy.deepcopy(fed_result['model_a'].encoder.state_dict())
            result['cv_a'] = cv_evaluate(
                data_hole['client_a'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=1e-3,
                batch_size=64, device=device_a,
                kan_grid=kan_grid, verbose=verbose,
                model_factory=model_factory,
            )
            if norm_stats_hole and 'client_a' in norm_stats_hole:
                a_std = norm_stats_hole['client_a'][1]
                result['cv_a']['MAE'] = (result['cv_a']['MAE'][0] * a_std, result['cv_a']['MAE'][1] * a_std)
                result['cv_a']['RMSE'] = (result['cv_a']['RMSE'][0] * a_std, result['cv_a']['RMSE'][1] * a_std)
                print(f"  [Denormalize] Client A: MAE={result['cv_a']['MAE'][0]:.4f} eV")

            # Client B CV
            print(f"\n  [Client B CV]")
            result['cv_b'] = cv_evaluate(
                data_hole['client_b'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=1e-3,
                batch_size=64, device=device_b,
                kan_grid=kan_grid, verbose=verbose,
                model_factory=model_factory,
            )
            if norm_stats_hole and 'client_b' in norm_stats_hole:
                b_std = norm_stats_hole['client_b'][1]
                result['cv_b']['MAE'] = (result['cv_b']['MAE'][0] * b_std, result['cv_b']['MAE'][1] * b_std)
                result['cv_b']['RMSE'] = (result['cv_b']['RMSE'][0] * b_std, result['cv_b']['RMSE'][1] * b_std)
                print(f"  [Denormalize] Client B: MAE={result['cv_b']['MAE'][0]:.4f} eV")

            # Client D CV
            print(f"\n  [Client D CV]")
            result['cv_d'] = cv_evaluate(
                data_hole['client_d'], head_type=head_type,
                n_folds=N_FOLDS_AB, n_epochs=100, lr=1e-3,
                batch_size=64, device=device_a,
                kan_grid=kan_grid, verbose=verbose,
                model_factory=model_factory,
            )
            if norm_stats_hole and 'client_d' in norm_stats_hole:
                d_std = norm_stats_hole['client_d'][1]
                result['cv_d']['MAE'] = (result['cv_d']['MAE'][0] * d_std, result['cv_d']['MAE'][1] * d_std)
                result['cv_d']['RMSE'] = (result['cv_d']['RMSE'][0] * d_std, result['cv_d']['RMSE'][1] * d_std)
                print(f"  [Denormalize] Client D: MAE={result['cv_d']['MAE'][0]:.4f} eV")

            # Client C LOOCV (hole) -- fast
            print(f"\n  [Client C LOOCV - hole (fast, 4-client)]")
            result['loocv_c_hole'] = federated_loocv_c_fast(
                data_hole, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=1e-3, batch_size_ab=64,
                device_a=device_a, device_b=device_b,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=None,
                model_factory=model_factory,
            )
            if norm_stats_hole and 'client_c' in norm_stats_hole:
                c_mean, c_std = norm_stats_hole['client_c']
                yt = result['loocv_c_hole']['y_true'] * c_std + c_mean
                yp = result['loocv_c_hole']['y_pred'] * c_std + c_mean
                result['loocv_c_hole']['MAE'] = float(np.mean(np.abs(yt - yp)))
                result['loocv_c_hole']['RMSE'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                result['loocv_c_hole']['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
                result['loocv_c_hole']['y_true'] = yt
                result['loocv_c_hole']['y_pred'] = yp
                print(f"  [Denormalize] Client C hole: MAE={result['loocv_c_hole']['MAE']:.4f} eV, "
                      f"R2={result['loocv_c_hole']['R2']:.4f}")

            # Client C LOOCV (triplet) -- fast
            print(f"\n  [Client C LOOCV - triplet (fast, 4-client)]")
            result['loocv_c_triplet'] = federated_loocv_c_fast(
                data_triplet, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=N_LOCAL_EPOCHS,
                lr=1e-3, batch_size_ab=64,
                device_a=device_a, device_b=device_b,
                kan_grid=kan_grid, in_dim=NODE_FEAT_DIM,
                verbose=verbose, norm_stats=None,
                model_factory=model_factory,
            )
            if norm_stats_triplet and 'client_c' in norm_stats_triplet:
                c_mean, c_std = norm_stats_triplet['client_c']
                yt = result['loocv_c_triplet']['y_true'] * c_std + c_mean
                yp = result['loocv_c_triplet']['y_pred'] * c_std + c_mean
                result['loocv_c_triplet']['MAE'] = float(np.mean(np.abs(yt - yp)))
                result['loocv_c_triplet']['RMSE'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                result['loocv_c_triplet']['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
                result['loocv_c_triplet']['y_true'] = yt
                result['loocv_c_triplet']['y_pred'] = yp
                print(f"  [Denormalize] Client C triplet: MAE={result['loocv_c_triplet']['MAE']:.4f} eV, "
                      f"R2={result['loocv_c_triplet']['R2']:.4f}")

            result['fed_history'] = fed_result['history']

        result['total_time'] = time.time() - start_time

        # Print results
        print(f"\n{'='*60}")
        print(f"*** {exp_id} Results ***")
        print(f"{'='*60}")
        if 'cv_a' in result:
            m = result['cv_a']
            print(f"  Client A: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_b' in result:
            m = result['cv_b']
            print(f"  Client B: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_d' in result:
            m = result['cv_d']
            print(f"  Client D: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R2={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'loocv_c_hole' in result:
            m = result['loocv_c_hole']
            print(f"  *** Client C (hole lambda):    MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        if 'loocv_c_triplet' in result:
            m = result['loocv_c_triplet']
            print(f"  *** Client C (triplet lambda): MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
        print(f"  Elapsed: {result['total_time']:.1f}s")

        all_results[exp_id] = result
        csv_rows.append(extract_metrics_row_batch6(exp_id, result))

    # Save
    save_ablation_csv(csv_rows, 'batch8_schnet.csv')
    save_full_results(all_results, 'batch8_schnet_full.json')

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"Batch 8 complete! Results saved to:")
    print(f"  {RESULTS_DIR / 'batch8_schnet.csv'}")
    print(f"  {RESULTS_DIR / 'batch8_schnet_full.json'}")
    print(f"{'='*60}")

    # Print comparison with historical experiments
    print(f"\n{'='*60}")
    print(f"*** SchNet vs GIN comparison (Client C hole lambda) ***")
    print(f"{'='*60}")
    for eid in ['E50_Local_C_SchNet', 'E51_FedAvg_SchNet', 'E52_FedPer_SchNet']:
        if eid in all_results and 'loocv_c_hole' in all_results[eid]:
            m = all_results[eid]['loocv_c_hole']
            print(f"  {eid:30s}: MAE={m['MAE']:.4f}, R2={m['R2']:.4f}")
    print(f"  {'(ref: E15 GIN FedPer)':30s}: see historical results in results/ directory")
    print(f"  {'(ref: E32 GIN FedBN 4-Client)':30s}: see batch6_4client.csv")

    return all_results


# ============ Main entry point ============

def print_summary_table(rows: list):
    """Print summary table"""
    print(f"\n{'='*100}")
    print(f"Experiment results summary")
    print(f"{'='*100}")
    header = f"{'ID':5s} {'Description':30s} {'MAE_A':16s} {'R2_A':16s} {'MAE_Ch':10s} {'R2_Ch':10s} {'Time':8s}"
    print(header)
    print("-" * 100)
    for r in rows:
        line = (f"{r['exp_id']:5s} {r['desc']:30s} "
                f"{r.get('MAE_A', ''):16s} {r.get('R2_A', ''):16s} "
                f"{r.get('MAE_Ch', ''):10s} {r.get('R2_Ch', ''):10s} "
                f"{r.get('time_s', ''):8s}")
        print(line)
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description='Fed-GNN-KAN-Per experiment matrix')
    parser.add_argument('--batch1', action='store_true',
                        help='Run Batch 1 (local baselines)')
    parser.add_argument('--batch2', action='store_true',
                        help='Run Batch 2 (core federated)')
    parser.add_argument('--batch3', action='store_true',
                        help='Run Batch 3 (SSL enhancement)')
    parser.add_argument('--batch4', action='store_true',
                        help='Run Batch 4 (physics enhancement)')
    parser.add_argument('--batch6', action='store_true',
                        help='Run Batch 6 (4-client federation)')
    parser.add_argument('--batch7', action='store_true',
                        help='Run Batch 7 (3D features)')
    parser.add_argument('--batch8', action='store_true',
                        help='Run Batch 8 (SchNet)')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--exp', type=str, default=None,
                        help='Run specific experiments (e.g., E14,E15)')
    parser.add_argument('--phase2', action='store_true',
                        help='Run Phase 2 ultimate experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (reduced rounds for debugging)')
    parser.add_argument('--target', type=str, default='hole',
                        choices=['hole', 'triplet', 'both'],
                        help='Client C target type')
    args = parser.parse_args()

    device_a, device_b = setup_devices()

    # Quick mode: reduce parameters
    if args.quick:
        from experiments import configs
        configs.N_FED_ROUNDS = 5
        configs.N_LOCAL_EPOCHS = 2
        configs.N_SSL_ROUNDS = 3
        configs.N_FOLDS_AB = 2
        configs.PATIENCE = 5
        # Update global variables
        global N_FED_ROUNDS, N_LOCAL_EPOCHS, N_SSL_ROUNDS, N_FOLDS_AB, PATIENCE
        N_FED_ROUNDS = 5
        N_LOCAL_EPOCHS = 2
        N_SSL_ROUNDS = 3
        N_FOLDS_AB = 2
        PATIENCE = 5
        print(">>> Quick mode: rounds=5, local_epochs=2, ssl_rounds=3, folds=2")

    all_results = {}
    csv_rows = []

    # Determine which experiments to run
    exp_ids = []
    if args.exp:
        exp_ids = [e.strip() for e in args.exp.split(',')]
    elif args.all:
        for b in ['1', '2', '3', '4']:
            exp_ids.extend(BATCHES[b])
    else:
        # Check individual batch flags
        if args.batch1:
            exp_ids.extend(BATCHES['1'])
        if args.batch2:
            exp_ids.extend(BATCHES['2'])
        if args.batch3:
            exp_ids.extend(BATCHES['3'])
        if args.batch4:
            exp_ids.extend(BATCHES['4'])

    # Only load 3-client data if needed (not when --batch6 only)
    needs_3client_data = bool(exp_ids) or args.phase2
    data_hole = None
    data_triplet = None
    if needs_3client_data:
        print("\nLoading data (3-client)...")
        data_hole = prepare_all_clients(target_type='hole')
        data_triplet = prepare_all_clients(target_type='triplet')

    # Run main experiments
    if exp_ids:
        print(f"\nRunning experiments: {exp_ids}")
        for exp_id in exp_ids:
            if exp_id not in EXPERIMENTS:
                print(f"Unknown experiment: {exp_id}, skipping")
                continue

            result = run_single_experiment(
                exp_id, data_hole, data_triplet,
                device_a, device_b, verbose=True,
            )
            all_results[exp_id] = result
            csv_rows.append(extract_metrics_row(exp_id, result))

            # Save after each experiment
            save_results_csv(csv_rows)

        # Print summary table
        print_summary_table(csv_rows)

    # Phase 2: Ultimate verification experiments
    if args.phase2:
        run_phase2_ultimate(data_hole, data_triplet, device_a, device_b)

    # Batch 6: 4-Client federated experiments (loads its own data, includes Client D)
    if args.batch6:
        run_batch6_4client(device_a, device_b, verbose=True)

    # Batch 7: 3D conformer feature enhancement experiments
    if args.batch7:
        run_batch7_3d_features(device_a, device_b, verbose=True)

    # Batch 8: SchNet 3D Encoder experiments
    if args.batch8:
        run_batch8_schnet(device_a, device_b, verbose=True)

    # Save full results
    if all_results:
        save_full_results(all_results)

    # If nothing specified, print help
    if not exp_ids and not args.phase2 and not args.batch6 and not args.batch7 and not args.batch8:
        print("\nUsage examples:")
        print("  python run_all.py --batch1                   # Run Batch 1 (local baselines)")
        print("  python run_all.py --batch1 --batch2          # Run Batch 1 and 2")
        print("  python run_all.py --all                      # Run all main experiments (Batch 1-4)")
        print("  python run_all.py --exp E1,E7                # Run specific experiments")
        print("  python run_all.py --batch1 --quick           # Quick mode for debugging")
        print("  python run_all.py --phase2                   # Run Phase 2 ultimate experiments")
        print("  python run_all.py --batch6                   # Run Batch 6 (4-client federation)")
        print("  python run_all.py --batch7                   # Run Batch 7 (3D features)")
        print("  python run_all.py --batch8                   # Run Batch 8 (SchNet)")
        print()
        print("Experiment list:")
        for eid, cfg in EXPERIMENTS.items():
            print(f"  {eid:4s}: {cfg['desc']}")
        print("\nBatch 6 (4-Client federation):")
        print("  python run_all.py --batch6              # E30-E33 with Client D + y-norm")
        for eid, cfg in BATCH6_EXPERIMENTS.items():
            print(f"    {eid}: {cfg['desc']}")
        print("\nBatch 7 (3D conformer features):")
        print("  python run_all.py --batch7              # E40-E42 with 3D conformer features")
        for eid, cfg in BATCH7_EXPERIMENTS.items():
            print(f"    {eid}: {cfg['desc']}")
        print("\nBatch 8 (SchNet 3D Encoder):")
        print("  python run_all.py --batch8              # E50-E52 with SchNet 3D encoder")
        for eid, cfg in BATCH8_EXPERIMENTS.items():
            print(f"    {eid}: {cfg['desc']}")


if __name__ == '__main__':
    main()
