"""
Training and evaluation module.
- Local training / federated training loops
- SSL pretraining -> fine-tuning pipeline
- 5-fold CV (Clients A/B) / LOOCV (Client C)
- Metrics: MAE, R², RMSE
"""

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from .data_utils import NODE_FEAT_DIM
    from .models import ReorgEnergyModel, count_parameters
    from .federated import get_aggregation_fn, distribute_params, FedProxRegularizer
    from .ssl_pretrain import federated_ssl_pretrain, SSLEncoder
except ImportError:
    from data_utils import NODE_FEAT_DIM
    from models import ReorgEnergyModel, count_parameters
    from federated import get_aggregation_fn, distribute_params, FedProxRegularizer
    from ssl_pretrain import federated_ssl_pretrain, SSLEncoder

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / 'checkpoints'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============ Encoder dimension adaptation ============

def _adapt_encoder_state_dict(state_dict: dict, old_in_dim: int, new_in_dim: int) -> dict:
    """Adapt encoder state_dict: zero-pad first GINConv layer weights to match new input dim.

    Used when loading SSL pretrained encoder (in_dim=11) into a physics-augmented model (in_dim=15).
    """
    adapted = copy.deepcopy(state_dict)
    key_weight = 'convs.0.nn.0.weight'
    if key_weight in adapted:
        old_weight = adapted[key_weight]  # [hidden_dim, old_in_dim]
        if old_weight.size(1) == old_in_dim and new_in_dim > old_in_dim:
            new_weight = torch.zeros(old_weight.size(0), new_in_dim, dtype=old_weight.dtype)
            new_weight[:, :old_in_dim] = old_weight
            adapted[key_weight] = new_weight
    return adapted


# ============ Metrics computation ============

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R²."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ============ Single-client training / evaluation ============

def train_one_epoch(model, loader, optimizer, device, criterion,
                    fedprox_reg: Optional[FedProxRegularizer] = None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)
        loss = criterion(pred, batch.y)

        # FedProx regularization
        if fedprox_reg is not None:
            loss = loss + fedprox_reg.compute_penalty(model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch.y.size(0)
        n_samples += batch.y.size(0)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model, return predictions and ground truth."""
    model.eval()
    y_true_list = []
    y_pred_list = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).squeeze(-1)
        y_true_list.append(batch.y.cpu().numpy())
        y_pred_list.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


# ============ Local training (no federation) ============

def train_local(
    data_list: list,
    head_type: str = 'kan',
    n_epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 512,
    patience: int = 15,
    device: torch.device = torch.device('cpu'),
    pretrained_encoder: dict = None,
    kan_grid: int = 5,
    use_physics: bool = False,
    verbose: bool = True,
) -> tuple:
    """Train a single model locally.

    Returns:
        model, train_losses, val_losses
    """
    model = ReorgEnergyModel(
        head_type=head_type, kan_grid=kan_grid,
        use_global_desc=True,
        use_physics_feat=use_physics,
    ).to(device)

    # Load pretrained encoder
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrained_encoder, strict=False)
        if verbose:
            print("  Loaded pretrained encoder")

    # Split train/validation (80/20)
    n = len(data_list)
    perm = torch.randperm(n).tolist()
    n_val = max(1, int(n * 0.2))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step()

        # Validation
        y_true, y_pred = evaluate(model, val_loader, device)
        val_loss = np.mean((y_true - y_pred) ** 2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


# ============ Cross-validation ============

def cv_evaluate(
    data_list: list,
    head_type: str = 'kan',
    n_folds: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 512,
    patience: int = 15,
    device: torch.device = torch.device('cpu'),
    pretrained_encoder: dict = None,
    kan_grid: int = 5,
    use_physics: bool = False,
    in_dim: int = NODE_FEAT_DIM,
    verbose: bool = True,
    phys_feats_dim: int = 0,
    encoder_out_dim: int = 256,
    model_factory=None,
) -> dict:
    """K-fold cross-validation.

    Returns:
        {'MAE': (mean, std), 'RMSE': (mean, std), 'R2': (mean, std), 'folds': [...]}
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = list(range(len(data_list)))

    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if verbose:
            print(f"  Fold {fold_i+1}/{n_folds}")

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        if model_factory is not None:
            model = model_factory().to(device)
        else:
            model = ReorgEnergyModel(
                head_type=head_type, in_dim=in_dim, kan_grid=kan_grid,
                encoder_out_dim=encoder_out_dim,
                use_global_desc=True,
                use_physics_feat=use_physics,
                phys_feats_dim=phys_feats_dim,
            ).to(device)

        if pretrained_encoder is not None and model_factory is None:
            model.encoder.load_state_dict(pretrained_encoder, strict=False)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, criterion)
            scheduler.step()

            y_true, y_pred = evaluate(model, val_loader, device)
            val_loss = np.mean((y_true - y_pred) ** 2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_true, y_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        fold_metrics.append(metrics)

        if verbose:
            print(f"    MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # Summary
    result = {}
    for key in ['MAE', 'RMSE', 'R2']:
        vals = [m[key] for m in fold_metrics]
        result[key] = (np.mean(vals), np.std(vals))

    if verbose:
        print(f"  CV result: MAE={result['MAE'][0]:.4f}+/-{result['MAE'][1]:.4f}, "
              f"R²={result['R2'][0]:.4f}+/-{result['R2'][1]:.4f}")

    result['folds'] = fold_metrics
    return result


def loocv_evaluate(
    data_list: list,
    head_type: str = 'kan',
    n_epochs: int = 200,
    lr: float = 1e-4,
    patience: int = 15,
    device: torch.device = torch.device('cpu'),
    pretrained_encoder: dict = None,
    kan_grid: int = 5,
    use_physics: bool = False,
    in_dim: int = NODE_FEAT_DIM,
    verbose: bool = True,
    phys_feats_dim: int = 0,
    encoder_out_dim: int = 256,
    model_factory=None,
) -> dict:
    """Leave-One-Out cross-validation (for Client C).

    Returns:
        {'MAE': float, 'RMSE': float, 'R2': float, 'y_true': array, 'y_pred': array}
    """
    n = len(data_list)
    y_true_all = []
    y_pred_all = []

    iterator = range(n)
    if verbose:
        iterator = tqdm(iterator, desc='  LOOCV')

    for i in iterator:
        # Leave one sample out for testing
        train_data = [data_list[j] for j in range(n) if j != i]
        test_data = [data_list[i]]

        if model_factory is not None:
            model = model_factory().to(device)
        else:
            model = ReorgEnergyModel(
                head_type=head_type, in_dim=in_dim, kan_grid=kan_grid,
                encoder_out_dim=encoder_out_dim,
                use_global_desc=True,
                use_physics_feat=use_physics,
                phys_feats_dim=phys_feats_dim,
            ).to(device)

        if pretrained_encoder is not None and model_factory is None:
            model.encoder.load_state_dict(pretrained_encoder, strict=False)

        # Full-batch training
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, drop_last=False)
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

    if verbose:
        print(f"  LOOCV result: MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    return metrics


# ============ GPU assignment ============

def _get_device_map(client_keys: list,
                    device_a: torch.device,
                    device_b: torch.device) -> dict:
    """GPU assignment for federated training: A,D -> device_a, B,C -> device_b."""
    device_map = {}
    for name in client_keys:
        if name in ('client_a', 'client_d'):
            device_map[name] = device_a
        else:
            device_map[name] = device_b
    return device_map


# ============ Federated training loop ============

def train_federated(
    client_data: dict,
    head_type: str = 'kan',
    fed_strategy: str = 'fedper',
    n_rounds: int = 50,
    n_local_epochs: int = 5,
    lr: float = 1e-4,
    batch_size_ab: int = 512,
    patience: int = 15,
    device_a: torch.device = torch.device('cuda:0'),
    device_b: torch.device = torch.device('cuda:1'),
    pretrained_encoder: dict = None,
    kan_grid: int = 5,
    use_physics_c: bool = False,
    in_dim: int = NODE_FEAT_DIM,
    mu: float = 0.01,
    verbose: bool = True,
    phys_feats_dim: int = 0,
    encoder_out_dim: int = 256,
    norm_stats: dict = None,
    model_factory=None,
) -> dict:
    """Federated training (supports N clients).

    Args:
        client_data: {'client_a': [...], 'client_b': [...], 'client_c': [...], optional 'client_d': [...]}
        fed_strategy: 'fedavg', 'fedper', 'fedbn'
        n_rounds: number of federation communication rounds
        n_local_epochs: number of local training epochs per round
        mu: FedProx regularization coefficient (only effective when fed_strategy contains 'prox')
        norm_stats: {client_name: (y_mean, y_std)} per-client label normalization parameters
        model_factory: optional callable returning a fresh model instance

    Returns:
        dict with models, metrics, loss history
    """
    # Identify participating client keys
    client_keys = [k for k in client_data if k.startswith('client_') and
                   k not in ('client_c_conformers',) and client_data[k]]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Federated training: strategy={fed_strategy}, head={head_type}, "
              f"rounds={n_rounds}, local_epochs={n_local_epochs}, clients={client_keys}")
        print(f"{'='*60}")

    use_fedprox = 'prox' in fed_strategy
    base_strategy = fed_strategy.replace('_prox', '').replace('prox_', '').replace('prox', '')
    if not base_strategy:
        base_strategy = 'fedavg'

    aggregate_fn = get_aggregation_fn(base_strategy)

    # GPU assignment
    device_map = _get_device_map(client_keys, device_a, device_b)

    # Create models
    models = {}
    if model_factory is not None:
        for name in client_keys:
            models[name] = model_factory().to(device_map[name])
    else:
        for name in client_keys:
            ht = 'physics_kan' if (use_physics_c and name == 'client_c') else head_type
            up = use_physics_c if name == 'client_c' else False
            models[name] = ReorgEnergyModel(
                head_type=ht, in_dim=in_dim, kan_grid=kan_grid,
                encoder_out_dim=encoder_out_dim,
                use_global_desc=True, use_physics_feat=up,
                phys_feats_dim=phys_feats_dim,
            ).to(device_map[name])

    # Load pretrained encoder
    if pretrained_encoder is not None and model_factory is None:
        for name in client_keys:
            models[name].encoder.load_state_dict(pretrained_encoder, strict=False)
        if verbose:
            print("  Loaded pretrained encoder")

    # Synchronize encoder initial parameters (using first client as reference)
    first_key = client_keys[0]
    init_encoder = copy.deepcopy(models[first_key].encoder.state_dict())
    for name in client_keys[1:]:
        models[name].encoder.load_state_dict(init_encoder)

    # DataLoaders -- large clients use 80/20 split, small client (C) uses full data for training
    def split_data(data_list, val_ratio=0.2):
        n = len(data_list)
        perm = torch.randperm(n).tolist()
        n_val = max(1, int(n * val_ratio))
        return [data_list[i] for i in perm[n_val:]], [data_list[i] for i in perm[:n_val]]

    train_data = {}
    val_data = {}
    train_loaders = {}
    val_loaders = {}

    for name in client_keys:
        data_list = client_data[name]
        if name == 'client_c':
            # Client C: full data training (LOOCV is done externally)
            train_data[name] = data_list
            val_data[name] = None
            c_bs = min(len(data_list), 64)
            train_loaders[name] = DataLoader(data_list, batch_size=c_bs,
                                             shuffle=True, drop_last=False)
        else:
            tr, va = split_data(data_list)
            train_data[name] = tr
            val_data[name] = va
            train_loaders[name] = DataLoader(tr, batch_size=batch_size_ab,
                                             shuffle=True, drop_last=False)
            val_loaders[name] = DataLoader(va, batch_size=batch_size_ab, shuffle=False)

    # Optimizers & schedulers
    optimizers = {name: torch.optim.Adam(models[name].parameters(), lr=lr)
                  for name in client_keys}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizers[name], T_max=n_rounds * n_local_epochs)
                  for name in client_keys}

    criterion = nn.MSELoss()

    # FedProx
    prox_regs = {}
    if use_fedprox:
        prox_regs = {name: FedProxRegularizer(mu=mu) for name in client_keys}

    # Aggregation weights
    counts = {name: len(train_data[name]) for name in client_keys}
    total_n = sum(counts.values())
    weights_list = [counts[name] / total_n for name in client_keys]

    # Validation keys (clients that have val splits)
    val_keys = [name for name in client_keys if val_data.get(name) is not None]

    # Training history
    history = {'round': []}
    for name in client_keys:
        history[f'loss_{name.replace("client_", "")}'] = []
    for name in val_keys:
        history[f'val_mae_{name.replace("client_", "")}'] = []

    best_val_loss = float('inf')
    best_states = None
    patience_counter = 0

    start_time = time.time()

    for rnd in range(1, n_rounds + 1):
        # Save global parameters (FedProx)
        if use_fedprox:
            for name in client_keys:
                prox_regs[name].save_global_params(models[name])

        # Local training
        loss_sums = {name: 0.0 for name in client_keys}
        for _ in range(n_local_epochs):
            for name in client_keys:
                prox = prox_regs.get(name)
                loss_sums[name] += train_one_epoch(
                    models[name], train_loaders[name], optimizers[name],
                    device_map[name], criterion, prox)
                schedulers[name].step()

        avg_losses = {name: loss_sums[name] / n_local_epochs for name in client_keys}

        # Aggregate (on CPU)
        models_cpu = [copy.deepcopy(models[name]).cpu() for name in client_keys]
        avg_dict = aggregate_fn(models_cpu, weights_list)

        # Distribute
        for name in client_keys:
            local_dict = models[name].state_dict()
            for key in avg_dict:
                if key in local_dict:
                    local_dict[key] = avg_dict[key].clone().to(local_dict[key].device)
            models[name].load_state_dict(local_dict)

        # Validation
        val_maes = {}
        for name in val_keys:
            y_true, y_pred = evaluate(models[name], val_loaders[name], device_map[name])
            val_maes[name] = np.mean(np.abs(y_true - y_pred))

        # Weighted combined val MAE for early stopping
        val_combined = sum(val_maes.get(name, 0) * (counts[name] / total_n)
                           for name in val_keys)

        # Record history
        history['round'].append(rnd)
        for name in client_keys:
            history[f'loss_{name.replace("client_", "")}'].append(avg_losses[name])
        for name in val_keys:
            history[f'val_mae_{name.replace("client_", "")}'].append(val_maes[name])

        # Early stopping
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            best_states = {name: copy.deepcopy(models[name].state_dict())
                           for name in client_keys}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (rnd % 5 == 0 or rnd == 1 or rnd == n_rounds):
            elapsed = time.time() - start_time
            loss_str = ' '.join(f"{name.replace('client_', '').upper()}={avg_losses[name]:.4f}"
                                for name in client_keys)
            val_str = ' '.join(f"{name.replace('client_', '').upper()}={val_maes[name]:.4f}"
                               for name in val_keys)
            print(f"  Round {rnd:3d}/{n_rounds} | Loss {loss_str} | ValMAE {val_str} | {elapsed:.0f}s")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at round {rnd}")
            break

    # Restore best models
    if best_states is not None:
        for name in client_keys:
            models[name].load_state_dict(best_states[name])

    total_time = time.time() - start_time
    if verbose:
        print(f"  Training completed in {total_time:.1f}s")

    result = {
        'history': history,
        'total_time': total_time,
        'best_round': n_rounds - patience_counter if patience_counter >= patience else n_rounds,
    }
    # Add models with both old and new naming for backward compat
    for name in client_keys:
        result[f'model_{name.replace("client_", "")}'] = models[name]

    return result


# ============ Federated training + LOOCV evaluation (Client C) ============

def federated_loocv_c(
    client_data: dict,
    head_type: str = 'kan',
    fed_strategy: str = 'fedper',
    n_rounds: int = 50,
    n_local_epochs: int = 5,
    lr: float = 1e-4,
    batch_size_ab: int = 512,
    device_a: torch.device = torch.device('cuda:0'),
    device_b: torch.device = torch.device('cuda:1'),
    pretrained_encoder: dict = None,
    kan_grid: int = 5,
    use_physics_c: bool = False,
    in_dim: int = NODE_FEAT_DIM,
    mu: float = 0.01,
    verbose: bool = True,
    norm_stats: dict = None,
) -> dict:
    """Full federated LOOCV for Client C: run complete federated training for each
    leave-one-out fold (supports N clients).

    Returns:
        dict with MAE, R², RMSE, y_true, y_pred
    """
    n_c = len(client_data['client_c'])
    y_true_all = []
    y_pred_all = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Federated LOOCV for Client C: {n_c} molecules")
        print(f"  strategy={fed_strategy}, head={head_type}")
        print(f"{'='*60}")

    # Determine device for Client C
    device_c = device_b

    for i in tqdm(range(n_c), desc='LOOCV', disable=not verbose):
        # Construct train/test data -- preserve all clients
        c_train = [client_data['client_c'][j] for j in range(n_c) if j != i]
        c_test = [client_data['client_c'][i]]

        data_fold = dict(client_data)  # shallow copy
        data_fold['client_c'] = c_train

        # Per-fold normalization for Client C if norm_stats requested
        fold_norm = None
        if norm_stats is not None:
            import torch as _torch
            fold_ys = _torch.cat([d.y for d in c_train])
            fold_mean = fold_ys.mean().item()
            fold_std = max(fold_ys.std().item(), 1e-6)
            # Normalize fold train data
            for d in c_train:
                d.y = (d.y - fold_mean) / fold_std
            fold_norm = (fold_mean, fold_std)

        # Federated training
        result = train_federated(
            data_fold, head_type=head_type, fed_strategy=fed_strategy,
            n_rounds=n_rounds, n_local_epochs=n_local_epochs, lr=lr,
            batch_size_ab=batch_size_ab, patience=15,
            device_a=device_a, device_b=device_b,
            pretrained_encoder=pretrained_encoder,
            kan_grid=kan_grid, use_physics_c=use_physics_c, in_dim=in_dim,
            mu=mu, verbose=False,
        )

        # Evaluate the held-out C sample
        test_loader = DataLoader(c_test, batch_size=1, shuffle=False)
        y_true, y_pred = evaluate(result['model_c'], test_loader, device_c)

        # Denormalize if needed
        if fold_norm is not None:
            y_pred = y_pred * fold_norm[1] + fold_norm[0]
            y_true = y_true * fold_norm[1] + fold_norm[0]

        y_true_all.append(y_true[0])
        y_pred_all.append(y_pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['y_true'] = y_true_all
    metrics['y_pred'] = y_pred_all

    if verbose:
        print(f"  Client C LOOCV: MAE={metrics['MAE']:.4f}, "
              f"R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    return metrics


# ============ Fast federated LOOCV (frozen encoder) ============

def federated_loocv_c_fast(
    client_data: dict,
    head_type: str = 'kan',
    fed_strategy: str = 'fedper',
    n_rounds: int = 50,
    n_local_epochs: int = 5,
    lr: float = 1e-4,
    batch_size_ab: int = 512,
    device_a: torch.device = torch.device('cuda:0'),
    device_b: torch.device = torch.device('cuda:1'),
    pretrained_encoder: dict = None,
    kan_grid: int = 5,
    use_physics_c: bool = False,
    in_dim: int = NODE_FEAT_DIM,
    mu: float = 0.01,
    n_finetune_epochs: int = 200,
    verbose: bool = True,
    phys_feats_dim: int = 0,
    encoder_out_dim: int = 256,
    norm_stats: dict = None,
    model_factory=None,
) -> dict:
    """Fast federated LOOCV: train full federation once to obtain shared encoder,
    then LOOCV retrain head only.
    Supports N clients and per-fold label denormalization.
    """
    n_c = len(client_data['client_c'])
    device_c = device_b

    if verbose:
        print(f"\n{'='*60}")
        print(f"Fast federated LOOCV for Client C: {n_c} molecules")
        print(f"  strategy={fed_strategy}, head={head_type}")
        print(f"  Step 1: Full federated training to obtain shared encoder")
        print(f"{'='*60}")

    # Step 1: Run full federated training with all data
    fed_result = train_federated(
        client_data, head_type=head_type, fed_strategy=fed_strategy,
        n_rounds=n_rounds, n_local_epochs=n_local_epochs, lr=lr,
        batch_size_ab=batch_size_ab, patience=15,
        device_a=device_a, device_b=device_b,
        pretrained_encoder=pretrained_encoder,
        kan_grid=kan_grid, use_physics_c=use_physics_c, in_dim=in_dim,
        mu=mu, verbose=verbose,
        phys_feats_dim=phys_feats_dim, encoder_out_dim=encoder_out_dim,
        norm_stats=norm_stats,
        model_factory=model_factory,
    )

    # Save the federated encoder
    fed_encoder = copy.deepcopy(fed_result['model_c'].encoder.state_dict())

    if verbose:
        print(f"\n  Step 2: LOOCV - freeze encoder, retrain head only ({n_c} folds)")

    # Step 2: LOOCV - freeze encoder, train head only
    head_c = 'physics_kan' if use_physics_c else head_type

    # Save raw y values for per-fold normalization
    c_data_all = client_data['client_c']
    raw_ys = [d.y.item() for d in c_data_all]

    y_true_all = []
    y_pred_all = []

    criterion = nn.MSELoss()

    for i in tqdm(range(n_c), desc='LOOCV(fast)', disable=not verbose):
        c_train = [c_data_all[j] for j in range(n_c) if j != i]
        c_test = [c_data_all[i]]

        # Per-fold normalization
        fold_norm = None
        if norm_stats is not None:
            train_ys = torch.tensor([d.y.item() for d in c_train])
            fold_mean = train_ys.mean().item()
            fold_std = max(train_ys.std().item(), 1e-6)
            fold_norm = (fold_mean, fold_std)
            # Create copies with normalized y
            c_train_norm = []
            for d in c_train:
                d_copy = d.clone()
                d_copy.y = (d_copy.y - fold_mean) / fold_std
                c_train_norm.append(d_copy)
            c_train = c_train_norm

        # Create new model, load federated encoder and freeze it
        if model_factory is not None:
            model_c = model_factory().to(device_c)
        else:
            model_c = ReorgEnergyModel(
                head_type=head_c, in_dim=in_dim, kan_grid=kan_grid,
                encoder_out_dim=encoder_out_dim,
                use_global_desc=True, use_physics_feat=use_physics_c,
                phys_feats_dim=phys_feats_dim,
            ).to(device_c)
        model_c.encoder.load_state_dict(fed_encoder, strict=False)

        # Freeze encoder parameters
        for param in model_c.encoder.parameters():
            param.requires_grad = False

        # Only optimize head parameters
        head_params = [p for p in model_c.head.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(head_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_finetune_epochs)

        c_bs = min(len(c_train), 64)
        train_loader = DataLoader(c_train, batch_size=c_bs, shuffle=True, drop_last=False)
        test_loader = DataLoader(c_test, batch_size=1, shuffle=False)

        # Train head (with early stopping)
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(n_finetune_epochs):
            model_c.train()
            epoch_loss = 0.0
            n_samples = 0
            for batch in train_loader:
                batch = batch.to(device_c)
                optimizer.zero_grad()
                pred = model_c(batch).squeeze(-1)
                loss = criterion(pred, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.y.size(0)
                n_samples += batch.y.size(0)
            scheduler.step()

            avg_loss = epoch_loss / max(n_samples, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model_c.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 30:
                break

        # Restore best model
        if best_state is not None:
            model_c.load_state_dict(best_state)

        # Predict
        y_true, y_pred = evaluate(model_c, test_loader, device_c)

        # Denormalize if needed
        if fold_norm is not None:
            y_pred = y_pred * fold_norm[1] + fold_norm[0]
            # y_true is raw (not normalized) since we used original c_test
        y_true_all.append(y_true[0])
        y_pred_all.append(y_pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['y_true'] = y_true_all
    metrics['y_pred'] = y_pred_all
    metrics['fed_history'] = fed_result['history']

    if verbose:
        print(f"  Client C LOOCV(fast): MAE={metrics['MAE']:.4f}, "
              f"R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    return metrics


# ============ Complete experiment pipeline ============

def run_experiment(
    client_data: dict,
    exp_name: str,
    head_type: str = 'kan',
    fed_strategy: Optional[str] = None,
    ssl_method: Optional[str] = None,
    n_rounds: int = 50,
    n_local_epochs: int = 5,
    n_ssl_rounds: int = 20,
    lr: float = 1e-4,
    batch_size_ab: int = 512,
    device_a: torch.device = torch.device('cuda:0'),
    device_b: torch.device = torch.device('cuda:1'),
    kan_grid: int = 5,
    use_physics_c: bool = False,
    mu: float = 0.01,
    eval_c_loocv: bool = True,
    n_folds_ab: int = 5,
    verbose: bool = True,
    norm_stats: dict = None,
) -> dict:
    """Run a complete experiment.

    Args:
        exp_name: experiment name
        fed_strategy: None=local training, 'fedavg'/'fedper'/'fedbn' etc.
        ssl_method: None=no pretraining, 'atom_mask'/'edge_pred'/'graph_cl'

    Returns:
        experiment results dict
    """
    if verbose:
        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp_name}")
        print(f"# head={head_type}, fed={fed_strategy}, ssl={ssl_method}")
        print(f"{'#'*60}")

    start_time = time.time()
    pretrained_encoder = None

    # Step 1: SSL pretraining (if needed)
    if ssl_method is not None:
        if verbose:
            print("\n[Step 1] SSL pretraining")
        encoder_state, ssl_history = federated_ssl_pretrain(
            client_data, ssl_method=ssl_method,
            n_rounds=n_ssl_rounds, n_local_epochs=5,
            lr=1e-3, batch_size_ab=batch_size_ab,
            device_a=device_a, device_b=device_b,
        )
        pretrained_encoder = encoder_state

    results = {'exp_name': exp_name, 'head_type': head_type,
               'fed_strategy': fed_strategy, 'ssl_method': ssl_method}

    # Step 2: Training + evaluation
    if fed_strategy is None:
        # ===== Local training =====
        if verbose:
            print("\n[Step 2] Local training + CV evaluation")

        # Client A
        if verbose:
            print("\n  --- Client A ---")
        results['cv_a'] = cv_evaluate(
            client_data['client_a'], head_type=head_type,
            n_folds=n_folds_ab, n_epochs=200, lr=lr,
            batch_size=batch_size_ab, device=device_a,
            pretrained_encoder=pretrained_encoder, kan_grid=kan_grid,
            verbose=verbose,
        )

        # Client B
        if verbose:
            print("\n  --- Client B ---")
        results['cv_b'] = cv_evaluate(
            client_data['client_b'], head_type=head_type,
            n_folds=n_folds_ab, n_epochs=200, lr=lr,
            batch_size=batch_size_ab, device=device_b,
            pretrained_encoder=pretrained_encoder, kan_grid=kan_grid,
            verbose=verbose,
        )

        # Client C (LOOCV)
        if eval_c_loocv:
            if verbose:
                print("\n  --- Client C (LOOCV) ---")
            results['loocv_c'] = loocv_evaluate(
                client_data['client_c'], head_type=head_type,
                n_epochs=200, lr=lr, device=device_b,
                pretrained_encoder=pretrained_encoder, kan_grid=kan_grid,
                use_physics=use_physics_c, verbose=verbose,
            )

    else:
        # ===== Federated training =====
        if verbose:
            print("\n[Step 2] Federated training")

        fed_result = train_federated(
            client_data, head_type=head_type, fed_strategy=fed_strategy,
            n_rounds=n_rounds, n_local_epochs=n_local_epochs, lr=lr,
            batch_size_ab=batch_size_ab, device_a=device_a, device_b=device_b,
            pretrained_encoder=pretrained_encoder, kan_grid=kan_grid,
            use_physics_c=use_physics_c, mu=mu, verbose=verbose,
            norm_stats=norm_stats,
        )
        results['fed_history'] = fed_result['history']

        # CV evaluation Client A
        if verbose:
            print("\n[Step 3] CV evaluation Client A")
        # Use the federated encoder for CV
        fed_encoder_a = copy.deepcopy(fed_result['model_a'].encoder.state_dict())
        results['cv_a'] = cv_evaluate(
            client_data['client_a'], head_type=head_type,
            n_folds=n_folds_ab, n_epochs=100, lr=lr,
            batch_size=batch_size_ab, device=device_a,
            pretrained_encoder=fed_encoder_a, kan_grid=kan_grid,
            verbose=verbose,
        )

        # CV evaluation Client B
        if verbose:
            print("\n[Step 3] CV evaluation Client B")
        fed_encoder_b = copy.deepcopy(fed_result['model_b'].encoder.state_dict())
        results['cv_b'] = cv_evaluate(
            client_data['client_b'], head_type=head_type,
            n_folds=n_folds_ab, n_epochs=100, lr=lr,
            batch_size=batch_size_ab, device=device_b,
            pretrained_encoder=fed_encoder_b, kan_grid=kan_grid,
            verbose=verbose,
        )

        # LOOCV Client C
        if eval_c_loocv:
            if verbose:
                print("\n[Step 3] LOOCV evaluation Client C")
            results['loocv_c'] = federated_loocv_c(
                client_data, head_type=head_type, fed_strategy=fed_strategy,
                n_rounds=n_rounds, n_local_epochs=n_local_epochs, lr=lr,
                batch_size_ab=batch_size_ab, device_a=device_a, device_b=device_b,
                pretrained_encoder=pretrained_encoder, kan_grid=kan_grid,
                use_physics_c=use_physics_c, mu=mu, verbose=verbose,
                norm_stats=norm_stats,
            )

    total_time = time.time() - start_time
    results['total_time'] = total_time

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment {exp_name} completed ({total_time:.1f}s)")
        print(f"{'='*60}")
        if 'cv_a' in results:
            m = results['cv_a']
            print(f"  Client A: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R²={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'cv_b' in results:
            m = results['cv_b']
            print(f"  Client B: MAE={m['MAE'][0]:.4f}+/-{m['MAE'][1]:.4f}, "
                  f"R²={m['R2'][0]:.4f}+/-{m['R2'][1]:.4f}")
        if 'loocv_c' in results:
            m = results['loocv_c']
            print(f"  Client C: MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")

    return results
