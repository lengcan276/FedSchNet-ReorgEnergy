"""
Federated learning strategies.

Legacy strategies (preserve E1-E59 reproducibility; only delta is the
universal calibration exclusion):
- FedAvg:  Weighted average of all parameters except calibration buffers
- FedProx: FedAvg + proximal term mu/2 ||w - w_global||^2 (excludes calibration)
- FedPer:  Excludes head + bn/norm + calibration (encoder non-BN shared)
- FedBN:   Excludes head + bn/norm + calibration (same exclusion set as FedPer
           in this codebase; kept as a separate strategy for experiment-tag
           clarity, mirroring the original implementation)

New strategies for PC²-FedReorg ablations (Phase 3+):
- fedper_pc2: Excludes head + adapter + calibration (encoder + its BN shared)
- fedbn_pc2:  Excludes bn/norm + calibration (head + adapter shared)
- pc2_fed:    Per-key routing aggregator; see experiments/pc2_fedreorg.py

Universal Phase 3 safety fix:
    'calibration' is excluded from EVERY strategy. Per-task-client (mu, sigma)
    buffers must never be cross-client averaged.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn


# ============ Parameter Filtering ============

def _get_exclude_keys(state_dict: dict, strategy: str) -> set:
    """Determine parameter keys to exclude based on federated strategy.

    Universal rule (Phase 3 safety fix):
        Any key containing 'calibration' is ALWAYS excluded -- per-task-client
        (mu, sigma) buffers must never be cross-client averaged.

    Legacy strategies (E1-E59 semantics preserved; only delta is the universal
    calibration exclusion):
        fedavg, fedprox: only 'calibration' excluded.
        fedper:          'calibration' + 'head' + 'bn'/'norm' excluded
                         (encoder non-BN shared, heads + BN private).
        fedbn:           'calibration' + 'head' + 'bn'/'norm' excluded
                         (same set as fedper in this codebase; kept as a
                         distinct tag so existing experiment names still
                         resolve. The original docstring noted that "FedPer
                         already skips BN, so FedBN has the same exclusion
                         set"; that invariant is preserved here.)

    New PC²-FedReorg ablation strategies (do NOT alter legacy semantics):
        fedper_pc2:      'calibration' + 'head' + 'adapter' excluded
                         (encoder incl. its BN shared, head + adapter private).
        fedbn_pc2:       'calibration' + 'bn'/'norm' excluded
                         (head + adapter shared; standard FedBN semantics).
        pc2_fed:         'calibration' excluded; per-key routing handled by
                         experiments/pc2_fedreorg.py:pc2_fed_aggregate.
    """
    exclude = set()
    for key in state_dict:
        # Universal: calibration buffers are never aggregated.
        if 'calibration' in key:
            exclude.add(key)
            continue

        if strategy in ('fedper', 'fedbn'):
            # Legacy semantics: head + bn/norm private (matches pre-Phase-3 behavior).
            if 'head' in key:
                exclude.add(key)
            if 'bn' in key or 'norm' in key:
                exclude.add(key)
        elif strategy == 'fedper_pc2':
            # New: head + adapter private (encoder incl. BN shared).
            if 'head' in key or 'adapter' in key:
                exclude.add(key)
        elif strategy == 'fedbn_pc2':
            # New: only BN private (head + adapter shared).
            if 'bn' in key or 'norm' in key:
                exclude.add(key)
        elif strategy in ('fedavg', 'fedprox', 'pc2_fed'):
            pass  # only the universal calibration exclusion applies
        else:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of "
                f"'fedavg', 'fedprox', 'fedper', 'fedbn', "
                f"'fedper_pc2', 'fedbn_pc2', 'pc2_fed'"
            )

    return exclude


# ============ FedAvg ============

def fedavg_aggregate(client_models: list, weights: list,
                     exclude_keys: set = None) -> OrderedDict:
    """Weighted average of model parameters.

    Args:
        client_models: list of nn.Module
        weights: aggregation weight list (sum=1)
        exclude_keys: set of parameter keys to exclude from aggregation

    Returns:
        Aggregated state_dict (only contains aggregated keys)
    """
    if exclude_keys is None:
        exclude_keys = set()

    state_dicts = [m.state_dict() for m in client_models]
    avg_dict = OrderedDict()

    for key in state_dicts[0]:
        if key in exclude_keys:
            continue

        # Skip non-floating-point parameters (e.g. num_batches_tracked)
        if not state_dicts[0][key].is_floating_point():
            continue

        avg_dict[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )

    return avg_dict


def fedavg_strategy(client_models: list, weights: list) -> OrderedDict:
    """FedAvg: aggregate all parameters except calibration buffers (universal rule)."""
    sample_dict = client_models[0].state_dict()
    exclude = _get_exclude_keys(sample_dict, 'fedavg')
    return fedavg_aggregate(client_models, weights, exclude_keys=exclude)


# ============ FedPer ============

def fedper_aggregate(client_models: list, weights: list) -> OrderedDict:
    """FedPer: aggregate only encoder non-BN parameters, heads remain private.

    Exclusion rule: keys containing 'head', 'bn', or 'norm'
    """
    sample_dict = client_models[0].state_dict()
    exclude = _get_exclude_keys(sample_dict, 'fedper')
    return fedavg_aggregate(client_models, weights, exclude_keys=exclude)


# ============ FedBN ============

def fedbn_aggregate(client_models: list, weights: list) -> OrderedDict:
    """FedBN: on top of FedPer, additionally skip all BatchNorm parameters.

    In practice FedPer already skips BN, so FedBN has the same exclusion set.
    Kept as a separate function for experimental code clarity.
    """
    sample_dict = client_models[0].state_dict()
    exclude = _get_exclude_keys(sample_dict, 'fedbn')
    return fedavg_aggregate(client_models, weights, exclude_keys=exclude)


# ============ PC²-FedReorg ablation strategies (Phase 3+) ============
# These do NOT alter the legacy fedper / fedbn semantics. They exist so PC²
# can compare against personalized baselines that differ in *which* layers
# are private (head + adapter, vs BN, vs head + BN). Keep separate from
# legacy strategy names so E1-E59 results stay reproducible.

def fedper_pc2_aggregate(client_models: list, weights: list) -> OrderedDict:
    """FedPer (PC² variant): exclude head + adapter + calibration. BN shared."""
    sample_dict = client_models[0].state_dict()
    exclude = _get_exclude_keys(sample_dict, 'fedper_pc2')
    return fedavg_aggregate(client_models, weights, exclude_keys=exclude)


def fedbn_pc2_aggregate(client_models: list, weights: list) -> OrderedDict:
    """FedBN (PC² variant): exclude bn/norm + calibration. Head/adapter shared."""
    sample_dict = client_models[0].state_dict()
    exclude = _get_exclude_keys(sample_dict, 'fedbn_pc2')
    return fedavg_aggregate(client_models, weights, exclude_keys=exclude)


# ============ Parameter Distribution ============

def distribute_params(client_models: list, avg_dict: OrderedDict):
    """Distribute aggregated parameters to each client (only update keys present in avg_dict)."""
    for model in client_models:
        local_dict = model.state_dict()
        for key in avg_dict:
            local_dict[key] = avg_dict[key].clone()
        model.load_state_dict(local_dict)


# ============ FedProx Proximal Regularizer ============

class FedProxRegularizer:
    """FedProx proximal regularizer: loss += (mu/2) * ||w - w_global||^2

    At the start of each federated communication round, save the global model parameters.
    During training, add the regularization term to the local loss.
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu
        self.global_params = None

    def save_global_params(self, model: nn.Module):
        """Save current global model parameters (called after each aggregation round)."""
        self.global_params = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    def compute_penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute (mu/2) * ||w - w_global||^2"""
        if self.global_params is None:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.global_params:
                global_param = self.global_params[name].to(param.device)
                penalty += (param - global_param).pow(2).sum()

        return (self.mu / 2.0) * penalty


# ============ Unified Interface ============

def get_aggregation_fn(strategy: str):
    """Get the aggregation function for a given strategy.

    Args:
        strategy: one of 'fedavg', 'fedper', 'fedbn', 'fedper_pc2', 'fedbn_pc2'.

        Note: 'pc2_fed' is NOT included here -- its aggregator
        (experiments/pc2_fedreorg.py:pc2_fed_aggregate) returns a per-target
        dict-of-dicts and has a different signature; the training loop must
        call it via its own dispatch path.

    Returns:
        Aggregation function (client_models, weights) -> OrderedDict
    """
    strategy_map = {
        'fedavg':     fedavg_strategy,
        'fedper':     fedper_aggregate,        # legacy semantics (head + bn private)
        'fedbn':      fedbn_aggregate,         # legacy semantics (same exclusion as fedper)
        'fedper_pc2': fedper_pc2_aggregate,    # new: head + adapter private, BN shared
        'fedbn_pc2':  fedbn_pc2_aggregate,     # new: BN private, head + adapter shared
    }
    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategy_map.keys())}")
    return strategy_map[strategy]


def get_exclude_keys_for_strategy(model: nn.Module, strategy: str) -> set:
    """Get the set of excluded keys for a given strategy (for external inspection)."""
    return _get_exclude_keys(model.state_dict(), strategy)
