"""
Federated learning strategies.
- FedAvg:  Weighted average of all parameters
- FedProx: FedAvg + proximal term mu/2 ||w - w_global||^2
- FedPer:  Aggregate only encoder non-BN parameters, heads remain private
- FedBN:   FedPer + additionally skip all BatchNorm parameters
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn


# ============ Parameter Filtering ============

def _get_exclude_keys(state_dict: dict, strategy: str) -> set:
    """Determine parameter keys to exclude based on federated strategy.

    FedAvg:  No keys excluded
    FedPer:  Exclude head + BN
    FedBN:   Exclude head + BN (same logic as FedPer, explicit declaration)
    """
    exclude = set()
    for key in state_dict:
        if strategy in ('fedper', 'fedbn'):
            # Exclude head parameters (private layers)
            if 'head' in key:
                exclude.add(key)
            # Exclude BatchNorm parameters
            if 'bn' in key or 'norm' in key:
                exclude.add(key)
        # FedAvg: no exclusion
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
    """FedAvg: aggregate all parameters."""
    return fedavg_aggregate(client_models, weights, exclude_keys=set())


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
        strategy: 'fedavg', 'fedper', 'fedbn'

    Returns:
        Aggregation function (client_models, weights) -> OrderedDict
    """
    strategy_map = {
        'fedavg': fedavg_strategy,
        'fedper': fedper_aggregate,
        'fedbn': fedbn_aggregate,
    }
    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategy_map.keys())}")
    return strategy_map[strategy]


def get_exclude_keys_for_strategy(model: nn.Module, strategy: str) -> set:
    """Get the set of excluded keys for a given strategy (for external inspection)."""
    return _get_exclude_keys(model.state_dict(), strategy)
