"""PC²-FedReorg layer-wise gated aggregation (Phase 3).

Replaces the single-output ``aggregate_fn`` returning one ``avg_dict`` with a
per-target aggregator returning, for each target client j, an aggregated
state_dict whose entries follow per-key gates:

    encoder*    -> T_repr
    adapter*    -> T_adapter
    head*       -> T_head
    calibration -> skip (always local)
    bn / norm / running_* -> skip (always local)

Aggregation weights are target-wise normalized via
``transferability.target_wise_normalize``: ``W[i,j] = T[i,j] * n_i /
sum_k(T[k,j] * n_k)``. When the full column for target j is zero, the
fallback degenerates to local-only (W[j,j]=1), so theta_j_new = theta_j_old.

This module is framework-light: it only consumes torch tensors held in
state_dicts and a few Python dicts. It does not import the model definition,
training loop, or federated strategy registry.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.transferability import target_wise_normalize  # noqa: E402


# ---------- Per-key routing ----------

LAYER_KIND_CALIBRATION = 'calibration'
LAYER_KIND_BN_NORM = 'bn_norm'
LAYER_KIND_ADAPTER = 'adapter'
LAYER_KIND_HEAD = 'head'
LAYER_KIND_ENCODER = 'encoder'

_BN_NORM_SUBSTRINGS = ('bn', 'norm', 'running_mean', 'running_var')


def classify_key(key: str) -> str:
    """Return which gate governs the given state_dict key.

    Order of precedence:
        'calibration' substring        -> 'calibration'  (skip; always local)
        'adapter' substring            -> 'adapter'      (gated by T_adapter)
        bn / norm / running_*          -> 'bn_norm'      (skip; always local)
        'head' substring               -> 'head'         (gated by T_head)
        otherwise                      -> 'encoder'      (gated by T_repr)

    The 'adapter' check is placed before bn/norm so that adapter-internal
    LayerNorm or BatchNorm parameters (if ever introduced) ride with the
    adapter gate rather than being silently kept local. Calibration takes
    absolute precedence; its keys are never aggregated by anything.
    """
    if 'calibration' in key:
        return LAYER_KIND_CALIBRATION
    if 'adapter' in key:
        return LAYER_KIND_ADAPTER
    if any(sub in key for sub in _BN_NORM_SUBSTRINGS):
        return LAYER_KIND_BN_NORM
    if 'head' in key:
        return LAYER_KIND_HEAD
    return LAYER_KIND_ENCODER


def _gate_for_kind(kind: str, T_matrices: dict) -> Optional[dict]:
    """Return the dict[i][j] -> float gate for a layer kind, or None for skip."""
    if kind == LAYER_KIND_CALIBRATION or kind == LAYER_KIND_BN_NORM:
        return None
    if kind == LAYER_KIND_ADAPTER:
        return T_matrices['T_adapter']
    if kind == LAYER_KIND_HEAD:
        return T_matrices['T_head']
    return T_matrices['T_repr']  # encoder default


# ---------- Aggregation entry point ----------

def pc2_fed_aggregate(client_state_dicts: Dict[str, OrderedDict],
                     client_sample_counts: Dict[str, int],
                     T_matrices: dict,
                     target_client: str) -> OrderedDict:
    """Aggregate state_dicts for a single ``target_client`` under PC²-FedReorg.

    Parameters
    ----------
    client_state_dicts : dict[str, OrderedDict]
        Map from client key (must be present in ``T_matrices``) to its
        full ``state_dict``. All clients must share the same key schema.
    client_sample_counts : dict[str, int]
        Per-client sample count ``n_i`` used as weight multiplier.
    T_matrices : dict
        ``{'T_repr', 'T_head', 'T_adapter'}``; each is ``dict[i][j] -> float``.
    target_client : str
        Which client j the result is aggregated for.

    Returns
    -------
    OrderedDict
        Aggregated tensors for the keys that are NOT skipped (i.e. excludes
        calibration, bn/norm, running_*). The caller keeps the target's own
        values for the absent keys.

    Notes
    -----
    - Calibration keys are dropped from the output even if they were present
      in the input state_dict; the universal "never aggregate calibration"
      contract is enforced here too, redundantly with src/federated.py:_get_exclude_keys.
    - Sample counts ``n_i`` enter only as a multiplier in target_wise_normalize;
      a zero ``T[i,j]`` produces a zero weight regardless of how large ``n_i``
      is. This is the ``T_gate >> n_i`` invariant.
    - All-zero column (every source gated out for j, including self T[j,j]=0)
      falls back to weight[j,j]=1 -> result equals the target's own values.
    """
    if target_client not in client_state_dicts:
        raise ValueError(f'target_client {target_client!r} not in state_dicts '
                         f'{list(client_state_dicts.keys())}')

    node_keys = list(client_state_dicts.keys())
    missing_n = [k for k in node_keys if k not in client_sample_counts]
    if missing_n:
        raise ValueError(f'client_sample_counts missing entries for {missing_n}')

    # Pre-compute target-wise weights once per kind (encoder / adapter / head).
    # Cache: kind -> dict[i][j] -> float (j fixed implicitly via target_client).
    weights_per_kind = {}
    for kind in (LAYER_KIND_ENCODER, LAYER_KIND_ADAPTER, LAYER_KIND_HEAD):
        T = _gate_for_kind(kind, T_matrices)
        weights_per_kind[kind] = target_wise_normalize(
            T, client_sample_counts, node_keys=node_keys
        )

    schema = client_state_dicts[node_keys[0]]
    out: OrderedDict = OrderedDict()

    for key in schema:
        kind = classify_key(key)
        # Skipped layers: omit from output.
        if kind in (LAYER_KIND_CALIBRATION, LAYER_KIND_BN_NORM):
            continue

        sample = schema[key]
        # Skip non-floating-point buffers (e.g. num_batches_tracked, ints).
        if not sample.is_floating_point():
            continue

        weights = weights_per_kind[kind]

        agg = None
        for source in node_keys:
            w = weights[source][target_client]
            if w == 0.0:
                continue
            tensor = client_state_dicts[source][key].float()
            if agg is None:
                agg = w * tensor
            else:
                agg = agg + w * tensor
        if agg is None:
            # Defensive fallback (target_wise_normalize already guarantees self
            # weight when the column would otherwise be all-zero).
            agg = client_state_dicts[target_client][key].float()
        out[key] = agg

    return out


def pc2_fed_aggregate_all(client_state_dicts: Dict[str, OrderedDict],
                          client_sample_counts: Dict[str, int],
                          T_matrices: dict,
                          targets: Optional[Iterable[str]] = None) -> Dict[str, OrderedDict]:
    """Convenience: aggregate for every target. Returns dict-of-dicts.

    If ``targets`` is given, only those clients receive an aggregated dict;
    the rest are absent from the result.
    """
    if targets is None:
        targets = list(client_state_dicts.keys())
    return {
        t: pc2_fed_aggregate(client_state_dicts, client_sample_counts,
                             T_matrices, t)
        for t in targets
    }


# ---------- Task-client mapping (Phase 4) ----------

def task_client_for(client_key: str, target_type: Optional[str]) -> str:
    """Map a federation client key (e.g. 'client_a') to the task-client node
    name used in the 5x5 transferability matrix ('A', 'B', 'C-hole',
    'C-triplet', 'D').

    Required because Client C is split into two task-client nodes (one per
    quantity); the rest map 1-to-1.

    Parameters
    ----------
    client_key : str
        One of {'client_a', 'client_b', 'client_c', 'client_d'}.
    target_type : Optional[str]
        'hole' or 'triplet'; required for ``client_c``, ignored otherwise.

    Returns
    -------
    str
        One of {'A', 'B', 'C-hole', 'C-triplet', 'D'}.
    """
    if client_key == 'client_a':
        return 'A'
    if client_key == 'client_b':
        return 'B'
    if client_key == 'client_d':
        return 'D'
    if client_key == 'client_c':
        if target_type == 'hole':
            return 'C-hole'
        if target_type == 'triplet':
            return 'C-triplet'
        raise ValueError(
            f"client_c requires target_type='hole' or 'triplet'; got {target_type!r}"
        )
    raise ValueError(f'unknown client_key {client_key!r}')


# ---------- High-level dispatch (Phase 4) ----------

def load_T_matrices_from_json(path) -> dict:
    """Read T_transferability.json and return a dict with the three matrices."""
    import json
    payload = json.loads(Path(path).read_text())
    return {
        'T_repr':    payload['T_repr'],
        'T_head':    payload['T_head'],
        'T_adapter': payload['T_adapter'],
    }


def pc2_fed_dispatch(client_models_cpu: list,
                     client_keys: list,
                     client_sample_counts: Dict[str, int],
                     target_type: Optional[str],
                     transferability_path) -> Dict[str, OrderedDict]:
    """High-level PC²-FedReorg aggregation entry for the training loop.

    Replaces a single global ``aggregate_fn(models, weights) -> avg_dict`` call
    with a per-target dict-of-dicts indexed by *client_key* (not task-client).

    Parameters
    ----------
    client_models_cpu : list[nn.Module]
        Aligned with ``client_keys``; each model already on CPU with its
        full ``state_dict`` reflecting the latest local training round.
    client_keys : list[str]
        Federation client keys (e.g. ['client_a', 'client_b', 'client_c']).
    client_sample_counts : dict[str, int]
        ``n_i`` per client_key (used by target_wise_normalize).
    target_type : Optional[str]
        Required when ``client_c`` participates; passed to ``task_client_for``.
    transferability_path : str | Path
        Path to ``T_transferability.json`` (output of Phase 1).

    Returns
    -------
    dict[client_key -> OrderedDict]
        Per-client aggregated state_dict to load back into each local model.
        Skipped layers (calibration / bn / norm) are absent and left local.
    """
    if len(client_keys) != len(client_models_cpu):
        raise ValueError('client_keys and client_models_cpu must align')

    task_map = {ck: task_client_for(ck, target_type) for ck in client_keys}

    # Build state_dict map keyed by task-client name (matching T_matrices).
    state_dicts = {
        task_map[ck]: m.state_dict()
        for ck, m in zip(client_keys, client_models_cpu)
    }
    n_kept_task = {task_map[ck]: client_sample_counts[ck] for ck in client_keys}

    T_matrices = load_T_matrices_from_json(transferability_path)

    targets_task = list(state_dicts.keys())
    aggregated_task = pc2_fed_aggregate_all(
        state_dicts, n_kept_task, T_matrices, targets=targets_task
    )

    return {ck: aggregated_task[task_map[ck]] for ck in client_keys}
