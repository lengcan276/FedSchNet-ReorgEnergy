"""PC²-FedReorg transferability matrices (Phase 1).

Three matrices over the 5 task-client nodes {A, B, C-hole, C-triplet, D}:

  T_repr     -- chemistry-driven (Tanimoto K + Murcko-scaffold S).
                T_repr[i, j] = max(eps, w_K * K[i, j] + w_S * S[i, j])
                K, S come from results/preverify/T_chemistry.json (Phase 0 output).

  T_head     -- conservative protocol gate. Off-diagonal entries are 0 unless
                quantity matches AND (same source_id OR all method fields known
                and matching). Even if compatible, T_head < cutoff is forced 0.
                When compatible, T_head = L_ij^gamma where L_ij combines
                Wasserstein-1 distance and the std ratio of label samples.

  T_adapter  -- geometric mean: max(eps, sqrt(T_repr * T_head)). Whenever
                T_head[i,j] = 0, T_adapter[i,j] floors to T_ADAPTER_FLOOR.

Diagonals are 1.0 by definition. The matrices are deterministic functions of
preverify outputs and PROTOCOL_META; they do not change during training.

This module is framework-agnostic: numpy / scipy / json only, no torch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
T_CHEMISTRY_JSON = PROJECT_ROOT / 'results' / 'preverify' / 'T_chemistry.json'
OUT_JSON = PROJECT_ROOT / 'results' / 'preverify' / 'T_transferability.json'
OUT_MD = PROJECT_ROOT / 'results' / 'preverify' / 'T_transferability.md'

# ---------- locked parameters (no post-hoc tuning per user instruction) ----------

NODE_KEYS = ['A', 'B', 'C-hole', 'C-triplet', 'D']

W_K = 0.6
W_S = 0.4
T_REPR_FLOOR = 1e-3

ALPHA_W = 5.0          # exp(-ALPHA_W * W1) decay in label-distribution score
GAMMA_L = 1.0          # L_ij exponent in T_head
T_HEAD_CUTOFF = 0.01   # if 0 < T_head < cutoff, force to 0

T_ADAPTER_FLOOR = 1e-3

# ---------- protocol metadata ----------

# Per-node DFT/measurement protocol. Any 'unknown' field on either side of an
# ordered pair drops T_head to 0 (clause 2). 'source_id' is a shortcut: same
# source means same protocol by construction (no need to match every field).
PROTOCOL_META = {
    'A': {
        'quantity': 'reorg_energy_general',  # column reorg_energy_eV; charge state not specified
        'source_id': 'qm9_public_reorg_15210',
        'functional': 'unknown',
        'basis': 'unknown',
        'geometry': 'unknown',
        'charge_state': 'unknown',
        'conformers': 'unknown',
    },
    'B': {
        'quantity': 'reorg_energy_general',
        'source_id': 'qm9_public_reorg_15210',  # same CSV as A
        'functional': 'unknown',
        'basis': 'unknown',
        'geometry': 'unknown',
        'charge_state': 'unknown',
        'conformers': 'unknown',
    },
    'C-hole': {
        'quantity': 'hole_lambda',
        'source_id': 'client_c_dft',
        'functional': 'wB97XD',
        'basis': 'unknown_in_csv',
        'geometry': 'adiabatic_4point_nelsen',
        'charge_state': 'neutral_to_cation',
        'conformers': 'one_per_molecule',
    },
    'C-triplet': {
        'quantity': 'triplet_lambda',
        'source_id': 'client_c_dft',
        'functional': 'wB97XD',
        'basis': 'unknown_in_csv',
        'geometry': 'adiabatic_4point_nelsen',
        'charge_state': 'neutral_to_T1',
        'conformers': 'one_per_molecule',
    },
    'D': {
        'quantity': 'hole_lambda',
        'source_id': 'atahan_2019',
        'functional': 'B3LYP',
        'basis': '6-31G*',
        'geometry': 'unknown',       # not stated in abstract
        'charge_state': 'unknown',
        'conformers': 'unknown',
    },
}


# ---------- T_head clauses ----------

def _is_unknown(value: str) -> bool:
    """A field counts as 'unknown' if it equals 'unknown' or starts with 'unknown'."""
    return isinstance(value, str) and value.lower().startswith('unknown')


def t_head_compatibility(src: str, dst: str,
                         meta_src: dict, meta_dst: dict) -> tuple:
    """Conservative protocol gate. Returns (is_compatible, reason).

    Clause order (the *first* failing clause is reported):
      1. quantity mismatch -> 0
      4. (src='D' and dst in {C-hole, C-triplet}) -> 0  (V1=FAIL hard rule;
         applied early so D->C is always reported with this reason even if a
         later clause would also fire)
      shortcut. same source_id -> compatible (same dataset/lab).
      2. any method field 'unknown' on either side -> 0
      3. method field differs -> 0
    """
    # Clause 1 (most common; check first so D->C-triplet is reported as
    # quantity-mismatch rather than D->C hard rule per user spec).
    if meta_src['quantity'] != meta_dst['quantity']:
        return False, 'clause 1: quantity mismatch'
    # Clause 4: explicit V1=FAIL hard rule for D->C* even when quantity matches.
    if src == 'D' and dst in ('C-hole', 'C-triplet'):
        return False, 'clause 4: V1=FAIL hard rule (D -> C)'
    # Same source -> compatible.
    if meta_src['source_id'] == meta_dst['source_id']:
        return True, None
    # Different source: every method field must be known and matching.
    for field in ('functional', 'basis', 'geometry', 'charge_state', 'conformers'):
        v_src, v_dst = meta_src.get(field, 'unknown'), meta_dst.get(field, 'unknown')
        if _is_unknown(v_src) or _is_unknown(v_dst):
            return False, f'clause 2: {field} unknown'
        if v_src != v_dst:
            return False, f'clause 3: {field} differs ({v_src} vs {v_dst})'
    return True, None


def label_distribution_score(samples_src: np.ndarray,
                             samples_dst: np.ndarray) -> float:
    """L_ij = exp(-alpha * W1) * min(sigma_i / sigma_j, sigma_j / sigma_i).

    alpha is ALPHA_W. Returns a scalar in [0, 1]; 1 only when distributions
    coincide and stds are equal.
    """
    samples_src = np.asarray(samples_src, dtype=float)
    samples_dst = np.asarray(samples_dst, dtype=float)
    w1 = float(wasserstein_distance(samples_src, samples_dst))
    sigma_src = float(np.std(samples_src))
    sigma_dst = float(np.std(samples_dst))
    if sigma_src <= 0 or sigma_dst <= 0:
        sigma_term = 0.0
    else:
        sigma_term = float(min(sigma_src / sigma_dst, sigma_dst / sigma_src))
    return math.exp(-ALPHA_W * w1) * sigma_term


# ---------- Top-level: compute matrices ----------

def compute_T_matrices(t_chemistry: Optional[dict] = None,
                       protocol_meta: Optional[dict] = None,
                       label_samples: Optional[dict] = None) -> dict:
    """Compute T_repr, T_head, T_adapter as nested dicts.

    Parameters
    ----------
    t_chemistry : optional override of T_chemistry.json contents.
    protocol_meta : optional override of PROTOCOL_META.
    label_samples : optional override, dict[node] -> 1-D np.ndarray of labels.
                    Required only for ordered pairs that survive the protocol
                    gate; if None, samples are loaded lazily on demand.
    """
    if t_chemistry is None:
        t_chemistry = json.loads(T_CHEMISTRY_JSON.read_text())
    K = t_chemistry['K']
    S = t_chemistry['S']

    if protocol_meta is None:
        protocol_meta = PROTOCOL_META

    # ----- T_repr -----
    T_repr = {i: {} for i in NODE_KEYS}
    for i in NODE_KEYS:
        for j in NODE_KEYS:
            if i == j:
                T_repr[i][j] = 1.0
            else:
                T_repr[i][j] = max(T_REPR_FLOOR, W_K * K[i][j] + W_S * S[i][j])

    # ----- T_head -----
    # Lazy label loading: only fetch samples once we hit a compatible pair.
    samples_cache = dict(label_samples) if label_samples else {}

    def _get_samples(node):
        if node not in samples_cache:
            if label_samples is not None:
                raise KeyError(f'label_samples missing entry for {node!r}')
            samples_cache.update(_load_label_samples_for_nodes([node]))
        return samples_cache[node]

    T_head = {i: {} for i in NODE_KEYS}
    reasons = {i: {} for i in NODE_KEYS}
    for i in NODE_KEYS:
        for j in NODE_KEYS:
            if i == j:
                T_head[i][j] = 1.0
                reasons[i][j] = None
                continue
            ok, reason = t_head_compatibility(i, j, protocol_meta[i], protocol_meta[j])
            if not ok:
                T_head[i][j] = 0.0
                reasons[i][j] = reason
                continue
            L = label_distribution_score(_get_samples(i), _get_samples(j))
            t_h = L ** GAMMA_L
            if t_h < T_HEAD_CUTOFF:
                T_head[i][j] = 0.0
                reasons[i][j] = f'cutoff: T_head={t_h:.3e} < {T_HEAD_CUTOFF}'
            else:
                T_head[i][j] = float(t_h)
                reasons[i][j] = None

    # ----- T_adapter -----
    T_adapter = {i: {} for i in NODE_KEYS}
    for i in NODE_KEYS:
        for j in NODE_KEYS:
            if i == j:
                T_adapter[i][j] = 1.0
            else:
                t_a = math.sqrt(max(0.0, T_repr[i][j] * T_head[i][j]))
                T_adapter[i][j] = max(T_ADAPTER_FLOOR, t_a)

    return {
        'T_repr': T_repr,
        'T_head': T_head,
        'T_adapter': T_adapter,
        'reasons': reasons,
    }


# ---------- Aggregation helper (used in Phase 3) ----------

def target_wise_normalize(T: dict, n_kept: dict,
                          node_keys: Optional[list] = None) -> dict:
    """Normalize T into per-target aggregation weights.

    weights[i, j] = T[i, j] * n_i / sum_k(T[k, j] * n_k)

    For each target j the weights sum to 1 over sources i. If the denominator
    is 0 (every source gated out, including self) the target is left local:
    weight[j, j] = 1 and all other entries are 0.

    Parameters
    ----------
    T, n_kept : nested dicts indexed by node_keys.
    node_keys : optional explicit ordering. Defaults to ``NODE_KEYS``; use a
        smaller subset to support tests / partial-federation scenarios.
    """
    if node_keys is None:
        node_keys = NODE_KEYS
    weights = {i: {} for i in node_keys}
    for j in node_keys:
        denom = sum(T[i][j] * n_kept[i] for i in node_keys)
        if denom <= 0:
            for i in node_keys:
                weights[i][j] = 1.0 if i == j else 0.0
        else:
            for i in node_keys:
                weights[i][j] = T[i][j] * n_kept[i] / denom
    return weights


# ---------- Label loaders (lazy) ----------

def _load_label_samples_for_nodes(nodes: list) -> dict:
    """Load label samples for a subset of nodes; cached implicitly by caller."""
    sys.path.insert(0, str(PROJECT_ROOT))
    out = {}
    need_pub = any(n in ('A', 'B') for n in nodes)
    if need_pub:
        from src.data_utils import load_public_data, split_public_by_aromaticity
        df_pub = load_public_data()
        a_df, b_df = split_public_by_aromaticity(df_pub)
        if 'A' in nodes:
            out['A'] = a_df['reorg_energy_eV'].to_numpy(dtype=float)
        if 'B' in nodes:
            out['B'] = b_df['reorg_energy_eV'].to_numpy(dtype=float)
    if 'C-hole' in nodes:
        from src.data_utils import load_client_c_hole
        out['C-hole'] = load_client_c_hole()['lambda_hole_eV'].to_numpy(dtype=float)
    if 'C-triplet' in nodes:
        from src.data_utils import load_client_c_triplet
        out['C-triplet'] = load_client_c_triplet()['lambda_T_total_eV'].to_numpy(dtype=float)
    if 'D' in nodes:
        from src.data_utils import load_client_d
        out['D'] = load_client_d()['reorg_eV'].to_numpy(dtype=float)
    return out


# ---------- Persistence ----------

def save_T_matrices(matrices: dict, n_kept: dict,
                    out_json: Path = OUT_JSON, out_md: Path = OUT_MD,
                    label_means: Optional[dict] = None,
                    label_stds: Optional[dict] = None):
    """Persist T matrices to JSON and a human-readable markdown table."""
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'description': (
            'PC²-FedReorg transferability matrices over 5 task-client nodes. '
            'T_repr is chemistry-driven; T_head is a conservative protocol '
            'gate; T_adapter is the geometric mean. Diagonals = 1.0. '
            'See src/transferability.py for the formulas and clause definitions.'
        ),
        'node_keys': NODE_KEYS,
        'parameters': {
            'w_K': W_K, 'w_S': W_S, 'T_repr_floor': T_REPR_FLOOR,
            'alpha_W': ALPHA_W, 'gamma_L': GAMMA_L, 'T_head_cutoff': T_HEAD_CUTOFF,
            'T_adapter_floor': T_ADAPTER_FLOOR,
        },
        'protocol_meta': PROTOCOL_META,
        'n_kept': n_kept,
        'label_means': label_means or {},
        'label_stds': label_stds or {},
        'T_repr': matrices['T_repr'],
        'T_head': matrices['T_head'],
        'T_adapter': matrices['T_adapter'],
        'reasons': matrices['reasons'],
    }
    out_json.write_text(json.dumps(payload, indent=2))

    # ----- markdown -----
    lines = ['# PC²-FedReorg transferability matrices (Phase 1)\n']
    lines.append('Computed from `T_chemistry.json` (Phase 0) and `PROTOCOL_META`')
    lines.append('via `src/transferability.py:compute_T_matrices`. **No post-hoc tuning** — ')
    lines.append(f'parameters are locked at w_K={W_K}, w_S={W_S}, alpha_W={ALPHA_W}, '
                 f'gamma_L={GAMMA_L}, cutoff={T_HEAD_CUTOFF}.')
    lines.append('')

    lines.append('## Protocol metadata\n')
    lines.append('| node | quantity | source_id | functional | basis | geometry | charge_state | conformers |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for k in NODE_KEYS:
        m = PROTOCOL_META[k]
        lines.append(f"| {k} | {m['quantity']} | {m['source_id']} | {m['functional']} | "
                     f"{m['basis']} | {m['geometry']} | {m['charge_state']} | {m['conformers']} |")
    lines.append('')

    def _matrix_md(label, M):
        out = [f'## {label} (rows = source i, cols = target j)\n']
        out.append('| i \\\\ j | ' + ' | '.join(NODE_KEYS) + ' |')
        out.append('|---|' + '---:|' * len(NODE_KEYS))
        for i in NODE_KEYS:
            cells = [f'{M[i][j]:.4f}' for j in NODE_KEYS]
            out.append(f'| {i} | ' + ' | '.join(cells) + ' |')
        out.append('')
        return out

    lines.extend(_matrix_md('T_repr', matrices['T_repr']))
    lines.extend(_matrix_md('T_head', matrices['T_head']))
    lines.extend(_matrix_md('T_adapter', matrices['T_adapter']))

    lines.append('## Why each off-diagonal T_head is 0\n')
    lines.append('| src | dst | T_head | reason |')
    lines.append('|---|---|---:|---|')
    for i in NODE_KEYS:
        for j in NODE_KEYS:
            if i == j:
                continue
            t = matrices['T_head'][i][j]
            r = matrices['reasons'][i][j] or '(compatible; T_head from L_ij)'
            lines.append(f'| {i} | {j} | {t:.4f} | {r} |')
    lines.append('')

    lines.append('## Locked invariants (must hold; verified in tests/test_transferability.py)\n')
    lines.append('- T_head[D, C-hole] == 0  (clause 4: V1=FAIL hard rule)')
    lines.append('- T_head[D, C-triplet] == 0  (clause 1: quantity mismatch)')
    lines.append('- T_head[C-hole, C-triplet] == 0  (clause 1)')
    lines.append('- T_head[C-triplet, C-hole] == 0  (clause 1)')
    lines.append('- T_repr[C-hole, C-triplet] > 0.95')
    lines.append('- T_repr[D, C-hole] < 0.10')
    lines.append('- T_repr[B, D] > 0.05  AND  T_head[B, D] == 0  '
                 '(chemistry-share OK; head-share NOT OK)')
    lines.append('')

    out_md.write_text('\n'.join(lines))


# ---------- entry ----------

def main():
    t_chem = json.loads(T_CHEMISTRY_JSON.read_text())
    n_kept = t_chem['n_kept']

    # Eagerly load all label samples so we can record their stats in the JSON.
    label_samples = _load_label_samples_for_nodes(NODE_KEYS)
    label_means = {k: float(np.mean(v)) for k, v in label_samples.items()}
    label_stds = {k: float(np.std(v)) for k, v in label_samples.items()}

    matrices = compute_T_matrices(
        t_chemistry=t_chem,
        protocol_meta=PROTOCOL_META,
        label_samples=label_samples,
    )

    save_T_matrices(matrices, n_kept,
                    label_means=label_means, label_stds=label_stds)
    print(f'[Phase 1] -> {OUT_JSON}')
    print(f'[Phase 1] -> {OUT_MD}')
    print()
    print('T_head off-diagonal nonzero entries:')
    any_nonzero = False
    for i in NODE_KEYS:
        for j in NODE_KEYS:
            if i != j and matrices['T_head'][i][j] > 0:
                print(f'  T_head[{i:>9s} -> {j:<9s}] = {matrices["T_head"][i][j]:.4f}')
                any_nonzero = True
    if not any_nonzero:
        print('  (all off-diagonal T_head are 0 under the conservative gate)')


if __name__ == '__main__':
    main()
