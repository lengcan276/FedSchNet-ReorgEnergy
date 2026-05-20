"""Phase 3 tests for PC²-FedReorg aggregation and federated safety fix.

Run with:
    python -m unittest tests.test_pc2_aggregation -v
or (if pytest is installed):
    python -m pytest tests/test_pc2_aggregation.py -v

Covers (per user spec for Phase 3 sign-off):

  - calibration keys are never aggregated (PC² + FedAvg + FedProx + FedPer + FedBN);
  - T_head[D, C-hole]=0 -> D's head params do NOT enter C-hole;
  - C-hole / C-triplet share encoder strongly but heads are private;
  - A and B share heads via T_head[A,B]=0.33 (only nonzero off-diagonal);
  - self-only column degenerates to local-only (no NaN, no uniform fallback);
  - target-wise normalization sums to 1 per target;
  - sample count n_i is just a multiplier; T gate of 0 dominates regardless of n_i;
  - per-key routing (classify_key) is correct for every layer kind.
"""

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.federated import (  # noqa: E402
    _get_exclude_keys, fedavg_strategy, fedper_aggregate, fedbn_aggregate,
    fedper_pc2_aggregate, fedbn_pc2_aggregate, get_aggregation_fn,
)
from src.transferability import (  # noqa: E402
    NODE_KEYS, compute_T_matrices, target_wise_normalize,
)
from experiments.pc2_fedreorg import (  # noqa: E402
    LAYER_KIND_ADAPTER, LAYER_KIND_BN_NORM, LAYER_KIND_CALIBRATION,
    LAYER_KIND_ENCODER, LAYER_KIND_HEAD,
    classify_key, pc2_fed_aggregate, pc2_fed_aggregate_all,
)
from tests.test_transferability import (  # noqa: E402
    _phase0_chemistry, _synthetic_label_samples,
)


# --------------------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------------------

def _make_state_dict(seed: int) -> OrderedDict:
    """Build a state_dict that mimics ReorgEnergyModel(use_adapter=True, use_calibration=True).

    Includes encoder convs + BN + projection, adapter (mlp residual), KAN head,
    and calibration buffers. Each tensor uses a per-seed RNG so different
    clients get different values.
    """
    g = torch.Generator().manual_seed(seed)
    return OrderedDict([
        ('encoder.convs.0.nn.0.weight',     torch.randn(8, 8, generator=g)),
        ('encoder.convs.0.nn.0.bias',       torch.randn(8, generator=g)),
        ('encoder.bns.0.weight',            torch.randn(8, generator=g)),
        ('encoder.bns.0.bias',              torch.randn(8, generator=g)),
        ('encoder.bns.0.running_mean',      torch.randn(8, generator=g)),
        ('encoder.bns.0.running_var',       torch.rand(8, generator=g) + 0.1),
        ('encoder.project.weight',          torch.randn(8, 8, generator=g)),
        ('encoder.project.bias',            torch.randn(8, generator=g)),
        ('adapter.fc1.weight',              torch.randn(4, 8, generator=g)),
        ('adapter.fc1.bias',                torch.randn(4, generator=g)),
        ('adapter.fc2.weight',              torch.randn(8, 4, generator=g)),
        ('adapter.fc2.bias',                torch.randn(8, generator=g)),
        ('head.kan1.base_weight',           torch.randn(4, 8, generator=g)),
        ('head.kan2.base_weight',           torch.randn(1, 4, generator=g)),
        ('calibration.mu',                  torch.tensor(1.0 + seed * 0.1)),
        ('calibration.sigma',               torch.tensor(0.5 + seed * 0.05)),
    ])


def _build_T_matrices_and_n():
    """Compute real T_matrices using Phase 0 chemistry + synthetic labels."""
    chem = _phase0_chemistry()
    samples = _synthetic_label_samples()
    M = compute_T_matrices(t_chemistry=chem, label_samples=samples)
    return M, chem['n_kept']


# --------------------------------------------------------------------------
# classify_key
# --------------------------------------------------------------------------

class TestClassifyKey(unittest.TestCase):

    def test_calibration_takes_precedence(self):
        self.assertEqual(classify_key('calibration.mu'), LAYER_KIND_CALIBRATION)
        self.assertEqual(classify_key('calibration.sigma'), LAYER_KIND_CALIBRATION)

    def test_adapter_routes_to_adapter(self):
        self.assertEqual(classify_key('adapter.fc1.weight'), LAYER_KIND_ADAPTER)
        self.assertEqual(classify_key('adapter.fc2.bias'), LAYER_KIND_ADAPTER)

    def test_adapter_norm_routes_to_adapter(self):
        # If a future adapter ever adds a LayerNorm, that param should ride
        # the adapter gate, not be silently kept local.
        self.assertEqual(classify_key('adapter.norm.weight'), LAYER_KIND_ADAPTER)

    def test_bn_running_stats_skipped(self):
        self.assertEqual(classify_key('encoder.bns.0.weight'),       LAYER_KIND_BN_NORM)
        self.assertEqual(classify_key('encoder.bns.0.running_mean'), LAYER_KIND_BN_NORM)
        self.assertEqual(classify_key('encoder.bns.0.running_var'),  LAYER_KIND_BN_NORM)

    def test_head_routes_to_head(self):
        self.assertEqual(classify_key('head.kan1.base_weight'), LAYER_KIND_HEAD)
        self.assertEqual(classify_key('head.mlp.2.bias'),       LAYER_KIND_HEAD)

    def test_encoder_default(self):
        self.assertEqual(classify_key('encoder.convs.0.nn.0.weight'), LAYER_KIND_ENCODER)
        self.assertEqual(classify_key('encoder.project.weight'),      LAYER_KIND_ENCODER)


# --------------------------------------------------------------------------
# Per-key correctness: PC² aggregation
# --------------------------------------------------------------------------

class TestPC2AggregationKeyRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.state_dicts = {k: _make_state_dict(i) for i, k in enumerate(NODE_KEYS)}
        cls.M, cls.n_kept = _build_T_matrices_and_n()

    def test_calibration_keys_absent_from_output(self):
        """Calibration is never aggregated -> never appears in PC² output."""
        for target in NODE_KEYS:
            out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, target)
            cal_keys = [k for k in out if 'calibration' in k]
            self.assertEqual(cal_keys, [], msg=f'target={target}: leaked {cal_keys}')

    def test_bn_running_keys_absent_from_output(self):
        """BN / running stats are kept local -> not in PC² output."""
        for target in NODE_KEYS:
            out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, target)
            bn_keys = [k for k in out if any(s in k for s in ('bn', 'running_'))]
            self.assertEqual(bn_keys, [],
                             msg=f'target={target}: BN/running leaked {bn_keys}')

    def test_d_head_does_not_enter_c_hole(self):
        """T_head[D, C-hole]=0 + every other off-diagonal source also 0 ->
        aggregated head for C-hole equals C-hole's own."""
        out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, 'C-hole')
        for key in self.state_dicts['C-hole']:
            if classify_key(key) != LAYER_KIND_HEAD:
                continue
            self.assertIn(key, out)
            self.assertTrue(
                torch.allclose(out[key], self.state_dicts['C-hole'][key].float()),
                msg=f'C-hole head key {key} aggregated; D or others leaked'
            )

    def test_c_hole_to_c_triplet_head_zero(self):
        """T_head[C-hole, C-triplet]=0 -> head for C-triplet equals own."""
        out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, 'C-triplet')
        for key in self.state_dicts['C-triplet']:
            if classify_key(key) != LAYER_KIND_HEAD:
                continue
            self.assertTrue(
                torch.allclose(out[key], self.state_dicts['C-triplet'][key].float()),
                msg=f'C-triplet head key {key} aggregated; C-hole or others leaked'
            )

    def test_c_triplet_encoder_aggregated(self):
        """Encoder for C-triplet must aggregate from other sources (T_repr > 0
        for every i, including high overlap from C-hole). Result should differ
        from C-triplet's own values."""
        out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, 'C-triplet')
        any_diff = False
        for key in self.state_dicts['C-triplet']:
            if classify_key(key) != LAYER_KIND_ENCODER:
                continue
            self.assertIn(key, out)
            if not torch.allclose(out[key], self.state_dicts['C-triplet'][key].float(),
                                  atol=1e-12):
                any_diff = True
                break
        self.assertTrue(any_diff, msg='no encoder key changed for C-triplet under PC² '
                                       '(expected aggregation from at least C-hole / A / B)')

    def test_a_b_head_aggregated_via_nonzero_t_head(self):
        """T_head[A, B] = 0.33 (only nonzero off-diagonal head). Aggregated
        head for B should differ from B's own."""
        out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, 'B')
        any_diff = False
        for key in self.state_dicts['B']:
            if classify_key(key) != LAYER_KIND_HEAD:
                continue
            self.assertIn(key, out)
            if not torch.allclose(out[key], self.state_dicts['B'][key].float(),
                                  atol=1e-12):
                any_diff = True
                break
        self.assertTrue(any_diff, msg='B head unchanged; expected nonzero contribution from A')

    def test_d_head_self_only_under_pc2(self):
        """D's column for T_head: T_head[i, D]=0 for all i!=D (D's protocol
        unknown, or quantity differs from A/B/C-triplet). So D's head stays
        local. (Only T_head[D, D] = 1.)"""
        out = pc2_fed_aggregate(self.state_dicts, self.n_kept, self.M, 'D')
        for key in self.state_dicts['D']:
            if classify_key(key) != LAYER_KIND_HEAD:
                continue
            self.assertTrue(
                torch.allclose(out[key], self.state_dicts['D'][key].float()),
                msg=f'D head key {key} aggregated; expected self-only'
            )


# --------------------------------------------------------------------------
# Sample-count vs T-gate semantics (rule 8)
# --------------------------------------------------------------------------

class TestSampleCountDoesNotOverrideTGate(unittest.TestCase):
    """Even with n_D enormous, T_head[D, C-hole] = 0 must keep D's head out."""

    def test_huge_n_d_still_zero_contribution(self):
        # Two-node sub-federation: just C-hole and D.
        state_dicts = {
            'C-hole': _make_state_dict(11),
            'D':      _make_state_dict(22),
        }
        # n_D set to absurd size; n_C_hole tiny.
        n_kept = {'C-hole': 53, 'D': 10_000_000}

        # 2x2 sub-matrices preserving the real values.
        full_M, _ = _build_T_matrices_and_n()
        sub = {}
        for kind in ('T_repr', 'T_head', 'T_adapter'):
            sub[kind] = {
                i: {j: full_M[kind][i][j] for j in state_dicts}
                for i in state_dicts
            }

        out = pc2_fed_aggregate(state_dicts, n_kept, sub, 'C-hole')

        # Head: T_head[D, C-hole]=0 -> aggregated head equals C-hole's own.
        for key in state_dicts['C-hole']:
            if classify_key(key) == LAYER_KIND_HEAD:
                self.assertTrue(
                    torch.allclose(out[key], state_dicts['C-hole'][key].float()),
                    msg=f'D head leaked into C-hole despite T_head=0; key={key}'
                )

        # Encoder: T_repr[D, C-hole] is tiny (~0.054) but n_D is huge,
        # so D's encoder should dominate. The T gate is preserved (nonzero),
        # the n_i multiplier just shifts the relative weight.
        # We just verify the aggregated encoder is not equal to C-hole's own.
        encoder_changed = False
        for key in state_dicts['C-hole']:
            if classify_key(key) != LAYER_KIND_ENCODER:
                continue
            if not torch.allclose(out[key], state_dicts['C-hole'][key].float(),
                                  atol=1e-12):
                encoder_changed = True
                break
        self.assertTrue(encoder_changed,
                        'encoder should be aggregated (T_repr nonzero) regardless of n_i scale')


# --------------------------------------------------------------------------
# Self-only fallback (rule 4)
# --------------------------------------------------------------------------

class TestSelfOnlyFallback(unittest.TestCase):
    """When the column for target j has zero off-diagonal weights, the
    aggregator must degenerate to local-only -- no NaN, no uniform fallback."""

    def test_zero_off_diagonal_yields_target_own(self):
        clients = ['x', 'y', 'z']
        state_dicts = OrderedDict([
            (c, OrderedDict([
                ('encoder.w', torch.tensor([float(i) + 1.0])),
                ('head.w',    torch.tensor([float(i) - 1.0])),
            ]))
            for i, c in enumerate(clients)
        ])
        n_kept = {c: 1 for c in clients}
        # All T off-diagonal = 0; diagonal = 1.
        T = {i: {j: (1.0 if i == j else 0.0) for j in clients} for i in clients}
        M = {'T_repr': T, 'T_head': T, 'T_adapter': T}

        out = pc2_fed_aggregate(state_dicts, n_kept, M, 'y')

        # encoder for y: weight[y, y] = 1, all others 0 -> equals y's own.
        self.assertTrue(torch.allclose(out['encoder.w'], state_dicts['y']['encoder.w'].float()))
        self.assertTrue(torch.allclose(out['head.w'],    state_dicts['y']['head.w'].float()))

    def test_zero_full_column_falls_back_to_target_own(self):
        """If even self T[j,j] is zero, fallback assigns weight[j,j]=1 -> local."""
        clients = ['x', 'y', 'z']
        state_dicts = OrderedDict([
            (c, OrderedDict([('encoder.w', torch.tensor([float(i) + 1.0]))]))
            for i, c in enumerate(clients)
        ])
        n_kept = {c: 1 for c in clients}
        # Full zero column for 'y'.
        T = {i: {j: 0.0 for j in clients} for i in clients}
        for i in clients:
            T[i][i] = 1.0  # diag stays 1 except we'll zero out 'y' below
        for i in clients:
            T[i]['y'] = 0.0
        M = {'T_repr': T, 'T_head': T, 'T_adapter': T}

        out = pc2_fed_aggregate(state_dicts, n_kept, M, 'y')
        self.assertTrue(torch.allclose(out['encoder.w'], state_dicts['y']['encoder.w'].float()),
                        msg='full-zero column for y should fall back to local')


# --------------------------------------------------------------------------
# Target-wise normalization (rule 3)
# --------------------------------------------------------------------------

class TestTargetWiseNormalization(unittest.TestCase):

    def test_sums_to_one_for_each_target_and_kind(self):
        M, n = _build_T_matrices_and_n()
        for kind in ('T_repr', 'T_head', 'T_adapter'):
            w = target_wise_normalize(M[kind], n)
            for j in NODE_KEYS:
                s = sum(w[i][j] for i in NODE_KEYS)
                self.assertAlmostEqual(s, 1.0, places=9,
                                       msg=f'{kind} target={j} sum={s}')

    def test_self_weight_present_when_self_t_one(self):
        """When T[j,j] = 1 (always, by construction), self weight must be the
        ratio n_j / sum_k(T[k,j] * n_k). This must be > 0."""
        M, n = _build_T_matrices_and_n()
        for kind in ('T_repr', 'T_head', 'T_adapter'):
            w = target_wise_normalize(M[kind], n)
            for j in NODE_KEYS:
                self.assertGreater(w[j][j], 0.0,
                                   msg=f'{kind} self weight w[{j},{j}] = 0')

    def test_subset_node_keys_works(self):
        """Parametric node_keys lets tests run on a subset of clients."""
        clients = ['A', 'B']
        T = {'A': {'A': 1.0, 'B': 0.5}, 'B': {'A': 0.5, 'B': 1.0}}
        n = {'A': 100, 'B': 100}
        w = target_wise_normalize(T, n, node_keys=clients)
        # For target B: sum = 0.5*100 + 1.0*100 = 150; self = 100/150 = 2/3.
        self.assertAlmostEqual(w['B']['B'], 2/3, places=9)
        self.assertAlmostEqual(w['A']['B'], 1/3, places=9)


# --------------------------------------------------------------------------
# Federated baseline safety (rule 6) — calibration excluded universally
# --------------------------------------------------------------------------

class TestFederatedExcludeCalibration(unittest.TestCase):

    def setUp(self):
        self.sample_dict = _make_state_dict(0)

    def test_fedavg_excludes_calibration(self):
        ex = _get_exclude_keys(self.sample_dict, 'fedavg')
        self.assertIn('calibration.mu', ex)
        self.assertIn('calibration.sigma', ex)
        # FedAvg only excludes calibration -- nothing else.
        non_calibration_excludes = [k for k in ex if 'calibration' not in k]
        self.assertEqual(non_calibration_excludes, [])

    def test_fedprox_excludes_calibration(self):
        ex = _get_exclude_keys(self.sample_dict, 'fedprox')
        self.assertIn('calibration.mu', ex)
        self.assertIn('calibration.sigma', ex)

    # ---- Legacy fedper / fedbn: PRESERVED semantics (head + bn + calibration) ----

    def test_fedper_legacy_excludes_head_bn_calibration(self):
        """Legacy fedper semantics MUST be preserved for E1-E59 reproducibility:
        exclude head + bn/norm + calibration. Adapter is NOT excluded under
        legacy fedper (legacy experiments did not use adapter)."""
        ex = _get_exclude_keys(self.sample_dict, 'fedper')
        # Universal calibration exclusion
        self.assertIn('calibration.mu', ex)
        self.assertIn('calibration.sigma', ex)
        # Legacy head exclusion
        self.assertIn('head.kan1.base_weight', ex)
        self.assertIn('head.kan2.base_weight', ex)
        # Legacy BN exclusion (preserved!)
        self.assertIn('encoder.bns.0.weight', ex)
        self.assertIn('encoder.bns.0.bias', ex)
        self.assertIn('encoder.bns.0.running_mean', ex)
        self.assertIn('encoder.bns.0.running_var', ex)
        # Adapter is NOT excluded under legacy fedper.
        self.assertNotIn('adapter.fc1.weight', ex)
        self.assertNotIn('adapter.fc2.bias', ex)
        # Encoder non-BN params shared (not excluded)
        self.assertNotIn('encoder.convs.0.nn.0.weight', ex)
        self.assertNotIn('encoder.project.weight', ex)

    def test_fedbn_legacy_excludes_head_bn_calibration(self):
        """Legacy fedbn semantics MUST be preserved: same exclusion set as
        legacy fedper (head + bn + calibration), as documented in the
        original codebase ("FedPer already skips BN, so FedBN has the same
        exclusion set")."""
        ex = _get_exclude_keys(self.sample_dict, 'fedbn')
        self.assertIn('calibration.mu', ex)
        self.assertIn('head.kan1.base_weight', ex)
        self.assertIn('encoder.bns.0.weight', ex)
        self.assertIn('encoder.bns.0.running_mean', ex)
        # Adapter NOT excluded under legacy fedbn.
        self.assertNotIn('adapter.fc1.weight', ex)
        # Encoder non-BN params shared.
        self.assertNotIn('encoder.convs.0.nn.0.weight', ex)

    def test_fedper_and_fedbn_legacy_have_same_exclusion(self):
        """Legacy invariant: fedper and fedbn produce the same exclude set
        on the standard model schema."""
        ex_fedper = _get_exclude_keys(self.sample_dict, 'fedper')
        ex_fedbn = _get_exclude_keys(self.sample_dict, 'fedbn')
        self.assertEqual(ex_fedper, ex_fedbn,
                         msg='legacy fedper / fedbn must match (E1-E59 invariant)')

    # ---- New PC²-FedReorg ablation strategies ----

    def test_fedper_pc2_excludes_head_adapter_calibration(self):
        """fedper_pc2: head + adapter + calibration excluded. BN SHARED."""
        ex = _get_exclude_keys(self.sample_dict, 'fedper_pc2')
        # Universal calibration exclusion
        self.assertIn('calibration.mu', ex)
        self.assertIn('calibration.sigma', ex)
        # head + adapter private
        self.assertIn('head.kan1.base_weight', ex)
        self.assertIn('head.kan2.base_weight', ex)
        self.assertIn('adapter.fc1.weight', ex)
        self.assertIn('adapter.fc2.bias', ex)
        # BN shared (not excluded)
        self.assertNotIn('encoder.bns.0.weight', ex)
        self.assertNotIn('encoder.bns.0.running_mean', ex)
        # Encoder non-BN shared
        self.assertNotIn('encoder.convs.0.nn.0.weight', ex)

    def test_fedbn_pc2_excludes_bn_norm_calibration_only(self):
        """fedbn_pc2: only BN/norm + calibration excluded. Head + adapter SHARED."""
        ex = _get_exclude_keys(self.sample_dict, 'fedbn_pc2')
        self.assertIn('calibration.mu', ex)
        # BN private
        self.assertIn('encoder.bns.0.weight', ex)
        self.assertIn('encoder.bns.0.bias', ex)
        self.assertIn('encoder.bns.0.running_mean', ex)
        # Head shared (not excluded)
        self.assertNotIn('head.kan1.base_weight', ex)
        self.assertNotIn('head.kan2.base_weight', ex)
        # Adapter shared
        self.assertNotIn('adapter.fc1.weight', ex)
        self.assertNotIn('adapter.fc2.bias', ex)

    def test_fedper_pc2_differs_from_legacy_fedper(self):
        """The new ablation strategy must produce a different exclude set from
        legacy fedper; otherwise it's not actually a different baseline."""
        ex_legacy = _get_exclude_keys(self.sample_dict, 'fedper')
        ex_new = _get_exclude_keys(self.sample_dict, 'fedper_pc2')
        self.assertNotEqual(ex_legacy, ex_new)

    # ---- pc2_fed and unknown ----

    def test_pc2_fed_excludes_calibration(self):
        ex = _get_exclude_keys(self.sample_dict, 'pc2_fed')
        self.assertIn('calibration.mu', ex)
        # Per-key routing handled by pc2_fed_aggregate; here we just ensure
        # calibration is filtered if anyone falls back to fedavg_aggregate.
        non_calibration_excludes = [k for k in ex if 'calibration' not in k]
        self.assertEqual(non_calibration_excludes, [])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            _get_exclude_keys(self.sample_dict, 'frobnicate')

    def test_get_aggregation_fn_dispatches_legacy_and_new(self):
        """get_aggregation_fn must register both legacy and new strategies and
        return distinct callables for them."""
        self.assertIs(get_aggregation_fn('fedavg'), fedavg_strategy)
        self.assertIs(get_aggregation_fn('fedper'), fedper_aggregate)
        self.assertIs(get_aggregation_fn('fedbn'), fedbn_aggregate)
        self.assertIs(get_aggregation_fn('fedper_pc2'), fedper_pc2_aggregate)
        self.assertIs(get_aggregation_fn('fedbn_pc2'), fedbn_pc2_aggregate)
        # pc2_fed has a different signature (returns dict-of-dicts) and is not
        # routed via get_aggregation_fn.
        with self.assertRaises(ValueError):
            get_aggregation_fn('pc2_fed')
        with self.assertRaises(ValueError):
            get_aggregation_fn('frobnicate')

    def test_fedavg_strategy_actually_skips_calibration(self):
        """End-to-end: fedavg_strategy must produce an avg_dict that excludes
        calibration buffers (fix for the previous empty-exclude bug)."""
        # Build two trivial nn.Modules carrying calibration buffers via state_dicts
        # we can construct an OrderedDict; fedavg_strategy expects nn.Modules with
        # state_dict(). Use a wrapper.

        class _StubModule(torch.nn.Module):
            def __init__(self, sd):
                super().__init__()
                self._sd = sd
            def state_dict(self, *a, **kw):
                return self._sd

        m1 = _StubModule(_make_state_dict(0))
        m2 = _StubModule(_make_state_dict(7))
        avg = fedavg_strategy([m1, m2], [0.5, 0.5])
        self.assertNotIn('calibration.mu',    avg)
        self.assertNotIn('calibration.sigma', avg)


# --------------------------------------------------------------------------
# pc2_fed_aggregate_all convenience wrapper
# --------------------------------------------------------------------------

class TestAggregateAll(unittest.TestCase):

    def test_dict_of_dicts_shape(self):
        state_dicts = {k: _make_state_dict(i) for i, k in enumerate(NODE_KEYS)}
        M, n = _build_T_matrices_and_n()
        out = pc2_fed_aggregate_all(state_dicts, n, M)
        self.assertEqual(set(out.keys()), set(NODE_KEYS))
        for target, agg in out.items():
            cal_keys = [k for k in agg if 'calibration' in k]
            self.assertEqual(cal_keys, [], msg=f'target={target}')

    def test_targets_subset(self):
        state_dicts = {k: _make_state_dict(i) for i, k in enumerate(NODE_KEYS)}
        M, n = _build_T_matrices_and_n()
        out = pc2_fed_aggregate_all(state_dicts, n, M, targets=['C-hole'])
        self.assertEqual(list(out.keys()), ['C-hole'])


if __name__ == '__main__':
    unittest.main()
