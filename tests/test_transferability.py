"""Unit tests for src/transferability.py — PC²-FedReorg locked invariants.

Run with:
    python -m unittest tests.test_transferability -v

The tests use synthetic label samples so they don't depend on the full
public_reorg_energy_15210.csv being loaded each invocation.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.transferability import (  # noqa: E402
    NODE_KEYS, T_REPR_FLOOR, T_ADAPTER_FLOOR, T_HEAD_CUTOFF,
    PROTOCOL_META, compute_T_matrices, target_wise_normalize,
    t_head_compatibility, label_distribution_score,
)


def _phase0_chemistry():
    """Return a stub T_chemistry payload using the actual Phase 0 measurements
    so the unit tests don't depend on T_chemistry.json being on disk during
    the test run."""
    K = {
        'A':         {'A': 1.0,    'B': 0.2455, 'C-hole': 0.1158, 'C-triplet': 0.1150, 'D': 0.1122},
        'B':         {'A': 0.2593, 'B': 1.0,    'C-hole': 0.1423, 'C-triplet': 0.1404, 'D': 0.1883},
        'C-hole':    {'A': 0.2435, 'B': 0.2392, 'C-hole': 1.0,    'C-triplet': 0.9718, 'D': 0.1241},
        'C-triplet': {'A': 0.2418, 'B': 0.2399, 'C-hole': 1.0,    'C-triplet': 1.0,    'D': 0.1241},
        'D':         {'A': 0.1887, 'B': 0.3141, 'C-hole': 0.0910, 'C-triplet': 0.0909, 'D': 1.0},
    }
    S = {
        'A':         {'A': 1.0,    'B': 0.0,    'C-hole': 0.0,    'C-triplet': 0.0,    'D': 0.0},
        'B':         {'A': 0.0,    'B': 1.0,    'C-hole': 0.0,    'C-triplet': 0.0,    'D': 0.0176},
        'C-hole':    {'A': 0.0,    'B': 0.0,    'C-hole': 1.0,    'C-triplet': 1.0,    'D': 0.0},
        'C-triplet': {'A': 0.0,    'B': 0.0,    'C-hole': 1.0,    'C-triplet': 1.0,    'D': 0.0},
        'D':         {'A': 0.0,    'B': 0.0020, 'C-hole': 0.0,    'C-triplet': 0.0,    'D': 1.0},
    }
    return {'K': K, 'S': S, 'node_keys': NODE_KEYS,
            'n_kept': {'A': 6020, 'B': 9190, 'C-hole': 53, 'C-triplet': 49, 'D': 5876}}


def _synthetic_label_samples(seed: int = 42) -> dict:
    """Synthetic per-node label samples with the means/stds reported by V1."""
    rng = np.random.default_rng(seed)
    return {
        'A':         rng.normal(loc=0.20, scale=0.10, size=6020).astype(float),
        'B':         rng.normal(loc=0.30, scale=0.15, size=9190).astype(float),
        'C-hole':    rng.normal(loc=1.22, scale=0.50, size=53).astype(float),
        'C-triplet': rng.normal(loc=2.64, scale=0.75, size=49).astype(float),
        'D':         rng.normal(loc=0.26, scale=0.05, size=5876).astype(float),
    }


class TestProtocolGate(unittest.TestCase):
    """Per-clause unit tests for t_head_compatibility (no chemistry needed)."""

    def test_d_to_c_hole_blocked(self):
        ok, reason = t_head_compatibility(
            'D', 'C-hole', PROTOCOL_META['D'], PROTOCOL_META['C-hole']
        )
        self.assertFalse(ok)
        # Quantity matches (both hole), so clause 1 doesn't fire; clause 4 should.
        self.assertIn('clause 4', reason)

    def test_d_to_c_triplet_blocked_quantity(self):
        ok, reason = t_head_compatibility(
            'D', 'C-triplet', PROTOCOL_META['D'], PROTOCOL_META['C-triplet']
        )
        self.assertFalse(ok)
        # quantity differs (hole vs triplet) — clause 1 fires before clause 4.
        self.assertIn('clause 1', reason)

    def test_c_hole_to_c_triplet_quantity_mismatch(self):
        ok, reason = t_head_compatibility(
            'C-hole', 'C-triplet', PROTOCOL_META['C-hole'], PROTOCOL_META['C-triplet']
        )
        self.assertFalse(ok)
        self.assertIn('clause 1', reason)

    def test_a_to_b_compatible(self):
        # A and B share source_id 'qm9_public_reorg_15210' -> compatible
        ok, reason = t_head_compatibility(
            'A', 'B', PROTOCOL_META['A'], PROTOCOL_META['B']
        )
        self.assertTrue(ok, msg=f'A->B should be compatible (same source); reason={reason}')
        self.assertIsNone(reason)

    def test_b_to_d_blocked_clause1(self):
        ok, reason = t_head_compatibility(
            'B', 'D', PROTOCOL_META['B'], PROTOCOL_META['D']
        )
        self.assertFalse(ok)
        # B is reorg_energy_general, D is hole_lambda — clause 1.
        self.assertIn('clause 1', reason)

    def test_c_hole_to_d_blocked_protocol_unknown(self):
        # quantity matches (hole), source differs, D's geometry/charge_state/conformers unknown
        ok, reason = t_head_compatibility(
            'C-hole', 'D', PROTOCOL_META['C-hole'], PROTOCOL_META['D']
        )
        self.assertFalse(ok)
        # Should fail at clause 2 or 3.
        self.assertTrue('clause 2' in reason or 'clause 3' in reason,
                        msg=f'expected clause 2 or 3, got {reason!r}')


class TestMatrixInvariants(unittest.TestCase):
    """The locked invariants the user spec'd in Phase 1, plus diagonals."""

    @classmethod
    def setUpClass(cls):
        chem = _phase0_chemistry()
        samples = _synthetic_label_samples()
        cls.M = compute_T_matrices(t_chemistry=chem, label_samples=samples)

    def test_diagonals_are_one(self):
        for k in NODE_KEYS:
            self.assertEqual(self.M['T_repr'][k][k], 1.0)
            self.assertEqual(self.M['T_head'][k][k], 1.0)
            self.assertEqual(self.M['T_adapter'][k][k], 1.0)

    # ---- spec'd invariants 1–3 (T_head) ----

    def test_t_head_d_to_c_hole_zero(self):
        self.assertEqual(self.M['T_head']['D']['C-hole'], 0.0)

    def test_t_head_d_to_c_triplet_zero(self):
        self.assertEqual(self.M['T_head']['D']['C-triplet'], 0.0)

    def test_t_head_c_hole_to_c_triplet_zero(self):
        self.assertEqual(self.M['T_head']['C-hole']['C-triplet'], 0.0)

    def test_t_head_c_triplet_to_c_hole_zero(self):
        self.assertEqual(self.M['T_head']['C-triplet']['C-hole'], 0.0)

    # ---- spec'd invariants 4–5 (T_repr) ----

    def test_t_repr_c_hole_to_c_triplet_above_0p95(self):
        self.assertGreater(self.M['T_repr']['C-hole']['C-triplet'], 0.95)

    def test_t_repr_d_to_c_hole_below_0p1(self):
        self.assertLess(self.M['T_repr']['D']['C-hole'], 0.10)

    # ---- spec'd invariant 6 (B->D nonzero T_repr but zero T_head) ----

    def test_b_to_d_chemistry_shares_head_does_not(self):
        self.assertGreater(self.M['T_repr']['B']['D'], 0.05)
        self.assertEqual(self.M['T_head']['B']['D'], 0.0)
        # Adapter must collapse to its floor whenever T_head is zero.
        self.assertEqual(self.M['T_adapter']['B']['D'], T_ADAPTER_FLOOR)

    # ---- additional spec invariants ----

    def test_t_adapter_zero_when_t_head_zero(self):
        for i in NODE_KEYS:
            for j in NODE_KEYS:
                if i != j and self.M['T_head'][i][j] == 0.0:
                    self.assertEqual(
                        self.M['T_adapter'][i][j], T_ADAPTER_FLOOR,
                        msg=f'T_adapter[{i},{j}] should floor when T_head=0',
                    )

    def test_t_repr_floor(self):
        for i in NODE_KEYS:
            for j in NODE_KEYS:
                self.assertGreaterEqual(self.M['T_repr'][i][j], T_REPR_FLOOR)

    def test_no_off_diagonal_t_head_above_one(self):
        for i in NODE_KEYS:
            for j in NODE_KEYS:
                self.assertLessEqual(self.M['T_head'][i][j], 1.0)

    def test_t_head_off_diagonals_only_a_b(self):
        """Under the conservative gate + current PROTOCOL_META, the only
        off-diagonal T_head entries that can be nonzero are A<->B (both share
        source_id). All other off-diagonals must be exactly 0."""
        allowed = {('A', 'B'), ('B', 'A')}
        for i in NODE_KEYS:
            for j in NODE_KEYS:
                if i == j or (i, j) in allowed:
                    continue
                self.assertEqual(
                    self.M['T_head'][i][j], 0.0,
                    msg=f'T_head[{i},{j}] should be 0 under conservative gate'
                )


class TestTargetWiseNormalize(unittest.TestCase):
    """Aggregation weights must sum to 1 per target (rule 5)."""

    def test_sum_to_one_per_target(self):
        chem = _phase0_chemistry()
        samples = _synthetic_label_samples()
        M = compute_T_matrices(t_chemistry=chem, label_samples=samples)
        n_kept = chem['n_kept']
        for label, T in (
            ('T_repr', M['T_repr']),
            ('T_head', M['T_head']),
            ('T_adapter', M['T_adapter']),
        ):
            w = target_wise_normalize(T, n_kept)
            for j in NODE_KEYS:
                s = sum(w[i][j] for i in NODE_KEYS)
                self.assertAlmostEqual(
                    s, 1.0, places=9,
                    msg=f'{label}: weights for target {j} sum to {s}, not 1.0'
                )

    def test_zero_column_falls_back_to_local(self):
        """If every source is gated out for target j, weight[j,j] = 1."""
        zero_T = {i: {j: (1.0 if i == j else 0.0) for j in NODE_KEYS} for i in NODE_KEYS}
        # Simulate "everything off-diag gated to 0" by zeroing diag too for one column.
        zero_T['C-hole']['C-triplet'] = 0.0  # already 0
        # For target C-triplet make every source 0 except keep diag pattern.
        # Remove diag of C-triplet to test fallback:
        zero_T_test = {i: dict(zero_T[i]) for i in NODE_KEYS}
        for i in NODE_KEYS:
            zero_T_test[i]['C-triplet'] = 0.0
        n_kept = {'A': 100, 'B': 100, 'C-hole': 100, 'C-triplet': 100, 'D': 100}
        w = target_wise_normalize(zero_T_test, n_kept)
        # Fallback: identity on C-triplet column
        self.assertEqual(w['C-triplet']['C-triplet'], 1.0)
        for i in NODE_KEYS:
            if i != 'C-triplet':
                self.assertEqual(w[i]['C-triplet'], 0.0)


class TestLabelDistributionScore(unittest.TestCase):
    """Sanity for label_distribution_score: monotone in W1."""

    def test_identical_distributions_give_one(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=500)
        y = x.copy()
        L = label_distribution_score(x, y)
        # exp(0) * 1 = 1
        self.assertAlmostEqual(L, 1.0, places=6)

    def test_far_distributions_give_small_value(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 0.1, size=500)
        y = rng.normal(5.0, 0.1, size=500)   # W1 ~= 5 -> exp(-25) ~= 0
        L = label_distribution_score(x, y)
        self.assertLess(L, 1e-6)


if __name__ == '__main__':
    unittest.main()
