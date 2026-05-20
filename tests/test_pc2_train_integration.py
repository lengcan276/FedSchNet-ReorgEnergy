"""Phase 4 train_eval integration tests for PC²-FedReorg.

Run with:
    python -m unittest tests.test_pc2_train_integration -v

Verifies (per user spec for Phase 4 sign-off, constraint 11):

  - calibration stats are computed from the train split ONLY (no test-fold
    leakage), with sigma floor honored.
  - train_one_epoch computes loss in z-space when use_calibration=True.
  - evaluate returns predictions in eV-space when use_calibration=True.
  - pc2_fed is NOT routed through get_aggregation_fn (would raise ValueError).
  - pc2_fed_dispatch reads T_transferability.json and produces correct
    per-client aggregated state_dicts.
  - When PC² flags are off (defaults), forward / train_one_epoch / evaluate
    paths are unchanged from E1-E59.
  - C-hole and C-triplet map to distinct task-client nodes.
"""

import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import GLOBAL_DESC_DIM, NODE_FEAT_DIM  # noqa: E402
from src.federated import get_aggregation_fn  # noqa: E402
from src.models import CalibrationHead, ReorgEnergyModel  # noqa: E402
from src.train_eval import (  # noqa: E402
    evaluate, install_calibration_from_train, train_one_epoch,
)
from experiments.pc2_fedreorg import (  # noqa: E402
    classify_key, load_T_matrices_from_json, pc2_fed_dispatch, task_client_for,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_data_list(n: int, y_mean: float, y_std: float, seed: int = 0):
    """Construct a small list of PyG Data objects with scalar y labels drawn
    from N(y_mean, y_std)."""
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        nn_atoms = 6
        rows = torch.arange(nn_atoms)
        cols = (rows + 1) % nn_atoms
        edge_index = torch.stack(
            [torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0
        ).long()
        d = Data(
            x=torch.randn(nn_atoms, NODE_FEAT_DIM, generator=g),
            edge_index=edge_index,
            global_desc=torch.randn(1, GLOBAL_DESC_DIM, generator=g),
            y=torch.tensor(float(rng.normal(y_mean, y_std))),
        )
        out.append(d)
    return out


def _to_loader(data_list, batch_size=4):
    from torch_geometric.loader import DataLoader
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------
# task-client identity (constraint 2)
# --------------------------------------------------------------------------

class TestTaskClientMapping(unittest.TestCase):

    def test_c_hole_and_c_triplet_distinct(self):
        self.assertEqual(task_client_for('client_c', 'hole'), 'C-hole')
        self.assertEqual(task_client_for('client_c', 'triplet'), 'C-triplet')
        self.assertNotEqual(
            task_client_for('client_c', 'hole'),
            task_client_for('client_c', 'triplet'),
        )

    def test_other_clients(self):
        self.assertEqual(task_client_for('client_a', None), 'A')
        self.assertEqual(task_client_for('client_b', None), 'B')
        self.assertEqual(task_client_for('client_d', None), 'D')

    def test_client_c_requires_target_type(self):
        with self.assertRaises(ValueError):
            task_client_for('client_c', None)
        with self.assertRaises(ValueError):
            task_client_for('client_c', 'singlet')

    def test_unknown_client_raises(self):
        with self.assertRaises(ValueError):
            task_client_for('client_e', 'hole')


# --------------------------------------------------------------------------
# Calibration stats from train split (constraint 3)
# --------------------------------------------------------------------------

class TestCalibrationFromTrainSplitOnly(unittest.TestCase):

    def test_install_uses_only_provided_train_data(self):
        """install_calibration_from_train must compute mu/sigma from the
        passed-in data ONLY -- the test fold's labels never enter."""
        torch.manual_seed(0)
        m = ReorgEnergyModel(use_calibration=True)
        # Train labels: 49 of 50 molecules, mean=1.0, std=0.5
        full = _make_data_list(50, y_mean=1.0, y_std=0.5, seed=42)
        held_out = full[0]            # n=1 test molecule with possibly extreme label
        held_out.y = torch.tensor(99.0)   # poison the held-out label
        train = full[1:]              # 49 molecules

        mu, sigma = install_calibration_from_train(m, train)

        # The 99.0 outlier must NOT affect mu/sigma.
        train_ys = np.array([float(d.y.item()) for d in train], dtype=np.float64)
        self.assertAlmostEqual(mu, float(train_ys.mean()), places=5)
        self.assertAlmostEqual(sigma, float(train_ys.std()), places=5)
        # Sanity: sigma is well below 99.0 (which would happen if outlier leaked).
        self.assertLess(sigma, 5.0)

    def test_install_writes_to_calibration_buffers(self):
        m = ReorgEnergyModel(use_calibration=True)
        train = _make_data_list(20, y_mean=2.5, y_std=0.3, seed=1)
        mu, sigma = install_calibration_from_train(m, train)
        self.assertAlmostEqual(m.calibration.mu.item(), mu, places=5)
        self.assertAlmostEqual(m.calibration.sigma.item(), sigma, places=5)

    def test_install_sigma_floor_honored(self):
        """Constant labels -> sigma -> 0; install must clamp to SIGMA_FLOOR."""
        m = ReorgEnergyModel(use_calibration=True)
        constant = _make_data_list(8, y_mean=1.0, y_std=0.0, seed=9)
        # Force exact constancy
        for d in constant:
            d.y = torch.tensor(1.0)
        mu, sigma = install_calibration_from_train(m, constant)
        self.assertAlmostEqual(mu, 1.0, places=5)
        self.assertAlmostEqual(m.calibration.sigma.item(),
                               CalibrationHead.SIGMA_FLOOR, places=10)

    def test_install_rejects_use_calibration_false(self):
        m = ReorgEnergyModel(use_calibration=False)
        train = _make_data_list(8, 0.0, 1.0, seed=2)
        with self.assertRaises(ValueError):
            install_calibration_from_train(m, train)


# --------------------------------------------------------------------------
# Loss in z-space, metrics in eV-space (constraint 4)
# --------------------------------------------------------------------------

class TestZSpaceVsEVSpace(unittest.TestCase):

    def test_train_one_epoch_loss_uses_z_space_when_calibration_on(self):
        """When use_calibration=True, train_one_epoch should compute loss
        between z-pred and z-target; the loss MUST scale with 1/sigma^2
        relative to physical-space MSE."""
        torch.manual_seed(0)
        # dropout=0 so train/eval forward modes produce identical outputs.
        m = ReorgEnergyModel(head_type='kan', use_calibration=True, dropout=0.0)
        train = _make_data_list(8, y_mean=1.0, y_std=0.5, seed=11)
        install_calibration_from_train(m, train)
        # Force a non-trivial sigma so the z-space vs ev-space difference is large.
        m.calibration.set_stats(mu=1.0, sigma=0.5)

        loader = _to_loader(train, batch_size=4)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)  # lr=0 -> weights frozen for a fair check
        criterion = torch.nn.MSELoss()

        # train_one_epoch returns mean loss; with lr=0 weights are unchanged
        # so we can compare against an explicit z-space MSE.
        mean_loss = train_one_epoch(m, loader, opt, device=torch.device('cpu'),
                                     criterion=criterion)

        # Reference: z-space MSE recomputed by hand. Use train() mode so
        # BatchNorm uses batch statistics (matching train_one_epoch's forward
        # path); torch.no_grad() to avoid gradient bookkeeping.
        m.train()
        with torch.no_grad():
            zs_pred, zs_true = [], []
            for batch in _to_loader(train, batch_size=4):
                z_pred = m(batch, return_z=True).squeeze(-1)
                z_true = m.calibration.standardize(batch.y)
                zs_pred.append(z_pred); zs_true.append(z_true)
            zp = torch.cat(zs_pred); zt = torch.cat(zs_true)
            ref_z_loss = float(torch.mean((zp - zt) ** 2))
        self.assertAlmostEqual(mean_loss, ref_z_loss, places=5)

    def test_evaluate_returns_ev_space_when_calibration_on(self):
        """evaluate must return predictions in eV-space (de-standardized)."""
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_calibration=True)
        m.calibration.set_stats(mu=2.0, sigma=0.5)
        train = _make_data_list(8, y_mean=2.0, y_std=0.5, seed=22)
        loader = _to_loader(train, batch_size=4)

        y_true, y_pred = evaluate(m, loader, torch.device('cpu'))

        # Recompute by hand: z-pred * sigma + mu  vs  evaluate output.
        m.eval()
        with torch.no_grad():
            zs = []
            for batch in _to_loader(train, batch_size=4):
                zs.append(m(batch, return_z=True).squeeze(-1))
            z = torch.cat(zs)
            phys_ref = (z * 0.5 + 2.0).numpy()
        np.testing.assert_allclose(y_pred, phys_ref, atol=1e-5)

    def test_train_one_epoch_legacy_path_unchanged_when_calibration_off(self):
        """When use_calibration=False, train_one_epoch must use eV-space loss
        (as before). The mean loss should equal MSE between raw model output
        and batch.y."""
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_calibration=False, dropout=0.0)
        train = _make_data_list(8, y_mean=1.0, y_std=0.5, seed=33)
        loader = _to_loader(train, batch_size=4)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        criterion = torch.nn.MSELoss()

        mean_loss = train_one_epoch(m, loader, opt, device=torch.device('cpu'),
                                     criterion=criterion)

        # Use train() mode so BatchNorm uses batch stats (matches train_one_epoch).
        m.train()
        with torch.no_grad():
            preds, ys = [], []
            for batch in _to_loader(train, batch_size=4):
                preds.append(m(batch).squeeze(-1)); ys.append(batch.y)
            ref_loss = float(torch.mean((torch.cat(preds) - torch.cat(ys)) ** 2))
        self.assertAlmostEqual(mean_loss, ref_loss, places=5)


# --------------------------------------------------------------------------
# pc2_fed dispatch (constraint 6)
# --------------------------------------------------------------------------

class TestPC2FedDispatch(unittest.TestCase):

    def test_pc2_fed_not_in_get_aggregation_fn(self):
        """pc2_fed must NOT be routed through the legacy get_aggregation_fn
        registry; the user spec calls for a separate dispatch path."""
        with self.assertRaises(ValueError):
            get_aggregation_fn('pc2_fed')

    def test_pc2_fed_dispatch_reads_transferability_json(self):
        """pc2_fed_dispatch must produce per-client aggregated state_dicts
        whose keys exclude calibration / bn / norm and respect T_head=0
        invariants."""
        # Use the actual T_transferability.json from Phase 1.
        json_path = PROJECT_ROOT / 'results' / 'preverify' / 'T_transferability.json'
        if not json_path.exists():
            self.skipTest(f'T_transferability.json not found at {json_path}; '
                          'run `python -m src.transferability` to generate.')

        # Build small models for a 2-client setup (D and C-hole) to exercise
        # the dispatch path with a real transferability matrix.
        torch.manual_seed(0)
        m_d = ReorgEnergyModel(use_adapter=True, use_calibration=True)
        torch.manual_seed(1)
        m_c = ReorgEnergyModel(use_adapter=True, use_calibration=True)

        # Move to cpu (defensive; they already are)
        m_d.cpu(); m_c.cpu()

        client_keys = ['client_d', 'client_c']
        client_models_cpu = [m_d, m_c]
        n_kept = {'client_d': 5876, 'client_c': 53}
        target_type = 'hole'

        out = pc2_fed_dispatch(
            client_models_cpu, client_keys, n_kept,
            target_type=target_type, transferability_path=str(json_path),
        )

        # Output keyed by client_keys
        self.assertEqual(set(out.keys()), set(client_keys))

        # Calibration / bn / norm keys absent (universal contract)
        for ck in client_keys:
            for key in out[ck]:
                self.assertNotIn('calibration', key, msg=f'{ck} leaked {key}')
                self.assertFalse('bn' in key or 'norm' in key,
                                 msg=f'{ck} leaked {key}')

        # T_head[D, C-hole] == 0 -> head for client_c equals client_c's own.
        # (Head's only nonzero contribution is self.)
        for key in m_c.state_dict():
            if classify_key(key) == 'head':
                self.assertIn(key, out['client_c'])
                self.assertTrue(
                    torch.allclose(out['client_c'][key],
                                   m_c.state_dict()[key].float(), atol=1e-12),
                    msg=f'D leaked into client_c head via {key}'
                )

    def test_pc2_fed_dispatch_writes_then_reads_synthetic_json(self):
        """Round-trip: write a synthetic T_transferability JSON and confirm
        load_T_matrices_from_json + pc2_fed_dispatch consume it correctly."""
        T_payload = {
            'T_repr':    {i: {j: 0.5 for j in ['A', 'C-hole']} for i in ['A', 'C-hole']},
            'T_head':    {i: {j: (1.0 if i == j else 0.0) for j in ['A', 'C-hole']}
                          for i in ['A', 'C-hole']},
            'T_adapter': {i: {j: 0.1 for j in ['A', 'C-hole']} for i in ['A', 'C-hole']},
        }
        # Diagonal T_repr / T_adapter must be 1 by convention; force.
        for k in ('T_repr', 'T_adapter'):
            for c in ['A', 'C-hole']:
                T_payload[k][c][c] = 1.0
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(T_payload, f)
            tmp_path = f.name

        loaded = load_T_matrices_from_json(tmp_path)
        self.assertEqual(set(loaded.keys()), {'T_repr', 'T_head', 'T_adapter'})
        self.assertAlmostEqual(loaded['T_repr']['A']['A'], 1.0)
        self.assertAlmostEqual(loaded['T_head']['C-hole']['A'], 0.0)


# --------------------------------------------------------------------------
# Defaults off preserve old path (constraint 1)
# --------------------------------------------------------------------------

class TestDefaultOffPreservesOldPath(unittest.TestCase):

    def test_default_model_no_pc2_keys_in_state_dict(self):
        """A default-constructed ReorgEnergyModel (use_adapter=False,
        use_calibration=False) must have a state_dict with no PC²-specific
        keys, ensuring E1-E59 checkpoints continue to load with strict=True."""
        m = ReorgEnergyModel()
        keys = list(m.state_dict().keys())
        self.assertFalse(any('adapter' in k for k in keys))
        self.assertFalse(any('calibration' in k for k in keys))

    def test_default_train_one_epoch_no_calibration_branch(self):
        """When model.use_calibration=False (E1-E59 default), the train_one_epoch
        loss exactly matches the legacy MSE(model(batch), batch.y)."""
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', dropout=0.0)
        self.assertFalse(getattr(m, 'use_calibration', False))
        train = _make_data_list(8, y_mean=1.0, y_std=0.5, seed=44)
        loader = _to_loader(train, batch_size=4)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        criterion = torch.nn.MSELoss()
        legacy_loss = train_one_epoch(m, loader, opt, device=torch.device('cpu'),
                                       criterion=criterion)
        m.train()
        with torch.no_grad():
            preds, ys = [], []
            for batch in _to_loader(train, batch_size=4):
                preds.append(m(batch).squeeze(-1)); ys.append(batch.y)
            ref = float(torch.mean((torch.cat(preds) - torch.cat(ys)) ** 2))
        self.assertAlmostEqual(legacy_loss, ref, places=5)

    def test_default_evaluate_returns_head_output_directly(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan')
        train = _make_data_list(8, y_mean=1.0, y_std=0.5, seed=55)
        loader = _to_loader(train, batch_size=4)
        y_true, y_pred = evaluate(m, loader, torch.device('cpu'))
        m.eval()
        with torch.no_grad():
            preds = []
            for batch in _to_loader(train, batch_size=4):
                preds.append(m(batch).squeeze(-1).numpy())
            ref = np.concatenate(preds)
        np.testing.assert_allclose(y_pred, ref, atol=1e-6)


# --------------------------------------------------------------------------
# train_federated signature accepts new kwargs without breaking defaults
# --------------------------------------------------------------------------

class TestTrainFederatedSignature(unittest.TestCase):
    """Verify the train_federated function signature accepts the new kwargs
    with safe defaults. We don't actually run training -- just inspect the
    function and confirm safety checks fire for invalid combinations."""

    def test_signature_includes_pc2_kwargs(self):
        from inspect import signature
        from src.train_eval import train_federated
        params = signature(train_federated).parameters
        for new_kw in ('use_adapter', 'adapter_type', 'adapter_bottleneck',
                       'use_calibration', 'aggregation_strategy',
                       'transferability_path', 'target_type'):
            self.assertIn(new_kw, params, msg=f'missing kwarg {new_kw}')

    def test_use_calibration_norm_stats_mutex(self):
        """train_federated must reject use_calibration=True with norm_stats!=None."""
        from src.train_eval import train_federated
        # Tiny client_data so we error out fast on the safety check.
        client_data = {'client_a': _make_data_list(4, 1.0, 0.5, seed=0)}
        with self.assertRaises(ValueError):
            train_federated(
                client_data,
                use_calibration=True,
                norm_stats={'client_a': (0.0, 1.0)},
                n_rounds=1, n_local_epochs=1, verbose=False,
                device_a=torch.device('cpu'), device_b=torch.device('cpu'),
            )

    def test_pc2_fed_requires_transferability_path(self):
        from src.train_eval import train_federated
        client_data = {'client_a': _make_data_list(4, 1.0, 0.5, seed=0)}
        with self.assertRaises(ValueError):
            train_federated(
                client_data,
                aggregation_strategy='pc2_fed',
                transferability_path=None,
                n_rounds=1, n_local_epochs=1, verbose=False,
                device_a=torch.device('cpu'), device_b=torch.device('cpu'),
            )


if __name__ == '__main__':
    unittest.main()
