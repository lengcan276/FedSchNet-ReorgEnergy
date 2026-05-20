"""Phase 2 model-structure tests for PC²-FedReorg.

Run with:
    python -m unittest tests.test_models_pc2 -v

Verifies (per user spec for Phase 2 sign-off):
  - Default flags: state_dict has no adapter / calibration keys (E1-E59 compat).
  - use_adapter=True: forward output shape unchanged; identity at init means
    adapter-on output equals adapter-off output when encoder/head weights match.
  - use_calibration=True: forward correctly de-standardizes (lambda = sigma*z + mu).
  - Calibration buffers do not require gradients.
  - Adapter-off forward path is unchanged from the legacy model.
"""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import NODE_FEAT_DIM, GLOBAL_DESC_DIM  # noqa: E402
from src.models import (  # noqa: E402
    AdapterLayer, CalibrationHead, ReorgEnergyModel,
)


def _dummy_batch(num_graphs: int = 2, nodes_per_graph: int = 6, seed: int = 0) -> Batch:
    """Construct a small PyG Batch for testing forward passes."""
    g = torch.Generator().manual_seed(seed)
    datas = []
    for _ in range(num_graphs):
        n = nodes_per_graph
        # Ensure edges form a connected ring + a few random extras (no self-loops)
        rows = torch.arange(n)
        cols = (rows + 1) % n
        edge_index = torch.stack([torch.cat([rows, cols]),
                                  torch.cat([cols, rows])], dim=0).long()
        d = Data(
            x=torch.randn(n, NODE_FEAT_DIM, generator=g),
            edge_index=edge_index,
            global_desc=torch.randn(1, GLOBAL_DESC_DIM, generator=g),
        )
        datas.append(d)
    return Batch.from_data_list(datas)


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------

class TestBackwardCompat(unittest.TestCase):

    def test_default_flags_state_dict_has_no_new_keys(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel()
        keys = list(m.state_dict().keys())
        self.assertFalse(any('adapter' in k for k in keys),
                         msg=f'unexpected adapter keys: {[k for k in keys if "adapter" in k]}')
        self.assertFalse(any('calibration' in k for k in keys),
                         msg=f'unexpected calibration keys: {[k for k in keys if "calibration" in k]}')

    def test_default_forward_returns_one_tensor(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan')
        m.eval()
        batch = _dummy_batch()
        out = m(batch)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (batch.num_graphs, 1))

    def test_default_forward_no_kwargs_unchanged(self):
        """The new `return_z` / `return_physical` kwargs default to False, so
        a kwargs-less call must produce the same tensor as before."""
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan')
        m.eval()
        batch = _dummy_batch()
        with torch.no_grad():
            out_default = m(batch)
            out_explicit = m(batch, return_z=False, return_physical=False)
        self.assertTrue(torch.equal(out_default, out_explicit))


# --------------------------------------------------------------------------
# Adapter: shape preservation + identity at init
# --------------------------------------------------------------------------

class TestAdapter(unittest.TestCase):

    def test_adapter_layer_residual_identity_at_init(self):
        """A standalone AdapterLayer initialized with zero output should map
        x to x (residual identity)."""
        torch.manual_seed(0)
        adapter = AdapterLayer(hidden_dim=16, bottleneck_dim=8, adapter_type='mlp')
        adapter.eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            y = adapter(x)
        self.assertTrue(torch.allclose(y, x, atol=1e-7))

    def test_use_adapter_true_output_shape_unchanged(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_adapter=True, adapter_type='mlp')
        m.eval()
        batch = _dummy_batch()
        out = m(batch)
        self.assertEqual(out.shape, (batch.num_graphs, 1))

    def test_adapter_off_vs_on_at_init(self):
        """At initialization, adapter-on output must equal adapter-off output
        when encoder + head weights are identical (because adapter starts as
        identity)."""
        torch.manual_seed(0)
        m_off = ReorgEnergyModel(head_type='kan', use_adapter=False)
        torch.manual_seed(0)
        m_on = ReorgEnergyModel(head_type='kan', use_adapter=True, adapter_type='mlp')
        # Sync encoder + head weights (adapter init differs from rng draw order).
        m_on.encoder.load_state_dict(m_off.encoder.state_dict())
        m_on.head.load_state_dict(m_off.head.state_dict())

        m_off.eval()
        m_on.eval()
        batch = _dummy_batch()
        with torch.no_grad():
            y_off = m_off(batch)
            y_on = m_on(batch)
        self.assertTrue(torch.allclose(y_off, y_on, atol=1e-6),
                        msg=f'adapter not identity at init: '
                            f'y_off[:3]={y_off.flatten()[:3].tolist()}, '
                            f'y_on[:3]={y_on.flatten()[:3].tolist()}')

    def test_adapter_off_state_dict_no_adapter_keys(self):
        m = ReorgEnergyModel(use_adapter=False)
        keys = list(m.state_dict().keys())
        self.assertFalse(any('adapter' in k for k in keys))

    def test_adapter_on_state_dict_has_adapter_keys(self):
        m = ReorgEnergyModel(use_adapter=True, adapter_type='mlp')
        keys = list(m.state_dict().keys())
        self.assertTrue(any('adapter' in k for k in keys))

    def test_adapter_kan_initializes_without_error(self):
        """KAN adapter is an ablation path; it must at least construct and
        forward cleanly."""
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_adapter=True, adapter_type='kan')
        m.eval()
        batch = _dummy_batch()
        out = m(batch)
        self.assertEqual(out.shape, (batch.num_graphs, 1))


# --------------------------------------------------------------------------
# CalibrationHead: buffers, sigma floor, de-standardize
# --------------------------------------------------------------------------

class TestCalibrationHead(unittest.TestCase):

    def test_buffers_no_grad(self):
        cal = CalibrationHead()
        self.assertFalse(cal.mu.requires_grad)
        self.assertFalse(cal.sigma.requires_grad)

    def test_set_stats_sigma_floor(self):
        # The buffer is float32, so 1e-8 is rounded to the nearest representable
        # value (~9.999999e-9). Compare with tolerance, not strict equality.
        cal = CalibrationHead()
        cal.set_stats(mu=1.0, sigma=0.0)
        self.assertAlmostEqual(cal.sigma.item(), CalibrationHead.SIGMA_FLOOR, places=10)
        cal.set_stats(mu=1.0, sigma=-5.0)
        self.assertAlmostEqual(cal.sigma.item(), CalibrationHead.SIGMA_FLOOR, places=10)
        # And ensure values above the floor are not clamped down.
        cal.set_stats(mu=1.0, sigma=0.5)
        self.assertAlmostEqual(cal.sigma.item(), 0.5, places=6)

    def test_calibration_de_standardize(self):
        cal = CalibrationHead()
        cal.set_stats(mu=2.0, sigma=3.0)
        z = torch.tensor([0.0, 1.0, -1.0])
        out = cal(z)
        # sigma * z + mu = 3z + 2 -> [2, 5, -1]
        self.assertTrue(torch.allclose(out, torch.tensor([2.0, 5.0, -1.0])))

    def test_calibration_standardize_inverse(self):
        cal = CalibrationHead()
        cal.set_stats(mu=2.0, sigma=3.0)
        y = torch.tensor([5.0, 2.0, -1.0])
        z = cal.standardize(y)
        self.assertTrue(torch.allclose(z, torch.tensor([1.0, 0.0, -1.0])))

    def test_calibration_in_state_dict_under_calibration_substring(self):
        """State-dict keys must contain 'calibration' so the federated
        aggregator can filter them in Phase 3."""
        m = ReorgEnergyModel(use_calibration=True)
        cal_keys = [k for k in m.state_dict().keys() if 'calibration' in k]
        self.assertGreaterEqual(len(cal_keys), 2,
                                msg=f'expected mu and sigma under calibration.*, got {cal_keys}')

    def test_calibration_buffers_have_no_grad_in_full_model(self):
        m = ReorgEnergyModel(use_calibration=True)
        for name, p in m.named_parameters():
            # parameters() should not include calibration buffers
            self.assertFalse('calibration' in name,
                             msg=f'unexpected calibration parameter: {name}')


# --------------------------------------------------------------------------
# Model with calibration: return_z / return_physical routing
# --------------------------------------------------------------------------

class TestModelWithCalibration(unittest.TestCase):

    def test_default_returns_physical_when_calibration_on(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_calibration=True)
        m.calibration.set_stats(mu=2.0, sigma=3.0)
        m.eval()
        batch = _dummy_batch()
        with torch.no_grad():
            phys = m(batch)
            z = m(batch, return_z=True)
        self.assertEqual(phys.shape, z.shape)
        # phys = sigma * z + mu = 3z + 2
        self.assertTrue(torch.allclose(phys, 3.0 * z + 2.0, atol=1e-5))

    def test_return_both_z_and_physical(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_calibration=True)
        m.calibration.set_stats(mu=-1.0, sigma=0.5)
        m.eval()
        batch = _dummy_batch()
        with torch.no_grad():
            z_solo = m(batch, return_z=True)
            phys_solo = m(batch, return_physical=True)
            z_tuple, phys_tuple = m(batch, return_z=True, return_physical=True)
        self.assertTrue(torch.allclose(z_solo, z_tuple))
        self.assertTrue(torch.allclose(phys_solo, phys_tuple))
        self.assertTrue(torch.allclose(phys_solo, 0.5 * z_solo + (-1.0), atol=1e-5))

    def test_return_z_when_calibration_off_returns_head_output(self):
        torch.manual_seed(0)
        m = ReorgEnergyModel(head_type='kan', use_calibration=False)
        m.eval()
        batch = _dummy_batch()
        with torch.no_grad():
            default = m(batch)
            z = m(batch, return_z=True)
            phys = m(batch, return_physical=True)
        # No calibration means z and physical are the same as default.
        self.assertTrue(torch.equal(default, z))
        self.assertTrue(torch.equal(default, phys))


if __name__ == '__main__':
    unittest.main()
