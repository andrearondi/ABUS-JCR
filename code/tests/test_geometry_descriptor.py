"""[4.4] The Axis-A relative-geometry descriptor g(m,n) must be BYTE-IDENTICAL to the
function that produced the recorded Axis-A numbers (``probe.pool_diag.relative_geometry``).

If the batched descriptor drifts from the scalar one, every recorded prior
([P3U2.PD]: TP-FP vs FP-FP max |δ| = 0.082; [P3U3.8] §3) stops describing what the model
actually sees, and the pre-registered A1-vs-B2 comparison becomes untestable.

The batched implementation is array-agnostic (numpy **and** torch flow through the same
code), so this parity test runs on the laptop; the torch path is additionally checked
wherever torch is installed.
"""

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.probe import pool_diag as PD
from abus_jcr.rescore.geometry_bias import relative_geometry_batch, relative_geometry_torch


def _random_boxes(n, seed=0):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(-500, 500, size=(n, 3))
    length = rng.uniform(0.5, 300.0, size=(n, 3))
    return coord, length


def test_matches_the_scalar_pool_diag_descriptor_on_every_ordered_pair():
    coord, length = _random_boxes(20, seed=1)
    g = relative_geometry_batch(coord, length)
    assert g.shape == (20, 20, 6)
    for m in range(20):
        for n in range(20):
            ref = PD.relative_geometry(*coord[m], *length[m], *coord[n], *length[n])
            np.testing.assert_allclose(np.asarray(g[m, n]), ref, rtol=0, atol=1e-9)


def test_parity_holds_on_a_larger_random_sample():
    coord, length = _random_boxes(60, seed=7)
    g = np.asarray(relative_geometry_batch(coord, length))
    rng = np.random.default_rng(0)
    for _ in range(200):
        m, n = int(rng.integers(60)), int(rng.integers(60))
        ref = PD.relative_geometry(*coord[m], *length[m], *coord[n], *length[n])
        assert np.max(np.abs(g[m, n] - ref)) < 1e-6


def test_eps_is_the_same_constant_as_the_probe_that_produced_the_recorded_numbers():
    assert C.RESC_GEOM_EPS == PD.EPS


def test_size_components_are_translation_invariant():
    coord, length = _random_boxes(12, seed=2)
    g0 = np.asarray(relative_geometry_batch(coord, length))
    g1 = np.asarray(relative_geometry_batch(coord + np.array([13.0, -7.0, 4.0]), length))
    np.testing.assert_allclose(g0[..., 3:], g1[..., 3:], atol=1e-12)


def test_distance_components_move_under_translation_of_one_box_only():
    coord, length = _random_boxes(6, seed=3)
    shifted = coord.copy()
    shifted[0] += 50.0
    g0 = np.asarray(relative_geometry_batch(coord, length))
    g1 = np.asarray(relative_geometry_batch(shifted, length))
    assert np.max(np.abs(g0[0, 1, :3] - g1[0, 1, :3])) > 1e-3


def test_self_pair_distance_components_hit_the_eps_floor():
    coord, length = _random_boxes(4, seed=4)
    g = np.asarray(relative_geometry_batch(coord, length))
    for m in range(4):
        np.testing.assert_allclose(g[m, m, :3], np.log(C.RESC_GEOM_EPS), atol=1e-9)
        np.testing.assert_allclose(g[m, m, 3:], 0.0, atol=1e-12)


def test_size_components_are_antisymmetric_in_m_and_n():
    coord, length = _random_boxes(10, seed=5)
    g = np.asarray(relative_geometry_batch(coord, length))
    np.testing.assert_allclose(g[..., 3:], -np.transpose(g, (1, 0, 2))[..., 3:], atol=1e-9)


def test_batched_leading_dimension_is_supported():
    coord, length = _random_boxes(8, seed=6)
    b_coord = np.stack([coord, coord + 1.0])
    b_length = np.stack([length, length])
    g = np.asarray(relative_geometry_batch(b_coord, b_length))
    assert g.shape == (2, 8, 8, 6)
    np.testing.assert_allclose(g[0], np.asarray(relative_geometry_batch(coord, length)), atol=1e-12)


def test_coordZ_is_the_depth_beam_axis():
    """State it, do not rediscover it: PERM_STORAGE_TO_ITK = (2,1,0), so the official
    ``coordZ``/``z_length`` IS storage d0 — the depth/beam axis Phase-0b implicates."""
    assert C.PERM_STORAGE_TO_ITK == (2, 1, 0)
    assert C.PERM_STORAGE_TO_ITK[2] == 0


def test_descriptor_is_unit_free_so_anisotropic_voxel_units_cancel():
    """Every component is a ratio, so scaling all coordinates AND lengths per axis by the
    same factor leaves g unchanged (up to the eps floor)."""
    coord, length = _random_boxes(10, seed=8)
    s = np.array([0.475674, 0.200, 0.073])          # the native ITK-order spacing
    g0 = np.asarray(relative_geometry_batch(coord, length))
    g1 = np.asarray(relative_geometry_batch(coord * s, length * s))
    off = ~np.eye(10, dtype=bool)
    assert np.max(np.abs(g0[off] - g1[off])) < 1e-3


def test_relative_geometry_torch_is_the_same_callable():
    assert relative_geometry_torch is relative_geometry_batch


def test_torch_tensors_flow_through_the_identical_code_path():
    torch = pytest.importorskip("torch")
    coord, length = _random_boxes(9, seed=9)
    g_np = np.asarray(relative_geometry_batch(coord, length))
    g_t = relative_geometry_batch(torch.as_tensor(coord, dtype=torch.float64),
                                  torch.as_tensor(length, dtype=torch.float64))
    assert isinstance(g_t, torch.Tensor)
    np.testing.assert_allclose(g_t.numpy(), g_np, atol=1e-9)


def test_torch_descriptor_is_differentiable_wrt_the_boxes():
    torch = pytest.importorskip("torch")
    coord, length = _random_boxes(5, seed=10)
    c = torch.as_tensor(coord, dtype=torch.float64, requires_grad=True)
    l = torch.as_tensor(length, dtype=torch.float64, requires_grad=True)
    relative_geometry_batch(c, l).sum().backward()
    assert torch.isfinite(c.grad).all() and torch.isfinite(l.grad).all()
