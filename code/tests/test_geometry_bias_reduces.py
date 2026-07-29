"""[4.4] THE Inv.-critical unit test: rung A1 must START as rung B2.

If the geometry bias is not exactly zero at initialisation, "A1 beats B2" could be an
artefact of a different starting point rather than of learned pairwise geometry — and the
pre-registered weak-prior comparison (TP-FP vs FP-FP max |δ| = 0.082) becomes unreadable.

torch is absent on the laptop, so this module SKIPs there and runs on the server.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abus_jcr import conventions as C
from abus_jcr.rescore.geometry_bias import GeometryBias
from abus_jcr.rescore.setmodel import RGFrocRescorer


D_IN, D_MODEL, N_HEADS, N_LAYERS = 32, 64, 4, 2


def _inputs(b=2, n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, n, D_IN, generator=g, dtype=torch.float64)
    coord = torch.rand(b, n, 3, generator=g, dtype=torch.float64) * 400.0
    length = torch.rand(b, n, 3, generator=g, dtype=torch.float64) * 50.0 + 1.0
    mask = torch.ones(b, n, dtype=torch.bool)
    return feats, coord, length, mask


def _build(use_geometry, mechanism=None, seed=0):
    torch.manual_seed(seed)
    m = RGFrocRescorer(d_in=D_IN, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       dropout=0.0, use_geometry=use_geometry, geom_mechanism=mechanism)
    return m.double().eval()


def test_additive_geometry_bias_is_zero_initialised():
    gb = GeometryBias(n_heads=N_HEADS, mechanism="additive").double()
    _, coord, length, _ = _inputs()
    out = gb(coord, length)
    assert out.shape == (coord.shape[0], N_HEADS, coord.shape[1], coord.shape[1])
    assert float(out.abs().max()) == 0.0


def test_A1_forward_is_numerically_identical_to_B2_at_initialisation():
    """max |delta logit| < 1e-6 — the spec's exit-check-1 number."""
    b2 = _build(use_geometry=False, seed=0)
    a1 = _build(use_geometry=True, mechanism="additive", seed=0)
    # the geometry branch adds parameters; every SHARED parameter must be identical
    a1_shared = {k: v for k, v in a1.state_dict().items() if not k.startswith("geom.")}
    b2.load_state_dict(a1_shared, strict=True)
    feats, coord, length, mask = _inputs()
    with torch.no_grad():
        l_b2 = b2(feats, coord, length, mask)
        l_a1 = a1(feats, coord, length, mask)
    assert float((l_a1 - l_b2).abs().max()) < 1e-6


def test_A1_diverges_from_B2_once_the_bias_weights_are_non_zero():
    """The negative control: if this passes trivially, the test above proves nothing."""
    b2 = _build(use_geometry=False, seed=0)
    a1 = _build(use_geometry=True, mechanism="additive", seed=0)
    b2.load_state_dict({k: v for k, v in a1.state_dict().items() if not k.startswith("geom.")},
                       strict=True)
    with torch.no_grad():
        a1.geom.proj.weight.normal_(0.0, 0.5)
    feats, coord, length, mask = _inputs()
    with torch.no_grad():
        d = (a1(feats, coord, length, mask) - b2(feats, coord, length, mask)).abs().max()
    assert float(d) > 1e-4


def test_multiplicative_mechanism_also_starts_at_B2():
    """omega starts at 1 -> log(1 + eps) ~ 0, so the FALLBACK mechanism is likewise a
    no-op at step 0 and 'no signal' stays distinguishable from 'wrong mechanism'."""
    b2 = _build(use_geometry=False, seed=1)
    a1 = _build(use_geometry=True, mechanism="multiplicative", seed=1)
    b2.load_state_dict({k: v for k, v in a1.state_dict().items() if not k.startswith("geom.")},
                       strict=True)
    feats, coord, length, mask = _inputs(seed=1)
    with torch.no_grad():
        d = (a1(feats, coord, length, mask) - b2(feats, coord, length, mask)).abs().max()
    assert float(d) < 1e-4


def test_multiplicative_bias_without_relu_and_unit_omega_is_log_eps_free():
    gb = GeometryBias(n_heads=N_HEADS, mechanism="multiplicative", relu=False).double()
    _, coord, length, _ = _inputs()
    out = gb(coord, length)
    np.testing.assert_allclose(out.detach().numpy(), np.log(1.0 + C.RESC_GEOM_EPS), atol=1e-12)


def test_geometry_bias_is_learnable():
    gb = GeometryBias(n_heads=N_HEADS, mechanism="additive").double()
    _, coord, length, _ = _inputs()
    gb(coord, length).sum().backward()
    assert torch.isfinite(gb.proj.weight.grad).all()
    assert float(gb.proj.weight.grad.abs().sum()) > 0.0


def test_default_mechanism_is_the_additive_one():
    assert C.RESC_GEOM_MECHANISM == "additive"
    assert GeometryBias(n_heads=2).mechanism == "additive"


def test_unknown_mechanism_is_rejected():
    with pytest.raises(ValueError, match="mechanism"):
        GeometryBias(n_heads=2, mechanism="bilinear")
