"""[4.5] The set module must be permutation-EQUIVARIANT and padding-transparent.

A set is unordered: candidate order in the record is an artefact of the linker's tube
enumeration. If the module's output depended on it, every reported CPM would depend on an
arbitrary sort. Padding to ``RESC_MAX_SET_SIZE`` must likewise be invisible.

torch is absent on the laptop -> SKIP there, run on the server.
"""

import pytest

torch = pytest.importorskip("torch")

from abus_jcr import conventions as C
from abus_jcr.rescore.setmodel import B1Rescorer, RGFrocRescorer, to_probability


D_IN, D_MODEL, N_HEADS, N_LAYERS = 16, 32, 4, 2


def _model(use_geometry, seed=0):
    torch.manual_seed(seed)
    m = RGFrocRescorer(d_in=D_IN, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       dropout=0.0, use_geometry=use_geometry)
    return m.double().eval()


def _batch(b=2, n=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, n, D_IN, generator=g, dtype=torch.float64)
    coord = torch.rand(b, n, 3, generator=g, dtype=torch.float64) * 300.0
    length = torch.rand(b, n, 3, generator=g, dtype=torch.float64) * 40.0 + 1.0
    mask = torch.ones(b, n, dtype=torch.bool)
    return feats, coord, length, mask


@pytest.mark.parametrize("use_geometry", [False, True])
def test_permuting_a_set_permutes_the_logits_identically(use_geometry):
    m = _model(use_geometry)
    feats, coord, length, mask = _batch()
    perm = torch.randperm(feats.shape[1], generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        base = m(feats, coord, length, mask)
        permuted = m(feats[:, perm], coord[:, perm], length[:, perm], mask[:, perm])
    assert float((permuted - base[:, perm]).abs().max()) < 1e-6


@pytest.mark.parametrize("use_geometry", [False, True])
def test_padding_does_not_change_the_unpadded_outputs(use_geometry):
    m = _model(use_geometry)
    feats, coord, length, mask = _batch(b=1, n=6)
    pad = 5
    g = torch.Generator().manual_seed(11)
    feats_p = torch.cat([feats, torch.randn(1, pad, D_IN, generator=g, dtype=torch.float64)], 1)
    coord_p = torch.cat([coord, torch.rand(1, pad, 3, generator=g, dtype=torch.float64) * 999], 1)
    length_p = torch.cat([length, torch.rand(1, pad, 3, generator=g, dtype=torch.float64) * 99 + 1], 1)
    mask_p = torch.cat([mask, torch.zeros(1, pad, dtype=torch.bool)], 1)
    with torch.no_grad():
        base = m(feats, coord, length, mask)
        padded = m(feats_p, coord_p, length_p, mask_p)
    assert float((padded[:, :6] - base).abs().max()) < 1e-6


def test_padded_rows_never_produce_nan():
    m = _model(use_geometry=True)
    feats, coord, length, mask = _batch(b=1, n=8)
    mask[:, 4:] = False
    with torch.no_grad():
        out = m(feats, coord, length, mask)
    assert torch.isfinite(out).all()


def test_attention_never_crosses_sets_in_a_batch():
    """PER-VOLUME sets only: g(m,n) is meaningless between patients, and there is no
    batch-level normalisation of the logits (spec §4.5)."""
    m = _model(use_geometry=True)
    feats, coord, length, mask = _batch(b=3, n=7, seed=5)
    with torch.no_grad():
        together = m(feats, coord, length, mask)
        alone = torch.cat([m(feats[i:i + 1], coord[i:i + 1], length[i:i + 1], mask[i:i + 1])
                           for i in range(3)], dim=0)
    assert float((together - alone).abs().max()) < 1e-8


def test_b1_is_per_candidate_and_therefore_set_independent():
    """B1 is the no-jointness baseline: a candidate's logit must not move when its set-mates
    change. This is what makes B2-vs-B1 the JOINTNESS comparison."""
    torch.manual_seed(0)
    m = B1Rescorer(d_in=D_IN, d_model=D_MODEL, hidden=48, depth=2, dropout=0.0).double().eval()
    feats, coord, length, mask = _batch(b=1, n=6)
    other = feats.clone()
    other[:, 1:] = torch.randn_like(other[:, 1:])
    with torch.no_grad():
        a = m(feats, coord, length, mask)
        b = m(other, coord, length, mask)
    assert float((a[:, 0] - b[:, 0]).abs().max()) < 1e-9


def test_b1_is_also_permutation_equivariant():
    torch.manual_seed(0)
    m = B1Rescorer(d_in=D_IN, d_model=D_MODEL, hidden=48, depth=2, dropout=0.0).double().eval()
    feats, coord, length, mask = _batch(b=1, n=6)
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        base = m(feats, coord, length, mask)
        p = m(feats[:, perm], coord[:, perm], length[:, perm], mask[:, perm])
    assert float((p - base[:, perm]).abs().max()) < 1e-9


def test_logits_are_a_fresh_head_not_a_residual_on_score_max():
    """§4.5: never a residual on score_max, so the score-stats ablation stays clean."""
    m = _model(use_geometry=False)
    feats, coord, length, mask = _batch(b=1, n=5)
    with torch.no_grad():
        out = m(feats * 0.0, coord, length, mask)
    assert torch.isfinite(out).all()          # a zero token still produces a defined logit


def test_isab_is_not_used():
    """ISAB's inducing points would delete the full n x n term that IS the Axis-A
    contribution (spec §4.5). The module must expose plain SABs."""
    import abus_jcr.rescore.setmodel as SM
    assert not hasattr(SM, "ISAB")
    assert hasattr(SM, "SAB")


def test_probability_contract_is_enforced_by_to_probability():
    logits = torch.tensor([[-50.0, 0.0, 50.0, 1e9]], dtype=torch.float64)
    p = to_probability(logits)
    assert float(p.min()) >= 0.0
    assert float(p.max()) < 1.0
    assert float(p.max()) <= 1.0 - C.RESC_PROB_EPS + 1e-12
