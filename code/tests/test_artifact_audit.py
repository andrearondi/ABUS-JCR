"""Artifact-signature audit — synthetic phantoms with known answers.

Every test plants a structure whose signature is known by construction (a dark column to the
floor, a bounded blob, a stripe of a given width, a dark margin, a periodic ladder, a column
stack, a sweep chain, regular bands) and checks the probe recovers it. Nothing here reads real
data.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.probe import artifact_audit as AA

D0, D1, D2 = 200, 120, 100          # lateral, depth, sweep (iso voxels, 0.4 mm)


def phantom(bright=0.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.full((D0, D1, D2), bright, np.float32)
    v += noise * rng.standard_normal(v.shape).astype(np.float32)
    return v


def plant_box(v, cen, ext, value):
    l0, l1 = int(cen[0] - ext[0] / 2), int(cen[0] + ext[0] / 2)
    t, b = int(cen[1] - ext[1] / 2), int(cen[1] + ext[1] / 2)
    s0, s1 = int(cen[2] - ext[2] / 2), int(cen[2] + ext[2] / 2)
    v[l0:l1, t:b, s0:s1] = value


# ------------------------------------------------------------------ Task 1: signatures

def test_bounded_blob_has_no_shadow_run():
    v = phantom(); cen, ext = (100, 40, 50), (20, 16, 20)
    plant_box(v, cen, ext, 0.1)
    f = AA.candidate_signature(v, cen, ext, D1 - 1, AA.Knobs())
    assert f["inside_contrast"] < -0.2
    assert f["shadow_run_mm"] == 0.0
    assert f["dark_above"] is False and f["reaches_floor"] is False


def test_column_to_floor_with_bright_cap():
    v = phantom(); cen, ext = (100, 40, 50), (10, 16, 20)
    v[95:105, 28:D1, 40:60] = 0.1          # dark column from row 28 (1.6 mm above the box) to the floor
    v[95:105, 25:28, 40:60] = 0.9          # bright cap (the rupture) just above the column
    f = AA.candidate_signature(v, cen, ext, D1 - 1, AA.Knobs())
    assert f["shadow_run_mm"] == pytest.approx((D1 - 48) * 0.4, abs=0.8)
    assert f["reaches_floor"] is True
    assert f["dark_above"] is True          # the column starts above the box
    assert f["cap_bright"] > 0.2


def test_shadowing_blob_starts_at_box():
    v = phantom(); cen, ext = (100, 40, 50), (20, 16, 20)
    v[90:110, 32:D1, 40:60] = 0.1          # dark from the box top downwards only
    f = AA.candidate_signature(v, cen, ext, D1 - 1, AA.Knobs())
    assert f["dark_above"] is False and f["shadow_run_mm"] > 6.0


def test_stripe_width_recovers_planted_width():
    v = phantom()
    v[97:103, :, :] = 0.1                   # 6 voxels = 2.4 mm stripe
    assert AA.stripe_width_mm(v, (100, 40, 50), (6, 16, 20), AA.Knobs()) == pytest.approx(2.4, abs=0.8)
    v2 = phantom(); v2[80:120, :, :] = 0.1  # 40 voxels = 16 mm
    assert AA.stripe_width_mm(v2, (100, 40, 50), (6, 16, 20), AA.Knobs()) == pytest.approx(16.0, abs=0.8)
    assert AA.stripe_width_mm(phantom(), (100, 40, 50), (6, 16, 20), AA.Knobs()) == 0.0


def test_contact_edge_distance():
    v = phantom(); v[170:, :, :] = 0.0     # no contact beyond lateral 170
    d = AA.contact_edge_mm(v, (150, 40, 50), (10, 16, 20), AA.Knobs())
    assert d == pytest.approx((170 - 155) * 0.4, abs=0.8)
    d2 = AA.contact_edge_mm(phantom(), (150, 40, 50), (10, 16, 20), AA.Knobs())
    assert d2 == pytest.approx((200 - 155) * 0.4, abs=0.8)   # field edge


def test_periodicity_detects_ladder():
    prof = np.full(30, 0.3); prof[::5] = 0.8      # period 5 rows = 2 mm
    assert AA.periodicity(prof, AA.Knobs()) > 0.5
    assert AA.periodicity(np.linspace(0.6, 0.2, 30), AA.Knobs()) < 0.2


def test_tissue_floor_row():
    v = phantom(); v[:, 90:, :] = 0.0
    assert AA.tissue_floor_row(v, AA.Knobs()) == 89


# ------------------------------------------------------------------ Task 2: partition + stats

def _row(**kw):
    base = dict(top_mm=12.0, depth_mm=15.0, contact_edge_mm=30.0, depth_frac=0.3, periodicity=0.0, dark_above=False,
                shadow_run_mm=0.0, stripe_width_mm=0.0, inside_contrast=-0.1, cap_bright=0.0)
    base.update(kw)
    return base


def test_partition_order_and_rules():
    k = AA.Knobs()
    assert AA.assign_mechanism(_row(depth_mm=3.0), k) == "skin"
    assert AA.assign_mechanism(_row(top_mm=1.0, depth_mm=12.0), k) == "bounded_blob"   # a big box reaching the skin is not a skin lesion
    assert AA.assign_mechanism(_row(contact_edge_mm=2.0), k) == "marginal"
    assert AA.assign_mechanism(_row(depth_frac=0.9), k) == "deep"
    assert AA.assign_mechanism(_row(depth_mm=6.0, periodicity=0.8), k) == "reverberation"
    assert AA.assign_mechanism(_row(dark_above=True, shadow_run_mm=10, stripe_width_mm=2), k) == "narrow_column"
    assert AA.assign_mechanism(_row(dark_above=True, shadow_run_mm=10, stripe_width_mm=9), k) == "broad_column"
    assert AA.assign_mechanism(_row(shadow_run_mm=10), k) == "shadowing_blob"
    assert AA.assign_mechanism(_row(), k) == "bounded_blob"
    assert AA.assign_mechanism(_row(inside_contrast=0.05), k) == "other"


def test_cliffs_and_size_binned():
    assert AA.cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    df = pd.DataFrame({"label": ["pos"] * 8 + ["neg"] * 8, "public_id": [1] * 16,
                       "detector_of_origin": ["d"] * 16,
                       "diag_mm": list(range(8)) * 2, "x": [1] * 8 + [0] * 8})
    r = AA.size_binned_delta(df, "x", nbins=2)
    assert all(d == -1.0 for d in r["delta"]) and r["weighted"] == -1.0


# ------------------------------------------------------------------ Task 3: arrangement

def test_column_stack_beats_null():
    rng = np.random.default_rng(0)
    base = rng.uniform([0, 5, 0], [150, 40, 150], size=(30, 3))            # random cloud (mm)
    stack = np.array([[60.0, 8 + 6 * k, 70.0] for k in range(6)])          # one column, 6 deep
    r = AA.arrangement_null(np.vstack([base, stack]), AA.Knobs(), rng, n_perm=100)
    assert r["same_column"]["exceeds"] is True and r["same_column"]["z"] > 2
    r0 = AA.arrangement_null(base, AA.Knobs(), rng, n_perm=100)
    assert r0["same_column"]["z"] < 2


def test_sweep_chain_and_bands():
    rng = np.random.default_rng(1)
    base = rng.uniform([0, 5, 0], [150, 40, 150], size=(30, 3))
    chain = np.array([[40.0, 12.0, 20 + 8 * k] for k in range(6)])
    r = AA.arrangement_null(np.vstack([base, chain]), AA.Knobs(), rng, n_perm=100)
    assert r["sweep_chain"]["exceeds"] is True
    bands = np.array([[20 + 6 * k, 12.0, 70.0] for k in range(12)])        # regular 6 mm bands
    r = AA.arrangement_null(np.vstack([base, bands]), AA.Knobs(), rng, n_perm=100)
    assert r["band_regularity"]["exceeds"] is True


def test_context_counts():
    p = np.array([[10, 10, 10], [11, 10, 30], [40, 10.5, 10], [90, 30, 90.0]])
    c = AA.context_counts(p)
    assert c.shape == (4, 4) and c[0, 0] == 2 and c[0, 2] == 1 and c[3, 0] == 0 and c[0, 3] == 4


# ------------------------------------------------------------------ Task 4: ranking information

def test_irls_separable_and_oof_no_leak():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3)); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    w, b = AA.irls_logistic(X, y, l2=0.1)
    assert w[0] > w[2] and AA.auc(X @ w + b, y) > 0.95
    groups = np.repeat(np.arange(20), 20)
    oof = AA.grouped_oof_scores(X, y, groups, fit_mask=np.ones(400, bool), k=5, seed=0)
    assert oof.shape == (400,) and (oof >= 0).all() and (oof < 1).all() and AA.auc(oof, y) > 0.9


def test_grouped_folds_never_split_a_group():
    groups = np.repeat(np.arange(10), 7)
    folds = AA.grouped_folds(groups, k=5, seed=0)
    for f in range(5):
        test_groups = set(groups[folds == f])
        train_groups = set(groups[folds != f])
        assert not (test_groups & train_groups)


def test_leave_one_volume_out_folds():
    groups = np.repeat(np.arange(7), 3)
    folds = AA.grouped_folds(groups, k=0, seed=0)
    assert len(np.unique(folds)) == 7 and all(len(np.unique(groups[folds == f])) == 1 for f in range(7))
