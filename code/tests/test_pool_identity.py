"""[Inv. 8] All six rungs re-rank ONE frozen pool — machine-checked.

Same rows, same boxes; only the ``probability`` column may differ. If a rung could move a
box, "rescored vs not" would be a comparison across different pools and the recall ceiling
would stop cancelling — the single most damaging silent failure available to Phase 4.

Also pins the ``B0-spread`` control (§4.7): a strictly rank-preserving remap of B0's scores
onto the full ``[0,1)`` grid, which bounds the FROC grid-quantisation artefact on the real
pool instead of assuming it away (the local Check-2 synthetic measured Δ = −0.0061).

Torch-free.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.candidates.record import CANDIDATE_COLUMNS, to_official_pred_csv
from abus_jcr.rescore.evaluate import assert_pool_identity, b0_spread_probability


def _record(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "public_id": np.repeat([100, 101], n // 2),
        "candidate_id": [f"full_seed0:{i}" for i in range(n)],
        "detector_of_origin": ["full_seed0"] * n, "split": ["val"] * n, "fold": [-1] * n,
        "coordX": rng.uniform(50, 400, n), "coordY": rng.uniform(50, 400, n),
        "coordZ": rng.uniform(20, 150, n),
        "x_length": rng.uniform(10, 60, n), "y_length": rng.uniform(10, 60, n),
        "z_length": rng.uniform(5, 40, n),
        "score_max": rng.uniform(0.08, 0.6, n), "score_mean": rng.uniform(0.02, 0.4, n),
        "score_std": rng.uniform(0, 0.1, n), "score_min": rng.uniform(0, 0.1, n),
        "slice_count": rng.integers(8, 60, n).astype(float),
        "z_span": rng.integers(8, 90, n).astype(float), "fill_ratio": rng.uniform(0.3, 1.0, n),
        "centroid_jitter": rng.uniform(0, 1, n), "area_cv": rng.uniform(0, 1, n),
        "rank": np.arange(1, n + 1), "rank_norm": np.linspace(0, 1, n),
        "label": ["pos"] + ["neg"] * (n - 2) + ["ignore"], "iou_gt": rng.uniform(0, 0.6, n),
        "cen_d0": rng.uniform(10, 140, n), "cen_d1": rng.uniform(10, 300, n),
        "cen_d2": rng.uniform(10, 380, n), "ext_d0": rng.uniform(5, 60, n),
        "ext_d1": rng.uniform(5, 180, n), "ext_d2": rng.uniform(5, 80, n),
        "preprocess_hash": ["abc"] * n,
    })[CANDIDATE_COLUMNS]


# --------------------------------------------------------------------------- Inv. 8
def test_two_rungs_pred_csvs_differ_only_in_probability():
    rec = _record()
    rescored = rec.copy()
    rescored["prob_full"] = np.random.default_rng(1).uniform(0, 0.99, len(rec))
    a = to_official_pred_csv(rec, "score_max")
    b = to_official_pred_csv(rescored, "prob_full")
    assert_pool_identity(a, b)                       # must not raise


def test_pool_identity_raises_when_a_box_moves():
    rec = _record()
    moved = rec.copy()
    moved.loc[0, "coordX"] = moved.loc[0, "coordX"] + 5.0
    a = to_official_pred_csv(rec, "score_max")
    b = to_official_pred_csv(moved, "score_max")
    with pytest.raises(AssertionError, match="coordX"):
        assert_pool_identity(a, b)


def test_pool_identity_raises_when_a_row_is_dropped():
    rec = _record()
    a = to_official_pred_csv(rec, "score_max")
    b = to_official_pred_csv(rec.iloc[:-1], "score_max")
    with pytest.raises(AssertionError, match="row count"):
        assert_pool_identity(a, b)


def test_ignore_band_candidates_are_scored_and_written():
    """Inv. 8/11: they are pool members, so they appear in the pred CSV even though they
    enter neither loss term."""
    rec = _record()
    assert (rec["label"] == "ignore").any()
    pred = to_official_pred_csv(rec, "score_max")
    assert len(pred) == len(rec)


def test_pred_csv_row_count_equals_the_record_row_count():
    rec = _record(n=20)
    assert len(to_official_pred_csv(rec, "score_max")) == 20


# --------------------------------------------------------------------------- B0-spread
def test_b0_spread_preserves_the_ranking_exactly():
    """Order-isomorphism, stated directly so tie-breaking of equal scores cannot make the
    assertion depend on numpy's sort stability: p_i < p_j => s_i < s_j, and p_i == p_j =>
    s_i == s_j."""
    p = np.array([0.02, 0.27, 0.11, 0.11, 0.05])
    s = b0_spread_probability(p)
    for i in range(len(p)):
        for j in range(len(p)):
            if p[i] < p[j]:
                assert s[i] < s[j], (i, j, p, s)
            elif p[i] == p[j]:
                assert s[i] == s[j], (i, j, p, s)


def test_b0_spread_lands_in_the_open_unit_interval():
    p = np.array([0.02, 0.27, 0.11, 0.05, 0.30])
    s = b0_spread_probability(p)
    assert float(s.min()) >= 0.0 and float(s.max()) < 1.0


def test_b0_spread_widens_a_compressed_band():
    """The measured pool sits in a narrow band (TP median 0.195 vs FP 0.112), which the
    official ``np.arange(0, 1, 0.005)`` sweep quantises differently from a spread one."""
    p = np.linspace(0.02, 0.272, 200)
    s = b0_spread_probability(p)
    assert (s.max() - s.min()) > (p.max() - p.min())


def test_b0_spread_handles_ties_deterministically():
    p = np.array([0.1, 0.1, 0.1, 0.2])
    a, b = b0_spread_probability(p), b0_spread_probability(p)
    np.testing.assert_array_equal(a, b)
    assert a[3] == a.max()


def test_b0_spread_output_is_accepted_by_the_official_writer():
    rec = _record()
    rec = rec.copy()
    rec["prob_spread"] = b0_spread_probability(rec["score_max"].to_numpy())
    pred = to_official_pred_csv(rec, "prob_spread")
    assert ((pred["probability"] >= 0.0) & (pred["probability"] < 1.0)).all()


def test_probability_clamp_constant_is_respected():
    from abus_jcr.rescore.losses import to_probability
    p = to_probability(np.array([1e9]))
    assert float(p[0]) <= 1.0 - C.RESC_PROB_EPS + 1e-15
