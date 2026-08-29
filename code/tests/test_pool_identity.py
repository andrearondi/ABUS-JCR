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
from abus_jcr.rescore.evaluate import (assert_pool_identity, b0_rank_probability,
                                       b0_spread_probability)


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


# --------------------------------------------------------------------------- B0-rank
# The `B0-rank` baseline (added 2026-08-29 after [I3.11]): replace the detector's score with
# its WITHIN-SET rank. Label-free, zero-parameter, deployable — and on the iso val pool it
# scores 0.7889 +- 0.0209 against B0' 0.7062 +- 0.0146, so it is a BASELINE the rescorer has
# to clear, not a diagnostic. The anchor (top grid cell left empty) is the whole finding: it
# is what gives the swept curve its empty-set operating point.

def test_b0_rank_discards_cross_set_level():
    """Two sets whose scores live on completely different scales but share a within-set order
    must come out IDENTICAL — that is what "cross-volume confidence discarded" means."""
    prob = np.array([0.90, 0.60, 0.30,      # set 100, high band
                     0.09, 0.06, 0.03])     # set 101, low band
    pid = np.array([100, 100, 100, 101, 101, 101])
    p = b0_rank_probability(prob, pid)
    np.testing.assert_allclose(p[:3], p[3:])


def test_b0_rank_preserves_order_inside_a_set():
    prob = np.array([0.10, 0.50, 0.30])
    pid = np.array([7, 7, 7])
    p = b0_rank_probability(prob, pid)
    assert p[1] > p[2] > p[0]


def test_b0_rank_leaves_the_top_grid_cell_empty():
    """THE anchor. `_interpolate_recall_at_fp` returns 0 for every key FP below the cheapest
    achievable one, so a column that saturates the top cell has no empty-set point and its
    three lowest key rates are forced to zero. Every real probability column has this anchor;
    the unanchored reading is what made the recorded [I3.7] volume_neutral a floor."""
    from abus_jcr.probe.calibration import GRID, TOP
    p = b0_rank_probability(np.array([0.9, 0.5]), np.array([1, 1]))
    assert float(p.max()) <= TOP - GRID + 1e-12
    assert float(p.max()) == pytest.approx(TOP - GRID)


def test_b0_rank_lands_on_the_official_threshold_grid():
    from abus_jcr.probe.calibration import GRID
    p = b0_rank_probability(np.linspace(0.9, 0.1, 9), np.zeros(9, dtype=int))
    np.testing.assert_allclose(p / GRID, np.round(p / GRID), atol=1e-9)


def test_b0_rank_clips_a_set_deeper_than_the_grid_at_zero():
    """A train set can hold 343 candidates against 198 usable grid cells. The tail must clip
    at 0, never go negative — `to_official_pred_csv` requires probability in [0, 1)."""
    from abus_jcr.probe.calibration import N_LEVELS
    n = N_LEVELS + 50
    p = b0_rank_probability(np.linspace(1.0, 0.0, n), np.zeros(n, dtype=int))
    assert float(p.min()) >= 0.0
    assert float(p[-1]) == 0.0


def test_b0_rank_output_is_accepted_by_the_official_writer():
    rec = _record().copy()
    rec["prob_rank"] = b0_rank_probability(rec["score_max"].to_numpy(),
                                           rec["public_id"].to_numpy())
    pred = to_official_pred_csv(rec, "prob_rank")
    assert ((pred["probability"] >= 0.0) & (pred["probability"] < 1.0)).all()


def test_b0_rank_breaks_ties_deterministically():
    prob = np.array([0.2, 0.2, 0.2])
    pid = np.zeros(3, dtype=int)
    a, b = b0_rank_probability(prob, pid), b0_rank_probability(prob, pid)
    np.testing.assert_array_equal(a, b)
    assert len(set(a.tolist())) == 3        # distinct ranks, stable by row order


def test_b0_rank_matches_the_diagnostic_anchored_assignment():
    """PARITY. `rescore/evaluate` (deployed path) and `probe/calibration` (diagnostic path)
    must agree exactly, or [I3.7]'s re-measured row and the [4.7] B0-rank rung would be two
    different rules wearing one name."""
    from abus_jcr.probe.calibration import volume_neutral_probability
    rec = _record(n=24, seed=3)
    mine = b0_rank_probability(rec["score_max"].to_numpy(), rec["public_id"].to_numpy())
    theirs = volume_neutral_probability(rec, anchored=True).to_numpy(float)
    np.testing.assert_allclose(mine, theirs)


def test_b0_rank_never_reads_labels():
    """It is a BASELINE, so it must be computable at inference. Permuting the label column
    cannot move it — this is what rules out the label-reading ``anchored="auto"`` path."""
    rec = _record(n=24, seed=4)
    a = b0_rank_probability(rec["score_max"].to_numpy(), rec["public_id"].to_numpy())
    shuffled = rec.copy()
    shuffled["label"] = list(reversed(rec["label"].tolist()))
    b = b0_rank_probability(shuffled["score_max"].to_numpy(), shuffled["public_id"].to_numpy())
    np.testing.assert_array_equal(a, b)
