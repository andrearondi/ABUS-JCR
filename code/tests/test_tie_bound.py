"""[MIG-5b] The tie-ambiguity bound must bracket the official evaluator exactly.

Written 2026-09-04 with the script, after the [MIG-5] audit flagged risers on every rung. The
pin: for any curve, the vendored ``_get_key_recall`` (whatever order pandas' non-stable sort
leaves the ties in) must land INSIDE [lo, hi]; for a tie-free curve the bounds must collapse
onto it exactly; and a riser away from every bracket endpoint must contribute zero spread.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.eval._official_det_score import _get_key_recall

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_tie_bound import cpm_bounds, key_recall_bounds  # noqa: E402

KEYS = tuple(C.KEY_FP)


def _official_cpm(fp, recall):
    return float(np.mean(_get_key_recall(list(fp), list(recall), list(KEYS))))


def test_tie_free_curve_collapses_onto_the_official_value():
    fp = [0.05, 0.2, 0.7, 1.5, 3.0, 6.0, 9.0]
    rc = [0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90]
    b = cpm_bounds(fp, rc)
    assert b["spread"] == pytest.approx(0.0, abs=1e-12)
    assert b["cpm_lo"] == pytest.approx(_official_cpm(fp, rc), abs=1e-9)
    assert b["keys_affected"] == 0


def test_riser_between_read_points_contributes_no_spread():
    """A vertical riser at fp=0.7 — flagged by the [MIG-5] audit pattern — is harmless when no
    key rate's bracket endpoint lands on it (keys 0.5 and 1 bracket through OTHER points)."""
    fp = [0.05, 0.2, 0.6, 0.7, 0.7, 0.9, 1.5, 3.0, 6.0, 9.0]
    rc = [0.10, 0.30, 0.40, 0.45, 0.55, 0.58, 0.60, 0.70, 0.80, 0.90]
    b = cpm_bounds(fp, rc)
    assert b["spread"] == pytest.approx(0.0, abs=1e-12)


def test_riser_at_a_bracket_endpoint_is_bounded_and_brackets_the_official_read():
    """Two entries tied at fp=0.4 with different recalls, and key 0.5's <=-side endpoint IS
    0.4: the official read depends on tie order; both orders must sit inside [lo, hi]."""
    fp = [0.1, 0.4, 0.4, 0.8, 2.0, 5.0, 9.0]
    rc = [0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90]
    lo, hi = key_recall_bounds(fp, rc, 0.5)
    assert hi - lo > 1e-6
    # both tie orders through the real evaluator land inside the bounds
    for order in ([1, 2], [2, 1]):
        perm = [0] + order + [3, 4, 5, 6]
        off = _get_key_recall([fp[i] for i in perm], [rc[i] for i in perm], [0.5])[0]
        assert lo - 1e-9 <= off <= hi + 1e-9
    b = cpm_bounds(fp, rc)
    assert b["keys_affected"] >= 1
    assert b["cpm_lo"] <= _official_cpm(fp, rc) <= b["cpm_hi"]


def test_key_exactly_on_a_tied_fp_value():
    """fp == key on both sides of the bracket (t ~ 0): the read collapses to the <=-side pick;
    bounds must still bracket every ordering."""
    fp = [0.1, 0.5, 0.5, 2.0, 9.0]
    rc = [0.10, 0.40, 0.60, 0.70, 0.90]
    lo, hi = key_recall_bounds(fp, rc, 0.5)
    assert lo == pytest.approx(0.40, abs=1e-6) and hi == pytest.approx(0.60, abs=1e-6)
    for order in ([1, 2], [2, 1]):
        perm = [0] + order + [3, 4]
        off = _get_key_recall([fp[i] for i in perm], [rc[i] for i in perm], [0.5])[0]
        assert lo - 1e-9 <= off <= hi + 1e-9


def test_below_cheapest_and_above_dearest_are_hard():
    fp = [1.0, 2.0]
    rc = [0.5, 0.8]
    assert key_recall_bounds(fp, rc, 0.125) == (0.0, 0.0)      # the [I3.11] empty-set rule
    assert key_recall_bounds(fp, rc, 8.0) == (0.8, 0.8)


def test_random_curves_always_bracket_the_official_evaluator():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(5, 40))
        fp = np.round(rng.uniform(0, 10, n), 1)               # coarse grid -> frequent ties
        rc = np.sort(rng.uniform(0, 1, n))
        b = cpm_bounds(fp, rc)
        off = _official_cpm(fp, rc)
        assert b["cpm_lo"] - 1e-9 <= off <= b["cpm_hi"] + 1e-9
