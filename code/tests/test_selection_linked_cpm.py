"""[P3-UPDATE D3/A1] Post-hoc checkpoint selection rule (torch-free core).

The heavy detect->link->oracle CPM computation is server-side; the DECISION lives in
``abus_jcr.detect.select`` and is unit-tested here: max CPM among epochs >= min_epoch,
tie-broken toward the later epoch, with the all-NaN fallback.
"""

import math

import pytest

from abus_jcr.detect.select import select_epoch, selection_stability


def test_picks_max_cpm_in_converged_window():
    cpms = {0: 0.9, 5: 0.2, 15: 0.30, 20: 0.35, 25: 0.31}   # epoch 0 is a high-LR fluke
    assert select_epoch(cpms, min_epoch=15) == 20            # 0 and 5 excluded; 0.35 wins


def test_min_epoch_excludes_early_lucky_epochs():
    cpms = {2: 0.99, 16: 0.40, 18: 0.42}
    assert select_epoch(cpms, min_epoch=15) == 18            # the 0.99 at epoch 2 is ignored


def test_tie_breaks_toward_later_epoch():
    cpms = {16: 0.40, 20: 0.40, 24: 0.40}
    assert select_epoch(cpms, min_epoch=15) == 24


def test_all_nan_falls_back_to_latest():
    cpms = {16: float("nan"), 20: float("nan")}
    assert select_epoch(cpms, min_epoch=15) == 20


def test_raises_when_no_epoch_qualifies():
    with pytest.raises(ValueError):
        select_epoch({0: 0.5, 10: 0.6}, min_epoch=15)


def test_stability_reports_spread():
    cpms = {15: 0.30, 18: 0.34, 21: 0.33}
    spread, top = selection_stability(cpms, min_epoch=15, top_k=3)
    assert top[0][0] == 18                       # highest CPM first
    assert math.isclose(spread, 0.34 - 0.30, rel_tol=0, abs_tol=1e-9)
