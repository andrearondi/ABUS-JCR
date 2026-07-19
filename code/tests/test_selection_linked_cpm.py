"""[P3-UPDATE D3/A1] Post-hoc checkpoint selection rule (torch-free core).

The heavy detect->link->oracle CPM computation is server-side; the DECISION lives in
``abus_jcr.detect.select`` and is unit-tested here: max CPM among epochs >= min_epoch,
tie-broken toward the later epoch, with the all-NaN fallback.
"""

import math

import pytest

from abus_jcr.detect.select import select_epoch, selection_stability


def test_legacy_picks_max_cpm_without_ceilings():
    # No ceilings + tol 0 -> legacy behaviour (max CPM, later-epoch tie-break).
    cpms = {0: 0.9, 5: 0.2, 4: 0.30, 6: 0.35, 8: 0.31}
    assert select_epoch(cpms, min_epoch=3) == 6              # 0 excluded by floor; 0.35 wins


def test_min_epoch_floor_excludes_preconvergence_epochs():
    cpms = {0: 0.99, 1: 0.95, 2: 0.9, 4: 0.40, 6: 0.42}      # epochs 0-2 are pre-convergence flukes
    assert select_epoch(cpms, min_epoch=3) == 6              # 0-2 ignored even though CPM is high


def test_ceiling_breaks_cpm_ties_the_seed0_case():
    # The real seed0 numbers: epoch 10 has the bare-max CPM but epoch 6 has a higher ceiling.
    cpms = {6: 0.4779, 10: 0.4798, 13: 0.4771, 15: 0.4667}
    ceilings = {6: 0.833, 10: 0.700, 13: 0.633, 15: 0.700}
    assert select_epoch(cpms, min_epoch=3, epoch_ceilings=ceilings, cpm_tol=0.02) == 6


def test_ceiling_tie_breaks_toward_earliest_epoch():
    cpms = {6: 0.48, 8: 0.48, 12: 0.48}
    ceilings = {6: 0.70, 8: 0.70, 12: 0.70}                  # all tied on CPM and ceiling
    assert select_epoch(cpms, min_epoch=3, epoch_ceilings=ceilings, cpm_tol=0.02) == 6  # earliest


def test_tol_zero_ignores_ceilings():
    cpms = {6: 0.4779, 10: 0.4798}
    ceilings = {6: 0.833, 10: 0.700}
    # tol 0 -> pure CPM argmax (later-epoch tie-break), ceilings unused
    assert select_epoch(cpms, min_epoch=3, epoch_ceilings=ceilings, cpm_tol=0.0) == 10


def test_all_nan_falls_back_to_latest():
    cpms = {6: float("nan"), 10: float("nan")}
    assert select_epoch(cpms, min_epoch=3) == 10


def test_raises_when_no_epoch_qualifies():
    with pytest.raises(ValueError):
        select_epoch({0: 0.5, 1: 0.6}, min_epoch=3)


def test_stability_reports_spread():
    cpms = {4: 0.30, 6: 0.34, 8: 0.33}
    spread, top = selection_stability(cpms, min_epoch=3, top_k=3)
    assert top[0][0] == 6                        # highest CPM first
    assert math.isclose(spread, 0.34 - 0.30, rel_tol=0, abs_tol=1e-9)
