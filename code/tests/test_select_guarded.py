"""[P3U3] Coverage-floor + CPM-guard selection (CONTINGENCY policy, not deployed). Torch-free."""

import pytest

from abus_jcr import conventions as C
from abus_jcr.detect.select import select_epoch, select_epoch_guarded


# The real fold2 table (abridged): the deployed rule picks 19 (CPM .429, ceiling .550); the guarded
# rule should reach epoch 10 (CPM .300, ceiling .900) because .300 >= .429 - .15.
FOLD2_CPM = {3: 0.20, 10: 0.300, 16: 0.414, 19: 0.429, 23: 0.430}
FOLD2_CEIL = {3: 0.85, 10: 0.900, 16: 0.526, 19: 0.550, 23: 0.500}
FOLD2_POOL = {3: 300.0, 10: 152.0, 16: 30.0, 19: 24.0, 23: 13.0}


def test_guarded_reaches_the_high_coverage_epoch():
    e = select_epoch_guarded(FOLD2_CPM, FOLD2_CEIL, min_epoch=3, coverage_floor=0.80,
                             cpm_guard=0.15, epoch_pools=FOLD2_POOL, pool_budget=230)
    assert e == 10                      # .900 ceiling, CPM .300 within .15 of the .430 max
    # the DEPLOYED rule picks 19: epochs 16/19/23 are CPM-tied (within .02 of the .430 max) and 19 has
    # the best ceiling among them (.550) — reproducing the real fold2 pick. The .900-ceiling epoch 10
    # is 0.13 below the max, far outside the tie band, so the deployed rule cannot reach it.
    assert select_epoch(FOLD2_CPM, 3, FOLD2_CEIL, C.DET_SELECT_CPM_TOL) == 19


def test_guard_blocks_a_degenerate_epoch_and_falls_back():
    # fold4-like: the only high-coverage epoch is unconverged (CPM .14 vs .41 max) -> guard blocks it,
    # so the rule must fall back to the DEPLOYED pick rather than deploy a broken ranking.
    cpm = {5: 0.14, 16: 0.411, 20: 0.400}
    ceil = {5: 0.74, 16: 0.579, 20: 0.526}
    e = select_epoch_guarded(cpm, ceil, min_epoch=3, coverage_floor=0.70, cpm_guard=0.15)
    assert e == 16 == select_epoch(cpm, 3, ceil, C.DET_SELECT_CPM_TOL)


def test_pool_budget_excludes_oversized_epochs():
    cpm = {5: 0.40, 9: 0.42}
    ceil = {5: 0.95, 9: 0.90}
    pools = {5: 900.0, 9: 150.0}        # epoch 5 clears coverage but blows the budget
    assert select_epoch_guarded(cpm, ceil, 3, 0.80, 0.15, pools, pool_budget=230) == 9


def test_falls_back_when_nothing_clears_the_floor():
    cpm = {4: 0.50, 8: 0.55}
    ceil = {4: 0.40, 8: 0.45}           # no epoch reaches an 0.80 coverage floor
    assert select_epoch_guarded(cpm, ceil, 3, 0.80, 0.15) == select_epoch(cpm, 3, ceil, 0.02)


def test_ties_break_to_the_earliest_epoch():
    cpm = {4: 0.40, 9: 0.40}
    ceil = {4: 0.90, 9: 0.95}           # equal CPM, both clear the floor -> earliest (least overfit)
    assert select_epoch_guarded(cpm, ceil, 3, 0.80, 0.15) == 4


def test_conventions_constants_present_and_not_deployed():
    assert C.DET_SELECT_COVERAGE_FLOOR == 0.80 and C.DET_SELECT_CPM_GUARD == 0.15
    # the DEPLOYED metric must still be the unchanged post-hoc linked-CPM rule
    assert C.DET_SELECTION_METRIC == "val_linked_cpm_3d@0.3_posthoc"
