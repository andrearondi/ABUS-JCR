"""[4.2b/c] Cost accounting and CI-cell selection.

The seed-0 `[4.2b]` run was estimated at ~35 min and took **~3 h**: the oracle costs ~7.6 s
per `evaluate()` call on this pool, not the ~2 s the Phase-4 runbook quotes, and 67 % of the
run was the per-epoch selection table. A step that cannot state its own cost gets planned
wrong, so the estimate is computed and printed before anything runs.
"""

import pytest

from abus_jcr.rescore.objective import estimate_oracle_calls, select_ci_cells


def _cells(cpms):
    return [{"name": f"c{i}", "raw": {"cpm": r}, "spread": {"cpm": s}}
            for i, (r, s) in enumerate(cpms)]


# ------------------------------------------------------------------ CI cell selection
def test_top_cells_are_ranked_by_the_better_of_raw_and_spread():
    """A cell that only wins after the rank-preserving remap is still a candidate."""
    got = select_ci_cells(_cells([(0.60, 0.61), (0.55, 0.70), (0.65, 0.64)]), 2)
    assert [c["name"] for c in got] == ["c1", "c2"]


def test_zero_requests_no_intervals():
    assert select_ci_cells(_cells([(0.6, 0.6)]), 0) == []


def test_k_above_the_grid_size_returns_every_cell():
    assert len(select_ci_cells(_cells([(0.6, 0.6), (0.5, 0.5)]), 99)) == 2


# ------------------------------------------------------------------ cost model
def test_the_seed0_run_is_reproduced_by_the_estimator():
    """16 cells x 60 epochs, B0 and B0-spread both at 200 draws — what actually ran."""
    assert estimate_oracle_calls(n_cells=16, epochs=60, n_boot_b0=200, n_boot_paired=0,
                                 ci_top_k=0, boot_b0_spread=True) == 1428


def test_dropping_the_b0_spread_interval_saves_exactly_its_draws():
    kw = dict(n_cells=16, epochs=60, n_boot_b0=200, n_boot_paired=0, ci_top_k=0)
    assert (estimate_oracle_calls(**kw, boot_b0_spread=True)
            - estimate_oracle_calls(**kw, boot_b0_spread=False)) == 200


def test_a_paired_draw_costs_two_calls():
    kw = dict(n_cells=4, epochs=60, n_boot_b0=0, boot_b0_spread=False)
    one = estimate_oracle_calls(**kw, n_boot_paired=100, ci_top_k=1)
    none = estimate_oracle_calls(**kw, n_boot_paired=0, ci_top_k=1)
    assert one - none == 200


def test_the_per_epoch_table_dominates_a_wide_grid():
    """67 % of the seed-0 run — the reason [4.2c] narrows the grid instead of the schedule."""
    total = estimate_oracle_calls(n_cells=16, epochs=60, n_boot_b0=200, n_boot_paired=0,
                                  ci_top_k=0, boot_b0_spread=True)
    assert 16 * 60 / total == pytest.approx(0.67, abs=0.01)
