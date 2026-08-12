"""[4.2b] The objective grid + its record-level target/weight builders."""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.objective import (OBJECTIVE_FACTORS, objective_grid, record_lesion_weights,
                                        record_targets, threshold_occupancy)


def _rec(iou, label, sets=None, dets=None):
    n = len(iou)
    return pd.DataFrame({
        "iou_gt": iou, "label": label,
        "public_id": sets if sets is not None else [0] * n,
        "detector_of_origin": dets if dets is not None else ["full_seed0"] * n,
    })


# ----------------------------------------------------------------------- the grid
def test_grid_is_the_full_factorial_of_the_declared_factors():
    n = 1
    for values in OBJECTIVE_FACTORS.values():
        n *= len(values)
    assert len(objective_grid()) == n


def test_the_deployed_objective_is_in_the_grid_as_the_reference_cell():
    """[4.3]/[4.6] ship gamma=2, hard labels + ignore mask, per-candidate weighting."""
    ref = [v for v in objective_grid()
           if v["gamma"] == C.RESC_FOCAL_GAMMA and not v["soft"] and not v["per_lesion"]
           and v["alpha"] == 0.25]
    assert len(ref) == 1, "the deployed cell must appear exactly once — it is the control"
    assert ref[0]["is_deployed"] is True


def test_exactly_one_cell_is_flagged_deployed():
    assert sum(bool(v["is_deployed"]) for v in objective_grid()) == 1


def test_every_variant_has_a_unique_name():
    names = [v["name"] for v in objective_grid()]
    assert len(set(names)) == len(names)


# ----------------------------------------------------------------------- targets
def test_hard_targets_reproduce_the_deployed_label_codes():
    rec = _rec([0.0, 0.5, 0.2], ["neg", "pos", "ignore"])
    assert record_targets(rec, soft=False) == pytest.approx([0.0, 1.0, -1.0])


def test_soft_targets_ramp_the_ignore_band_and_mask_nothing():
    rec = _rec([0.0, 0.5, 0.20], ["neg", "pos", "ignore"])
    t = record_targets(rec, soft=True)
    assert t == pytest.approx([0.0, 1.0, 0.5])
    assert (t >= 0.0).all(), "no row may carry the ignore code under the soft target"


# ----------------------------------------------------------------------- lesion weights
def test_lesion_weights_split_one_lesions_worth_across_a_sets_positives():
    rec = _rec([0.9, 0.8, 0.0, 0.9], ["pos", "pos", "neg", "pos"],
               sets=[1, 1, 1, 2])
    w = record_lesion_weights(rec)
    assert w[0] + w[1] == pytest.approx(1.0)
    assert w[2] == pytest.approx(1.0)          # negatives untouched
    assert w[3] == pytest.approx(1.0)          # lone hit in its own set


def test_lesion_weights_split_per_set_not_per_volume():
    """A SET is (detector, volume) — Inv. 7. Two seeds on one volume are two lesions' worth."""
    rec = _rec([0.9, 0.9], ["pos", "pos"], sets=[1, 1],
               dets=["full_seed0", "full_seed1"])
    assert record_lesion_weights(rec) == pytest.approx([1.0, 1.0])


# ----------------------------------------------------------------------- spread diagnostic
def test_threshold_occupancy_counts_the_sweep_bins_the_scores_actually_reach():
    """The oracle sweeps np.arange(0, 1, 0.005); a saturated score visits almost none."""
    saturated = np.concatenate([np.full(500, 1e-7), np.full(20, 1.0 - 1e-7)])
    spread = np.linspace(0.0, 1.0 - 1e-7, 520)
    assert threshold_occupancy(saturated) < 5
    assert threshold_occupancy(spread) > 150


def test_threshold_occupancy_is_at_most_the_sweep_length():
    assert threshold_occupancy(np.linspace(0.0, 0.999, 10000)) <= 200
