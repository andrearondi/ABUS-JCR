"""[4.7 / 4.8] The ablation ladder, the fairness contract, set batching, and the
Phase-4 constant tripwire.

Exit check 5 is machine-checked here: every rung gets the same epochs, the same schedule,
**exactly 4** hyperparameter trials, and a B1 head whose parameter count is within ±10 % of
the selected set module — the do-not-drift rule "never make the independent baseline weaker
than the joint variants".

Torch-free: the trial table, the parameter arithmetic, the epoch-selection rule and the set
batching are all pure python/numpy.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.datasets import collate_sets, group_sets, set_sizes
from abus_jcr.rescore.variants import (
    LADDER,
    SUB_ABLATION_BLOCKS,
    VARIANTS,
    assert_fairness,
    b1_param_count,
    fairness_table,
    match_b1_capacity,
    select_epoch_by_val_cpm,
    trials_for,
)


# --------------------------------------------------------------------------- the ladder
def test_the_ladder_is_exactly_the_six_preregistered_rungs():
    assert LADDER == ("B0", "B1", "B2", "A1", "A2", "FULL")


def test_each_rung_isolates_what_the_spec_says_it_isolates():
    assert VARIANTS["B1"]["module"] == "mlp" and VARIANTS["B1"]["geometry"] is False
    assert VARIANTS["B2"]["module"] == "set" and VARIANTS["B2"]["geometry"] is False
    assert VARIANTS["A1"]["module"] == "set" and VARIANTS["A1"]["geometry"] is True
    assert VARIANTS["A2"]["module"] == "set" and VARIANTS["A2"]["geometry"] is False
    assert VARIANTS["FULL"]["module"] == "set" and VARIANTS["FULL"]["geometry"] is True


def test_w_rank_separates_the_ce_rungs_from_the_ranking_rungs():
    assert [VARIANTS[v]["w_rank"] for v in ("B1", "B2", "A1")] == [0.0, 0.0, 0.0]
    assert [VARIANTS[v]["w_rank"] for v in ("A2", "FULL")] == [1.0, 1.0]


def test_B0_is_not_a_trained_variant():
    assert "B0" not in VARIANTS          # B0 is the recorded score_max ranking, not a model


# --------------------------------------------------------------------------- trial budget
@pytest.mark.parametrize("variant", ["B1", "B2", "A1", "A2", "FULL"])
def test_every_rung_gets_exactly_four_hyperparameter_trials(variant):
    b2 = {"alpha": 0.25, "lr": 1e-3}
    assert len(trials_for(variant, b2_choice=b2)) == 4


def test_ce_rungs_sweep_alpha_and_lr():
    t = trials_for("B2")
    assert {(x["alpha"], x["lr"]) for x in t} == {(a["alpha"], a["lr"]) for a in C.RESC_CE_SEARCH}
    assert all(x["w_rank"] == 0.0 and x["lam"] == 1.0 for x in t)


def test_ranking_rungs_sweep_lambda_at_B2s_selected_alpha_and_lr():
    b2 = {"alpha": 0.50, "lr": 3e-4}
    t = trials_for("FULL", b2_choice=b2)
    assert sorted(x["lam"] for x in t) == sorted(C.RESC_LAMBDA_SEARCH)
    assert all(x["alpha"] == 0.50 and x["lr"] == 3e-4 and x["w_rank"] == 1.0 for x in t)


def test_ranking_rungs_require_the_B2_choice():
    with pytest.raises(ValueError, match="b2_choice"):
        trials_for("A2")


def test_A1_receives_the_same_four_trial_budget_as_B2():
    """Spec reconciliation (recorded): §4.6's fairness contract ("exactly 4 trials per
    rung", machine-checked by exit check 5) governs over §4.7's table cell, which reads as
    a single A1 trial. B2's selected config is ALWAYS one of A1's four, so the clean
    'A1 at B2's alpha/lr' isolation is still available for the primary delta."""
    b2 = {"alpha": 0.25, "lr": 1e-3}
    t = trials_for("A1", b2_choice=b2)
    assert len(t) == 4
    assert any(x["alpha"] == b2["alpha"] and x["lr"] == b2["lr"] for x in t)


def test_the_diagnostic_lambda_zero_run_is_outside_the_selection_budget():
    b2 = {"alpha": 0.25, "lr": 1e-3}
    assert C.RESC_LAMBDA_DIAGNOSTIC not in [x["lam"] for x in trials_for("FULL", b2_choice=b2)]
    assert C.RESC_LAMBDA_DIAGNOSTIC == 0.0


def test_sub_ablation_blocks_are_the_three_declared_ones():
    assert SUB_ABLATION_BLOCKS == ("rank", "score_stats", "tube_geom")


# --------------------------------------------------------------------------- fairness
def test_b1_param_count_matches_an_explicit_hand_computation():
    d_in, d_model, hidden, depth = 10, 8, 4, 2
    proj = d_in * d_model + d_model + 2 * d_model                   # Linear + LayerNorm
    mlp = (d_model * hidden + hidden) + (hidden * hidden + hidden)  # depth = 2
    head = hidden + 1
    assert b1_param_count(d_in, d_model, hidden, depth) == proj + mlp + head


def test_match_b1_capacity_lands_within_ten_percent_of_the_target():
    target = 400_000
    h = match_b1_capacity(d_in=161, d_model=128, target_params=target)
    got = b1_param_count(161, 128, h, 2)
    assert abs(got - target) / target <= 0.10, (h, got, target)


def test_match_b1_capacity_is_monotone_in_the_target():
    a = match_b1_capacity(d_in=161, d_model=128, target_params=200_000)
    b = match_b1_capacity(d_in=161, d_model=128, target_params=800_000)
    assert b > a


def test_fairness_table_passes_when_every_rung_matches():
    tbl = fairness_table(
        params={"B1": 402_000, "B2": 400_000, "A1": 425_000, "A2": 400_000, "FULL": 425_000},
        epochs={v: C.RESC_SET_EPOCHS for v in ("B1", "B2", "A1", "A2", "FULL")},
        trials={v: 4 for v in ("B1", "B2", "A1", "A2", "FULL")},
        reference="B2")
    assert_fairness(tbl)
    assert tbl["reference_params"] == 400_000


def test_fairness_table_fails_when_B1_is_handicapped():
    tbl = fairness_table(
        params={"B1": 100_000, "B2": 400_000, "A1": 400_000, "A2": 400_000, "FULL": 400_000},
        epochs={v: C.RESC_SET_EPOCHS for v in ("B1", "B2", "A1", "A2", "FULL")},
        trials={v: 4 for v in ("B1", "B2", "A1", "A2", "FULL")},
        reference="B2")
    with pytest.raises(AssertionError, match="B1"):
        assert_fairness(tbl)


def test_fairness_table_fails_on_an_unequal_trial_budget():
    tbl = fairness_table(
        params={v: 400_000 for v in ("B1", "B2", "A1", "A2", "FULL")},
        epochs={v: C.RESC_SET_EPOCHS for v in ("B1", "B2", "A1", "A2", "FULL")},
        trials={"B1": 4, "B2": 4, "A1": 4, "A2": 4, "FULL": 8},
        reference="B2")
    with pytest.raises(AssertionError, match="trial"):
        assert_fairness(tbl)


def test_fairness_table_fails_on_an_unequal_epoch_budget():
    tbl = fairness_table(
        params={v: 400_000 for v in ("B1", "B2", "A1", "A2", "FULL")},
        epochs={"B1": 30, "B2": 60, "A1": 60, "A2": 60, "FULL": 60},
        trials={v: 4 for v in ("B1", "B2", "A1", "A2", "FULL")},
        reference="B2")
    with pytest.raises(AssertionError, match="epoch"):
        assert_fairness(tbl)


def test_geometry_branch_params_are_excluded_from_the_fairness_comparison():
    """A1/FULL legitimately carry the extra GeometryBias projection — that IS the treatment.
    The contract compares the SHARED capacity, so the tolerance is applied to B1 vs the
    reference set module only."""
    tbl = fairness_table(
        params={"B1": 400_000, "B2": 400_000, "A1": 600_000, "A2": 400_000, "FULL": 600_000},
        epochs={v: C.RESC_SET_EPOCHS for v in ("B1", "B2", "A1", "A2", "FULL")},
        trials={v: 4 for v in ("B1", "B2", "A1", "A2", "FULL")},
        reference="B2")
    assert_fairness(tbl)


# --------------------------------------------------------------------------- epoch selection
def test_selection_takes_the_earliest_epoch_within_the_tolerance():
    """Inv.-2/A1 lesson: on 30 val lesions CPM moves in ~1/30 steps, so a bare argmax
    selects noise."""
    epochs = [0, 1, 2, 3, 4]
    cpm = [0.40, 0.58, 0.55, 0.59, 0.57]
    assert select_epoch_by_val_cpm(epochs, cpm, tol=0.02) == 1     # 0.58 is within 0.02 of 0.59


def test_selection_falls_back_to_the_argmax_when_nothing_ties():
    assert select_epoch_by_val_cpm([0, 1, 2], [0.10, 0.20, 0.60], tol=0.02) == 2


def test_selection_uses_the_declared_tolerance_by_default():
    assert C.RESC_SELECT_CPM_TOL == 0.02
    assert select_epoch_by_val_cpm([0, 1], [0.585, 0.60]) == 0


def test_selection_never_reads_a_val_loss_column():
    """RESC_SELECT_METRIC is the official average_recall; a val-loss-selected checkpoint is
    forbidden by exit check 8."""
    assert C.RESC_SELECT_METRIC == "val_cpm"
    tbl = pd.DataFrame({"epoch": [0, 1, 2], "val_cpm": [0.30, 0.60, 0.59],
                        "val_loss": [0.9, 0.9, 0.1]})
    assert select_epoch_by_val_cpm(tbl["epoch"], tbl["val_cpm"]) == 1


def test_selection_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        select_epoch_by_val_cpm([0, 1], [0.1])


# --------------------------------------------------------------------------- constants tripwire
def test_the_frozen_phase3_substrate_has_not_drifted():
    """Phase 4 touches NONE of these (spec §Context).

    **Legacy-profile tripwire.** These are the values frozen on the DEPLOYED substrate
    ([P3U2.7]/[P3U2.8], re-derived unchanged on the corrected-flip folds at [F.4]), and they
    are what every recorded Phase-3/4 number stands on. On the `measured` axis profile the
    same constants are **provisional and gate-derived** by construction — `[I2.6]`/`[I3.2]`
    of `iso/RB_ISO_REBUILD.md` re-derive them on a cache whose in-plane voxel size is
    2.7x finer laterally, so pinning the deployed literals there would assert that a
    different substrate must produce identical constants, which is the opposite of what the
    branch is measuring. Split rather than relaxed: the structural half below binds on both.
    """
    # --- profile-independent: Inv. 4 structure, not substrate-specific values -------------
    assert C.LINK_3DNMS_IOU is None,        "3D NMS is OFF on both substrates (not recall-neutral)"
    assert C.LINK_CONTAINMENT_THRESH == 1.0, "containment suppression is OFF on both"
    assert C.RESCORER_POOL_BUDGET == 200,   "a Phase-4 O(n^2) budget, not a physical quantity"
    assert C.LINK_SCORE_AGG == "max"        # the committed per-tube aggregation
    assert C.LINK_OP_SCORE_THRESH == C.DET_SELECT_OP_THRESH, \
        "selection and deployment must read the pool at the SAME operating point ([P3U2.9])"

    if C.AXIS_PROFILE != "legacy":
        pytest.skip(f"substrate values are gate-derived on the {C.AXIS_PROFILE!r} profile "
                    "(iso/RB_ISO_REBUILD.md [I2.6], [I3.2]); the structural half above ran")

    expected = dict(LINK_IOU=0.30, LINK_MAX_Z_GAP=1, LINK_MIN_TUBE_LEN=8,
                    LINK_MAX_TUBE_ZSPAN=182, LINK_MAX_CENTROID_DRIFT=342,
                    LINK_CONTAINMENT_THRESH=1.0, LINK_3DNMS_IOU=None,
                    PREFILTER_SCORE_FLOOR=0.08, LINK_NMS_THRESH=0.70,
                    LINK_OP_SCORE_THRESH=0.03, RESCORER_POOL_BUDGET=200)
    bad = {k: (getattr(C, k), v) for k, v in expected.items() if getattr(C, k) != v}
    assert not bad, f"frozen Phase-3 constants drifted: {bad}"


def test_phase4_constants_match_the_spec():
    assert (C.RESC_CROP_OUT, C.RESC_CROP_CONTEXT, C.RESC_CROP_MIN_SIDE, C.RESC_CROP_MAX_SIDE) \
        == (48, 1.5, 48, 160)
    assert C.RESC_D_APP == 128 and C.RESC_RANK_PE_DIM == 16 and C.RESC_GEOM_PE_DIM == 64
    # 576, raised from 320 on 2026-08-09: the PROMOTED pool's worst set is train fold0 vol14
    # at 509 candidates ([F.7]), so the old cap would have failed [4.1]'s assertion on the
    # first run. The cap must strictly exceed the largest set either pool contains.
    assert C.RESC_MAX_SET_SIZE == 576 and C.RESC_NEG_POS_RATIO is None
    assert C.RESC_MAX_SET_SIZE > 509, "the promoted train pool contains a 509-candidate set"
    assert C.RESC_SEEDS == (0, 1, 2) and C.RESC_SET_EPOCHS == 60 and C.RESC_ENC_EPOCHS == 30
    assert C.RESC_TOKEN_BLOCKS == ("appearance", "abs_geom", "score_stats", "tube_geom", "rank")
    assert len(C.RESC_CE_SEARCH) == 4 and len(C.RESC_LAMBDA_SEARCH) == 4
    assert len(C.RESC_SET_CAPACITY_GRID) == 2


# --------------------------------------------------------------------------- set batching
def _rec(spec):
    rows = []
    for (det, pid), n in spec.items():
        for k in range(n):
            rows.append({"detector_of_origin": det, "public_id": pid,
                         "candidate_id": f"{det}:{pid}:{k}", "label": "neg"})
    return pd.DataFrame(rows)


def test_a_set_is_a_detector_volume_pair_never_a_bare_volume():
    """A Val volume contributes 3 independent sets, one per seed (Inv. 7 / Inv. 14)."""
    rec = _rec({("full_seed0", 100): 5, ("full_seed1", 100): 4, ("full_seed2", 100): 6})
    groups = group_sets(rec)
    assert len(groups) == 3
    assert {k[0] for k in groups} == {"full_seed0", "full_seed1", "full_seed2"}


def test_group_sets_preserves_record_row_indices():
    rec = _rec({("fold0", 1): 2, ("fold1", 2): 3})
    groups = group_sets(rec)
    all_rows = sorted(int(i) for idx in groups.values() for i in idx)
    assert all_rows == list(range(len(rec)))


def test_set_sizes_reports_the_max_the_pad_width_must_cover():
    rec = _rec({("fold0", 1): 7, ("fold1", 2): 3})
    s = set_sizes(rec)
    assert s.max() == 7 and len(s) == 2


def test_collate_pads_to_the_declared_width_and_masks_the_padding():
    feats = np.arange(20, dtype=float).reshape(5, 4)
    labels = np.array([1.0, 0.0, 0.0, -1.0, 0.0])
    coord = np.zeros((5, 3)); length = np.ones((5, 3))
    batch = collate_sets([np.array([0, 1, 2]), np.array([3, 4])],
                         feats, labels, coord, length, max_set_size=6)
    assert batch["feats"].shape == (2, 6, 4)
    assert batch["mask"].tolist() == [[True] * 3 + [False] * 3, [True] * 2 + [False] * 4]
    np.testing.assert_allclose(batch["feats"][0, :3], feats[:3])
    np.testing.assert_allclose(batch["labels"][1, :2], labels[3:])


def test_collate_pads_to_the_largest_set_when_no_width_is_given():
    feats = np.zeros((5, 2)); labels = np.zeros(5)
    coord = np.zeros((5, 3)); length = np.ones((5, 3))
    batch = collate_sets([np.array([0, 1, 2]), np.array([3, 4])], feats, labels, coord, length)
    assert batch["feats"].shape[1] == 3


def test_collate_refuses_a_set_larger_than_the_pad_width():
    feats = np.zeros((5, 2)); labels = np.zeros(5)
    coord = np.zeros((5, 3)); length = np.ones((5, 3))
    with pytest.raises(ValueError, match="RESC_MAX_SET_SIZE|pad width"):
        collate_sets([np.arange(5)], feats, labels, coord, length, max_set_size=4)


def test_padded_lengths_are_positive_so_the_geometry_descriptor_stays_finite():
    """log(w_n/w_m) with a zero-padded length would be -inf and could poison the softmax
    even where the mask should have removed it."""
    feats = np.zeros((3, 2)); labels = np.zeros(3)
    coord = np.zeros((3, 3)); length = np.ones((3, 3))
    batch = collate_sets([np.array([0, 1, 2])], feats, labels, coord, length, max_set_size=6)
    assert np.isfinite(np.log(batch["length"])).all()
    assert (batch["length"] > 0).all()
