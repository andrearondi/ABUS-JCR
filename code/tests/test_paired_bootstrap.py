"""[4.9] ``paired_bootstrap_delta`` — the CI every Phase-4/5 comparison must carry (Inv. 12).

Because every rung re-ranks the IDENTICAL pool (Inv. 8), a paired bootstrap — resample the
volumes ONCE per draw and score BOTH conditions on that same resample — is the only correct
interval for a within-pool comparison, and it is far tighter than the difference of two
marginal intervals. Phase 5 requires it for the test-split condition deltas.

**Why the draw counts here are tiny.** The vendored official oracle sweeps 200 thresholds
per call and costs ~2 s; a 1000-draw paired bootstrap is ~1 h of wall clock and belongs on
the server, not in a unit test. So the local suite proves the *mechanism* with few draws.
In particular the pairing itself is tested directly rather than statistically: when the two
conditions differ on exactly ONE volume, every draw that misses that volume must yield a
delta of **exactly 0** — which can only happen if both conditions saw the same resample.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.conventions import GT_COLUMNS, PRED_COLUMNS
from abus_jcr.eval.froc import cpm, evaluate_froc, paired_bootstrap_delta

N_VOL = 6


def _gt(n=N_VOL):
    return pd.DataFrame([{"public_id": i, "coordX": 100.0, "coordY": 100.0, "coordZ": 100.0,
                          "x_length": 40.0, "y_length": 40.0, "z_length": 40.0}
                         for i in range(n)], columns=GT_COLUMNS)


def _pred(hit_prob, miss_prob, n_fp=2, n=N_VOL, only_vol=None, other=None):
    """One TP (exactly on the GT box) + ``n_fp`` far-away FPs per volume.

    ``only_vol`` / ``other``: use ``(hit_prob, miss_prob)`` for that one volume and
    ``other = (hit, miss)`` for the rest — the single-volume difference the pairing test needs.
    """
    rows = []
    for i in range(n):
        hp, mp = (hit_prob, miss_prob) if (only_vol is None or i == only_vol) else other
        rows.append([i, 100.0, 100.0, 100.0, 40.0, 40.0, 40.0, hp])
        for k in range(n_fp):
            rows.append([i, 400.0 + 50 * k, 400.0, 400.0, 20.0, 20.0, 20.0, mp + 0.001 * k])
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def test_identical_predictions_give_a_zero_delta_with_an_interval_covering_zero():
    gt = _gt()
    p = _pred(0.9, 0.1)
    out = paired_bootstrap_delta(gt, p, p, n_boot=5, seed=0)
    assert out["delta_point"] == pytest.approx(0.0, abs=1e-12)
    assert out["lo"] <= 0.0 <= out["hi"]
    assert out["frac_positive"] == pytest.approx(0.0, abs=1e-12)


def test_a_strictly_better_ranking_gives_a_positive_delta():
    gt = _gt()
    good = _pred(0.9, 0.1)          # TP above every FP
    bad = _pred(0.05, 0.5)          # TP below every FP
    out = paired_bootstrap_delta(gt, good, bad, n_boot=5, seed=0)
    assert out["delta_point"] > 0.0
    assert out["frac_positive"] > 0.5
    assert -1.0 <= out["lo"] <= out["hi"] <= 1.0


def test_the_delta_point_equals_the_difference_of_the_two_unresampled_cpms():
    gt = _gt()
    a, b = _pred(0.9, 0.1), _pred(0.05, 0.5)
    out = paired_bootstrap_delta(gt, a, b, n_boot=2, seed=0)
    expected = cpm(evaluate_froc(gt, a)) - cpm(evaluate_froc(gt, b))
    assert out["delta_point"] == pytest.approx(expected, abs=1e-12)


def test_the_sign_flips_when_the_conditions_are_swapped():
    gt = _gt()
    a, b = _pred(0.9, 0.1), _pred(0.05, 0.5)
    fwd = paired_bootstrap_delta(gt, a, b, n_boot=4, seed=0)
    rev = paired_bootstrap_delta(gt, b, a, n_boot=4, seed=0)
    assert rev["delta_point"] == pytest.approx(-fwd["delta_point"], abs=1e-12)
    assert rev["lo"] == pytest.approx(-fwd["hi"], abs=1e-9)
    assert rev["hi"] == pytest.approx(-fwd["lo"], abs=1e-9)


def test_both_conditions_are_scored_on_the_SAME_resample():
    """THE pairing property (spec §4.9). The two conditions differ on volume 0 only, so any
    draw that does not sample volume 0 must give a delta of EXACTLY zero. Under independent
    resampling that would essentially never happen."""
    gt = _gt()
    a = _pred(0.9, 0.1, only_vol=0, other=(0.9, 0.1))
    b = _pred(0.02, 0.6, only_vol=0, other=(0.9, 0.1))     # only volume 0 is degraded
    out = paired_bootstrap_delta(gt, a, b, n_boot=10, seed=0)
    boot = np.asarray(out["boot"])
    exact_zero = float(np.mean(np.abs(boot) < 1e-12))
    assert exact_zero > 0.15, f"only {exact_zero:.0%} of draws were exactly paired: {boot}"
    assert exact_zero < 1.0, "the conditions must differ on at least some draws"


def test_the_bootstrap_is_seeded_and_reproducible():
    gt = _gt()
    a, b = _pred(0.9, 0.1), _pred(0.6, 0.5)
    x = paired_bootstrap_delta(gt, a, b, n_boot=3, seed=7)
    y = paired_bootstrap_delta(gt, a, b, n_boot=3, seed=7)
    assert (x["lo"], x["hi"]) == (y["lo"], y["hi"])
    np.testing.assert_array_equal(x["boot"], y["boot"])


def test_a_different_seed_changes_the_draws():
    gt = _gt()
    a, b = _pred(0.9, 0.1, only_vol=0, other=(0.9, 0.1)), _pred(0.02, 0.6, only_vol=0,
                                                                other=(0.9, 0.1))
    x = paired_bootstrap_delta(gt, a, b, n_boot=4, seed=1)
    y = paired_bootstrap_delta(gt, a, b, n_boot=4, seed=2)
    assert not np.array_equal(x["boot"], y["boot"])


def test_boot_array_and_count_are_returned_for_downstream_reporting():
    gt = _gt()
    a, b = _pred(0.9, 0.1), _pred(0.6, 0.5)
    out = paired_bootstrap_delta(gt, a, b, n_boot=4, seed=0)
    assert len(out["boot"]) == 4 and out["n_boot"] == 4
    assert set(out) >= {"delta_point", "lo", "hi", "frac_positive", "n_boot", "boot"}


def test_missing_probability_column_is_rejected():
    gt = _gt()
    a = _pred(0.9, 0.1)
    with pytest.raises(ValueError, match="probability"):
        paired_bootstrap_delta(gt, a.drop(columns=["probability"]), a, n_boot=2)


def test_the_interval_brackets_the_bootstrap_distribution():
    gt = _gt()
    a = _pred(0.9, 0.1, only_vol=0, other=(0.9, 0.1))
    b = _pred(0.02, 0.6, only_vol=0, other=(0.9, 0.1))
    out = paired_bootstrap_delta(gt, a, b, n_boot=6, seed=3)
    boot = np.asarray(out["boot"])
    assert out["lo"] >= boot.min() - 1e-12 and out["hi"] <= boot.max() + 1e-12
    assert out["lo"] <= out["hi"]


# --------------------------------------------------------------------------- deferred CI
# `phase4_pretrain_encoder` used to run a 200-draw CI on EVERY epoch. Measured 2026-08-31 on
# job 17425385: ~2.9 s per draw, so ~10 min per epoch, 5 h per arm — and `pretrain_encoder`
# selects with `select_epoch_by_val_cpm`, which reads `val_cpm` and NEVER the interval. 29 of
# 30 intervals were computed and discarded, and the GPU idled through all of it.
#
# The fix defers the CI to the selected epoch. It is only result-invariant if the interval is
# a pure function of (gt, pred, n_boot, seed) — i.e. computing it later gives byte-identical
# bounds. That is what these pin.

def test_marginal_ci_is_a_pure_function_of_its_inputs():
    """Deferring the CI to after the loop must not change it."""
    from abus_jcr.eval.froc import bootstrap_cpm_ci
    gt, pred = _gt(), _pred(0.9, 0.2)
    now = bootstrap_cpm_ci(gt, pred, n_boot=8, seed=0)
    later = bootstrap_cpm_ci(gt, pred, n_boot=8, seed=0)
    assert (now["lo"], now["hi"], now["point"]) == (later["lo"], later["hi"], later["point"])


def test_marginal_ci_does_not_depend_on_how_many_were_computed_before_it():
    """Interleaving other bootstraps must not perturb it — no shared RNG stream."""
    from abus_jcr.eval.froc import bootstrap_cpm_ci
    gt = _gt()
    target = _pred(0.9, 0.2)
    first = bootstrap_cpm_ci(gt, target, n_boot=8, seed=0)
    for p in (_pred(0.8, 0.3), _pred(0.7, 0.4)):        # the discarded per-epoch CIs
        bootstrap_cpm_ci(gt, p, n_boot=8, seed=0)
    after = bootstrap_cpm_ci(gt, target, n_boot=8, seed=0)
    assert (first["lo"], first["hi"]) == (after["lo"], after["hi"])


def _record_from(pred: pd.DataFrame) -> pd.DataFrame:
    """Widen a PRED frame to a full candidate record — ``evaluate_variant`` validates the
    whole schema before writing the official CSV."""
    from abus_jcr.candidates.record import CANDIDATE_COLUMNS
    n = len(pred)
    rec = pred.rename(columns={"probability": "score_max"}).copy()
    defaults = {"candidate_id": [f"s0:{i}" for i in range(n)], "detector_of_origin": "s0",
                "split": "val", "fold": -1, "score_mean": 0.1, "score_std": 0.05,
                "score_min": 0.05, "slice_count": 20.0, "z_span": 25.0, "fill_ratio": 0.8,
                "centroid_jitter": 0.05, "area_cv": 0.4, "rank": 1, "rank_norm": 0.5,
                "label": "neg", "iou_gt": 0.4, "cen_d0": 50.0, "cen_d1": 50.0, "cen_d2": 50.0,
                "ext_d0": 20.0, "ext_d1": 20.0, "ext_d2": 20.0, "preprocess_hash": "abc"}
    for c, v in defaults.items():
        rec[c] = v
    return rec[CANDIDATE_COLUMNS]


def test_evaluate_variant_with_zero_draws_keeps_the_point_estimate():
    """What the training loop now runs on every epoch. ``evaluate_variant`` is the layer that
    documents ``n_boot <= 0`` as "no CI", and the point estimate — the ONLY selection signal —
    must survive it; only the bounds go away."""
    from abus_jcr.rescore.evaluate import evaluate_variant
    gt, pred = _gt(), _pred(0.9, 0.2)
    rec = _record_from(pred)
    res = evaluate_variant(rec, rec["score_max"].to_numpy(float), gt, "t", n_boot=0)
    assert res["cpm"] == pytest.approx(cpm(evaluate_froc(gt, pred)))
    assert np.isnan(res["ci"]["lo"]) and np.isnan(res["ci"]["hi"])


def test_bootstrap_cpm_ci_refuses_zero_draws_by_name():
    """The primitive's whole job is to produce an interval, so zero draws is a caller bug.
    It used to fall through to ``np.quantile`` on an empty array and die with
    ``IndexError: index -1 is out of bounds`` from inside numpy — forty frames from the
    mistake. ``evaluate_variant`` remains the layer that legitimately means "no CI"."""
    from abus_jcr.eval.froc import bootstrap_cpm_ci
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_cpm_ci(_gt(), _pred(0.9, 0.2), n_boot=0, seed=0)
