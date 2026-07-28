"""[P3U2.CAL] Cross-volume calibration headroom — the three probability assignments.

The decomposition is only meaningful if the two synthetic assignments really do trace the curves they
claim to, THROUGH the official evaluator. Part B builds a pool whose knapsack answer is known by hand
and checks the vendored ``evaluate()`` reproduces it.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.probe.calibration import (GRID, TOP, per_set_cost, volume_neutral_probability,
                                        per_volume_oracle_probability, headroom_curve, assignments)


def _pool(spec):
    """Build a candidate frame from ``{public_id: [labels by DESCENDING score]}``."""
    rows = []
    for pid, labels in spec.items():
        for i, lab in enumerate(labels):
            rows.append({"public_id": pid, "detector_of_origin": "s0", "label": lab,
                         "score_max": 1.0 - 0.001 * i})     # rank == position in the list
    return pd.DataFrame(rows)


# ---- Part A: the pure pieces --------------------------------------------------

def test_per_set_cost_is_best_tp_rank_minus_one():
    df = _pool({1: ["pos", "neg", "neg"],            # hit already on top  -> 0 FP
                2: ["neg", "neg", "pos"],            # 2 artefacts above it -> 2 FP
                3: ["neg", "neg"]})                  # no hit at all       -> unbuyable
    c = per_set_cost(df).set_index("public_id")
    assert c.loc[1, "best_tp_rank"] == 1 and c.loc[1, "fp_cost"] == 0
    assert c.loc[2, "best_tp_rank"] == 3 and c.loc[2, "fp_cost"] == 2
    assert pd.isna(c.loc[3, "best_tp_rank"]) and pd.isna(c.loc[3, "fp_cost"])


def test_volume_neutral_is_the_same_map_in_every_set():
    df = _pool({1: ["pos", "neg", "neg"], 2: ["neg", "pos"]})
    p = volume_neutral_probability(df)
    # rank-1 of BOTH sets gets the same probability; rank-2 the next grid cell down
    assert p.iloc[0] == pytest.approx(TOP)          # set 1 rank 1
    assert p.iloc[3] == pytest.approx(TOP)          # set 2 rank 1
    assert p.iloc[1] == pytest.approx(TOP - GRID)
    assert p.iloc[4] == pytest.approx(TOP - GRID)
    # distinct ranks land in distinct official grid cells (nothing lost to quantization)
    lv = np.round((TOP - p.to_numpy()) / GRID)
    assert np.allclose((TOP - p.to_numpy()) / GRID, lv)


def test_per_volume_oracle_buys_cheapest_sets_first():
    df = _pool({1: ["neg", "neg", "pos"],            # cost 2
                2: ["pos", "neg"],                   # cost 0  -> bought FIRST
                3: ["neg", "pos"]})                  # cost 1  -> bought SECOND
    p = per_volume_oracle_probability(df)
    df = df.assign(p=p)
    lvl = lambda pid, i: round((TOP - df[df.public_id == pid].p.iloc[i]) / GRID)
    assert lvl(2, 0) == 0                            # cheapest set, its rank-1 (the hit)
    assert lvl(3, 0) == 1 and lvl(3, 1) == 1         # next cheapest: its FP + its hit, same level
    assert lvl(1, 0) == 2 and lvl(1, 2) == 2         # dearest: 2 FPs + the hit
    assert lvl(2, 1) > 2 and lvl(1, 1) == 2          # unbought tail strictly below every purchase


def test_headroom_curve_matches_the_hand_knapsack():
    df = _pool({1: ["pos"], 2: ["pos"], 3: ["neg", "pos"], 4: ["neg", "neg", "pos"], 5: ["neg"]})
    hc = headroom_curve(df)
    assert hc["n_sets"] == 5 and hc["n_sets_with_hit"] == 4 and hc["n_sets_free"] == 2
    # costs sorted = [0, 0, 1, 2] -> cumulative FP 0,0,1,3 over 5 sets
    got = [(c["fp_per_vol"], c["recall"]) for c in hc["curve"]]
    assert got == [(0.0, 0.0), (0.0, 0.2), (0.0, 0.4), (0.2, 0.6), (0.6, 0.8)]


def test_assignments_keeps_score_max_untouched():
    df = _pool({1: ["pos", "neg"]})
    a = assignments(df)
    assert set(a) == {"score_max", "volume_neutral", "per_vol_oracle"}
    assert np.allclose(a["score_max"].to_numpy(), df["score_max"].to_numpy())


# ---- Part B: end-to-end through the OFFICIAL evaluator ------------------------

_HIT = {"coordX": 100.0, "coordY": 100.0, "coordZ": 100.0,      # IoU 1.0 with the GT -> a hit
        "x_length": 20.0, "y_length": 20.0, "z_length": 20.0, "label": "pos"}
_MISS = {"coordX": 400.0, "coordY": 400.0, "coordZ": 400.0,     # IoU 0 -> an FP
         "x_length": 20.0, "y_length": 20.0, "z_length": 20.0, "label": "neg"}


def _official_pool(per_vol):
    """``per_vol = {public_id: [(is_hit, score), ...]}`` -> (GT frame, candidate frame). 1 GT/volume."""
    gt_rows, cand_rows = [], []
    for pid, cands in per_vol.items():
        gt_rows.append({"public_id": pid, **{k: _HIT[k] for k in
                                             ("coordX", "coordY", "coordZ", "x_length", "y_length", "z_length")}})
        for is_hit, sc in cands:
            cand_rows.append(dict(_HIT if is_hit else _MISS, public_id=pid, score_max=sc))
    df = pd.DataFrame(cand_rows)
    df["detector_of_origin"] = "s0"
    return pd.DataFrame(gt_rows)[C.GT_COLUMNS], df


def _cpm_of(gt, df, prob):
    from abus_jcr.eval.froc import evaluate_froc, cpm
    pred = pd.DataFrame({c: df[c] for c in C.GT_COLUMNS})
    pred["probability"] = np.clip(np.asarray(prob, float), 0, 1 - 1e-9)
    return cpm(evaluate_froc(gt, pred[C.PRED_COLUMNS]))


def _eight_volume_pool():
    """4 sets with the hit at rank 1 (cost 0) + 4 with one artefact above it (cost 1). The raw scores
    are adversarial across volumes: the expensive sets' artefacts outscore the cheap sets' hits."""
    spec = {}
    for v in range(8):
        spec[v] = [(True, 0.30), (False, 0.20)] if v < 4 else [(False, 0.90), (True, 0.80)]
    return _official_pool(spec)


def test_official_cpm_of_the_oracle_equals_the_hand_knapsack():
    """The whole decomposition rests on this: the synthetic assignment must make the OFFICIAL
    evaluator walk the knapsack curve exactly, not approximately."""
    gt, df = _eight_volume_pool()
    # costs [0,0,0,0,1,1,1,1] over 8 sets -> curve (0,.5),(.125,.625),(.25,.75),(.375,.875),(.5,1.0)
    # key recalls .625/.75/1/1/1/1/1 -> CPM = 6.375/7
    got = _cpm_of(gt, df, assignments(df)["per_vol_oracle"].to_numpy())
    assert got == pytest.approx(6.375 / 7, abs=1e-6)


def test_oracle_dominates_every_within_set_order_preserving_assignment():
    """``score_max`` and ``volume_neutral`` both preserve within-set order, so both lie inside the
    family the oracle maximises over. It must beat or equal them — on ANY pool."""
    for gt, df in (_eight_volume_pool(), _miscalibrated_pool()):
        a = assignments(df)
        oracle = _cpm_of(gt, df, a["per_vol_oracle"].to_numpy())
        assert oracle >= _cpm_of(gt, df, a["score_max"].to_numpy()) - 1e-9
        assert oracle >= _cpm_of(gt, df, a["volume_neutral"].to_numpy()) - 1e-9


def test_both_synthetic_maps_preserve_within_set_ordering():
    """They may only move score LEVELS between volumes — never the order inside one."""
    _gt, df = _miscalibrated_pool()
    a = assignments(df)
    for name in ("volume_neutral", "per_vol_oracle"):
        for _pid, g in df.assign(p=a[name]).groupby("public_id", sort=False):
            assert list((-g["score_max"]).rank(method="first")) == list((-g["p"]).rank(method="first"))


def _miscalibrated_pool():
    """9 sets whose hit is already rank-1 but scores LOW, + 1 noisy set whose 9 artefacts all outscore
    them. The pathology PHASE_4 §2's "iff within a volume" wording misses: within-set ranking is
    near-perfect in 9/10 sets, yet a global threshold sees nothing but the noisy set's artefacts."""
    spec = {v: [(True, 0.30), (False, 0.10)] for v in range(9)}
    spec[9] = [(False, 0.90 - 0.01 * i) for i in range(9)] + [(True, 0.10)]
    return _official_pool(spec)


def test_cross_volume_miscalibration_is_what_the_decomposition_exposes():
    """Here within-set ordering is already excellent, so B0' must lose badly while volume_neutral —
    which changes ONLY the cross-volume levels — recovers most of it."""
    gt, df = _miscalibrated_pool()
    a = assignments(df)
    base = _cpm_of(gt, df, a["score_max"].to_numpy())
    neutral = _cpm_of(gt, df, a["volume_neutral"].to_numpy())
    oracle = _cpm_of(gt, df, a["per_vol_oracle"].to_numpy())
    assert base < 0.60                       # 9 artefacts sit above every hit in the global order
    assert neutral > base + 0.20             # same within-set order, levels equalised -> big recovery
    assert oracle >= neutral - 1e-9


def test_oracle_cannot_exceed_the_recall_ceiling():
    """Sets with no hitting candidate are unbuyable at any FP budget (Inv. 8)."""
    from abus_jcr.eval.froc import evaluate_froc, recall_ceiling
    gt, df = _eight_volume_pool()
    df = df[~((df.public_id == 7) & (df.label == "pos"))].reset_index(drop=True)   # drop one hit
    a = assignments(df)
    pred = pd.DataFrame({c: df[c] for c in C.GT_COLUMNS})
    pred["probability"] = np.clip(a["per_vol_oracle"].to_numpy(), 0, 1 - 1e-9)
    ceiling = recall_ceiling(evaluate_froc(gt, pred[C.PRED_COLUMNS]))
    assert ceiling == pytest.approx(7 / 8, abs=1e-6)
    assert _cpm_of(gt, df, a["per_vol_oracle"].to_numpy()) <= ceiling + 1e-9
