"""Phase 5 — per-candidate-count stratification (PHASE_5_SPEC §5.3; written test-first 2026-09-05).

The leakage rule this file pins: **bin edges come from the VAL pool** (design-sanctioned data) and
are applied FIXED to the test volumes — no test-derived edge exists anywhere. The per-bin numbers
are ordinary ``evaluate_variant`` calls on the volume subset (same vendored oracle, same bootstrap
seed), pinned by equality against a direct subset evaluation so the stratifier cannot drift from
the reported machinery.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase5_stratify as S  # noqa: E402

from abus_jcr.rescore.evaluate import evaluate_variant  # noqa: E402


def _pred(pids, probs):
    """One candidate per volume, spatially separated boxes (official pred schema)."""
    n = len(pids)
    return pd.DataFrame({
        "public_id": pids,
        "coordX": np.arange(n) * 50.0, "coordY": [0.0] * n, "coordZ": [0.0] * n,
        "x_length": [10.0] * n, "y_length": [10.0] * n, "z_length": [10.0] * n,
        "probability": probs})


def _gt_from(pred):
    """GT at the candidate boxes themselves — every candidate is a hit."""
    return pred[["public_id", "coordX", "coordY", "coordZ",
                 "x_length", "y_length", "z_length"]].copy()


def test_tercile_edges_and_bin_assignment_low_mid_high():
    e1, e2 = S.tercile_edges(pd.Series([10, 20, 30, 40, 50, 60]))
    assert e1 == pytest.approx(26.6667, abs=1e-3)
    assert e2 == pytest.approx(43.3333, abs=1e-3)
    bins = S.assign_bins({130: 10, 131: 30, 132: 44, 133: 200}, e1, e2)
    assert bins == {130: "low", 131: "mid", 132: "high", 133: "high"}


def test_boundary_sizes_go_to_the_lower_bin():
    bins = S.assign_bins({1: 20, 2: 40}, 20.0, 40.0)     # size <= edge -> lower bin
    assert bins == {1: "low", 2: "mid"}


def test_per_bin_numbers_equal_a_direct_subset_evaluation():
    pred = _pred([130, 131, 132, 133], [0.9, 0.8, 0.7, 0.6])
    gt = _gt_from(pred)
    bins = {130: "low", 131: "low", 132: "high", 133: "high"}
    out = S.stratified_eval(pred, gt, bins, n_boot=8, tag="B0")

    sub_p = pred[pred["public_id"].isin([130, 131])].reset_index(drop=True)
    sub_g = gt[gt["public_id"].isin([130, 131])].reset_index(drop=True)
    direct = evaluate_variant(S.as_record_frame(sub_p),
                              sub_p["probability"].to_numpy(float), sub_g,
                              "B0_low", n_boot=8)
    assert out["low"]["cpm"] == pytest.approx(direct["cpm"])
    assert out["low"]["ci"]["lo"] == pytest.approx(direct["ci"]["lo"])
    assert out["low"]["ci"]["hi"] == pytest.approx(direct["ci"]["hi"])
    assert out["low"]["n_volumes"] == 2
    assert out["high"]["n_volumes"] == 2


def test_an_empty_bin_is_absent_not_faked():
    pred = _pred([130, 131], [0.9, 0.8])
    out = S.stratified_eval(pred, _gt_from(pred), {130: "low", 131: "low"},
                            n_boot=4, tag="B0")
    assert set(out) == {"low"}          # no "mid"/"high" rows invented


def test_per_rate_deltas_mean_and_std_over_seeds():
    kr_a = {"0.125": 0.5, "0.25": 0.6}
    kr_b = {"0.125": 0.4, "0.25": 0.65}
    grid = {"per_seed": {s: {"A": {"key_recall": dict(kr_a)},
                             "B": {"key_recall": dict(kr_b)}} for s in ("0", "1")}}
    out = S.per_rate_deltas(grid, [("A", "B")], seeds=("0", "1"))
    assert out["A-B"]["0.125"]["mean"] == pytest.approx(0.1)
    assert out["A-B"]["0.125"]["std"] == pytest.approx(0.0)
    assert out["A-B"]["0.25"]["mean"] == pytest.approx(-0.05)
