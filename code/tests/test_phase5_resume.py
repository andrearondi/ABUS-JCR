"""[5.5] resume persistence — written 2026-09-09, after the second pod death mid-reportable.

What is pinned: a persisted unit loads back EXACTLY (floats round-trip through JSON and the
pandas CSV writer bit-for-bit), a torn or inconsistent unit is treated as absent (recompute,
never a silent half-result), and ``cached_unit`` recomputes only what is missing. Together with
the purity of ``bootstrap_cpm_ci``/``compare_variants`` (fixed seeds, pinned by
``test_paired_bootstrap``), these make a resumed run identical to an uninterrupted one by
construction.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_eval_grid import cached_unit, load_unit, save_unit  # noqa: E402


def _payload():
    return {"cpm": 0.5685714285714286, "ceiling": 0.8571428571428571, "n_rows": 3,
            "ci": {"lo": 0.46428571428571425, "hi": 0.6955555555555556, "point": 0.5686},
            "key_recall": {"0.125": 0.1, "1": 0.6},
            "fp": [0.125, 8.0], "recall": [0.1, 0.86], "train_cpm": 0.49}


def _pred():
    return pd.DataFrame({
        "public_id": [130, 131, 132],
        "coordX": [1.0000000001, 2.5, 3.3333333333333335],
        "coordY": [0.1, 0.2, 0.3], "coordZ": [0.0] * 3,
        "x_length": [10.0] * 3, "y_length": [10.0] * 3, "z_length": [10.0] * 3,
        "probability": [0.123456789012345, 0.5, 0.0499999999999999]})


def test_unit_round_trips_exactly(tmp_path):
    save_unit(tmp_path, "rung_B0_seed0", _payload(), _pred())
    payload, pred = load_unit(tmp_path, "rung_B0_seed0")
    assert payload == _payload()                       # floats exact through JSON
    pdt.assert_frame_equal(pred, _pred(), check_exact=True)


def test_comparison_unit_without_pred(tmp_path):
    row = {"comparison": "A - B", "delta": 0.0653, "lo": 0.0012, "hi": 0.0695,
           "frac_positive": 0.983, "n_boot": 1000}
    save_unit(tmp_path, "cmp_A-B_seed0", row)
    got = load_unit(tmp_path, "cmp_A-B_seed0", with_pred=False)
    assert got is not None and got[0] == row and got[1] is None


def test_absent_torn_or_inconsistent_units_recompute(tmp_path):
    assert load_unit(tmp_path, "missing") is None
    # torn json (the pod died mid-write of a non-atomic file)
    save_unit(tmp_path, "torn", _payload(), _pred())
    (tmp_path / "torn.json").write_text('{"cpm": 0.5')
    assert load_unit(tmp_path, "torn") is None
    # json/csv row-count disagreement
    save_unit(tmp_path, "short", {**_payload(), "n_rows": 99}, _pred())
    assert load_unit(tmp_path, "short") is None
    # pred file missing entirely
    save_unit(tmp_path, "nopred", _payload(), _pred())
    (tmp_path / "nopred.csv").unlink()
    assert load_unit(tmp_path, "nopred") is None


def test_no_tmp_files_left_behind(tmp_path):
    save_unit(tmp_path, "u", _payload(), _pred())
    assert not list(tmp_path.glob("*.tmp"))


def test_cached_unit_recomputes_only_whats_missing(tmp_path):
    calls = []

    def make(name):
        def compute():
            calls.append(name)
            return {**_payload(), "name": name}, _pred()
        return compute

    # "first run": computes A and B, then the pod dies before C
    cached_unit(True, tmp_path, "A", make("A"))
    cached_unit(True, tmp_path, "B", make("B"))
    assert calls == ["A", "B"]
    # "second run": A and B load, only C computes; loaded payloads are the persisted ones
    pA, _ = cached_unit(True, tmp_path, "A", make("A"))
    pB, _ = cached_unit(True, tmp_path, "B", make("B"))
    pC, _ = cached_unit(True, tmp_path, "C", make("C"))
    assert calls == ["A", "B", "C"]
    assert (pA["name"], pB["name"], pC["name"]) == ("A", "B", "C")


def test_resume_off_never_touches_disk(tmp_path):
    cached_unit(False, tmp_path, "X", lambda: (_payload(), _pred()))
    assert not list(tmp_path.iterdir()) if tmp_path.exists() else True
