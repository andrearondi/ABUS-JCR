"""[MIG-6] The seed-split merge must equal what a single 3-seed run would assemble.

Written 2026-09-04 with the tool. The reference is constructed the same way
``phase4_eval_grid`` does it — ``seed_summary`` over the per-seed dicts, mean/std over the
comparison deltas — so a drift in either path breaks the equality, not just the merge.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from abus_jcr.rescore.evaluate import seed_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase4_merge_grid import merge_grids  # noqa: E402


def _rung(cpm, ceiling=0.93, train=None):
    d = {"cpm": cpm, "ceiling": ceiling, "key_recall": {"1": cpm},
         "fp": [0.5, 1.0], "recall": [cpm - 0.1, cpm],
         "ci": {"lo": cpm - 0.1, "hi": cpm + 0.1, "point": cpm}}
    if train is not None:
        d["train_cpm"] = train
    return d


def _part(seed, cpms, gates=None):
    per_seed = {str(seed): {r: _rung(c, train=(c - 0.2 if r not in ("B0-spread", "B0-rank")
                                               else None))
                            for r, c in cpms.items()}}
    per_rung = {r: {**seed_summary([per_seed[str(seed)][r]]), "key_recall": {"1": c}}
                for r, c in cpms.items()}
    comparisons = {"B1-B0": {"per_seed": [{"delta": cpms["B1"] - cpms["B0"],
                                           "lo": -0.1, "hi": 0.2, "frac_positive": 0.8}],
                             "delta_mean": cpms["B1"] - cpms["B0"], "delta_std": 0.0}}
    return {"per_seed": per_seed, "per_rung": per_rung, "comparisons": comparisons,
            "gates": gates or {"pool_identity": True,
                               "ceiling_invariant_and_respected": True}}


CPMS = [{"B0": 0.71, "B0-rank": 0.81, "B1": 0.77, "FULL-P": 0.83},
        {"B0": 0.72, "B0-rank": 0.76, "B1": 0.75, "FULL-P": 0.81},
        {"B0": 0.69, "B0-rank": 0.79, "B1": 0.72, "FULL-P": 0.81}]


def test_merged_summaries_equal_the_single_run_construction():
    merged = merge_grids([_part(s, c) for s, c in enumerate(CPMS)])
    for rung in ("B0", "B0-rank", "B1", "FULL-P"):
        vals = [c[rung] for c in CPMS]
        assert merged["per_rung"][rung]["cpm_mean"] == pytest.approx(np.mean(vals))
        assert merged["per_rung"][rung]["cpm_std"] == pytest.approx(np.std(vals))
        assert merged["per_rung"][rung]["n_seeds"] == 3
    # train mean only over rungs that carry it, exactly like the single-run path
    assert merged["per_rung"]["B1"]["train_cpm_mean"] == pytest.approx(
        np.mean([c["B1"] - 0.2 for c in CPMS]))
    assert merged["per_rung"]["B0-rank"]["train_cpm_mean"] is None


def test_comparison_rows_concatenate_and_aggregates_recompute():
    merged = merge_grids([_part(s, c) for s, c in enumerate(CPMS)])
    comp = merged["comparisons"]["B1-B0"]
    deltas = [c["B1"] - c["B0"] for c in CPMS]
    assert len(comp["per_seed"]) == 3
    assert comp["delta_mean"] == pytest.approx(np.mean(deltas))
    assert comp["delta_std"] == pytest.approx(np.std(deltas))


def test_gates_recompute_and_record_parts():
    merged = merge_grids([_part(s, c) for s, c in enumerate(CPMS)])
    assert merged["gates"]["pool_identity"] is True
    assert merged["gates"]["b1_beats_b0"] is True          # 0.7467 > 0.7067
    assert merged["gates"]["b0_cpm_mean_measured"] == pytest.approx(np.mean([0.71, 0.72, 0.69]))
    assert len(merged["gates"]["parts"]) == 3


def test_one_failing_part_gate_fails_the_merged_gate():
    parts = [_part(s, c) for s, c in enumerate(CPMS)]
    parts[1]["gates"]["pool_identity"] = False
    assert merge_grids(parts)["gates"]["pool_identity"] is False


def test_duplicated_seed_refuses():
    with pytest.raises(SystemExit, match="more than one part"):
        merge_grids([_part(0, CPMS[0]), _part(0, CPMS[1])])


def test_mismatched_rung_sets_refuse():
    short = _part(1, {k: v for k, v in CPMS[1].items() if k != "FULL-P"})
    with pytest.raises(SystemExit, match="disagree on the rung set"):
        merge_grids([_part(0, CPMS[0]), short])
