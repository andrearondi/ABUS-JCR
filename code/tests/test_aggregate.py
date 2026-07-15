"""Score-stats, within-volume rank, and IoU-band labeling (Phase 3, torch-free)."""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.link.aggregate import score_stats, within_volume_rank, label_candidate


def test_score_stats_on_hand_tube():
    # scores 0.2, 0.8, 0.5 on slices 3, 4, 6 (z=5 missing -> z_span 4, count 3)
    tube = [(3, (0, 0, 1, 1), 0.2), (4, (0, 0, 1, 1), 0.8), (6, (0, 0, 1, 1), 0.5)]
    s = score_stats(tube)
    assert s["score_max"] == pytest.approx(0.8)
    assert s["score_min"] == pytest.approx(0.2)
    assert s["score_mean"] == pytest.approx(0.5)
    assert s["score_std"] == pytest.approx(np.std([0.2, 0.8, 0.5]))  # ddof=0
    assert s["slice_count"] == 3
    assert s["z_span"] == 4          # 6 - 3 + 1
    assert s["fill_ratio"] == pytest.approx(3 / 4)


def test_score_stats_single_slice_fill_ratio_one():
    s = score_stats([(9, (0, 0, 1, 1), 0.3)])
    assert s["slice_count"] == 1 and s["z_span"] == 1
    assert s["fill_ratio"] == pytest.approx(1.0)
    assert s["score_std"] == pytest.approx(0.0)


def test_within_volume_rank_orders_by_score_max_desc():
    df = pd.DataFrame([
        {"public_id": 100, "score_max": 0.3},
        {"public_id": 100, "score_max": 0.9},
        {"public_id": 100, "score_max": 0.6},
        {"public_id": 101, "score_max": 0.5},
    ])
    out = within_volume_rank(df)
    v100 = out[out["public_id"] == 100].sort_values("rank")
    assert list(v100["score_max"]) == [0.9, 0.6, 0.3]
    assert list(v100["rank"]) == [1, 2, 3]
    assert v100["rank_norm"].tolist() == pytest.approx([1 / 3, 2 / 3, 3 / 3])
    v101 = out[out["public_id"] == 101]
    assert v101["rank"].tolist() == [1] and v101["rank_norm"].tolist() == pytest.approx([1.0])


def _box(cx, L=1.0):
    """Official box centred at (cx,0,0), full extent L on every axis."""
    return (float(cx), 0.0, 0.0, float(L), float(L), float(L))


def test_label_candidate_clear_cases():
    gt = _box(0.0)
    assert label_candidate(_box(0.0), gt) == ("pos", pytest.approx(1.0))  # identical
    lab, iou = label_candidate(_box(100.0), gt)                            # disjoint
    assert lab == "neg" and iou == pytest.approx(0.0)


def test_label_candidate_band_edges_exclusive():
    # Cubes side L=1 offset by d along x have IoU = (1-d)/(1+d). Straddle the two
    # band edges (pos: iou > 0.30; neg: iou < 0.10; both edges land in 'ignore').
    gt = _box(0.0)

    def lab(d):
        return label_candidate(_box(d), gt)[0]

    # pos edge (iou 0.30 at d≈0.5385):
    assert lab(0.53) == "pos"     # iou 0.3072 > 0.30
    assert lab(0.55) == "ignore"  # iou 0.2903 in [0.10, 0.30]
    # neg edge (iou 0.10 at d≈0.8182):
    assert lab(0.80) == "ignore"  # iou 0.1111 in [0.10, 0.30]
    assert lab(0.83) == "neg"     # iou 0.0929 < 0.10

    # sanity: the engineered IoUs are where we claim
    assert label_candidate(_box(0.53), gt)[1] == pytest.approx((1 - 0.53) / (1 + 0.53), abs=1e-6)
