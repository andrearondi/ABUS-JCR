"""[P3U2] Candidate-pool diagnostic core (torch-free)."""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.geometry import storage_box_to_official, mask_to_box_storage
from abus_jcr.probe.candidate_diag import (build_candidate_frame, score_floor_sweep,
                                           separability, tp_fp_split_stats, cluster_counts,
                                           CAND_DIAG_COLUMNS)

IDENTITY_META = {"zoom_factors": [1.0, 1.0, 1.0]}   # native == iso -> iso_iou ~= official_iou


def _cube_tube(x0, y0, z0, side, n, score=0.5):
    """A tube of ``n`` stacked identical boxes forming a cube of edge ~side."""
    return [(z0 + k, (float(x0), float(y0), float(x0 + side), float(y0 + side)), float(score))
            for k in range(n)]


def test_build_frame_columns_and_tp_flag():
    # GT iso box == a cube [0,10)^3; a candidate matching it should be a TP.
    gt_iso_storage = (0, 0, 0, 9, 9, 9)                 # inclusive min/max
    gt_iso_off = storage_box_to_official(gt_iso_storage)
    gt_official = gt_iso_off                            # identity meta -> official == iso
    tube = _cube_tube(0, 0, 0, 10, 10, score=0.7)
    frame = build_candidate_frame(1, [tube], gt_official, gt_iso_off, IDENTITY_META)
    assert list(frame.columns) == CAND_DIAG_COLUMNS
    assert frame.loc[0, "is_tp"]                        # perfect overlap
    assert frame.loc[0, "official_iou"] > 0.9
    assert abs(frame.loc[0, "recon_loss"]) < 0.05      # identity zoom -> no reconstruction loss


def test_score_floor_sweep_recall_and_pool():
    # 3 volumes, each: 1 high-score TP (0.8) + many low-score FP (0.05). A 0.1 floor should
    # cut every FP but keep the TP -> recall 1.0, pool 1/vol.
    rows = []
    for vid in (1, 2, 3):
        rows.append({"public_id": vid, "score_max": 0.8, "is_tp": True})
        for _ in range(20):
            rows.append({"public_id": vid, "score_max": 0.05, "is_tp": False})
    frame = pd.DataFrame(rows)
    sweep = score_floor_sweep(frame, floors=[0.0, 0.1, 0.9], n_vol=3).set_index("floor")
    assert sweep.loc[0.0, "recall"] == pytest.approx(1.0)
    assert sweep.loc[0.0, "pool_mean"] == pytest.approx(21.0)
    assert sweep.loc[0.1, "recall"] == pytest.approx(1.0)      # TP survives the floor
    assert sweep.loc[0.1, "pool_mean"] == pytest.approx(1.0)   # FP tail cut
    assert sweep.loc[0.9, "recall"] == pytest.approx(0.0)      # floor above the TP score -> recall lost


def test_separability_clean_split():
    # TP scores all above FP scores -> perfectly separable
    frame = pd.DataFrame(
        [{"score_max": s, "is_tp": True} for s in (0.6, 0.7, 0.8)] +
        [{"score_max": s, "is_tp": False} for s in (0.05, 0.1, 0.2)])
    sep = separability(frame, "score_max")
    assert sep["best_balacc"] == pytest.approx(1.0)
    assert sep["frac_fp_below_tp_median"] == pytest.approx(1.0)


def test_size_split_and_clusters():
    frame = pd.DataFrame([{"box_diag": 50.0, "is_tp": True}, {"box_diag": 10.0, "is_tp": False},
                          {"box_diag": 12.0, "is_tp": False}])
    st = tp_fp_split_stats(frame, "box_diag")
    assert st["TP"]["mean"] == pytest.approx(50.0) and st["FP"]["mean"] == pytest.approx(11.0)
    # two tight points + one far -> 2 clusters at radius 5
    ncl, npts, redund = cluster_counts([(0, 0, 0), (1, 0, 0), (100, 0, 0)], radius=5.0)
    assert ncl == 2 and npts == 3
