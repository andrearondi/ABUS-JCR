"""[P3U2.PD] Frozen-pool deep-diagnostic core (torch-free)."""

import numpy as np
import pandas as pd
import pytest

from abus_jcr.probe.pool_diag import (feature_discriminability, ranking_headroom,
                                      pairwise_geometry, set_structure, relative_geometry,
                                      confidence_iou_stats, top_k_candidates, POOL_FEATURES)


def _rec(rows):
    """Build a minimal candidate-record frame from (public_id, score_max, label, geom...) dicts."""
    cols = {"public_id": [], "score_max": [], "score_mean": [], "score_std": [], "score_min": [],
            "slice_count": [], "z_span": [], "fill_ratio": [], "centroid_jitter": [], "area_cv": [],
            "rank_norm": [], "coordX": [], "coordY": [], "coordZ": [],
            "x_length": [], "y_length": [], "z_length": [],
            "ext_d0": [], "ext_d1": [], "ext_d2": [], "label": [], "detector_of_origin": [], "iou_gt": []}
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k, 0.0))
    return pd.DataFrame(cols)


def _cand(pid, score, label, cx=0.0, cy=0.0, cz=0.0, L=10.0, det="s0", **kw):
    d = dict(public_id=pid, score_max=score, score_mean=score * 0.6, score_std=0.02, score_min=score * 0.3,
             slice_count=20, z_span=25, fill_ratio=0.8, centroid_jitter=0.05, area_cv=0.4, rank_norm=0.5,
             coordX=cx, coordY=cy, coordZ=cz, x_length=L, y_length=L, z_length=L,
             ext_d0=L, ext_d1=L, ext_d2=L, label=label, detector_of_origin=det)
    d.update(kw)
    return d


# --- Task 1: feature_discriminability ---
def test_feature_discriminability_ranks_separating_feature_first():
    rng = np.random.RandomState(0)
    rows = []
    for i in range(40):
        rows.append(_cand(i, score=0.5 + 0.3, label="pos", area_cv=float(rng.rand())))   # TP score high
    for i in range(40, 200):
        rows.append(_cand(i, score=0.05, label="neg", area_cv=float(rng.rand())))         # FP score low
    out = feature_discriminability(_rec(rows))
    assert list(out.columns.intersection(["feature", "cliffs_delta", "balacc", "best_thresh"])).__len__() == 4
    # score_max perfectly separates -> top row, |delta|≈1, balacc≈1
    assert out.iloc[0]["feature"] == "score_max"
    assert abs(out.iloc[0]["cliffs_delta"]) == pytest.approx(1.0, abs=1e-9)
    assert out.iloc[0]["balacc"] == pytest.approx(1.0, abs=1e-9)
    # the random area_cv should sit near the bottom with tiny |delta|
    area = out[out["feature"] == "area_cv"].iloc[0]
    assert abs(area["cliffs_delta"]) < 0.3
    assert set(POOL_FEATURES).issubset(set(out["feature"]))


# --- Task 2: ranking_headroom ---
def test_ranking_headroom_finds_buried_tp():
    # vol A: best TP is the top-scored candidate (rank 1); vol B: best TP is 5th
    rows = [_cand("A", 0.9, "pos"), _cand("A", 0.5, "neg"), _cand("A", 0.4, "neg")]
    rows += [_cand("B", 0.9, "neg"), _cand("B", 0.8, "neg"), _cand("B", 0.7, "neg"),
             _cand("B", 0.6, "neg"), _cand("B", 0.55, "pos")]
    hr = ranking_headroom(_rec(rows))
    per = {d["public_id"]: d for d in hr["per_vol"]}
    assert per["A"]["best_tp_rank"] == 1
    assert per["B"]["best_tp_rank"] == 5
    assert hr["frac_best_tp_not_rank1"] == pytest.approx(0.5)
    # recall@fp is non-decreasing as the FP budget grows
    rec = [hr["recall_at_fp"][b] for b in sorted(hr["recall_at_fp"])]
    assert all(rec[i] <= rec[i + 1] + 1e-9 for i in range(len(rec) - 1))


# --- Task 3: pairwise_geometry ---
def test_relative_geometry_matches_hand_computation():
    # m at origin size 10; n offset (20,0,0) size 20
    g = relative_geometry(cx_m=0, cy_m=0, cz_m=0, w_m=10, h_m=10, d_m=10,
                          cx_n=20, cy_n=0, cz_n=0, w_n=20, h_n=20, d_n=20)
    eps = 1e-6
    assert g[0] == pytest.approx(np.log(20 / 10 + eps))
    assert g[1] == pytest.approx(np.log(0 / 10 + eps))
    assert g[3] == pytest.approx(np.log(20 / 10))
    assert len(g) == 6 and all(np.isfinite(g))


def test_pairwise_geometry_pair_types_and_shapes():
    rows = [_cand("V", 0.9, "pos", cx=0), _cand("V", 0.3, "neg", cx=30), _cand("V", 0.2, "neg", cx=60)]
    pg = pairwise_geometry(_rec(rows), max_fpfp_per_vol=100)
    # 1 TP + 2 FP -> TP-TP:0 ordered pairs, TP-FP:2*1*2=... count both directions
    assert pg["counts"]["TP-TP"] == 0
    assert pg["counts"]["TP-FP"] == 4     # (TP->FP1, TP->FP2, FP1->TP, FP2->TP)
    assert pg["counts"]["FP-FP"] == 2     # (FP1->FP2, FP2->FP1)
    for kind in ("TP-FP", "FP-FP"):
        arr = np.asarray(pg["g"][kind])
        assert arr.ndim == 2 and arr.shape[1] == 6 and np.isfinite(arr).all()
    assert "separability_per_component" in pg and len(pg["separability_per_component"]) == 6
    assert "separability_tptp_vs_tpfp" in pg and len(pg["separability_tptp_vs_tpfp"]) == 6


def test_pairwise_tptp_vs_tpfp_flags_colocated_tps():
    # 2 TPs sitting essentially on top of each other + 2 FPs far away and spread.
    rows = [_cand("V", 0.9, "pos", cx=100, cy=100, cz=100),
            _cand("V", 0.8, "pos", cx=101, cy=100, cz=100),          # TP peer: tiny offset
            _cand("V", 0.3, "neg", cx=300, cy=300, cz=300),
            _cand("V", 0.2, "neg", cx=60, cy=350, cz=40)]
    pg = pairwise_geometry(_rec(rows))
    # a TP's relation to its co-located TP peer (tiny |dx|/w) differs from its relation to a far FP -> high |δ|
    assert np.nanmax(pg["separability_tptp_vs_tpfp"]) > 0.9


# --- confidence vs IoU ---
def test_confidence_iou_stats():
    # score correlates with iou; top-2 per set are the high-score TPs
    rows = [_cand("A", 0.9, "pos", cx=0), _cand("A", 0.6, "pos", cx=1), _cand("A", 0.1, "neg", cx=50),
            _cand("A", 0.05, "neg", cx=60)]
    ci = confidence_iou_stats(_rec([dict(r, iou_gt=(0.7 if r["label"] == "pos" else 0.0)) for r in rows]), k=2)
    assert ci["pearson_score_iou"] > 0.5          # higher score -> higher iou here
    assert ci["topk_frac_tp"] == pytest.approx(1.0)   # the top-2 by score are the TPs
    assert ci["topk_mean_iou"] == pytest.approx(0.7)
    assert len(ci["scatter"]["score_max"]) == 4


def test_top_k_candidates_orders_by_score():
    rows = [_cand("A", 0.3, "neg"), _cand("A", 0.9, "pos"), _cand("A", 0.6, "neg")]
    tk = top_k_candidates(_rec(rows), k=2)
    assert list(tk["score_max"]) == [0.9, 0.6] and len(tk) == 2


# --- Task 4: set_structure ---
def test_set_structure_counts():
    rows = [_cand("A", 0.9, "pos"), _cand("A", 0.5, "neg"), _cand("A", 0.2, "ignore"),
            _cand("B", 0.8, "neg", cx=100)]
    ss = set_structure(_rec(rows))
    per = {d["public_id"]: d for d in ss["per_vol"]}
    assert per["A"]["n"] == 3 and per["A"]["pos"] == 1 and per["A"]["neg"] == 1 and per["A"]["ignore"] == 1
    assert per["B"]["n"] == 1 and per["B"]["neg"] == 1
    assert ss["aggregate"]["n_volumes"] == 2
    assert ss["aggregate"]["pos_per_vol_median"] == pytest.approx(0.5)
