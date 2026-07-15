"""OOF checkpoint routing + torch-free candidate assembly (Phase 3, Inv. 10/14).

``plan_generation`` is the pure routing core: a Train volume in fold ``f`` must be
scored by ``retinanet_fold{f}.pt`` (the detector that never saw it), and Val/Test by
the 3 full-train seeds with no ensembling. ``generate_split`` is exercised end-to-end
with injected fakes (no torch): a stub loader + a stub detector returning synthetic
per-slice boxes, verifying candidate_id/routing/label/rank wiring.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.detect import schema as S
from abus_jcr.candidates.generate import plan_generation, generate_split
from abus_jcr.candidates.record import CANDIDATE_COLUMNS


def _manifest():
    rows = []
    # 6 train volumes across folds 0,1,2 (2 each); 2 val; 2 test
    for vid, fold in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]:
        rows.append({"volume_id": vid, "split": "train", "fold": fold, "label": "B"})
    for vid in (100, 101):
        rows.append({"volume_id": vid, "split": "val", "fold": -1, "label": "M"})
    for vid in (200, 201):
        rows.append({"volume_id": vid, "split": "test", "fold": -1, "label": "B"})
    return pd.DataFrame(rows)


def test_plan_train_is_out_of_fold():
    jobs = plan_generation(_manifest(), "train", Path("/ckpts"))
    by_det = {j["detector_of_origin"]: j for j in jobs}
    assert set(by_det) == {"fold0", "fold1", "fold2"}
    # fold-f volumes are scored by retinanet_fold{f}.pt (OOF for those volumes)
    assert by_det["fold0"]["volume_ids"] == [0, 1]
    assert by_det["fold0"]["checkpoint"] == Path("/ckpts/retinanet_fold0.pt")
    assert by_det["fold1"]["volume_ids"] == [2, 3]
    assert by_det["fold2"]["volume_ids"] == [4, 5]


def test_plan_val_uses_three_full_seeds_no_ensemble():
    jobs = plan_generation(_manifest(), "val", Path("/ckpts"))
    dets = [j["detector_of_origin"] for j in jobs]
    assert dets == [f"full_seed{s}" for s in C.DET_FULL_SEEDS]
    for j in jobs:  # every seed scores ALL val volumes independently (stacked, not merged)
        assert j["volume_ids"] == [100, 101]
        assert j["checkpoint"].name.startswith("retinanet_full_seed")


def test_plan_test_uses_three_full_seeds():
    jobs = plan_generation(_manifest(), "test", Path("/ckpts"))
    assert [j["detector_of_origin"] for j in jobs] == [f"full_seed{s}" for s in C.DET_FULL_SEEDS]
    assert all(j["volume_ids"] == [200, 201] for j in jobs)


# ---- end-to-end assembly with injected fakes (no torch) --------------------

def _fake_detect_factory():
    """A stub detector: two IoU-aligned boxes on slices 4,5 -> exactly one tube."""
    def detect_fn(model, cache_root, volume_id, *, score_thresh, nms_thresh, detections_per_img):
        rows = [
            {"volume_id": int(volume_id), "slice_z": 4, "x1": 10.0, "y1": 20.0,
             "x2": 30.0, "y2": 40.0, "score": 0.9},
            {"volume_id": int(volume_id), "slice_z": 5, "x1": 10.0, "y1": 20.0,
             "x2": 30.0, "y2": 40.0, "score": 0.7},
        ]
        df = pd.DataFrame(rows, columns=S.DETECTION_COLUMNS)
        df["volume_id"] = df["volume_id"].astype("int64")
        df["slice_z"] = df["slice_z"].astype("int64")
        return S.validate_detections(df)
    return detect_fn


def test_generate_split_val_assembly_wiring():
    from abus_jcr.preprocess import zoom_factors

    manifest = _manifest()
    # official GT far from the synthetic candidate -> label 'neg' (audit path exercised)
    gt = pd.DataFrame([
        {"public_id": 100, "coordX": 500.0, "coordY": 500.0, "coordZ": 500.0,
         "x_length": 5.0, "y_length": 5.0, "z_length": 5.0},
        {"public_id": 101, "coordX": 500.0, "coordY": 500.0, "coordZ": 500.0,
         "x_length": 5.0, "y_length": 5.0, "z_length": 5.0},
    ])
    meta = {"zoom_factors": list(zoom_factors())}

    pool = generate_split(
        manifest, cache_root="UNUSED", checkpoints_dir=Path("/ckpts"), split="val",
        gt_gt_df=gt,
        load_checkpoint_fn=lambda p: (("MODEL", str(p)), {"cfg": True}),
        detect_fn=_fake_detect_factory(),
        read_meta_fn=lambda cr, vid: meta,
    )
    # 3 seeds x 2 val volumes x 1 tube = 6 candidates
    assert len(pool) == 6
    assert list(pool.columns) == CANDIDATE_COLUMNS
    assert set(pool["detector_of_origin"]) == {f"full_seed{s}" for s in C.DET_FULL_SEEDS}
    # each (detector, volume) has exactly one candidate -> rank 1, rank_norm 1.0
    assert set(pool["rank"]) == {1}
    assert pool["rank_norm"].tolist() == pytest.approx([1.0] * 6)
    # candidate_id namespaced by detector; local_idx runs across the detector's pool
    # (vol 100 -> idx 0, vol 101 -> idx 1). Unique across the 3 seed pools.
    assert pool["candidate_id"].nunique() == 6
    assert "full_seed1:101:1" in set(pool["candidate_id"])
    assert "full_seed1:100:0" in set(pool["candidate_id"])
    # distant GT -> all 'neg'
    assert set(pool["label"]) == {"neg"}
    assert (pool["iou_gt"] == 0.0).all()
    # score_max is the tube peak
    assert pool["score_max"].tolist() == pytest.approx([0.9] * 6)
