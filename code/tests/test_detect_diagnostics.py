"""Detection recall diagnostics (torch-free).

Pins the 2D IoU, the GT-box recall, the lesion-slice fire-rate, and the
size/IoU-stratified breakdown used to interpret the [2.4] dump. Synthetic data,
no torch, no cache.
"""

import numpy as np
import pandas as pd

from abus_jcr.detect import diagnostics as DG


def _det(rows):
    cols = ["volume_id", "slice_z", "x1", "y1", "x2", "y2", "score"]
    return pd.DataFrame(rows, columns=cols)


def _gt(rows):
    cols = ["volume_id", "slice_z", "r0", "c0", "r1", "c1", "component_id"]
    return pd.DataFrame(rows, columns=cols)


def test_iou_2d_basic():
    assert DG.iou_2d((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert DG.iou_2d((0, 0, 2, 2), (2, 2, 4, 4)) == 0.0
    # half overlap: intersection 1x2=2, union 2*4-2=6 -> 1/3
    assert abs(DG.iou_2d((0, 0, 2, 2), (1, 0, 3, 2)) - (2.0 / 6.0)) < 1e-9


def test_gt_recall_hits_and_misses():
    # GT box (r0,c0,r1,c1)=(0,0,3,3) -> half-open (c0,r0,c1+1,r1+1)=(0,0,4,4)
    gt = _gt([{"volume_id": 1, "slice_z": 5, "r0": 0, "c0": 0, "r1": 3, "c1": 3, "component_id": 0}])
    # a near-perfect detection on the same slice, plus a far one
    det = _det([
        {"volume_id": 1, "slice_z": 5, "x1": 0, "y1": 0, "x2": 4, "y2": 4, "score": 0.9},
        {"volume_id": 1, "slice_z": 5, "x1": 20, "y1": 20, "x2": 24, "y2": 24, "score": 0.9},
    ])
    hits, total, recall = DG.gt_recall(det, gt, [1], score_thresh=0.05, iou_thresh=0.3)
    assert (hits, total) == (1, 1) and recall == 1.0


def test_gt_recall_score_threshold_excludes_low():
    gt = _gt([{"volume_id": 1, "slice_z": 5, "r0": 0, "c0": 0, "r1": 3, "c1": 3, "component_id": 0}])
    det = _det([{"volume_id": 1, "slice_z": 5, "x1": 0, "y1": 0, "x2": 4, "y2": 4, "score": 0.01}])
    hits, total, recall = DG.gt_recall(det, gt, [1], score_thresh=0.05, iou_thresh=0.3)
    assert (hits, total, recall) == (0, 1, 0.0)


def test_lesion_slice_fire_rate():
    gt = _gt([
        {"volume_id": 1, "slice_z": 5, "r0": 0, "c0": 0, "r1": 3, "c1": 3, "component_id": 0},
        {"volume_id": 1, "slice_z": 6, "r0": 0, "c0": 0, "r1": 3, "c1": 3, "component_id": 0},
    ])
    # a detection fires on slice 5 (far from GT) but nothing on slice 6
    det = _det([{"volume_id": 1, "slice_z": 5, "x1": 90, "y1": 90, "x2": 95, "y2": 95, "score": 0.9}])
    fired, total, rate = DG.lesion_slice_fire_rate(det, gt, [1], score_thresh=0.05)
    assert (fired, total) == (1, 2) and rate == 0.5


def test_breakdown_size_and_iou_sweep():
    # one small GT (diag ~ hypot(2,2)=2.8) recalled loosely, one large (diag ~ hypot(40,40)=56.6) recalled tightly
    gt = _gt([
        {"volume_id": 1, "slice_z": 1, "r0": 0, "c0": 0, "r1": 1, "c1": 1, "component_id": 0},   # small
        {"volume_id": 1, "slice_z": 2, "r0": 0, "c0": 0, "r1": 39, "c1": 39, "component_id": 0},  # large
    ])
    det = _det([
        # small GT half-open (0,0,2,2): a shifted det (1,1,3,3) -> IoU = 1/7 ~0.14 (hits 0.1, not 0.3)
        {"volume_id": 1, "slice_z": 1, "x1": 1, "y1": 1, "x2": 3, "y2": 3, "score": 0.9},
        # large GT half-open (0,0,40,40): near-perfect
        {"volume_id": 1, "slice_z": 2, "x1": 0, "y1": 0, "x2": 40, "y2": 40, "score": 0.9},
    ])
    rep = DG.recall_breakdown(det, gt, [1], score_thresh=0.05,
                              iou_threshs=(0.1, 0.3), size_edges=(0, 16, np.inf))
    # IoU sweep: at 0.1 both recalled (small IoU~0.14>0.1); at 0.3 only the large one
    assert rep["by_iou"][0.1] == 1.0
    assert rep["by_iou"][0.3] == 0.5
    # size buckets at iou_thresh=0.3: small bucket [0,16) misses, large bucket [16,inf) hits
    small = rep["by_size"]["[0,16)"]
    large = rep["by_size"]["[16,inf)"]
    assert small["recall"] == 0.0 and large["recall"] == 1.0
