"""2D detection metrics for Phase-2-UPDATE model selection (Inv. 2 amended).

Data-independent (numpy/pandas only, torch-free). Pins val 2D-AP@IoU and the two
logged recall diagnostics on synthetic detections/GT so the selection signal is
verified without the model.
"""

import numpy as np
import pandas as pd

from abus_jcr.detect.metrics import val_ap_2d, per_slice_recall_2d, per_volume_recall_2d

DET_COLS = ["volume_id", "slice_z", "x1", "y1", "x2", "y2", "score"]
GT_COLS = ["volume_id", "slice_z", "x1", "y1", "x2", "y2"]


def _det(rows):
    return pd.DataFrame(rows, columns=DET_COLS)


def _gt(rows):
    return pd.DataFrame(rows, columns=GT_COLS)


def test_perfect_detection_ap_is_one():
    gt = _gt([[1, 5, 0, 0, 10, 10], [1, 6, 0, 0, 10, 10]])
    det = _det([[1, 5, 0, 0, 10, 10, 0.9], [1, 6, 0, 0, 10, 10, 0.8]])
    assert val_ap_2d(det, gt, iou_thresh=0.3) == 1.0


def test_no_overlap_ap_is_zero():
    gt = _gt([[1, 5, 0, 0, 10, 10]])
    det = _det([[1, 5, 50, 50, 60, 60, 0.9]])
    assert val_ap_2d(det, gt, iou_thresh=0.3) == 0.0


def test_iou_threshold_gates_the_match():
    # det overlaps gt with IoU ~0.14 (a 10x10 gt, det shifted so overlap 4x10=40 / union 160)
    gt = _gt([[1, 5, 0, 0, 10, 10]])
    det = _det([[1, 5, 6, 0, 16, 10, 0.9]])  # overlap cols 6..10 => 4x10=40; union=100+100-40=160; IoU=0.25
    assert val_ap_2d(det, gt, iou_thresh=0.3) == 0.0    # 0.25 < 0.30 -> miss
    assert val_ap_2d(det, gt, iou_thresh=0.1) == 1.0    # 0.25 >= 0.10 -> hit


def test_detection_cannot_match_gt_in_another_image():
    gt = _gt([[1, 5, 0, 0, 10, 10]])
    det = _det([[2, 5, 0, 0, 10, 10, 0.9]])   # right box, wrong volume
    assert val_ap_2d(det, gt, iou_thresh=0.3) == 0.0


def test_one_gt_one_detection_only_counts_once():
    # two identical detections on one GT: one TP, one FP (not two TPs)
    gt = _gt([[1, 5, 0, 0, 10, 10]])
    det = _det([[1, 5, 0, 0, 10, 10, 0.9], [1, 5, 0, 0, 10, 10, 0.8]])
    # rec reaches 1.0 at the first det; the FP drops precision but AP interpolation
    # keeps the max-to-the-right so AP stays 1.0 for a single recallable GT.
    assert val_ap_2d(det, gt, iou_thresh=0.3) == 1.0


def test_ap_between_zero_and_one_for_mixed():
    # 2 GT; detections: TP(0.9), FP(0.8), TP(0.7). Classic PR: AP = (1*.5 + (2/3)*.5)=... check bounds
    gt = _gt([[1, 5, 0, 0, 10, 10], [1, 6, 0, 0, 10, 10]])
    det = _det([[1, 5, 0, 0, 10, 10, 0.9],
                [1, 5, 50, 50, 60, 60, 0.8],   # FP
                [1, 6, 0, 0, 10, 10, 0.7]])
    ap = val_ap_2d(det, gt, iou_thresh=0.3)
    assert 0.5 < ap < 1.0


def test_no_gt_no_det_is_one_no_gt_with_fp_is_zero():
    assert val_ap_2d(_det([]), _gt([]), iou_thresh=0.3) == 1.0
    assert val_ap_2d(_det([[1, 5, 0, 0, 10, 10, 0.9]]), _gt([]), iou_thresh=0.3) == 0.0


def test_per_slice_recall_counts_hit_gt_boxes():
    gt = _gt([[1, 5, 0, 0, 10, 10], [1, 6, 0, 0, 10, 10], [1, 7, 0, 0, 10, 10]])
    det = _det([[1, 5, 0, 0, 10, 10, 0.9],       # hits z5
                [1, 6, 0, 0, 10, 10, 0.02]])      # below score_thresh
    r = per_slice_recall_2d(det, gt, score_thresh=0.05, iou_thresh=0.3)
    assert r == 1 / 3   # only z5 recalled at score>=0.05


def test_per_volume_recall_any_hit_per_volume():
    gt = _gt([[1, 5, 0, 0, 10, 10], [2, 9, 0, 0, 10, 10]])
    det = _det([[1, 5, 0, 0, 10, 10, 0.9]])       # vol 1 hit, vol 2 missed
    assert per_volume_recall_2d(det, gt, score_thresh=0.05, iou_thresh=0.3) == 0.5
