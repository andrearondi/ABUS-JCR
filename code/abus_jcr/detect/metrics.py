"""2D detection metrics for Phase-2-UPDATE model selection (Inv. 2 amended).

``val_ap_2d`` is the checkpoint-selection signal (a detection metric, replacing
val-loss); ``per_slice_recall_2d`` / ``per_volume_recall_2d`` are logged
diagnostics. All torch-free (operate on the schema DataFrames), so they are
unit-tested on the laptop and reused by the training loop and the dump script.

Boxes are half-open ``(x1, y1, x2, y2)`` in the iso-slice frame; an "image" is one
``(volume_id, slice_z)`` slice. Detections carry a ``score``; GT does not.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C

_BOX = ["x1", "y1", "x2", "y2"]


def _iou(a, b) -> float:
    """IoU of two half-open boxes ``(x1,y1,x2,y2)``."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _gt_by_image(gt_df: pd.DataFrame) -> Dict[Tuple[int, int], List[List[float]]]:
    out: Dict[Tuple[int, int], List[List[float]]] = {}
    for vid, z, x1, y1, x2, y2 in gt_df[["volume_id", "slice_z", *_BOX]].to_numpy():
        out.setdefault((int(vid), int(z)), []).append([x1, y1, x2, y2])
    return out


def _match_sorted(det_df: pd.DataFrame, gt_df: pd.DataFrame, iou_thresh: float):
    """Greedy score-ranked TP/FP labelling per image. Returns ``(tp, fp)`` arrays
    aligned to detections sorted by descending score (one GT per detection)."""
    gt_img = {k: {"boxes": np.asarray(v, dtype=float), "used": np.zeros(len(v), dtype=bool)}
              for k, v in _gt_by_image(gt_df).items()}
    det = det_df.sort_values("score", ascending=False, kind="mergesort")
    tp = np.zeros(len(det), dtype=float)
    fp = np.zeros(len(det), dtype=float)
    for i, (vid, z, x1, y1, x2, y2, _score) in enumerate(
            det[["volume_id", "slice_z", *_BOX, "score"]].to_numpy()):
        entry = gt_img.get((int(vid), int(z)))
        if entry is None:
            fp[i] = 1.0
            continue
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(entry["boxes"]):
            if entry["used"][j]:
                continue
            iou = _iou((x1, y1, x2, y2), gbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            entry["used"][best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    return tp, fp


def val_ap_2d(det_df: pd.DataFrame, gt_df: pd.DataFrame, iou_thresh: float = 0.30) -> float:
    """Single-class 2D Average Precision at ``iou_thresh`` over all slices.

    Detections are ranked globally by descending score and greedily matched to
    unmatched GT within the SAME ``(volume_id, slice_z)``; a match with IoU >=
    ``iou_thresh`` is a TP (one GT per detection), else a FP. AP is the all-point
    (VOC2010+) area under the precision-recall envelope. Empty GT: 1.0 iff no
    detections, else 0.0.
    """
    total_gt = int(len(gt_df))
    if total_gt == 0:
        return 1.0 if len(det_df) == 0 else 0.0
    if len(det_df) == 0:
        return 0.0

    tp, fp = _match_sorted(det_df, gt_df, iou_thresh)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / total_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(float).eps)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def recall_at_fp_budgets_2d(det_df: pd.DataFrame, gt_df: pd.DataFrame, n_slices: int,
                            budgets=C.DET_SELECTION_FP_BUDGETS, iou_thresh: float = 0.30):
    """2D CPM-proxy: per-slice recall at each fixed FP/slice budget + their mean.

    For budget ``B``, the operating threshold is where cumulative FP == ``B * n_slices``
    (reading recall in the FP regime we deploy at, unlike AP which averages over all
    regimes). Returns ``(mean_recall, {budget: recall})``. The mean is the selection
    signal (Inv. 2 amended) — a pre-linking foreshadow of the Inv.-3 CPM. Empty GT:
    1.0 iff no detections else 0.0, at every budget.
    """
    budgets = tuple(float(b) for b in budgets)
    total_gt = int(len(gt_df))
    if total_gt == 0:
        per = {b: (1.0 if len(det_df) == 0 else 0.0) for b in budgets}
        return (float(np.mean(list(per.values()))) if per else float("nan")), per
    if len(det_df) == 0 or n_slices <= 0:
        per = {b: 0.0 for b in budgets}
        return 0.0, per

    tp, fp = _match_sorted(det_df, gt_df, iou_thresh)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    per = {}
    for b in budgets:
        fp_allowed = b * n_slices
        # last detection admitted before cumulative FP would exceed the budget
        idx = int(np.searchsorted(fp_cum, fp_allowed, side="right")) - 1
        per[b] = float(tp_cum[idx] / total_gt) if idx >= 0 else 0.0
    return float(np.mean(list(per.values()))), per


def per_slice_recall_2d(det_df: pd.DataFrame, gt_df: pd.DataFrame,
                        score_thresh: float, iou_thresh: float = 0.30) -> float:
    """Fraction of GT boxes recalled by some detection with ``score >= score_thresh``
    at ``IoU >= iou_thresh`` (the dump's per-slice 2D recall)."""
    total_gt = int(len(gt_df))
    if total_gt == 0:
        return float("nan")
    det = det_df[det_df["score"] >= score_thresh]
    det_img: Dict[Tuple[int, int], np.ndarray] = {}
    for vid, z, x1, y1, x2, y2 in det[["volume_id", "slice_z", *_BOX]].to_numpy():
        det_img.setdefault((int(vid), int(z)), []).append([x1, y1, x2, y2])
    hits = 0
    for vid, z, x1, y1, x2, y2 in gt_df[["volume_id", "slice_z", *_BOX]].to_numpy():
        dets = det_img.get((int(vid), int(z)), [])
        if any(_iou((x1, y1, x2, y2), d) >= iou_thresh for d in dets):
            hits += 1
    return hits / total_gt


def per_volume_recall_2d(det_df: pd.DataFrame, gt_df: pd.DataFrame,
                         score_thresh: float, iou_thresh: float = 0.30) -> float:
    """Fraction of volumes (with >=1 GT box) that have >=1 GT box hit by some
    detection with ``score >= score_thresh`` at ``IoU >= iou_thresh``."""
    vols = sorted({int(v) for v in gt_df["volume_id"].to_numpy()})
    if not vols:
        return float("nan")
    det = det_df[det_df["score"] >= score_thresh]
    det_img: Dict[Tuple[int, int], list] = {}
    for vid, z, x1, y1, x2, y2 in det[["volume_id", "slice_z", *_BOX]].to_numpy():
        det_img.setdefault((int(vid), int(z)), []).append([x1, y1, x2, y2])
    hit_vols = 0
    for v in vols:
        gv = gt_df[gt_df["volume_id"] == v]
        hit = False
        for vid, z, x1, y1, x2, y2 in gv[["volume_id", "slice_z", *_BOX]].to_numpy():
            dets = det_img.get((int(vid), int(z)), [])
            if any(_iou((x1, y1, x2, y2), d) >= iou_thresh for d in dets):
                hit = True
                break
        hit_vols += int(hit)
    return hit_vols / len(vols)
