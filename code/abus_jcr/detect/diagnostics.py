"""Per-slice detection recall diagnostics (Phase 2 [2.4], torch-free).

These read the common detection schema against the 2D GT-box table and quantify
*why* per-slice recall is what it is: an IoU-threshold sweep (loose vs tight
localisation), a GT-size breakdown (small specks vs real lesions), and a
localisation-agnostic **lesion-slice fire-rate** (does the detector fire at all on
a lesion slice). All are 2D diagnostics foreshadowing the Phase-3 3D recall
ceiling — never an operating point (Inv. 2).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .slice_det_dataset import boxes_halfopen_for


def iou_2d(a, b) -> float:
    """2D IoU of two half-open boxes ``(x1, y1, x2, y2)``."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def gt_matches(
    det_df: pd.DataFrame, gt_df: pd.DataFrame, volume_ids: Sequence[int], score_thresh: float
) -> List[Tuple[float, float]]:
    """Per GT box over ``volume_ids``: ``(diag_px, best_iou)`` vs same-slice detections.

    ``best_iou`` is the max 2D IoU against any detection with ``score >=
    score_thresh`` on that GT's slice (0.0 if none). ``diag_px`` is the GT box
    diagonal in iso pixels.
    """
    out: List[Tuple[float, float]] = []
    for vid in volume_ids:
        gv = gt_df[gt_df["volume_id"] == vid]
        dv = det_df[(det_df["volume_id"] == vid) & (det_df["score"] >= score_thresh)]
        for z in sorted(gv["slice_z"].unique()):
            gts = boxes_halfopen_for(gt_df, vid, int(z))
            dets = dv[dv["slice_z"] == z][["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            for g in gts:
                best = max((iou_2d(g, d) for d in dets), default=0.0)
                out.append((float(np.hypot(g[2] - g[0], g[3] - g[1])), best))
    return out


def gt_recall(
    det_df: pd.DataFrame, gt_df: pd.DataFrame, volume_ids: Sequence[int],
    score_thresh: float = 0.05, iou_thresh: float = 0.30,
) -> Tuple[int, int, float]:
    """Fraction of GT boxes with a same-slice detection at ``IoU > iou_thresh``."""
    matches = gt_matches(det_df, gt_df, volume_ids, score_thresh)
    total = len(matches)
    hits = sum(1 for _, best in matches if best > iou_thresh)
    return hits, total, (hits / total if total else float("nan"))


def lesion_slice_fire_rate(
    det_df: pd.DataFrame, gt_df: pd.DataFrame, volume_ids: Sequence[int], score_thresh: float = 0.05
) -> Tuple[int, int, float]:
    """Fraction of lesion-bearing slices carrying >=1 detection (IoU-agnostic).

    Separates "detector silent on the slice" from "detector fires but localises
    loosely" — a high fire-rate with low IoU-recall points at localisation, not
    missed slices.
    """
    fired, total = 0, 0
    for vid in volume_ids:
        gv = gt_df[gt_df["volume_id"] == vid]
        det_slices = set(det_df[(det_df["volume_id"] == vid)
                                & (det_df["score"] >= score_thresh)]["slice_z"].tolist())
        for z in sorted(gv["slice_z"].unique()):
            total += 1
            if int(z) in det_slices:
                fired += 1
    return fired, total, (fired / total if total else float("nan"))


def recall_breakdown(
    det_df: pd.DataFrame, gt_df: pd.DataFrame, volume_ids: Sequence[int],
    score_thresh: float = 0.05,
    iou_threshs: Sequence[float] = (0.1, 0.2, 0.3),
    size_edges: Sequence[float] = (0, 16, 32, 64, 128, np.inf),
) -> Dict:
    """IoU-sweep + size-stratified recall + lesion-slice fire-rate.

    ``by_iou[thr]`` = overall recall at that IoU. ``by_size[label]`` = per GT-diag
    bucket ``{n, hits, recall}`` at ``iou_threshs[-1]`` (the strictest, i.e. the
    headline 0.3). ``fire_rate`` = :func:`lesion_slice_fire_rate`.
    """
    matches = gt_matches(det_df, gt_df, volume_ids, score_thresh)
    diags = np.array([d for d, _ in matches], dtype=float)
    bests = np.array([b for _, b in matches], dtype=float)
    total = len(matches)

    by_iou = {}
    for thr in iou_threshs:
        by_iou[thr] = float((bests > thr).mean()) if total else float("nan")

    headline_iou = iou_threshs[-1]
    by_size: Dict[str, Dict] = {}
    for lo, hi in zip(size_edges[:-1], size_edges[1:]):
        label = f"[{int(lo)},{'inf' if hi == np.inf else int(hi)})"
        mask = (diags >= lo) & (diags < hi)
        n = int(mask.sum())
        hits = int(((bests > headline_iou) & mask).sum())
        by_size[label] = {"n": n, "hits": hits, "recall": (hits / n if n else float("nan"))}

    fired, n_slices, fire_rate = lesion_slice_fire_rate(det_df, gt_df, volume_ids, score_thresh)
    return {
        "n_gt_boxes": total,
        "score_thresh": score_thresh,
        "by_iou": by_iou,
        "by_size": by_size,
        "fire_rate": {"fired": fired, "lesion_slices": n_slices, "rate": fire_rate},
    }
