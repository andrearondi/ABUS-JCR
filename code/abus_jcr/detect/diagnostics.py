"""Per-slice detection recall diagnostics (Phase 2 [2.4], torch-free).

These read the common detection schema against the 2D GT-box table and quantify
*why* per-slice recall is what it is: an IoU-threshold sweep (loose vs tight
localisation), a GT-size breakdown (small specks vs real lesions), and a
localisation-agnostic **lesion-slice fire-rate** (does the detector fire at all on
a lesion slice). All are 2D diagnostics foreshadowing the Phase-3 3D recall
ceiling — never an operating point (Inv. 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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


def tight_bbox_from_mask(mask_slice) -> Optional[Tuple[float, float, float, float]]:
    """Single tight half-open bbox ``(x1, y1, x2, y2)`` around **all** set pixels of a 2D mask slice.

    ``x = d1`` (col), ``y = d0`` (row), matching the schema. Unions every component
    into one enclosing box (the "GT box built around the 2D mask" used by the
    overlay visualisation). Returns ``None`` for an empty mask.
    """
    m = np.asarray(mask_slice) > 0
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1.0), float(ys.max() + 1.0))


def overlay_annotations(boxes, scores, gt_box, top_k: int = 5) -> Dict:
    """IoU of each detection vs ``gt_box`` + the best-IoU index + the top-k-by-score indices.

    Returns ``{ious, best_idx, top_idx}``. ``best_idx`` is the detection with the
    highest 2D IoU against ``gt_box`` (``-1`` if there are no detections or no GT).
    ``top_idx`` is the (stable) score-descending index list, truncated to ``top_k``.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(boxes)
    if n == 0:
        return {"ious": np.zeros(0), "best_idx": -1, "top_idx": []}
    if gt_box is None:
        ious = np.zeros(n)
        best_idx = -1
    else:
        ious = np.array([iou_2d(gt_box, b) for b in boxes])
        best_idx = int(np.argmax(ious))
    top_idx = [int(i) for i in np.argsort(-scores, kind="stable")[:top_k]]
    return {"ious": ious, "best_idx": best_idx, "top_idx": top_idx}


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


def per_volume_recall(
    det_df: pd.DataFrame, gt_df: pd.DataFrame, volume_ids: Sequence[int],
    score_thresh: float = 0.05, iou_thresh: float = 0.30,
) -> Dict:
    """Per-LESION recall: fraction of volumes whose GT is hit on >=1 slice at ``IoU > iou_thresh``.

    **No linking is performed** — this is a pure 2D, *same-slice* GT↔detection IoU
    test OR-ed across each volume's slices (linking is Phase 3, frozen once per
    Inv. 4). The dataset is single-lesion-dominant (Phase 0: 99/100), so per-volume
    ≈ per-lesion.

    It is a **correlated proxy** for the Phase-3 3D recall ceiling, *not a strict
    bound*: it can overcount (a 1-slice 2D hit rarely survives 3D-tube IoU>0.3) and
    can undercount (loose per-slice boxes below this threshold may still link into a
    3D box that clears 3D IoU>0.3). ``hit_slice_counts`` (distinct hit slices per
    volume) gauges 3D linkability — a lesion hit on only one slice is a weak 3D
    candidate. The definitive number is Phase 3's linked 3D recall (Inv. 3, 8).
    """
    hit_slice_counts: List[int] = []
    vols_with_hit = 0
    for vid in volume_ids:
        gv = gt_df[gt_df["volume_id"] == vid]
        dv = det_df[(det_df["volume_id"] == vid) & (det_df["score"] >= score_thresh)]
        hit_slices = set()
        for z in sorted(gv["slice_z"].unique()):
            gts = boxes_halfopen_for(gt_df, vid, int(z))
            dets = dv[dv["slice_z"] == z][["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            if any(max((iou_2d(g, d) for d in dets), default=0.0) > iou_thresh for g in gts):
                hit_slices.add(int(z))
        if hit_slices:
            vols_with_hit += 1
        hit_slice_counts.append(len(hit_slices))
    n = len(list(volume_ids))
    return {
        "vols_with_hit": vols_with_hit,
        "n_vols": n,
        "recall": (vols_with_hit / n if n else float("nan")),
        "iou_thresh": iou_thresh,
        "hit_slice_counts": hit_slice_counts,
    }


def missed_lesion_detail(
    det_df: pd.DataFrame, gt_df: pd.DataFrame, volume_ids: Sequence[int],
    score_thresh: float = 0.05, iou_thresh: float = 0.30,
) -> List[Dict]:
    """Characterise the volumes with **0 hit-slices** — the recall-ceiling limiters.

    For each such volume returns ``{volume_id, n_gt_boxes, max_gt_diag, best_iou,
    fired_frac}`` where ``best_iou`` is the max 2D IoU any same-slice detection
    achieved against any of its GT boxes (0 => detector produced nothing
    overlapping; ~0.25 => loose boxes just under the bar, i.e. recoverable) and
    ``fired_frac`` is the fraction of the lesion's GT slices carrying >=1
    detection. Small ``max_gt_diag`` => intrinsic small-lesion tail.
    """
    out: List[Dict] = []
    for vid in volume_ids:
        gv = gt_df[gt_df["volume_id"] == vid]
        dv = det_df[(det_df["volume_id"] == vid) & (det_df["score"] >= score_thresh)]
        det_slices = set(dv["slice_z"].tolist())
        gt_slices = sorted(int(z) for z in gv["slice_z"].unique())
        best_iou, n_gt, max_diag, fired = 0.0, 0, 0.0, 0
        for z in gt_slices:
            gts = boxes_halfopen_for(gt_df, vid, int(z))
            dets = dv[dv["slice_z"] == z][["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            if z in det_slices:
                fired += 1
            for g in gts:
                n_gt += 1
                max_diag = max(max_diag, float(np.hypot(g[2] - g[0], g[3] - g[1])))
                best_iou = max(best_iou, max((iou_2d(g, d) for d in dets), default=0.0))
        if best_iou <= iou_thresh:  # 0 hit-slices
            out.append({
                "volume_id": int(vid),
                "n_gt_boxes": n_gt,
                "max_gt_diag": max_diag,
                "best_iou": best_iou,
                "fired_frac": (fired / len(gt_slices) if gt_slices else float("nan")),
            })
    return out


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
