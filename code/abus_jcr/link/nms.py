"""3D non-maximum suppression over reconstructed candidate boxes (Phase 3 dedup).

The linker (``tubes.py``) is greedy and *consume-once*: every per-slice detection box
ends up in exactly one tube. With the loosened per-slice NMS (``LINK_NMS_THRESH = 0.7``)
a single object carries a thick stack of near-duplicate boxes on each slice, so the
linker spawns one parallel tube per box in the stack — the pool tracks NMS density, not
object count. ``nms_3d`` collapses those spatially-redundant candidates the way the
per-slice NMS deliberately did not: keep the highest detector-score candidate in each
3D cluster, suppress the rest at ``iou_thr``. This is the standard NoduleSAT/Seq-NMS
final step and is a candidate for the frozen aggregation (Inv. 4) — see [3.4b].

IoU here is axis-aligned 3D IoU on official centre+extent boxes, matching the semantics
of ``geometry.iou_official`` (the scoring/labeling IoU); vectorised for speed.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C
from ..geometry import OfficialBox


def _official_to_corners(boxes: np.ndarray):
    """(n, 6) official ``(cx,cy,cz,lx,ly,lz)`` -> ``(mins (n,3), maxs (n,3))`` corners."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 6)
    c = b[:, :3]
    half = b[:, 3:6] / 2.0
    return c - half, c + half


def containment_suppress_2d(boxes: np.ndarray, scores: np.ndarray,
                            thresh: float = C.LINK_CONTAINMENT_THRESH) -> List[int]:
    """[P3-UPDATE L4] Per-slice containment suppression -> kept indices (score-desc order).

    Drops a lower-score box ``b`` if it is >= ``thresh`` contained in a higher-score box
    ``a`` (``inter / area_b >= thresh``). This removes the nested small-in-big duplicates
    that IoU-NMS structurally cannot suppress (``IoU(small,big) = area_small/area_big`` is
    tiny for very different scales, so it never reaches the 0.5-0.7 IoU-NMS bar) — the
    mechanism behind the ~226 near-duplicate tubes/object. ``boxes`` are half-open
    ``(x1,y1,x2,y2)`` for ONE slice; complements, never replaces, torchvision's per-slice
    IoU-NMS. ``thresh >= 1.0`` keeps everything.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(boxes)
    if n == 0 or thresh >= 1.0:
        return list(range(n))
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    for i in order:
        i = int(i)
        drop = False
        for k in keep:  # k has a higher score (kept earlier)
            x1 = max(boxes[i, 0], boxes[k, 0]); y1 = max(boxes[i, 1], boxes[k, 1])
            x2 = min(boxes[i, 2], boxes[k, 2]); y2 = min(boxes[i, 3], boxes[k, 3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if areas[i] > 0 and inter / areas[i] >= thresh:
                drop = True
                break
        if not drop:
            keep.append(i)
    return keep


def containment_suppress_detections(det_df: pd.DataFrame,
                                    thresh: float = C.LINK_CONTAINMENT_THRESH) -> pd.DataFrame:
    """Apply ``containment_suppress_2d`` per ``(volume_id, slice_z)`` on a detection frame.

    Torch-free; runs after the detector's per-slice IoU-NMS and before linking. Preserves
    the detection schema and row semantics (``x=d1, y=d0``, half-open). ``thresh >= 1.0`` is
    a no-op passthrough.
    """
    if thresh >= 1.0 or len(det_df) == 0:
        return det_df
    parts = []
    for _key, grp in det_df.groupby(["volume_id", "slice_z"], sort=False):
        boxes = grp[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        scores = grp["score"].to_numpy(dtype=float)
        keep = containment_suppress_2d(boxes, scores, thresh)
        parts.append(grp.iloc[keep])
    return pd.concat(parts, ignore_index=True) if parts else det_df.iloc[0:0]


def nms_3d(boxes: Sequence[OfficialBox], scores: Sequence[float], iou_thr: float) -> List[int]:
    """Greedy 3D NMS. Returns kept indices in descending-score order.

    ``boxes`` are official centre+extent boxes; ``scores`` the per-candidate detector
    score (``score_max``). Ties in score are broken by original index (stable). A box is
    suppressed if its 3D IoU with an already-kept, higher-score box is ``>= iou_thr``.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 6)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = len(boxes)
    if n == 0:
        return []
    mins, maxs = _official_to_corners(boxes)
    vols = np.prod(np.clip(maxs - mins, 0.0, None), axis=1)
    order = np.argsort(-scores, kind="stable")

    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        lo = np.maximum(mins[i], mins[rest])
        hi = np.minimum(maxs[i], maxs[rest])
        inter = np.prod(np.clip(hi - lo, 0.0, None), axis=1)
        iou = inter / (vols[i] + vols[rest] - inter + 1e-12)
        order = rest[iou < iou_thr]
    return keep


def reduce_pool_3dnms(boxes: Sequence[OfficialBox], scores: Sequence[float],
                      iou_thr=C.LINK_3DNMS_IOU) -> List[int]:
    """[P3U2 3.C] The single guarded frozen-pool reduction. Returns kept indices.

    ``iou_thr is None`` -> keep everything (``list(range(n))``; the pre-Update-2
    behaviour). A float -> ``nms_3d(boxes, scores, iou_thr)`` (membership-only 3D NMS
    keyed by ``score_max``; coordinates are never changed). This is the ONE place the
    None-guard lives, so every pool-construction path (generation, selection,
    calibration, the freeze sweep, the reducer gate) applies an identical reduction and
    the frozen pool == the deployed pool everywhere it is measured (Inv. 4, 8).
    """
    n = len(boxes)
    if iou_thr is None:
        return list(range(n))
    return nms_3d(boxes, scores, float(iou_thr))
