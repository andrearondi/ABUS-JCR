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

from ..geometry import OfficialBox


def _official_to_corners(boxes: np.ndarray):
    """(n, 6) official ``(cx,cy,cz,lx,ly,lz)`` -> ``(mins (n,3), maxs (n,3))`` corners."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 6)
    c = b[:, :3]
    half = b[:, 3:6] / 2.0
    return c - half, c + half


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
