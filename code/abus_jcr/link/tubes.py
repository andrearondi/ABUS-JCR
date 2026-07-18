"""The linker: per-slice detections -> 3D tubes (Phase 3, frozen; Inv. 4).

Greedy, score-seeded Seq-NMS-style linking over ONE volume's detections. The
aggregation is detector-agnostic and set ONCE (``conventions.LINK_*``), then reused
unmodified for every detector (Inv. 4). This module only *links*; per-tube score
aggregation, reconstruction, and labeling live in ``reconstruct`` / ``aggregate``.

A ``Tube`` is a z-ordered list of ``(slice_z, (x1, y1, x2, y2), score)`` where the
box is the frozen half-open iso-slice frame (``x = d1``, ``y = d0``) from
``detect/schema``. 2D IoU matches ``detect.diagnostics.iou_2d`` (half-open) exactly.

**Implementation note (perf, output-preserving):** the neighbour search is vectorised
with numpy over each window's live boxes. This is byte-identical to the reference
scalar greedy match — ``np.argmax`` returns the first occurrence of the maximum, which
reproduces the scalar loop's "keep the first box achieving the running-max IoU"
tie-break exactly (``test_link_tubes_differential`` pins this against a naive
implementation). It turns a per-volume O(N x boxes/slice) Python scan into a handful of
vectorised ops per link step, so recall-saturated pools (~1e5 dets/vol) link in seconds.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C
from ..detect import schema as S

Box = Tuple[float, float, float, float]
Tube = List[Tuple[int, Box, float]]


def _iou_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Half-open 2D IoU of one ``box`` (4,) against ``boxes`` (m, 4) — matches ``iou_2d``."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def _box_centre(b: np.ndarray) -> Tuple[float, float]:
    """Centre ``((x1+x2)/2, (y1+y2)/2)`` of one ``(x1,y1,x2,y2)`` box."""
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def link_tubes(
    det_df: pd.DataFrame,
    *,
    link_iou: float = C.LINK_IOU,
    max_z_gap: int = C.LINK_MAX_Z_GAP,
    min_tube_len: int = C.LINK_MIN_TUBE_LEN,
    max_tube_zspan=C.LINK_MAX_TUBE_ZSPAN,
    max_centroid_drift=C.LINK_MAX_CENTROID_DRIFT,
    containment_thresh: float = C.LINK_CONTAINMENT_THRESH,
) -> List[Tube]:
    """Greedy, score-seeded Seq-NMS-style tube linking over ONE volume's detections.

    Algorithm (deterministic; ties broken by ``(slice_z, x1, y1)``):

    1. Validate schema; group boxes by ``slice_z``.
    2. Repeat until no unconsumed box remains:
       a. Seed = the highest-score unconsumed box.
       b. Extend FORWARD: from the current head at ``z``, look in slices
          ``z+1 .. z+max_z_gap+1``; pick the unconsumed box with the highest 2D IoU
          ``>= link_iou`` to the head; if appending it would keep the tube within BOTH
          drift caps (below), append, consume, advance the head to it. Stop when no
          slice within the gap yields a match, OR the best match violates a cap.
       c. Extend BACKWARD symmetrically from the seed.
    3. Order each tube by ``slice_z``; drop tubes with ``< min_tube_len`` boxes.

    [P3-UPDATE L1] Drift caps bound the reconstructed union hull, so a tube can no longer
    random-walk across the whole volume and consume a real lesion's boxes (the pre-P3-UPDATE
    linker produced whole-volume "candidates" and made linked recall non-monotone in the
    threshold). ``max_tube_zspan`` caps ``max_z - min_z + 1``; ``max_centroid_drift`` caps a
    candidate box's centre distance from the tube's running-mean centre. ``None`` = uncapped
    (the pre-P3-UPDATE behaviour, kept for parity tests). Both are Train-GT-derived and frozen
    at [3.3'] (Inv. 4), never per-detector. When the best-IoU candidate violates a cap the
    extension STOPS (it does not fall through to a worse, more distant box).

    Input MUST be a single volume's rows (asserts one ``volume_id``). Returns the
    surviving tubes.
    """
    S.validate_detections(det_df)
    if len(det_df) == 0:
        return []
    vids = det_df["volume_id"].unique()
    assert len(vids) == 1, f"link_tubes expects one volume's rows, got volume_ids {list(vids)}"

    # [P3-UPDATE L4] Per-slice containment suppression before linking: drops nested small-in-big
    # duplicates the detector's IoU-NMS structurally cannot (single frozen aggregation, Inv. 4).
    if containment_thresh < 1.0:
        from .nms import containment_suppress_detections
        det_df = containment_suppress_detections(det_df, containment_thresh)
        if len(det_df) == 0:
            return []

    # Deterministic global order: (slice_z, x1, y1, x2, y2) — the tie-break spec.
    recs = det_df[["slice_z", "x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=float)
    order = sorted(range(len(recs)),
                   key=lambda i: (recs[i][0], recs[i][1], recs[i][2], recs[i][3], recs[i][4]))
    recs = recs[order]
    n = len(recs)
    zs = recs[:, 0].astype(np.int64)
    coords = np.ascontiguousarray(recs[:, 1:5])            # (n, 4), global order
    scores = recs[:, 5]
    consumed = np.zeros(n, dtype=bool)

    # slice_z -> int array of global indices (ascending == global order within the slice)
    by_z: dict = {}
    for i in range(n):
        by_z.setdefault(int(zs[i]), []).append(i)
    by_z = {z: np.asarray(idx, dtype=np.int64) for z, idx in by_z.items()}

    def best_match(head_box: np.ndarray, z_lo: int, z_hi: int) -> int:
        """First live box of maximal IoU (>= link_iou) over slices ``z_lo..z_hi``.

        Candidates are gathered in ``(slice_z asc, global-index asc)`` order — exactly
        the scalar loop's iteration order — and ``argmax`` takes the first occurrence of
        the maximum, matching the scalar "keep first box achieving the running max".
        """
        parts = [by_z[z] for z in range(z_lo, z_hi + 1) if z in by_z]
        if not parts:
            return -1
        cand = np.concatenate(parts)
        cand = cand[~consumed[cand]]
        if cand.size == 0:
            return -1
        ious = _iou_vec(head_box, coords[cand])
        j = int(np.argmax(ious))
        if ious[j] < link_iou:
            return -1
        return int(cand[j])

    # Seeds in descending score, ties by the deterministic global index order.
    seed_order = sorted(range(n), key=lambda i: (-scores[i], i))

    zspan_cap = None if max_tube_zspan is None else int(max_tube_zspan)
    drift_cap = None if max_centroid_drift is None else float(max_centroid_drift)

    tubes: List[Tube] = []
    for s in seed_order:
        if consumed[s]:
            continue
        consumed[s] = True
        members = [s]
        # running tube state for the drift caps
        min_z = max_z = int(zs[s])
        cx0, cy0 = _box_centre(coords[s])
        sum_cx, sum_cy, cnt = cx0, cy0, 1

        def _within_caps(cand: int) -> bool:
            """True iff appending ``cand`` keeps the tube inside both drift caps."""
            if zspan_cap is not None:
                zc = int(zs[cand])
                if (max(max_z, zc) - min(min_z, zc) + 1) > zspan_cap:
                    return False
            if drift_cap is not None:
                bx, by = _box_centre(coords[cand])
                mx, my = sum_cx / cnt, sum_cy / cnt
                if ((bx - mx) ** 2 + (by - my) ** 2) ** 0.5 > drift_cap:
                    return False
            return True

        head = s  # forward
        while True:
            z = int(zs[head])
            j = best_match(coords[head], z + 1, z + max_z_gap + 1)
            if j == -1 or not _within_caps(j):
                break
            consumed[j] = True
            members.append(j)
            min_z, max_z = min(min_z, int(zs[j])), max(max_z, int(zs[j]))
            bx, by = _box_centre(coords[j]); sum_cx += bx; sum_cy += by; cnt += 1
            head = j

        head = s  # backward from the seed
        while True:
            z = int(zs[head])
            j = best_match(coords[head], z - max_z_gap - 1, z - 1)
            if j == -1 or not _within_caps(j):
                break
            consumed[j] = True
            members.append(j)
            min_z, max_z = min(min_z, int(zs[j])), max(max_z, int(zs[j]))
            bx, by = _box_centre(coords[j]); sum_cx += bx; sum_cy += by; cnt += 1
            head = j

        if len(members) < min_tube_len:
            continue
        members.sort(key=lambda m: (int(zs[m]), m))
        tube = [(int(zs[m]), (float(coords[m][0]), float(coords[m][1]),
                              float(coords[m][2]), float(coords[m][3])), float(scores[m]))
                for m in members]
        tubes.append(tube)

    return tubes
