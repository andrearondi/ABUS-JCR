"""The linker: per-slice detections -> 3D tubes (Phase 3, frozen; Inv. 4).

Greedy, score-seeded Seq-NMS-style linking over ONE volume's detections. The
aggregation is detector-agnostic and set ONCE (``conventions.LINK_*``), then reused
unmodified for every detector (Inv. 4). This module only *links*; per-tube score
aggregation, reconstruction, and labeling live in ``reconstruct`` / ``aggregate``.

A ``Tube`` is a z-ordered list of ``(slice_z, (x1, y1, x2, y2), score)`` where the
box is the frozen half-open iso-slice frame (``x = d1``, ``y = d0``) from
``detect/schema``. 2D IoU is the existing ``detect.diagnostics.iou_2d`` (half-open),
so continuity uses byte-identical geometry to the Phase-2 diagnostics.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .. import conventions as C
from ..detect import schema as S
from ..detect.diagnostics import iou_2d

Box = Tuple[float, float, float, float]
Tube = List[Tuple[int, Box, float]]


def link_tubes(
    det_df: pd.DataFrame,
    *,
    link_iou: float = C.LINK_IOU,
    max_z_gap: int = C.LINK_MAX_Z_GAP,
    min_tube_len: int = C.LINK_MIN_TUBE_LEN,
) -> List[Tube]:
    """Greedy, score-seeded Seq-NMS-style tube linking over ONE volume's detections.

    Algorithm (deterministic; ties broken by ``(slice_z, x1, y1)``):

    1. Validate schema; group boxes by ``slice_z``.
    2. Repeat until no unconsumed box remains:
       a. Seed = the highest-score unconsumed box.
       b. Extend FORWARD: from the current head at ``z``, look in slices
          ``z+1 .. z+max_z_gap+1``; pick the unconsumed box with the highest 2D IoU
          ``>= link_iou`` to the head; append, consume, advance the head to it. Stop
          when no slice within the gap yields a match.
       c. Extend BACKWARD symmetrically from the seed.
    3. Order each tube by ``slice_z``; drop tubes with ``< min_tube_len`` boxes.

    Input MUST be a single volume's rows (asserts one ``volume_id``). Returns the
    surviving tubes.
    """
    S.validate_detections(det_df)
    if len(det_df) == 0:
        return []
    vids = det_df["volume_id"].unique()
    assert len(vids) == 1, f"link_tubes expects one volume's rows, got volume_ids {list(vids)}"

    # Materialise boxes as an index-stable list; a parallel `consumed` flag array.
    # Deterministic global order: (slice_z, x1, y1, x2, y2) — the tie-break spec.
    recs = det_df[["slice_z", "x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=float)
    order = sorted(range(len(recs)),
                   key=lambda i: (recs[i][0], recs[i][1], recs[i][2], recs[i][3], recs[i][4]))
    boxes = [(int(recs[i][0]), (recs[i][1], recs[i][2], recs[i][3], recs[i][4]), float(recs[i][5]))
             for i in order]
    n = len(boxes)
    consumed = [False] * n

    # slice_z -> list of box indices (in the deterministic global order)
    by_z: dict = {}
    for i, (z, _, _) in enumerate(boxes):
        by_z.setdefault(z, []).append(i)

    def best_match(head_box: Box, z_lo: int, z_hi: int) -> int:
        """Highest-IoU unconsumed box in slices [z_lo, z_hi] with IoU >= link_iou.

        Ties (equal IoU) are broken by the deterministic global index order, so the
        first candidate encountered (smallest (slice_z, x1, y1, ...)) wins.
        """
        best_i, best_iou = -1, link_iou
        for z in range(z_lo, z_hi + 1):
            for j in by_z.get(z, ()):
                if consumed[j]:
                    continue
                iou = iou_2d(head_box, boxes[j][1])
                if iou >= best_iou and iou >= link_iou:
                    # strict '>' keeps the first (lowest-index) box on exact ties
                    if iou > best_iou or best_i == -1:
                        best_iou, best_i = iou, j
        return best_i

    # Seeds in descending score, ties by the deterministic global index order.
    seed_order = sorted(range(n), key=lambda i: (-boxes[i][2], i))

    tubes: List[Tube] = []
    for s in seed_order:
        if consumed[s]:
            continue
        consumed[s] = True
        members = [s]

        # forward
        head = s
        while True:
            z = boxes[head][0]
            j = best_match(boxes[head][1], z + 1, z + max_z_gap + 1)
            if j == -1:
                break
            consumed[j] = True
            members.append(j)
            head = j

        # backward from the seed
        head = s
        while True:
            z = boxes[head][0]
            j = best_match(boxes[head][1], z - max_z_gap - 1, z - 1)
            if j == -1:
                break
            consumed[j] = True
            members.append(j)
            head = j

        tube = sorted((boxes[m] for m in members), key=lambda b: b[0])
        tubes.append(tube)

    tubes = [t for t in tubes if len(t) >= min_tube_len]
    return tubes
