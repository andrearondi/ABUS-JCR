"""Synthetic per-slice detections link into the expected 3D tubes (Phase 3, Inv. 4).

Torch-free. Exercises the frozen linker on hand-built detection frames: continuity
by 2D IoU, gap bridging within ``max_z_gap``, min-tube-length pruning, the
single-volume assertion, and deterministic (stable) tie-breaking.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.detect import schema as S
from abus_jcr.link.tubes import link_tubes


def _det(vid, z, x1, y1, x2, y2, score):
    return {"volume_id": int(vid), "slice_z": int(z),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "score": float(score)}


def _frame(rows):
    df = pd.DataFrame(rows, columns=S.DETECTION_COLUMNS)
    df["volume_id"] = df["volume_id"].astype("int64")
    df["slice_z"] = df["slice_z"].astype("int64")
    return S.validate_detections(df)


def test_two_aligned_boxes_form_one_tube():
    df = _frame([_det(7, 0, 0, 0, 10, 10, 0.9),
                 _det(7, 1, 0, 0, 10, 10, 0.8)])
    tubes = link_tubes(df, min_tube_len=2)   # pin: tests linking mechanics, not the frozen production min_len
    assert len(tubes) == 1
    tube = tubes[0]
    assert [b[0] for b in tube] == [0, 1]  # z-ordered
    assert tube[0][1] == (0.0, 0.0, 10.0, 10.0)


def test_min_tube_len_drops_single_slice_spike():
    # An isolated one-slice detection with no IoU-neighbour is a tube of length 1.
    df = _frame([_det(7, 0, 0, 0, 10, 10, 0.9),
                 _det(7, 1, 0, 0, 10, 10, 0.8),
                 _det(7, 5, 100, 100, 110, 110, 0.7)])  # far away, isolated
    tubes = link_tubes(df, min_tube_len=2)
    assert len(tubes) == 1
    assert len(tubes[0]) == 2
    # with min_tube_len=1 the spike survives as its own tube
    assert len(link_tubes(df, min_tube_len=1)) == 2


def test_gap_is_bridged_within_max_z_gap():
    # z=0 and z=2 overlap; z=1 is missing. max_z_gap=1 bridges the single gap.
    df = _frame([_det(7, 0, 0, 0, 10, 10, 0.9),
                 _det(7, 2, 0, 0, 10, 10, 0.8)])
    tubes = link_tubes(df, max_z_gap=1, min_tube_len=2)
    assert len(tubes) == 1
    assert [b[0] for b in tubes[0]] == [0, 2]
    # with no gap tolerance the two boxes cannot join -> both dropped by min_tube_len
    assert link_tubes(df, max_z_gap=0, min_tube_len=2) == []


def test_low_iou_neighbour_does_not_link():
    # Adjacent slice box overlaps too little (IoU < LINK_IOU) -> separate spikes.
    df = _frame([_det(7, 0, 0, 0, 10, 10, 0.9),
                 _det(7, 1, 9, 9, 19, 19, 0.8)])  # tiny corner overlap
    tubes = link_tubes(df, link_iou=0.30, min_tube_len=1)
    assert len(tubes) == 2


def test_rejects_multiple_volumes():
    df = _frame([_det(7, 0, 0, 0, 10, 10, 0.9),
                 _det(8, 0, 0, 0, 10, 10, 0.9)])
    with pytest.raises(AssertionError):
        link_tubes(df)


def test_empty_input_returns_empty():
    assert link_tubes(S.empty_detections()) == []


def _naive_link_tubes(det_df, *, link_iou, max_z_gap, min_tube_len):
    """Reference scalar implementation of the frozen greedy linker (pre-vectorisation).

    Kept in the test only, to pin that the vectorised ``link_tubes`` is byte-identical.
    """
    from abus_jcr.detect.diagnostics import iou_2d
    S.validate_detections(det_df)
    if len(det_df) == 0:
        return []
    recs = det_df[["slice_z", "x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=float)
    order = sorted(range(len(recs)),
                   key=lambda i: (recs[i][0], recs[i][1], recs[i][2], recs[i][3], recs[i][4]))
    boxes = [(int(recs[i][0]), (recs[i][1], recs[i][2], recs[i][3], recs[i][4]), float(recs[i][5]))
             for i in order]
    n = len(boxes)
    consumed = [False] * n
    by_z = {}
    for i, (z, _, _) in enumerate(boxes):
        by_z.setdefault(z, []).append(i)

    def best_match(head_box, z_lo, z_hi):
        best_i, best_iou = -1, link_iou
        for z in range(z_lo, z_hi + 1):
            for j in by_z.get(z, ()):
                if consumed[j]:
                    continue
                iou = iou_2d(head_box, boxes[j][1])
                if iou >= best_iou and iou >= link_iou:
                    if iou > best_iou or best_i == -1:
                        best_iou, best_i = iou, j
        return best_i

    seed_order = sorted(range(n), key=lambda i: (-boxes[i][2], i))
    tubes = []
    for s in seed_order:
        if consumed[s]:
            continue
        consumed[s] = True
        members = [s]
        head = s
        while True:
            z = boxes[head][0]
            j = best_match(boxes[head][1], z + 1, z + max_z_gap + 1)
            if j == -1:
                break
            consumed[j] = True; members.append(j); head = j
        head = s
        while True:
            z = boxes[head][0]
            j = best_match(boxes[head][1], z - max_z_gap - 1, z - 1)
            if j == -1:
                break
            consumed[j] = True; members.append(j); head = j
        tube = sorted((boxes[m] for m in members), key=lambda b: b[0])
        tubes.append(tube)
    return [t for t in tubes if len(t) >= min_tube_len]


def _tube_key(tubes):
    """Order-independent canonical form: sorted tuples of (z, rounded box, rounded score)."""
    out = []
    for t in tubes:
        out.append(tuple((z, tuple(round(c, 6) for c in b), round(sc, 6)) for z, b, sc in t))
    return sorted(out)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("link_iou,max_z_gap,min_tube_len",
                         [(0.30, 1, 2), (0.20, 0, 1), (0.50, 2, 3)])
def test_link_tubes_differential(seed, link_iou, max_z_gap, min_tube_len):
    # Random overlapping detections across many slices — exercises seeding, gap
    # bridging, tie-breaks, and pruning. The vectorised linker must match the naive one.
    rng = np.random.default_rng(seed)
    rows = []
    for z in range(25):
        for _ in range(int(rng.integers(0, 12))):
            x1 = rng.uniform(0, 60); y1 = rng.uniform(0, 40)
            rows.append(_det(3, z, x1, y1, x1 + rng.uniform(3, 20), y1 + rng.uniform(3, 20),
                             round(float(rng.uniform(0.05, 0.99)), 4)))
    df = _frame(rows)
    # The naive reference models only the linking core, so disable the P3-UPDATE aggregation extras
    # (containment suppression L4, drift caps L1) here — they are pinned by their own tests.
    fast = link_tubes(df, link_iou=link_iou, max_z_gap=max_z_gap, min_tube_len=min_tube_len,
                      max_tube_zspan=None, max_centroid_drift=None, containment_thresh=1.0)
    ref = _naive_link_tubes(df, link_iou=link_iou, max_z_gap=max_z_gap, min_tube_len=min_tube_len)
    assert _tube_key(fast) == _tube_key(ref)


def test_deterministic_stable_tiebreak():
    # Two identical-score seeds on the same slice; linking must be reproducible.
    rows = [_det(7, 0, 0, 0, 10, 10, 0.5),
            _det(7, 0, 50, 50, 60, 60, 0.5),
            _det(7, 1, 0, 0, 10, 10, 0.5),
            _det(7, 1, 50, 50, 60, 60, 0.5)]
    a = link_tubes(_frame(rows), min_tube_len=1)
    b = link_tubes(_frame(list(reversed(rows))), min_tube_len=1)
    key = lambda tubes: sorted((t[0][0], round(t[0][1][0], 3), len(t)) for t in tubes)
    assert key(a) == key(b)
    assert len(a) == 2  # the two spatially-separate columns
