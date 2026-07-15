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
    tubes = link_tubes(df)
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
