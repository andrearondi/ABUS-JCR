"""[P3-UPDATE L4] Per-slice containment suppression kills nested small-in-big duplicates.

Torch-free. IoU-NMS cannot suppress a small box nested in a big one (IoU is tiny); containment
(inter/area_small) does. Equal-scale overlaps are untouched.
"""

import numpy as np
import pandas as pd

from abus_jcr.detect import schema as S
from abus_jcr.link.nms import containment_suppress_2d, containment_suppress_detections


def test_nested_small_box_is_suppressed():
    # big high-score box; small box fully inside it but with low IoU (area ratio tiny)
    boxes = np.array([[0, 0, 100, 100],   # big, score 0.9
                      [40, 40, 50, 50]])  # small, inside, score 0.5
    scores = np.array([0.9, 0.5])
    keep = containment_suppress_2d(boxes, scores, thresh=0.8)
    assert keep == [0]   # the small nested box is dropped


def test_partial_overlap_is_kept():
    boxes = np.array([[0, 0, 100, 100],
                      [90, 90, 190, 190]])  # only a corner overlaps -> low containment
    scores = np.array([0.9, 0.5])
    keep = sorted(containment_suppress_2d(boxes, scores, thresh=0.8))
    assert keep == [0, 1]


def test_thresh_one_is_noop():
    boxes = np.array([[0, 0, 100, 100], [40, 40, 50, 50]])
    scores = np.array([0.9, 0.5])
    assert sorted(containment_suppress_2d(boxes, scores, thresh=1.0)) == [0, 1]


def test_detections_frame_per_slice():
    rows = [
        {"volume_id": 1, "slice_z": 5, "x1": 0, "y1": 0, "x2": 100, "y2": 100, "score": 0.9},
        {"volume_id": 1, "slice_z": 5, "x1": 40, "y1": 40, "x2": 50, "y2": 50, "score": 0.5},  # nested -> drop
        {"volume_id": 1, "slice_z": 6, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.7},     # alone -> keep
    ]
    df = pd.DataFrame(rows, columns=S.DETECTION_COLUMNS)
    df["volume_id"] = df["volume_id"].astype("int64"); df["slice_z"] = df["slice_z"].astype("int64")
    out = containment_suppress_detections(S.validate_detections(df), thresh=0.8)
    assert len(out) == 2
    # the nested box (small, on slice 5) is gone; the slice-6 box survives
    assert not (((out["slice_z"] == 5) & (out["x2"] == 50)).any())
    assert ((out["slice_z"] == 6)).any()
