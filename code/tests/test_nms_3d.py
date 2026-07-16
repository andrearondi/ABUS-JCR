"""3D NMS over official candidate boxes (Phase 3 dedup primitive)."""

import numpy as np
import pytest

from abus_jcr.geometry import iou_official
from abus_jcr.link.nms import nms_3d, _official_to_corners


def _box(cx, L=10.0):
    return (float(cx), 0.0, 0.0, float(L), float(L), float(L))


def test_keeps_highest_score_suppresses_overlap():
    # three near-identical boxes (high overlap) + one far away
    boxes = [_box(0.0), _box(0.5), _box(1.0), _box(100.0)]
    scores = [0.5, 0.9, 0.7, 0.4]
    keep = nms_3d(boxes, scores, iou_thr=0.3)
    assert keep[0] == 1                       # highest score first
    assert 3 in keep                          # the far box survives
    assert len(keep) == 2                     # the overlapping cluster collapses to one
    assert set(keep) == {1, 3}


def test_disjoint_boxes_all_survive():
    boxes = [_box(0.0), _box(100.0), _box(200.0)]
    keep = nms_3d(boxes, [0.3, 0.2, 0.1], iou_thr=0.2)
    assert sorted(keep) == [0, 1, 2]


def test_empty():
    assert nms_3d([], [], iou_thr=0.2) == []


def test_internal_iou_matches_iou_official():
    # the vectorised corner IoU must agree with the scoring IoU it stands in for
    a, b = _box(0.0), _box(4.0)
    mins, maxs = _official_to_corners(np.array([a, b]))
    lo = np.maximum(mins[0], mins[1]); hi = np.minimum(maxs[0], maxs[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    vols = np.prod(maxs - mins, axis=1)
    iou_vec = inter / (vols[0] + vols[1] - inter)
    assert iou_vec == pytest.approx(iou_official(a, b), abs=1e-9)
