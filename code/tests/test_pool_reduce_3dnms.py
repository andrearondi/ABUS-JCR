"""[P3U2 3.C] Membership-only 3D-NMS frozen-pool reducer (torch-free).

`reduce_pool_3dnms` is the single guarded reduction used by every pool-construction
path. None -> keep everything (pre-Update-2 behaviour). A float -> greedy 3D NMS keyed
by score_max, coordinates untouched. It must never drop a candidate that is the unique
hitter of its region (recall-preserving), and must leave low-IoU depth-collinear boxes
(a shadow train, the Axis-A signal) intact.
"""

import pytest

from abus_jcr.link.nms import reduce_pool_3dnms


def _box(cx, cy, cz, L=2.0):
    """Official centre+extent box (cx,cy,cz,lx,ly,lz)."""
    return (float(cx), float(cy), float(cz), float(L), float(L), float(L))


def test_none_keeps_everything():
    boxes = [_box(0, 0, 0), _box(0, 0, 0), _box(10, 10, 10)]
    scores = [0.9, 0.8, 0.7]
    assert sorted(reduce_pool_3dnms(boxes, scores, iou_thr=None)) == [0, 1, 2]


def test_duplicate_suppressed_higher_score_kept():
    # two (near-)identical boxes overlap at IoU 1.0 -> keep the higher score only
    boxes = [_box(0, 0, 0), _box(0, 0, 0)]
    scores = [0.4, 0.9]
    kept = reduce_pool_3dnms(boxes, scores, iou_thr=0.5)
    assert kept == [1]                       # index 1 has the higher score_max


def test_unique_hitter_never_dropped():
    # an isolated candidate (no higher-score overlapper) always survives
    boxes = [_box(0, 0, 0), _box(0, 0, 0), _box(100, 100, 100)]
    scores = [0.9, 0.8, 0.5]
    kept = set(reduce_pool_3dnms(boxes, scores, iou_thr=0.5))
    assert 2 in kept                          # the isolated (unique) box is kept
    assert 0 in kept and 1 not in kept        # of the overlapping pair, the top score survives


def test_depth_collinear_train_preserved():
    # a shadow train: boxes stacked along z, low mutual 3D IoU -> all survive
    boxes = [_box(0, 0, z * 10) for z in range(4)]
    scores = [0.5, 0.5, 0.5, 0.5]
    kept = reduce_pool_3dnms(boxes, scores, iou_thr=0.3)
    assert len(kept) == 4                     # Axis-A structure is not collapsed


def test_default_thresh_is_conventions_value():
    # with LINK_3DNMS_IOU defaulting to None, the reducer is a no-op passthrough
    from abus_jcr import conventions as C
    boxes = [_box(0, 0, 0), _box(0, 0, 0)]
    scores = [0.4, 0.9]
    kept = reduce_pool_3dnms(boxes, scores)   # iou_thr defaults to C.LINK_3DNMS_IOU
    if C.LINK_3DNMS_IOU is None:
        assert sorted(kept) == [0, 1]
