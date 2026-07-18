"""[P3-UPDATE D6] ATSS adaptive assignment core (numpy; torch-free, laptop-tested).

Only the pure ``atss_match`` algorithm is exercised here (the correctness-critical piece);
the thin torch subclass wiring is smoke-tested where torch is installed (test_retinanet_build
covers the build path under DET_ASSIGNER). ATSS returns torchvision Matcher conventions:
per-anchor matched-GT index, -1 = background.
"""

import numpy as np

from abus_jcr.detect.atss import atss_match, BELOW_LOW_THRESHOLD


def _grid_anchors(centres, size):
    """Axis-aligned square anchors of side ``size`` centred at each (cx,cy)."""
    c = np.asarray(centres, dtype=float)
    h = size / 2.0
    return np.stack([c[:, 0] - h, c[:, 1] - h, c[:, 0] + h, c[:, 1] + h], axis=1)


def test_empty_gt_all_background():
    anc = _grid_anchors([(0, 0), (10, 10)], 4)
    m = atss_match(np.zeros((0, 4)), anc, [2], topk=2)
    assert (m == BELOW_LOW_THRESHOLD).all()


def test_centre_inside_constraint_excludes_a_candidate_outside_the_box():
    # GT centre (5,5). Anchor 0 centre inside; anchor 1 centre (5,14) OUTSIDE the box but near
    # enough to be a top-k candidate; anchor 2 far away (not a candidate). topk=2 -> candidates
    # {0,1}; the centre-inside constraint must still drop anchor 1.
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    anc = _grid_anchors([(5, 5), (5, 14), (100, 100)], size=8)
    m = atss_match(gt, anc, [3], topk=2)
    assert m[0] == 0                              # inside, high IoU -> positive for GT 0
    assert m[1] == BELOW_LOW_THRESHOLD            # candidate but centre outside -> dropped
    assert m[2] == BELOW_LOW_THRESHOLD            # far -> not even a candidate


def test_adaptive_threshold_matches_hand_computation():
    # Single level, single GT; the ATSS threshold is mean+std of candidate IoUs.
    gt = np.array([[0, 0, 20, 20]], dtype=float)
    # three anchors all centred inside, varied sizes -> varied IoU
    anc = np.array([[0, 0, 20, 20],      # perfect overlap, IoU 1.0
                    [0, 0, 10, 10],      # quarter, IoU 0.25
                    [5, 5, 25, 25]], dtype=float)  # shifted, partial
    napl = [3]
    m = atss_match(gt, anc, napl, topk=3)
    # replicate: candidates = all 3 (topk>=3); t_g = mean+std of their IoUs; centre-inside all.
    from abus_jcr.detect.atss import _iou_matrix, _centres
    ious = _iou_matrix(gt, anc)[0]
    thr = ious.mean() + ious.std()
    inside = np.ones(3, dtype=bool)  # all centres inside [0,0,20,20]? check
    ac = _centres(anc)
    inside = (ac[:, 0] >= 0) & (ac[:, 0] <= 20) & (ac[:, 1] >= 0) & (ac[:, 1] <= 20)
    expected_pos = set(np.where((ious >= thr) & inside)[0].tolist())
    got_pos = set(np.where(m == 0)[0].tolist())
    assert got_pos == expected_pos
    assert 0 in got_pos                           # the perfect-overlap anchor is always positive
