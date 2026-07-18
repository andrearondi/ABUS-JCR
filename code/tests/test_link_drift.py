"""[P3-UPDATE L1] Tube drift/length caps bound the reconstructed hull (Inv. 4).

Torch-free. A detection stream that would random-walk across the whole volume must yield
tubes each within LINK_MAX_TUBE_ZSPAN / LINK_MAX_CENTROID_DRIFT; a compact-lesion stream is
byte-identical to the uncapped linker (the cap is inert when unviolated) and never SPLITS a
genuine within-cap tube.
"""

import pandas as pd

from abus_jcr.detect import schema as S
from abus_jcr.link.tubes import link_tubes


def _det(vid, z, x1, y1, x2, y2, score):
    return {"volume_id": int(vid), "slice_z": int(z),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "score": float(score)}


def _frame(rows):
    df = pd.DataFrame(rows, columns=S.DETECTION_COLUMNS)
    df["volume_id"] = df["volume_id"].astype("int64")
    df["slice_z"] = df["slice_z"].astype("int64")
    return S.validate_detections(df)


def test_zspan_cap_bounds_a_runaway_tube():
    # A high-IoU chain across 40 slices (each box overlaps its neighbour) — a runaway.
    rows = [_det(1, z, 0, 0, 10, 10, 0.9 - z * 0.001) for z in range(40)]
    df = _frame(rows)
    # uncapped: one 40-slice tube
    uncapped = link_tubes(df, max_tube_zspan=None, max_centroid_drift=None, containment_thresh=1.0)
    assert max(b[0] for b in uncapped[0]) - min(b[0] for b in uncapped[0]) + 1 == 40
    # capped at z-span 10: no tube may span more than 10 slices
    capped = link_tubes(df, max_tube_zspan=10, max_centroid_drift=None, containment_thresh=1.0)
    for tube in capped:
        zs = [b[0] for b in tube]
        assert max(zs) - min(zs) + 1 <= 10


def test_centroid_drift_cap_stops_lateral_wander():
    # Boxes that overlap neighbour-to-neighbour but march steadily in +x (a shadow drift).
    rows = [_det(2, z, z * 4, 0, z * 4 + 10, 10, 0.9 - z * 0.001) for z in range(30)]
    df = _frame(rows)
    capped = link_tubes(df, max_tube_zspan=None, max_centroid_drift=12.0, containment_thresh=1.0)
    for tube in capped:
        cxs = [(b[1][0] + b[1][2]) / 2.0 for b in tube]
        # every member's centre stays within the running-mean drift budget of the seed region
        assert max(cxs) - min(cxs) <= 12.0 * 2 + 1e-6  # generous span bound (running-mean based)


def test_caps_inert_on_a_compact_lesion():
    # A short, stationary 4-slice tube — well within any sane cap.
    rows = [_det(3, z, 0, 0, 10, 10, 0.9 - z * 0.01) for z in range(4)]
    df = _frame(rows)
    a = link_tubes(df, max_tube_zspan=None, max_centroid_drift=None, containment_thresh=1.0)
    b = link_tubes(df, max_tube_zspan=50, max_centroid_drift=50, containment_thresh=1.0)
    assert a == b  # identical tubes; cap never fired
    assert len(b) == 1 and len(b[0]) == 4
