"""Phase-0b FP-structure probe: does the FP geometry have exploitable structure?

Synthetic check: FP candidates that are (a) spatially clustered and (b) depth-elongated
(ext_d0 >> ext_d1,d2) vs TP candidates that are isotropic and isolated -> verdict
PRESENT. When FPs are also isotropic and isolated -> verdict ABSENT. This pins the
Phase-4 geometry-term claim scope ("relational" vs "set-level calibration").
"""

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.candidates.record import CANDIDATE_COLUMNS
from abus_jcr.probe.fp_structure import fp_structure_probe


def _elongated_along_probe_axis(long_vox=40.0, short_vox=5.0):
    """Extents elongated about whichever axis the probe is configured to measure.

    Planted via ``C.FP_PROBE_ANISO_DEPTH_AXIS`` rather than hard-coded to ``d0``. The
    hard-coded version silently assumed the legacy profile, so under
    ``ABUS_AXIS_PROFILE=measured`` the fixture built a candidate elongated along the axis the
    probe does *not* read and the test failed for a reason that had nothing to do with the
    probe. Deriving the fixture from the constant tests the mechanism on either substrate —
    the same repair the flip-axis tests needed (HISTORY.md §8).
    """
    ext = [short_vox, short_vox, short_vox]
    ext[int(C.FP_PROBE_ANISO_DEPTH_AXIS)] = long_vox
    return tuple(ext)


def _cand(vid, label, cen, ext):
    row = {c: 0 for c in CANDIDATE_COLUMNS}
    row.update({
        "public_id": vid, "candidate_id": f"d:{vid}:{cen[0]}", "detector_of_origin": "full_seed0",
        "split": "val", "fold": -1, "label": label,
        "cen_d0": float(cen[0]), "cen_d1": float(cen[1]), "cen_d2": float(cen[2]),
        "ext_d0": float(ext[0]), "ext_d1": float(ext[1]), "ext_d2": float(ext[2]),
        "score_max": 0.5, "preprocess_hash": "x",
    })
    return row


def _clustered_elongated_fp_dataset():
    # Per volume: 1 isotropic isolated TP; FPs in TWO tight, depth-elongated clusters.
    # TP positions differ across volumes (pooled TP NN is large); within-FP-cluster
    # spacing is tiny (pooled FP NN is small); >1 FP cluster/vol.
    tp_pos = {100: (0, 0, 0), 101: (500, 500, 500)}
    fp_clusters = [(200, 200, 200), (300, 300, 200)]  # separated by > cluster radius
    rows = []
    for vid in (100, 101):
        rows.append(_cand(vid, "pos", tp_pos[vid], (10, 10, 10)))  # isotropic
        for base in fp_clusters:
            for k in range(3):
                cen = (base[0] + k, base[1] + k, base[2])
                # elongated about the probe's configured axis (aniso ~8), not a fixed d0
                rows.append(_cand(vid, "neg", cen, _elongated_along_probe_axis()))
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _isotropic_isolated_fp_dataset():
    rng = np.random.default_rng(0)
    tp_pos = {100: (0, 0, 0), 101: (500, 500, 500)}
    rows = []
    for vid in (100, 101):
        rows.append(_cand(vid, "pos", tp_pos[vid], (10, 10, 10)))
        for k in range(6):
            c = rng.integers(0, 500, size=3)
            rows.append(_cand(vid, "neg", tuple(int(x) for x in c), (10, 10, 10)))  # isotropic
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def test_structure_present_detected():
    res = fp_structure_probe(_clustered_elongated_fp_dataset(), split_filter="val")
    assert res["verdict"]["structure_present"] is True
    # FP anisotropy (ext along FP_PROBE_ANISO_DEPTH_AXIS / mean of the other two) ~ 8 >> TP ~ 1
    assert res["fp"]["anisotropy_median"] > res["tp"]["anisotropy_median"]
    # FPs are more clustered: smaller NN distance + >1 cluster/vol
    assert res["fp"]["nn_dist_median"] < res["tp"]["nn_dist_median"]
    assert res["fp"]["clusters_per_vol_median"] > 1


def test_structure_absent_when_isotropic_isolated():
    res = fp_structure_probe(_isotropic_isolated_fp_dataset(), split_filter="val")
    assert res["verdict"]["structure_present"] is False
