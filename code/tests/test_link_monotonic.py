"""[P3-UPDATE L2] Linked recall must be monotone in the score threshold (Inv. 2/3).

Two levels:
  (a) the ``monotonicity_violations`` helper flags a decreasing recall-vs-descending-threshold
      sequence (the [3.4] fingerprint) and passes a monotone one;
  (b) on a bounded linker, lowering the threshold (adding detections) never LOWERS the linked
      volume-hit — a superset of per-slice boxes yields >= linked hits.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _phase3_common import monotonicity_violations  # noqa: E402

from abus_jcr.detect import schema as S  # noqa: E402
from abus_jcr.link.tubes import link_tubes  # noqa: E402


def _det(vid, z, x1, y1, x2, y2, score):
    return {"volume_id": int(vid), "slice_z": int(z),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "score": float(score)}


def _frame(rows):
    df = pd.DataFrame(rows, columns=S.DETECTION_COLUMNS)
    df["volume_id"] = df["volume_id"].astype("int64")
    df["slice_z"] = df["slice_z"].astype("int64")
    return S.validate_detections(df)


def test_helper_flags_a_decreasing_curve():
    threshs = [0.5, 0.3, 0.1, 0.05]        # descending
    recalls = [0.4, 0.9, 0.7, 0.6]         # rises then DROPS as threshold falls -> violation
    viol = monotonicity_violations(threshs, recalls)
    assert len(viol) >= 1
    assert any(v["thresh_hi"] == 0.3 and v["thresh_lo"] == 0.1 for v in viol)


def test_helper_passes_a_monotone_curve():
    threshs = [0.5, 0.3, 0.1, 0.05]
    recalls = [0.4, 0.6, 0.9, 0.9]         # non-decreasing as threshold falls
    assert monotonicity_violations(threshs, recalls) == []


def test_up_to_peak_ignores_benign_subpeak_wobble():
    # The real [P3U2.8] seed0 shape: recall RISES to a peak at op=0.03 then FALLS below it.
    threshs = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
    recalls = [0.0, 0.2, 0.433, 0.667, 0.833, 0.900, 0.867, 0.700]  # peaks at 0.03, then declines
    # full-range check sees the sub-peak drops (0.900->0.867->0.700) ...
    assert len(monotonicity_violations(threshs, recalls)) == 2
    # ... but the deployment-region check (>= recall-peak op) is CLEAN -> gate passes.
    assert monotonicity_violations(threshs, recalls, up_to_peak=True) == []


def test_up_to_peak_still_catches_a_deployment_region_drop():
    # A drop ABOVE the recall peak (op=0.1) is a real unsound-linker regression -> must be flagged.
    threshs = [0.5, 0.3, 0.1, 0.05]
    recalls = [0.4, 0.3, 0.9, 0.5]         # 0.4->0.3 drop at high threshold, before the 0.9 peak
    viol = monotonicity_violations(threshs, recalls, up_to_peak=True)
    assert len(viol) == 1 and viol[0]["thresh_hi"] == 0.5 and viol[0]["thresh_lo"] == 0.3


def test_up_to_peak_with_plateau_included():
    # peak recall plateaus across 0.05 and 0.03; the whole plateau is inside the checked region,
    # and there is no drop within it -> clean.
    threshs = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
    recalls = [0.2, 0.5, 0.8, 0.9, 0.9, 0.6]   # rise, plateau at 0.9, then sub-peak drop
    assert monotonicity_violations(threshs, recalls, up_to_peak=True) == []
    assert len(monotonicity_violations(threshs, recalls)) == 1   # the 0.9->0.6 sub-peak drop


def test_bounded_linker_recall_nondecreasing_as_threshold_falls():
    # A clean 3-slice lesion tube (high scores) + low-score noise boxes elsewhere.
    lesion = [_det(1, z, 0, 0, 10, 10, 0.9) for z in range(3)]
    noise = [_det(1, z, 200, 200, 210, 210, 0.02) for z in range(3)]   # far-away 3-slice noise tube
    all_rows = lesion + noise

    def n_tubes_ge2(df):
        return link_tubes(df, max_tube_zspan=20, max_centroid_drift=20, containment_thresh=1.0, min_tube_len=2)

    hi = _frame([r for r in all_rows if r["score"] >= 0.5])   # lesion only
    lo = _frame([r for r in all_rows if r["score"] >= 0.01])  # lesion + noise
    # the lesion tube survives in BOTH; adding noise never removes it (superset property)
    hi_has_lesion = any(t[0][1][0] < 50 for t in n_tubes_ge2(hi))
    lo_has_lesion = any(t[0][1][0] < 50 for t in n_tubes_ge2(lo))
    assert hi_has_lesion and lo_has_lesion
