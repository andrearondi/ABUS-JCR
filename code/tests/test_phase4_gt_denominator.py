"""Phase 4 must score against the FULL split GT, exactly as [F.8] did.

The official ``evaluate()`` iterates ``for pid in gt_pids`` and divides by ``len(df_list)``
(volumes) and ``sum(num_gts)`` (lesions) — so the GT csv sets BOTH denominators. Handing it a
GT table restricted to the volumes a pool happens to cover credits the pool for the volumes it
failed on.

Measured consequence, seed 0: ``full_seed0`` produces zero candidates for vol 129, so the
restricted table held 29 volumes and [4.2] reported ceiling **0.8621 = 25/29** and B0 CPM
0.6882, where ``[F.8]`` — which passes the whole 30-volume table to every seed
(``phase3_baseline_froc.py``) — recorded **0.8333 = 25/30** and 0.6673 on the identical pool
(``n_candidates`` 2225 both times).

Aligned 2026-08-10 with the user's approval. Phase 5 scores the untouched test split, where
volumes a detector missed cannot be dropped either, so this also removes a val-vs-test bias.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _phase4_common import gt_for_pool  # noqa: E402


def _gt(pids):
    return pd.DataFrame({"public_id": list(pids),
                         "coordX": [1.0] * len(pids), "coordY": [1.0] * len(pids),
                         "coordZ": [1.0] * len(pids), "x_length": [2.0] * len(pids),
                         "y_length": [2.0] * len(pids), "z_length": [2.0] * len(pids)})


def _pool(pids):
    return pd.DataFrame({"public_id": list(pids), "detector_of_origin": ["full_seed0"] * len(pids)})


def test_uncovered_volume_stays_in_the_gt_denominator():
    """The seed-0 case: 30 GT volumes, a pool covering only 29. All 30 must survive."""
    gt = _gt(range(100, 130))
    pool = _pool([p for p in range(100, 130) if p != 129])
    out = gt_for_pool(gt, pool)
    assert len(out) == 30, "an uncovered volume was dropped — it is a miss, not an absence"
    assert 129 in set(out["public_id"].astype(int))


def test_full_coverage_is_unchanged():
    gt = _gt(range(100, 130))
    out = gt_for_pool(gt, _pool(range(100, 130)))
    assert len(out) == 30
