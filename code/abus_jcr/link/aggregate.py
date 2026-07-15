"""Per-tube score statistics, within-volume ranking, and IoU-band labeling (Phase 3).

The score-stats vector column names are FROZEN (``conventions.SCORE_STAT_COLUMNS``)
and consumed verbatim by the Phase-4 feature record. Labeling reuses
``geometry.iou_official`` (== the vendored scoring ``iou_3d``) so a candidate's
training label is decided by byte-identical IoU to the FROC hit test (Inv. 3), with
the Inv.-11 ignore band (pos > 0.30, neg < 0.10, drop the middle).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .. import conventions as C
from ..geometry import OfficialBox, iou_official
from .tubes import Tube


def score_stats(tube: Tube) -> dict:
    """Frozen per-tube score/geometry summary.

    ``{score_max, score_mean, score_std(ddof=0), score_min, slice_count, z_span,
    fill_ratio}`` where ``slice_count`` = number of boxes, ``z_span = max_z - min_z
    + 1``, ``fill_ratio = slice_count / z_span`` in ``(0, 1]``.
    """
    if not tube:
        raise ValueError("score_stats: empty tube")
    scores = np.asarray([s for _, _, s in tube], dtype=float)
    zs = [z for z, _, _ in tube]
    z_span = int(max(zs) - min(zs) + 1)
    slice_count = int(len(tube))
    return {
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=0)),
        "score_min": float(scores.min()),
        "slice_count": slice_count,
        "z_span": z_span,
        "fill_ratio": float(slice_count / z_span),
    }


def within_volume_rank(cand_df: pd.DataFrame) -> pd.DataFrame:
    """Per ``public_id``: add ``rank`` (1 = highest ``score_max``) and ``rank_norm``.

    Sort is stable and descending on ``score_max``; ``rank_norm = rank /
    n_candidates_in_volume`` in ``(0, 1]``. Returns a copy with the two columns
    added, preserving the input row order otherwise (so a caller can align by index).
    """
    if len(cand_df) == 0:
        out = cand_df.copy()
        out["rank"] = pd.Series(dtype="int64")
        out["rank_norm"] = pd.Series(dtype="float64")
        return out

    out = cand_df.copy()
    rank = pd.Series(index=out.index, dtype="int64")
    rank_norm = pd.Series(index=out.index, dtype="float64")
    for _, grp in out.groupby("public_id", sort=False):
        ordered = grp["score_max"].sort_values(ascending=False, kind="stable").index
        n = len(ordered)
        for r, idx in enumerate(ordered, start=1):
            rank.loc[idx] = r
            rank_norm.loc[idx] = r / n
    out["rank"] = rank
    out["rank_norm"] = rank_norm
    return out


def label_candidate(cand_official: OfficialBox, gt_official: OfficialBox) -> Tuple[str, float]:
    """IoU-band label vs the single official GT box (Inv. 11).

    ``iou = geometry.iou_official(cand, gt)``; ``'pos'`` if ``iou > LABEL_POS_IOU``,
    ``'neg'`` if ``iou < LABEL_NEG_IOU``, else ``'ignore'`` (the [0.10, 0.30] band is
    dropped from the loss). Returns ``(label, iou)``; the IoU is kept for audit.
    """
    iou = float(iou_official(cand_official, gt_official))
    if iou > C.LABEL_POS_IOU:
        return "pos", iou
    if iou < C.LABEL_NEG_IOU:
        return "neg", iou
    return "ignore", iou
