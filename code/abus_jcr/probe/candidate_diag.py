"""[P3U2] Candidate-pool diagnostics — the torch-free core (unit-tested).

Given a volume's linked tubes + the official GT box + the GT's ISO-space box, build a
per-candidate frame that carries, for every 3D candidate:
  - the frozen score-stats + tube-geometry block,
  - the candidate's official size (extents, 3D diagonal),
  - ``official_iou``  = 3D IoU vs the GT in the OFFICIAL scoring space (the hit test), and
  - ``iso_iou``       = 3D IoU vs the GT in ISO space (before the iso->native->official
    round-trip). The gap ``iso_iou - official_iou`` is the **reconstruction loss** the
    0.4 mm resampling imposes ([3.2] recon-IoU ceiling) — it separates "the linker/detector
    could not cover the lesion" (low iso_iou) from "the resampling stole the hit"
    (iso_iou > 0.3 but official_iou < 0.3).
  - ``is_tp`` = ``official_iou > IOU_HIT_THRESHOLD`` (== the FROC/labeling hit, Inv. 3).

The score-floor sweep answers the LUNA/NoduleSAT question directly: how far does a
per-candidate ``score_max`` floor shrink the pool, and at what recall cost.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C
from ..geometry import iou_official, storage_box_to_official
from ..link.tubes import Tube
from ..link.reconstruct import tube_to_iso_storage_box, iso_tube_to_official
from ..link.aggregate import score_stats, tube_geometry_stats

# per-candidate diagnostic columns (superset of the record; adds iso_iou + official size)
CAND_DIAG_COLUMNS = [
    "public_id",
    "score_max", "score_mean", "score_std", "score_min", "slice_count", "z_span", "fill_ratio",
    "centroid_jitter", "area_cv", "area_peak_pos", "area_monotonicity",
    "coordX", "coordY", "coordZ", "ext_x", "ext_y", "ext_z", "box_diag",
    "official_iou", "iso_iou", "recon_loss", "is_tp",
]


def build_candidate_frame(public_id: int, tubes: Sequence[Tube], gt_official,
                          gt_iso_official, meta) -> pd.DataFrame:
    """Per-candidate diagnostic rows for ONE volume (torch-free).

    ``gt_official`` = the official (cx,cy,cz,lx,ly,lz) GT box; ``gt_iso_official`` = the GT
    box mapped to ISO centre+extent (``storage_box_to_official(mask_to_box_storage(mask_iso))``).
    """
    rows: List[dict] = []
    for tube in tubes:
        s = score_stats(tube)
        g = tube_geometry_stats(tube)
        off = iso_tube_to_official(tube, meta)                       # official scoring box
        iso_off = storage_box_to_official(tube_to_iso_storage_box(tube))  # iso centre+extent
        official_iou = float(iou_official(off, gt_official))
        iso_iou = float(iou_official(iso_off, gt_iso_official))
        ext_x, ext_y, ext_z = float(off[3]), float(off[4]), float(off[5])
        rows.append({
            "public_id": int(public_id),
            **{k: s[k] for k in ["score_max", "score_mean", "score_std", "score_min",
                                 "slice_count", "z_span", "fill_ratio"]},
            **{k: g[k] for k in ["centroid_jitter", "area_cv", "area_peak_pos", "area_monotonicity"]},
            "coordX": float(off[0]), "coordY": float(off[1]), "coordZ": float(off[2]),
            "ext_x": ext_x, "ext_y": ext_y, "ext_z": ext_z,
            "box_diag": float(np.sqrt(ext_x ** 2 + ext_y ** 2 + ext_z ** 2)),
            "official_iou": official_iou, "iso_iou": iso_iou,
            "recon_loss": float(iso_iou - official_iou), "is_tp": bool(official_iou > C.IOU_HIT_THRESHOLD),
        })
    if not rows:
        return pd.DataFrame(columns=CAND_DIAG_COLUMNS)
    return pd.DataFrame(rows, columns=CAND_DIAG_COLUMNS)


def score_floor_sweep(frame: pd.DataFrame, floors: Sequence[float], n_vol: int) -> pd.DataFrame:
    """LUNA-style per-candidate score_max floor sweep: recall + pool cost per floor.

    For each floor ``f``: keep candidates with ``score_max >= f``; ``recall`` = fraction of
    the ``n_vol`` volumes still holding >= 1 kept TP; pool mean/max over volumes that keep
    any candidate (volumes reduced to 0 count as pool 0). Returns one row per floor.
    """
    out = []
    for f in floors:
        kept = frame[frame["score_max"] >= f]
        # recall = volumes with >=1 kept TP, over the FULL volume count
        tp_vols = kept[kept["is_tp"]]["public_id"].nunique()
        per_vol = kept.groupby("public_id").size()
        # volumes that lost all candidates contribute 0 to the pool distribution
        pools = np.zeros(n_vol, dtype=float)
        pools[:len(per_vol)] = np.sort(per_vol.to_numpy())[::-1] if len(per_vol) else 0.0
        out.append({
            "floor": float(f),
            "recall": tp_vols / n_vol if n_vol else float("nan"),
            "kept_total": int(len(kept)),
            "pool_mean": float(len(kept) / n_vol) if n_vol else float("nan"),
            "pool_max": int(per_vol.max()) if len(per_vol) else 0,
            "n_tp_kept": int(kept["is_tp"].sum()),
        })
    return pd.DataFrame(out)


def _pct(a: np.ndarray, qs=(0, 10, 25, 50, 75, 90, 100)) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"mean": float("nan"), "n": 0}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean()), "n": int(a.size)}


def tp_fp_split_stats(frame: pd.DataFrame, col: str) -> dict:
    """Percentile summary of ``col`` split by TP vs FP (the confidence/size separation)."""
    tp = frame[frame["is_tp"]][col].to_numpy()
    fp = frame[~frame["is_tp"]][col].to_numpy()
    return {"TP": _pct(tp), "FP": _pct(fp)}


def separability(frame: pd.DataFrame, col: str = "score_max") -> dict:
    """How separable are TP vs FP on ``col``. Reports the fraction of FP below the median
    TP value, and the best single-threshold TP-vs-FP balanced accuracy (a cheap AUC proxy)."""
    tp = np.sort(frame[frame["is_tp"]][col].to_numpy())
    fp = np.sort(frame[~frame["is_tp"]][col].to_numpy())
    if tp.size == 0 or fp.size == 0:
        return {"frac_fp_below_tp_median": float("nan"), "best_balacc": float("nan"),
                "best_thresh": float("nan"), "tp_min": float("nan")}
    tp_med = float(np.median(tp))
    frac_fp_below = float((fp < tp_med).mean())
    # best threshold by balanced accuracy over candidate thresholds
    cands = np.unique(np.concatenate([tp, fp]))
    best_ba, best_t = -1.0, float("nan")
    for t in cands:
        tpr = float((tp >= t).mean())
        tnr = float((fp < t).mean())
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return {"frac_fp_below_tp_median": frac_fp_below, "best_balacc": best_ba,
            "best_thresh": best_t, "tp_min": float(tp.min())}


def cluster_counts(centres: Sequence, radius: float) -> tuple:
    """(n_clusters, n_points, redundancy) via single-linkage at ``radius`` (iso voxels)."""
    from ..link.dedup import single_linkage_labels
    pts = np.asarray(centres, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return 0, 0, float("nan")
    labels = single_linkage_labels(pts, radius)
    n_clusters = int(len(set(labels.tolist())))
    return n_clusters, len(pts), float(len(pts) / max(n_clusters, 1))
