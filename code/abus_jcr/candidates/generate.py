"""Per-split candidate generation (Inv. 9, 10, 14).

TRAIN candidates are strictly out-of-fold: a Train volume in fold ``f`` is scored by
``retinanet_fold{f}.pt`` — the detector trained on ``manifest.fold != f``, which never
saw it (Inv. 10). VAL/TEST candidates come from each of the 3 full-train seed detectors
(``retinanet_full_seed{s}.pt``), producing 3 stacked pools distinguished by
``detector_of_origin`` — **never an ensemble** (Inv. 14). Test is code-complete and
importable here but executed only in Phase 5 (Inv. 9).

``detector_of_origin`` is bookkeeping only; it is never a Phase-4 model feature (Inv. 7).
The checkpoint-selection ``plan_generation`` is a pure, torch-free function so the OOF
routing is unit-tested without loading a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C
from ..preprocess import preprocess_hash
from ..detect.infer import run_detector_on_volume
from ..link.tubes import link_tubes
from ..link.reconstruct import iso_tube_to_official, iso_centre_of_tube, iso_extents_of_tube
from ..link.aggregate import score_stats, within_volume_rank, label_candidate
from .record import CANDIDATE_COLUMNS


def plan_generation(manifest: pd.DataFrame, split: str, checkpoints_dir) -> List[dict]:
    """Torch-free routing: which checkpoint scores which volumes (Inv. 10, 14).

    Returns a list of ``{detector_of_origin, checkpoint, volume_ids, split, fold_of}``
    jobs. ``split`` is the canonical ``train``/``val``/``test``.

    - **train**: one job per fold ``f`` present in the Train volumes, scoring exactly
      the fold-``f`` volumes with ``retinanet_fold{f}.pt`` (OOF). ``fold_of`` maps each
      volume to its fold (== f for that job).
    - **val**/**test**: one job per seed in ``DET_FULL_SEEDS``, each scoring **all** the
      split's volumes with ``retinanet_full_seed{s}.pt``; ``fold_of`` is ``-1`` for all.
    """
    checkpoints_dir = Path(checkpoints_dir)
    sel = manifest[manifest["split"] == split]
    jobs: List[dict] = []
    if split == "train":
        for f in sorted(int(x) for x in sel["fold"].unique()):
            vids = sorted(int(v) for v in sel[sel["fold"] == f]["volume_id"])
            jobs.append({
                "detector_of_origin": f"fold{f}",
                "checkpoint": checkpoints_dir / f"retinanet_fold{f}.pt",
                "volume_ids": vids,
                "split": split,
                "fold_of": {v: f for v in vids},
            })
    elif split in ("val", "test"):
        vids = sorted(int(v) for v in sel["volume_id"])
        for s in C.DET_FULL_SEEDS:
            jobs.append({
                "detector_of_origin": f"full_seed{s}",
                "checkpoint": checkpoints_dir / f"retinanet_full_seed{s}.pt",
                "volume_ids": vids,
                "split": split,
                "fold_of": {v: -1 for v in vids},
            })
    else:
        raise ValueError(f"unknown split {split!r}")
    return jobs


def generate_volume_candidates(
    model, cache_root, volume_id: int, meta: dict, gt_official,
    *, detector_of_origin: str, split: str, fold: int, op_score_thresh: float,
    local_idx0: int, detect_fn: Callable = run_detector_on_volume,
) -> pd.DataFrame:
    """Detections -> tubes -> per-tube candidate rows for ONE volume, ONE detector.

    ``detect_fn(model, cache_root, volume_id, score_thresh, nms_thresh,
    detections_per_img)`` yields the per-slice schema frame at the recall-saturating
    operating point with loosened NMS (Inv. 2). Each surviving tube becomes a row with
    the official scoring box, the frozen score-stats, iso centre/extents, and the
    ignore-band label vs ``gt_official``. ``rank``/``rank_norm`` are placeholders here
    (finalised per (detector_of_origin, public_id) by :func:`generate_split`).

    ``candidate_id = f"{detector_of_origin}:{volume_id}:{local_idx}"`` with
    ``local_idx`` starting at ``local_idx0``. Returns a ``CANDIDATE_COLUMNS`` frame.
    """
    det_df = detect_fn(
        model, cache_root, int(volume_id),
        score_thresh=float(op_score_thresh),
        nms_thresh=C.LINK_NMS_THRESH,
        detections_per_img=C.LINK_DETECTIONS_PER_IMG,
    )
    tubes = link_tubes(det_df)
    phash = preprocess_hash()

    rows: List[dict] = []
    local_idx = int(local_idx0)
    for tube in tubes:
        stats = score_stats(tube)
        if C.PREFILTER_SCORE_FLOOR > 0.0 and stats["score_max"] < C.PREFILTER_SCORE_FLOOR:
            continue  # NoduleSAT-style pool floor; recorded recall cost when raised
        official = iso_tube_to_official(tube, meta)          # (coordX,coordY,coordZ,x_len,y_len,z_len)
        cen = iso_centre_of_tube(tube)
        ext = iso_extents_of_tube(tube)
        label, iou_gt = label_candidate(official, gt_official)
        rows.append({
            "public_id": int(volume_id),
            "candidate_id": f"{detector_of_origin}:{int(volume_id)}:{local_idx}",
            "detector_of_origin": detector_of_origin,
            "split": split,
            "fold": int(fold),
            "coordX": official[0], "coordY": official[1], "coordZ": official[2],
            "x_length": official[3], "y_length": official[4], "z_length": official[5],
            "score_max": stats["score_max"], "score_mean": stats["score_mean"],
            "score_std": stats["score_std"], "score_min": stats["score_min"],
            "slice_count": stats["slice_count"], "z_span": stats["z_span"],
            "fill_ratio": stats["fill_ratio"],
            "rank": 0, "rank_norm": 0.0,                     # finalised by generate_split
            "label": label, "iou_gt": iou_gt,
            "cen_d0": cen[0], "cen_d1": cen[1], "cen_d2": cen[2],
            "ext_d0": ext[0], "ext_d1": ext[1], "ext_d2": ext[2],
            "preprocess_hash": phash,
        })
        local_idx += 1

    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _rank_within_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`within_volume_rank` independently per ``detector_of_origin``.

    So a Val volume's 3 seed pools are ranked separately (rank 1 = that seed's
    top candidate for that volume), never mixed across seeds.
    """
    if len(df) == 0:
        return within_volume_rank(df)
    parts = []
    for _, grp in df.groupby("detector_of_origin", sort=False):
        ranked = within_volume_rank(grp.drop(columns=["rank", "rank_norm"]))
        parts.append(ranked)
    out = pd.concat(parts, ignore_index=True)
    return out[CANDIDATE_COLUMNS]


def generate_split(
    manifest: pd.DataFrame, cache_root, checkpoints_dir, split: str, gt_gt_df: pd.DataFrame,
    *, op_score_thresh: float = C.LINK_OP_SCORE_THRESH,
    load_checkpoint_fn: Optional[Callable] = None,
    detect_fn: Callable = run_detector_on_volume,
    read_meta_fn: Optional[Callable] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Generate the full candidate pool for ``split`` (Inv. 9, 10, 14).

    Routes checkpoints via :func:`plan_generation`, loads each once, scores its
    volumes, then applies within-(detector, volume) rank and the ignore-band label.
    ``gt_gt_df`` is the official GT table (columns ``GT_COLUMNS``), indexed by
    ``public_id`` to fetch each volume's single scoring box. Returns a
    ``CANDIDATE_COLUMNS`` frame (Val = 3 seed pools stacked).
    """
    from .. import cache as K
    from ..detect.retinanet import load_checkpoint as _load_ckpt

    load_checkpoint_fn = load_checkpoint_fn or _load_ckpt
    read_meta_fn = read_meta_fn or K.read_meta

    gt_idx = gt_gt_df.set_index("public_id")
    jobs = plan_generation(manifest, split, checkpoints_dir)

    all_rows: List[pd.DataFrame] = []
    for job in jobs:
        model, _cfg = load_checkpoint_fn(job["checkpoint"])
        local_idx = 0
        for vid in job["volume_ids"]:
            meta = read_meta_fn(cache_root, int(vid))
            row = gt_idx.loc[int(vid)]
            gt_official = (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]),
                           float(row["x_length"]), float(row["y_length"]), float(row["z_length"]))
            df_v = generate_volume_candidates(
                model, cache_root, int(vid), meta, gt_official,
                detector_of_origin=job["detector_of_origin"], split=split,
                fold=job["fold_of"][int(vid)], op_score_thresh=op_score_thresh,
                local_idx0=local_idx, detect_fn=detect_fn,
            )
            local_idx += len(df_v)
            all_rows.append(df_v)
            if progress:
                print(f"  {job['detector_of_origin']} vol {vid}: {len(df_v)} candidates")

    pool = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return _rank_within_detector(pool)
