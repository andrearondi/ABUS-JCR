"""Shared path resolution + linking helpers for the Phase-3 scripts.

Phase 3 consumes the Phase-1 substrate (iso cache, manifest) under ``--phase1-out``
and the Phase-2 checkpoints under ``--phase2-out``; it writes its own artefacts under
``--out-root``. Official GT boxes come from each split's shipped ``bbx_labels.csv``.
Paths default to SERVER_LAYOUT.md; override for local runs.

The linked-recall helper is the calibration crux and is torch-free (operates on
already-computed detection frames), so [3.3] freeze-linking and [3.4] operating-point
sweeps reuse the SAME code — a param sweep never re-runs the detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.gt_labels import load_gt_documented, to_official_gt
from abus_jcr.geometry import iou_official
from abus_jcr.detect import schema as S
from abus_jcr.link.tubes import link_tubes
from abus_jcr.link.reconstruct import iso_tube_to_official
from abus_jcr.link.aggregate import score_stats
from abus_jcr.link.nms import reduce_pool_3dnms

DEFAULT_PHASE1_OUT = "/home/maia-user/Andre2/outputs/phase1"
DEFAULT_PHASE2_OUT = "/home/maia-user/Andre2/outputs/phase2"
DEFAULT_PHASE3_OUT = "/home/maia-user/Andre2/outputs/phase3"
DEFAULT_DATA_ROOT = "/home/maia-user/Andre2/data"

_SPLIT_DIR = {"train": "Train", "val": "Validation", "test": "Test"}


def add_phase3_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase1-out", default=DEFAULT_PHASE1_OUT,
                        help=f"Phase-1 output root (cache/, manifest.csv); default {DEFAULT_PHASE1_OUT}")
    parser.add_argument("--phase2-out", default=DEFAULT_PHASE2_OUT,
                        help=f"Phase-2 output root (checkpoints/); default {DEFAULT_PHASE2_OUT}")
    parser.add_argument("--out-root", default=DEFAULT_PHASE3_OUT,
                        help=f"Phase-3 output root; default {DEFAULT_PHASE3_OUT}")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"dataset root holding the split dirs; default {DEFAULT_DATA_ROOT}")


_PROFILE_ANNOUNCED = False


def profile_banner(cache_dir_path: Path | None = None) -> None:
    """Print the axis-profile banner once per process (idempotent).

    Every Phase-3 artefact is substrate-specific under Inv. 6: a frozen linker, a
    candidate pool and a B0' number mean different things on ``legacy`` and on
    ``measured``. So each Phase-3 stdout has to name the substrate that produced it —
    an output must be attributable from its own log, not from whichever shell happened
    to run it. Mirrors the banner ``phase2_train_stats.py`` / ``phase2_train_retinanet.py``
    already print (``iso/RB_ISO_REBUILD.md`` Trap 1).
    """
    global _PROFILE_ANNOUNCED
    if _PROFILE_ANNOUNCED:
        return
    _PROFILE_ANNOUNCED = True
    print(f"# axis profile = {C.AXIS_PROFILE} | spacing_storage_mm = {C.SPACING_STORAGE_MM}")
    if cache_dir_path is not None:
        print(f"# iso cache    = {cache_dir_path}")


def cache_root(args) -> Path:
    """Resolve the iso cache root, banner the profile, and REFUSE a foreign cache.

    The guard lives HERE rather than in each caller so that a Phase-3 script cannot
    forget it — every script that reads the cache goes through this function, and a new
    one gets the guard by construction. Without it a missing
    ``export ABUS_AXIS_PROFILE=measured`` surfaced as a ``FileNotFoundError`` on a
    ``META_*.json`` deep in the call stack, which states the symptom and not the cause.
    """
    root = Path(args.phase1_out) / "cache"
    from abus_jcr import cache as K
    profile_banner(K.cache_dir(root))
    try:
        K.assert_hash(root)
    except K.CacheHashMismatch as exc:
        raise K.CacheHashMismatch(
            f"{exc}\n"
            f"  axis profile in this shell : {C.AXIS_PROFILE}\n"
            f"  cache dir it resolved to   : {K.cache_dir(root)}\n"
            f"  HINT: the two profiles name different cache dirs (legacy ab1fdf28…, "
            f"measured 3f94f17f…). If you meant the isotropic substrate, run "
            f"`export ABUS_AXIS_PROFILE=measured` and retry; if you meant the deployed "
            f"one, point --phase1-out at outputs/phase1.") from exc
    return root


def checkpoints_dir(args) -> Path:
    return Path(args.phase2_out) / "checkpoints"


def load_manifest(args) -> pd.DataFrame:
    return pd.read_csv(Path(args.phase1_out) / "manifest.csv")


def split_root(args, split: str) -> Path:
    return Path(args.data_root) / _SPLIT_DIR[split]


def load_official_gt(args, split: str) -> pd.DataFrame:
    """Official GT box table (``GT_COLUMNS``) for a split, from its ``bbx_labels.csv``."""
    return to_official_gt(load_gt_documented(split_root(args, split) / "bbx_labels.csv"))


def gt_official_tuple(gt_idx: pd.DataFrame, vid: int) -> Tuple[float, ...]:
    """The single official scoring box for ``vid`` as a 6-tuple (from a public_id-indexed GT)."""
    r = gt_idx.loc[int(vid)]
    return (float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]),
            float(r["x_length"]), float(r["y_length"]), float(r["z_length"]))


def assert_device(device: str) -> None:
    """Fail loudly if CUDA is requested but unavailable (reused Phase-2 guard)."""
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"--device {device} requested but torch is not installed ({e}). "
                         "Activate the abus-jcr env on the GPU host.")
    if not torch.cuda.is_available():
        raise SystemExit(
            f"--device {device} requested but torch reports no CUDA GPU "
            f"(is_available=False, device_count={torch.cuda.device_count()}). "
            "Move to the A6000 GPU host and re-check (the linking/aggregation is CPU-only, "
            "but candidate generation runs the detector on the GPU).")


def linked_recall(
    det_by_vid: Dict[int, pd.DataFrame],
    gt_by_vid: Dict[int, Tuple[float, ...]],
    meta_by_vid: Dict[int, dict],
    *,
    link_iou: float = C.LINK_IOU,
    max_z_gap: int = C.LINK_MAX_Z_GAP,
    min_tube_len: int = C.LINK_MIN_TUBE_LEN,
    max_tube_zspan=C.LINK_MAX_TUBE_ZSPAN,
    max_centroid_drift=C.LINK_MAX_CENTROID_DRIFT,
    containment_thresh: float = C.LINK_CONTAINMENT_THRESH,
    nms_iou=C.LINK_3DNMS_IOU,
    score_floor: float = C.PREFILTER_SCORE_FLOOR,
    hit_iou: float = C.IOU_HIT_THRESHOLD,
) -> Dict:
    """Linked 3D recall + pool size over a set of volumes (torch-free; the sweep crux).

    For each volume: link its (already-computed) detections into tubes, reconstruct each
    tube to an official box, apply the [P3U2 3.C] membership-only 3D-NMS reduction
    (``nms_iou``; None -> keep all), and count the volume HIT iff any SURVIVING candidate
    reaches ``iou_official > hit_iou`` (== the FROC/labeling hit rule, Inv. 3). Returns
    ``{recall, n_hit, n_vol, cands_per_vol_mean, cands_per_vol_median, pool_total}`` on
    the deployed (post-NMS) pool. Passes the P3-UPDATE drift caps + containment + the
    3D-NMS IoU through so [3.3']/[P3U2.7]/the reducer gate can sweep them.
    """
    n_vol = len(det_by_vid)
    n_hit = 0
    pool_sizes: List[int] = []
    for vid, det_df in det_by_vid.items():
        tubes = link_tubes(det_df, link_iou=link_iou, max_z_gap=max_z_gap,
                           min_tube_len=min_tube_len, max_tube_zspan=max_tube_zspan,
                           max_centroid_drift=max_centroid_drift,
                           containment_thresh=containment_thresh)
        gt = gt_by_vid[int(vid)]
        meta = meta_by_vid[int(vid)]
        offs = [iso_tube_to_official(t, meta) for t in tubes]
        scs = [float(score_stats(t)["score_max"]) for t in tubes]
        # [P3U2] LUNA-style per-candidate score_max floor BEFORE the 3D-NMS reduction (deployed-pool order:
        # generate applies PREFILTER_SCORE_FLOOR then 3D NMS, so recall/pool here match candidate generation).
        if score_floor > 0.0:
            keep0 = [i for i, sc in enumerate(scs) if sc >= score_floor]
            offs = [offs[i] for i in keep0]; scs = [scs[i] for i in keep0]
        kept = reduce_pool_3dnms(offs, scs, iou_thr=nms_iou)
        pool_sizes.append(len(kept))
        hit = any(iou_official(offs[i], gt) > hit_iou for i in kept)
        n_hit += int(hit)
    pool = np.asarray(pool_sizes, dtype=float) if pool_sizes else np.zeros(0)
    return {
        "recall": (n_hit / n_vol) if n_vol else float("nan"),
        "n_hit": n_hit,
        "n_vol": n_vol,
        "cands_per_vol_mean": float(pool.mean()) if pool.size else float("nan"),
        "cands_per_vol_median": float(np.median(pool)) if pool.size else float("nan"),
        "pool_total": int(pool.sum()) if pool.size else 0,
    }


def monotonicity_violations(threshs: Sequence[float], recalls: Sequence[float],
                            tol: float = 1e-9, up_to_peak: bool = False) -> List[dict]:
    """[P3-UPDATE L2 / P3U2 A2-refinement] Detect non-monotone linked recall.

    A sound aggregation is a superset relation: LOWERING ``op_score_thresh`` adds detections,
    which can only keep or raise linked recall. Sorting the sweep by DESCENDING threshold,
    recall must be non-decreasing. Returns the adjacent (higher->lower threshold) steps where
    recall DROPPED by more than ``tol`` — empty iff monotone. Torch-free.

    ``up_to_peak=True`` restricts the check to the **deployment region** — from the highest
    threshold DOWN TO the recall-peak threshold (the lowest threshold still at max recall).
    BELOW the recall peak the pool floods and the greedy consume-once linker sheds TPs by
    construction; that region is never a deployed operating point (the op is chosen at recall
    saturation, A2), so its non-monotonicity is benign and must not trip the gate. Above/at the
    peak, a drop is a real unsound-linker regression and IS flagged. [P3U2.8].
    """
    idx = sorted(range(len(threshs)), key=lambda i: -float(threshs[i]))  # descending threshold
    if up_to_peak and idx:
        recs = [float(recalls[i]) for i in idx]
        peak = max(recs)
        # last position (lowest threshold) still at peak recall -> include the whole peak plateau
        peak_pos = max(k for k in range(len(recs)) if recs[k] >= peak - tol)
        idx = idx[:peak_pos + 1]
    out: List[dict] = []
    for a, b in zip(idx, idx[1:]):   # a = higher thresh, b = next-lower thresh (more boxes)
        if float(recalls[b]) < float(recalls[a]) - tol:
            out.append({"thresh_hi": float(threshs[a]), "thresh_lo": float(threshs[b]),
                        "recall_hi": float(recalls[a]), "recall_lo": float(recalls[b]),
                        "drop": float(recalls[a]) - float(recalls[b])})
    return out


def derive_link_caps(slice_boxes_train: pd.DataFrame,
                     zspan_safety: float = 1.8, drift_safety: float = 1.5) -> dict:
    """[P3-UPDATE L1] Derive the Train-GT tube drift caps (iso space; no leakage, Inv. 4).

    Both caps are computed from the Phase-1 Train union GT boxes (already iso-space, so directly
    comparable to a tube's z-span and box-centre drift):
      - ``LINK_MAX_TUBE_ZSPAN``     = round(``zspan_safety`` * p99 of per-volume lesion z-extent
        in iso slices) — a lesion cannot span more slices than this.
      - ``LINK_MAX_CENTROID_DRIFT`` = round(``drift_safety`` * p99 of per-box in-plane extent
        max(width,height) in iso px) — a tube's boxes cannot wander farther than a lesion's own
        in-plane size.
    Single-lesion dominance (99/100 Train) makes the per-volume z-extent exact for all but the
    multifocal case; the safety factor absorbs it. Returns the derived ints + the percentiles.
    """
    df = slice_boxes_train
    zspans = []
    for _vid, grp in df.groupby("volume_id", sort=False):
        zs = grp["slice_z"].to_numpy()
        zspans.append(int(zs.max() - zs.min() + 1))
    widths = (df["c1"].to_numpy() - df["c0"].to_numpy() + 1)
    heights = (df["r1"].to_numpy() - df["r0"].to_numpy() + 1)
    inplane = np.maximum(widths, heights).astype(float)
    z_p99 = float(np.percentile(np.asarray(zspans, dtype=float), 99)) if zspans else float("nan")
    e_p99 = float(np.percentile(inplane, 99)) if len(inplane) else float("nan")
    return {
        "zspan_p99": z_p99, "inplane_extent_p99": e_p99,
        "LINK_MAX_TUBE_ZSPAN": int(round(zspan_safety * z_p99)),
        "LINK_MAX_CENTROID_DRIFT": int(round(drift_safety * e_p99)),
        "zspan_safety": zspan_safety, "drift_safety": drift_safety,
    }


def det_cache_path(out_root, tag: str, vid: int) -> Path:
    """Per-volume detection-cache path: ``<out_root>/detections_cache/<tag>/det_<vid>``."""
    return Path(out_root) / "detections_cache" / tag / f"det_{int(vid)}"


def load_or_run_detections(out_root, tag: str, vid: int, model, croot, op_score_thresh: float,
                           device, use_cache: bool = True) -> pd.DataFrame:
    """Read cached per-volume detections if present, else run the detector and cache them.

    The cache key ``tag`` must encode the detector identity + operating point (e.g.
    ``fold0_op0.05``), so a filtered/re-run never mixes regimes. Makes [3.3]/[3.4]/[3.5]
    resumable across disconnects and lets a re-run skip already-scored volumes. Pass
    ``use_cache=False`` to force recomputation.
    """
    p = det_cache_path(out_root, tag, vid)
    if use_cache and (p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists()):
        return S.read_detections(p)
    from abus_jcr.detect.infer import run_detector_on_volume
    df = run_detector_on_volume(
        model, croot, int(vid), score_thresh=float(op_score_thresh),
        nms_thresh=C.LINK_NMS_THRESH, detections_per_img=C.LINK_DETECTIONS_PER_IMG, device=device)
    if use_cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        S.write_detections(df, p)
    return df


def filter_by_score(det_df: pd.DataFrame, score_thresh: float) -> pd.DataFrame:
    """Rows with ``score >= score_thresh`` — reproduces a higher-threshold detector run.

    Running the detector once at the sweep minimum and filtering upward is exact for
    the frozen NMS regime: an extra low-score box can neither win NMS against nor
    displace a higher-score box, and the per-slice top-K cap keeps the highest scores
    first. Documented in RB_PHASE_3 [3.4].
    """
    return det_df[det_df["score"] >= float(score_thresh)]
