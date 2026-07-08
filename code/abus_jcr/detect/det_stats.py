"""The ``[2.0]`` Train design-constant probe + the pinned derivation rule (Inv. 9).

The single place the data-dependent detector constants (input size, intensity
normalisation, anchors) are decided — **on the Train split, in iso space, before
any training**. ``derive_constants`` is the deterministic, unit-pinned rule;
``probe_train_stats`` gathers the raw Train statistics it consumes.

Torch-free: pure numpy/pandas, so the rule runs in the laptop env and is verified
data-independently by ``tests/test_det_stats_rule.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C
from .. import cache as K

# percentiles reported for box size / diagonal / aspect distributions.
_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def round_up(x: float, m: int) -> int:
    """Smallest multiple of ``m`` that is >= ``x`` (``m > 0``)."""
    return int(math.ceil(float(x) / m) * m)


def _snap_to_grid(value: float, grid: Sequence[float]) -> float:
    """Nearest grid entry to ``value`` (ties -> the smaller entry)."""
    g = np.asarray(sorted(grid), dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(value))))])


def derive_constants(stats: Dict, rule: Dict = C.DET_RULE) -> Dict:
    """The pinned rule: Train iso-space ``stats`` -> the 6 data-dependent constants.

    - ``min_size = round_up(frame_d0_max, min_size_round)``,
      ``max_size = round_up(frame_d1_max, max_size_round)``.
    - ``image_mean = intensity_mean``, ``image_std = intensity_std``.
    - anchors: ``lo = diag_pct[anchor_diag_lo_pct]``, ``hi =
      diag_pct[anchor_diag_hi_pct]``. Smallest base ``b0 = 2**round(log2(lo))``;
      place ``anchor_n_levels`` bases geometric with ratio 2. If the top base does
      not satisfy ``base_max * 2**(2/3) >= hi``, shift the whole (ratio-2,
      power-of-two) ladder up by whole octaves until it does.
    - aspect ratios: the ``h/w`` values at ``aspect_pcts``, each snapped to the
      nearest ``aspect_grid`` entry, unioned with ``1.0``, deduped, sorted.
    """
    min_size = round_up(stats["frame_d0_max"], rule["min_size_round"])
    max_size = round_up(stats["frame_d1_max"], rule["max_size_round"])

    diag = stats["diag_pct"]
    lo = float(diag[str(rule["anchor_diag_lo_pct"])])
    hi = float(diag[str(rule["anchor_diag_hi_pct"])])
    n = int(rule["anchor_n_levels"])

    b0 = 2 ** int(round(math.log2(lo)))
    bases = [b0 * (2 ** i) for i in range(n)]
    top_needed = hi / (2 ** (2 / 3))
    # grow the ladder up whole octaves until the largest anchor covers hi.
    while bases[-1] < top_needed:
        b0 *= 2
        bases = [b0 * (2 ** i) for i in range(n)]
    anchor_base_sizes = tuple(int(round(b)) for b in bases)

    asp = stats["aspect_pct"]
    snapped = {_snap_to_grid(asp[str(p)], rule["aspect_grid"]) for p in rule["aspect_pcts"]}
    snapped.add(1.0)
    anchor_aspect_ratios = tuple(sorted(snapped))

    return {
        "min_size": min_size,
        "max_size": max_size,
        "image_mean": stats["intensity_mean"],
        "image_std": stats["intensity_std"],
        "anchor_base_sizes": anchor_base_sizes,
        "anchor_aspect_ratios": anchor_aspect_ratios,
    }


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {str(p): float("nan") for p in _PCTS}
    q = np.percentile(values, _PCTS)
    return {str(p): float(v) for p, v in zip(_PCTS, q)}


def probe_train_stats(
    cache_root,
    manifest: pd.DataFrame,
    slice_boxes_train_df: pd.DataFrame,
    rule: Dict = C.DET_RULE,
) -> Dict:
    """Compute the ``[2.0]`` Train statistics directly in iso space (no Val, no zoom proxy).

    Reads **Train** volumes only (``manifest.split == "train"``) and reports:
    frame ``(d0, d1)`` min/median/max from cache meta ``iso_shape``; lesion box
    ``h = r1-r0+1``, ``w = c1-c0+1``, ``diag = hypot(h, w)``, ``aspect = h/w``
    percentiles; global intensity mean/std over a seeded slice sample; and
    components-per-slice max & fraction ``> 1``. Also embeds
    ``derive_constants(stats)`` under ``"derived"`` for the reconciliation gate.
    """
    train_ids = sorted(int(v) for v in manifest.loc[manifest["split"] == "train", "volume_id"])

    # --- frame sizes (iso d0, d1) from cache meta ---
    d0s, d1s = [], []
    n_slices = {}
    for vid in train_ids:
        meta = K.read_meta(cache_root, vid)
        d0, d1, d2 = meta["iso_shape"]
        d0s.append(int(d0)); d1s.append(int(d1))
        n_slices[vid] = int(meta["iso_shape"][C.SLICE_AXIS])
    d0s = np.asarray(d0s); d1s = np.asarray(d1s)

    # --- lesion box sizes from slice_boxes_Train (inclusive iso voxels) ---
    df = slice_boxes_train_df
    df = df[df["volume_id"].isin(train_ids)]
    h = (df["r1"].to_numpy() - df["r0"].to_numpy() + 1).astype(np.float64)
    w = (df["c1"].to_numpy() - df["c0"].to_numpy() + 1).astype(np.float64)
    diag = np.hypot(h, w)
    aspect = h / w

    # --- components-per-slice ---
    per_slice = df.groupby(["volume_id", "slice_z"]).size().to_numpy() if len(df) else np.array([])
    comp_max = int(per_slice.max()) if per_slice.size else 0
    comp_frac_gt1 = float((per_slice > 1).mean()) if per_slice.size else 0.0

    # --- intensity mean/std over a seeded sample of Train iso slices ---
    rng = np.random.default_rng(int(rule["intensity_seed"]))
    n_sample = int(rule["intensity_sample_slices"])
    picks = []
    for _ in range(n_sample):
        vid = int(rng.choice(train_ids))
        z = int(rng.integers(0, n_slices[vid]))
        picks.append((vid, z))
    # accumulate mean/std in one pass over the sampled slices
    tot = 0.0; tot_sq = 0.0; count = 0
    for vid, z in picks:
        vol = K.open_vol(cache_root, vid)
        frame = np.asarray(np.take(vol, z, axis=C.SLICE_AXIS), dtype=np.float64)
        tot += frame.sum(); tot_sq += np.square(frame).sum(); count += frame.size
    intensity_mean = float(tot / count) if count else float("nan")
    intensity_std = float(math.sqrt(max(tot_sq / count - intensity_mean ** 2, 0.0))) if count else float("nan")

    stats = {
        "n_train_volumes": len(train_ids),
        "frame_d0_min": int(d0s.min()), "frame_d0_median": float(np.median(d0s)), "frame_d0_max": int(d0s.max()),
        "frame_d1_min": int(d1s.min()), "frame_d1_median": float(np.median(d1s)), "frame_d1_max": int(d1s.max()),
        "n_boxes": int(len(df)),
        "box_h_pct": _percentiles(h),
        "box_w_pct": _percentiles(w),
        "diag_pct": _percentiles(diag),
        "aspect_pct": _percentiles(aspect),
        "intensity_mean": intensity_mean,
        "intensity_std": intensity_std,
        "intensity_n_slices": len(picks),
        "components_per_slice_max": comp_max,
        "components_per_slice_frac_gt1": comp_frac_gt1,
    }
    stats["derived"] = derive_constants(stats, rule)
    return stats


def write_stats(stats: Dict, out_dir) -> Path:
    """Persist the probe output to ``<out_dir>/train_det_stats.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train_det_stats.json"
    path.write_text(json.dumps(stats, sort_keys=True, indent=2))
    return path
