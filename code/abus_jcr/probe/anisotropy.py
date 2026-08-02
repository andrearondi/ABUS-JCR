"""Candidate elongation about each axis, in physically correct millimetres.

Why this exists. The deployed feature is

    anisotropy = ext_d0 / mean(ext_d1, ext_d2)          (rescore/tokens._block_abs_geom)

on **iso-cache voxel** extents. Two things make that not what its name says:

1. The cache was built with ``preprocess.zoom_factors``, which scales axis ``a`` by
   ``SPACING_STORAGE_MM[a] / ISO_SPACING_MM`` using the DECLARED spacing. If the declared
   spacing does not match the physical one, a cache voxel is not 0.4 mm cubed — it spans
   ``ISO_SPACING_MM * true[a] / declared[a]`` mm, which differs per axis.
2. ``d0`` is not the beam axis (``results/AXIS_CHECK.md``), so the ratio measures elongation
   about **lateral**, not about depth.

So the near-null effect sizes recorded for ``anisotropy`` in every pool-diagnostics table
are not evidence that depth-elongated candidates are absent — that ratio was never computed.
This module computes all three, in true millimetres, from the frozen record alone.

**What this can and cannot license.** It is a *diagnostic*, not a feature proposal. The
``abs_geom`` block already carries ``log1p(ext_d0), log1p(ext_d1), log1p(ext_d2)``, so every
ratio of extents is a difference of logs and is linearly recoverable from what the set model
already receives. A "corrected" anisotropy scalar would therefore add no information. What
these numbers answer is a different question: does elongation about the beam axis separate
TP from FP *at all* — i.e. is there a ray-shaped-FP population in the pool?
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C

#: Storage-order official/ITK length columns: ``z_length`` is d0, ``x_length`` is d2.
_ITK_EXT = ("z_length", "y_length", "x_length")

#: The three elongation ratios, by the storage axis each is measured ABOUT.
RATIO_NAMES = ("elong_d0", "elong_d1", "elong_d2")


def iso_voxel_mm(true_spacing: Sequence[float],
                 declared_spacing: Sequence[float] = None,
                 iso_spacing: float = None) -> np.ndarray:
    """Physical size (mm) of one cached voxel along each storage axis.

    ``preprocess`` resampled axis ``a`` by ``declared[a] / iso``, so the cache has
    ``n_a * declared[a] / iso`` samples spanning the true ``n_a * true[a]`` mm. One cache
    voxel therefore spans ``iso * true[a] / declared[a]`` mm — equal to ``iso`` on every
    axis only when the declared spacing is right.
    """
    declared = C.SPACING_STORAGE_MM if declared_spacing is None else declared_spacing
    iso = C.ISO_SPACING_MM if iso_spacing is None else iso_spacing
    t = np.asarray(true_spacing, dtype=np.float64)
    d = np.asarray(declared, dtype=np.float64)
    return float(iso) * t / d


def deployed_anisotropy(df: pd.DataFrame) -> np.ndarray:
    """``ext_d0 / mean(ext_d1, ext_d2)`` on iso voxels — byte-identical to the shipped feature.

    Reproduced here (rather than imported) so this module can be read on its own and so a
    future edit to the token block shows up as a disagreement rather than silently.
    """
    ext = df[["ext_d0", "ext_d1", "ext_d2"]].to_numpy(float)
    lat = (ext[:, 1] + ext[:, 2]) / 2.0
    return np.divide(ext[:, 0], lat, out=np.full(len(ext), np.nan), where=lat > 0)


def cubic_reference(iso_mm: Sequence[float]) -> float:
    """What the DEPLOYED feature reads for a physically cubic candidate.

    1.0 only if the cache is truly isotropic. Anything else means the recorded medians have
    been read against the wrong origin: a value below 1 does not imply "compressed".
    """
    v = 1.0 / np.asarray(iso_mm, dtype=np.float64)   # voxel counts of a 1 mm cube
    return float(v[0] / ((v[1] + v[2]) / 2.0))


def extents_mm(df: pd.DataFrame, true_spacing: Sequence[float],
               iso_mm: Sequence[float] = None) -> Dict[str, np.ndarray]:
    """Per-candidate box extents in true mm, computed TWO independent ways.

    - ``native``: the official scoring box lengths x/y/z times the true native spacing.
    - ``iso``: the cached iso extents times the true cache-voxel size.

    They must agree up to tube-reconstruction rounding. Returning both — and their
    disagreement — makes a coordinate bug visible instead of averaging it away.
    """
    t = np.asarray(true_spacing, dtype=np.float64)
    native = np.stack([np.asarray(df[c], dtype=np.float64) for c in _ITK_EXT], axis=1) * t[None, :]
    out = {"native_mm": native}
    if iso_mm is not None:
        iso_ext = df[["ext_d0", "ext_d1", "ext_d2"]].to_numpy(float)
        out["iso_mm"] = iso_ext * np.asarray(iso_mm, dtype=np.float64)[None, :]
    return out


def elongation_ratios(ext_mm: np.ndarray) -> Dict[str, np.ndarray]:
    """Elongation about each axis: ``ext[a] / mean(the other two)``, in mm.

    A physically cubic box scores 1.0 on all three, so the three ratios are directly
    comparable to each other and across candidates. ``elong`` about the beam axis is the
    quantity a posterior-shadow "ray" would push far above 1.
    """
    ext_mm = np.asarray(ext_mm, dtype=np.float64)
    out = {}
    for a in range(3):
        others = [b for b in range(3) if b != a]
        den = (ext_mm[:, others[0]] + ext_mm[:, others[1]]) / 2.0
        out[RATIO_NAMES[a]] = np.divide(ext_mm[:, a], den,
                                        out=np.full(len(ext_mm), np.nan), where=den > 0)
    return out


def ray_fractions(elong_depth: np.ndarray, is_tp: np.ndarray,
                  thresholds: Sequence[float] = (1.0, 1.5, 2.0)) -> Dict[str, float]:
    """Fraction of candidates that are actually elongated ALONG the beam, by class.

    A median tells you where the bulk sits; it cannot tell you whether a ray-shaped
    sub-population exists. A posterior-shadow ray is a box much longer in depth than
    across it, so it must clear ``elong_depth > 1`` by a margin. If both classes are
    almost entirely below 1, there is no ray population to explain — whatever the medians
    do relative to each other.
    """
    out: Dict[str, float] = {}
    e = np.asarray(elong_depth, dtype=np.float64)
    ok = np.isfinite(e)
    for t in thresholds:
        for tag, m in (("tp", is_tp & ok), ("fp", (~is_tp) & ok)):
            out[f"frac_{tag}_gt{t:g}"] = float((e[m] > t).mean()) if m.any() else float("nan")
    return out


def per_volume_delta(values: np.ndarray, is_tp: np.ndarray, vol_ids: np.ndarray,
                     min_per_group: int = 3) -> Tuple[float, float, int]:
    """Median within-volume Cliff's delta (TP vs FP), its sign consistency, and n volumes.

    Pooling every candidate once would overstate precision: the split is single-lesion
    dominant, so a volume's TPs are many redundant tubes on ONE object. Volumes with fewer
    than ``min_per_group`` of either class are skipped rather than contributing a delta
    computed on one or two points.
    """
    from .intensity_geom import cliffs_delta
    deltas = []
    for v in np.unique(vol_ids):
        m = vol_ids == v
        a, b = values[m & is_tp], values[m & ~is_tp]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) >= min_per_group and len(b) >= min_per_group:
            deltas.append(cliffs_delta(a, b))
    if not deltas:
        return float("nan"), float("nan"), 0
    d = np.asarray(deltas, dtype=np.float64)
    med = float(np.median(d))
    sign = float(np.mean(np.sign(d) == np.sign(med))) if med != 0 else float("nan")
    return med, sign, len(d)
