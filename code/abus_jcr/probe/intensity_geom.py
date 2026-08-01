"""Intensity <-> candidate-localisation probe primitives (read-only, descriptive).

This module answers two questions with as few assumptions as possible:

1. **Which storage axis is the beam/depth axis?** Measured from image content only
   (:func:`axis_profile`, :func:`beam_axis_vote`). No constant from
   :mod:`abus_jcr.conventions` is an input — that is the whole point, since the
   declared assignment is itself what is under test.
2. **Does candidate placement track dark regions?** Per plane
   (:func:`plane_intensity_map` vs :func:`plane_count_map`, compared by
   :func:`map_spearman`) and in 3D (:func:`box_intensity_stats`).

Design constraints, deliberately:

- **No shadow segmentation.** There is no threshold that turns voxels into
  "shadow"/"not shadow". The shadow proxy is the raw mean intensity, which is what
  a reader actually looks at. Every knob that does exist (grid cell size, distal
  box length) is an explicit argument with a stated default.
- **Unit-free where possible.** Correspondence is measured by rank correlation on
  voxel/cell grids, so the numbers do not depend on the spacing map. Spacing enters
  only where a millimetre is genuinely required (grid cell size, spread fractions),
  and is always passed in — never imported.
- **Per-volume.** Every function operates on ONE volume. Pooling across volumes is
  the caller's job, because a pooled statistic hides that one lesion contributes
  many correlated redundant tubes.

Arrays are in storage order ``(d0, d1, d2)`` throughout.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Storage <-> official/ITK axis mapping. ITK x is the LAST storage axis; the
# permutation is self-inverse. Mirrors conventions.PERM_STORAGE_TO_ITK, restated
# here as a data-layout fact (which column is which), not as a physical claim.
_ITK_CEN = ("coordZ", "coordY", "coordX")      # -> storage d0, d1, d2
_ITK_EXT = ("z_length", "y_length", "x_length")  # -> storage d0, d1, d2


# --------------------------------------------------------------------------
# Stage 1 — axis identification from image content
# --------------------------------------------------------------------------

def axis_profile(vol: np.ndarray, axis: int) -> np.ndarray:
    """Mean intensity along ``axis`` (averaged over the other two axes).

    Spacing-free by construction: it is a function of array indices only.

    Accumulates in float64 without materialising a float64 copy — these volumes are
    ~200 MB of uint8, so a copy would be 1.6 GB.
    """
    other = tuple(a for a in range(vol.ndim) if a != axis)
    return np.asarray(vol).mean(axis=other, dtype=np.float64)


def _plane(vol: np.ndarray, axis: int, i: int, step: int = 1) -> np.ndarray:
    """A subsampled VIEW of hyperplane ``i`` along ``axis`` (no gather, no copy)."""
    sl = [slice(None, None, step)] * vol.ndim
    sl[axis] = i
    return vol[tuple(sl)]


def adjacent_correlation(vol: np.ndarray, axis: int, max_planes: int = 20,
                         target_samples: int = 40_000) -> float:
    """Mean Pearson correlation between adjacent hyperplanes along ``axis``.

    A second, independent axis descriptor: the more finely an axis is sampled, the
    more each plane resembles its neighbour. It therefore orders the three axes by
    sampling pitch without using any spacing constant, which is what lets the
    coarsest (elevational sweep) axis be identified separately from the beam axis.
    A heuristic ordering, not a measurement of millimetres — reported as one line of
    evidence among several, never on its own.

    Planes are read as strided VIEWS and subsampled to ~``target_samples`` points:
    a correlation over 40k samples is already far tighter than the between-axis gap
    this is used to resolve, and ``np.take`` on a 200 MB uint8 volume is orders of
    magnitude slower than a view.
    """
    vol = np.asarray(vol)
    n = vol.shape[axis]
    if n < 2:
        return float("nan")
    plane_size = int(np.prod([s for a, s in enumerate(vol.shape) if a != axis]))
    step = max(1, int(np.sqrt(plane_size / max(target_samples, 1))))
    idx = np.unique(np.linspace(0, n - 2, min(max_planes, n - 1)).astype(int))
    rs = []
    for i in idx:
        a = _plane(vol, axis, int(i), step).astype(np.float32).ravel()
        b = _plane(vol, axis, int(i) + 1, step).astype(np.float32).ravel()
        a = a - a.mean()
        b = b - b.mean()
        den = float(np.sqrt(float((a * a).sum()) * float((b * b).sum())))
        if den > 0:
            rs.append(float((a * b).sum()) / den)
    return float(np.mean(rs)) if rs else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho without a SciPy dependency (average ranks, Pearson on ranks)."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (equivalent to ``scipy.stats.rankdata``)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average tied groups
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i + 1
        while j < len(sa) and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def profile_stats(prof: np.ndarray) -> Dict[str, float]:
    """Three spacing-free descriptors of a 1-D intensity profile.

    - ``spearman``: rank correlation of (index, intensity). The beam axis is the
      only one along which the tissue attenuates monotonically, so it is strongly
      negative there and unstable in sign elsewhere.
    - ``peak_pos``: argmax as a fraction of the axis. The beam axis peaks at the
      near-field entrance, i.e. very close to 0.
    - ``decile_ratio``: mean of the first decile / mean of the last decile. > 1
      means the axis starts bright and ends dark.
    """
    prof = np.asarray(prof, dtype=np.float64)
    n = len(prof)
    k = max(1, n // 10)
    lo, hi = float(prof[:k].mean()), float(prof[-k:].mean())
    return {
        "spearman": _spearman(np.arange(n, dtype=np.float64), prof),
        "peak_pos": float(int(np.argmax(prof)) / max(n - 1, 1)),
        "decile_ratio": float(lo / hi) if hi > 0 else float("nan"),
        "first_decile": lo,
        "last_decile": hi,
    }


def beam_axis_vote(vol: np.ndarray) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """Vote for the beam/depth axis of ONE volume, from image content alone.

    The beam axis is taken to be the axis with the **most negative** Spearman
    between index and mean intensity — the attenuation gradient that TGC never
    fully cancels. Returns ``(axis, {axis: profile_stats})``.

    This is a single, stated criterion with no free parameter. The two companion
    descriptors (``peak_pos``, ``decile_ratio``) are returned so the caller can
    check that the winner also behaves like a beam axis rather than merely
    winning a weak contest.
    """
    stats = {a: profile_stats(axis_profile(vol, a)) for a in range(vol.ndim)}
    axis = min(stats, key=lambda a: stats[a]["spearman"])
    return int(axis), stats


# --------------------------------------------------------------------------
# Stage 2 — plane correspondence
# --------------------------------------------------------------------------

def _bin_edges(n: int, n_bins: int) -> np.ndarray:
    """Bin index per array position, splitting ``n`` positions into ``n_bins``."""
    return np.minimum((np.arange(n) * n_bins) // max(n, 1), n_bins - 1)


def grid_shape(shape2d: Sequence[int], spacing2d: Sequence[float], cell_mm: float) -> Tuple[int, int]:
    """Coarse-grid shape covering ``shape2d`` with roughly ``cell_mm`` cells."""
    return tuple(max(1, int(round(s * sp / cell_mm))) for s, sp in zip(shape2d, spacing2d))  # type: ignore[return-value]


def plane_intensity_map(vol: np.ndarray, drop_axis: int, out_shape: Tuple[int, int]) -> np.ndarray:
    """Mean intensity projected along ``drop_axis``, block-averaged to ``out_shape``.

    Projecting along the beam axis gives the ABUS coronal (C-plane) view, in which a
    beam-direction shadow collapses to a dark *point*; projecting along either other
    axis leaves it as a dark *stripe*. That contrast is the whole diagnostic, so the
    projection is a plain mean — no max/min intensity projection, which would trade
    the shadow's own contrast for whatever is brightest along the ray.
    """
    proj = np.asarray(vol).mean(axis=drop_axis, dtype=np.float64)
    return block_mean(proj, out_shape)


def block_mean(img: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Mean-pool a 2-D array onto ``out_shape`` (handles non-divisible sizes)."""
    img = np.asarray(img, dtype=np.float64)
    r = _bin_edges(img.shape[0], out_shape[0])
    c = _bin_edges(img.shape[1], out_shape[1])
    tot = np.zeros(out_shape, dtype=np.float64)
    cnt = np.zeros(out_shape, dtype=np.float64)
    np.add.at(tot, (r[:, None], c[None, :]), img)
    np.add.at(cnt, (r[:, None], c[None, :]), np.ones_like(img))
    return np.where(cnt > 0, tot / np.maximum(cnt, 1.0), np.nan)


def plane_count_map(points_rc: np.ndarray, shape2d: Sequence[int],
                    out_shape: Tuple[int, int]) -> np.ndarray:
    """Count of 2-D points per coarse cell, on the SAME grid as :func:`plane_intensity_map`.

    ``points_rc`` are (row, col) positions in full-resolution voxel units.
    """
    out = np.zeros(out_shape, dtype=np.float64)
    if len(points_rc) == 0:
        return out
    pts = np.asarray(points_rc, dtype=np.float64)
    r = np.clip((pts[:, 0] * out_shape[0]) // max(shape2d[0], 1), 0, out_shape[0] - 1).astype(int)
    c = np.clip((pts[:, 1] * out_shape[1]) // max(shape2d[1], 1), 0, out_shape[1] - 1).astype(int)
    np.add.at(out, (r, c), 1.0)
    return out


def map_spearman(a: np.ndarray, b: np.ndarray, where: Optional[np.ndarray] = None) -> float:
    """Rank correlation between two same-shaped maps over their finite cells.

    Negative = candidates sit where the plane is BRIGHT; positive = where it is DARK
    only if ``a`` is passed as *negated* intensity. The caller decides the sign
    convention; this function is agnostic. Returns NaN if either map is constant
    (e.g. no candidates at all), which is a real "cannot say", not a zero.

    ``where`` restricts the correlation to a subset of cells. This matters more than it
    looks: over ALL cells the statistic is dominated by the difference between tissue and
    the dark out-of-contact margins, where no candidate can ever land, so it reports
    "candidates are in tissue" — true, and not the question. Passing the cells that hold
    at least one candidate answers the question actually asked: among the places
    candidates go, do they prefer the darker ones?
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if where is not None:
        ok &= np.asarray(where).ravel().astype(bool)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return _spearman(a, b)


# --------------------------------------------------------------------------
# Stage 2 — 3-D: intensity at the candidate itself, and distal to it
# --------------------------------------------------------------------------

def storage_boxes(df, spacing: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """Native storage-order centres and extents from the official ITK columns.

    ``coordX``/``x_length`` are the LAST storage axis; ``coordZ``/``z_length`` the
    first. Returns ``{"cen": (N,3), "ext": (N,3)}`` in voxels, plus ``"cen_mm"`` when
    ``spacing`` is given. No cache, no resampling — these are the native indices the
    scoring space already uses.
    """
    cen = np.stack([np.asarray(df[c], dtype=np.float64) for c in _ITK_CEN], axis=1)
    ext = np.stack([np.asarray(df[c], dtype=np.float64) for c in _ITK_EXT], axis=1)
    out = {"cen": cen, "ext": ext}
    if spacing is not None:
        out["cen_mm"] = cen * np.asarray(spacing, dtype=np.float64)[None, :]
        out["ext_mm"] = ext * np.asarray(spacing, dtype=np.float64)[None, :]
    return out


def _clip_slice(lo: float, hi: float, n: int) -> Tuple[int, int]:
    """``[lo, hi]`` (inclusive max, per DATA_INFO.md) clipped to a NON-EMPTY ``[a, b)``.

    A candidate box may legitimately hang off the edge of the volume, and a box that
    starts beyond the last index must still yield a readable voxel rather than an empty
    slice — an empty slice silently becomes a NaN mean, which then propagates into an
    effect size as a missing value rather than as an error.
    """
    a = int(min(max(0, np.floor(lo)), max(n - 1, 0)))
    b = int(min(n, max(a + 1, np.ceil(hi) + 1)))
    return a, b


def box_intensity_stats(vol: np.ndarray, cen: np.ndarray, ext: np.ndarray, depth_axis: int,
                        distal_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Per-candidate intensity inside the box, and in the slab directly distal to it.

    Returns four arrays of length N:

    - ``inside``  — mean intensity within the candidate's own native box.
    - ``depth_baseline`` — mean intensity of the WHOLE volume over the same depth
      range. This is the necessary control: intensity falls steeply with depth, so
      without it a "candidates are dark" finding is just a restatement of "candidates
      are deep".
    - ``contrast`` = ``inside - depth_baseline``. Negative = darker than its own depth
      band. Only comparable WITHIN a volume (the baseline includes the dark FOV
      margins, an offset shared by every candidate in that volume).
    - ``distal_contrast`` — the same quantity for a slab of the same in-plane footprint
      placed immediately distal along ``depth_axis``, ``distal_scale`` x the box's own
      depth extent long. This is the literal definition of "casts a posterior shadow".
      NaN where the candidate is already at the far edge of the volume.

    The only knob is ``distal_scale``, and it is self-scaling (a multiple of the
    candidate's own depth extent), not an absolute millimetre figure.
    """
    vol = np.asarray(vol)   # kept in its native dtype; means accumulate in float64
    # Per-depth-index mean over the whole volume. Every depth slab has the same voxel
    # count, so the mean over a depth RANGE is exactly the mean of this profile over
    # that range — O(1) per candidate instead of re-reducing the volume each time.
    depth_prof = axis_profile(vol, depth_axis)

    n = len(cen)
    inside = np.full(n, np.nan)
    base = np.full(n, np.nan)
    distal = np.full(n, np.nan)
    lo_all = cen - ext / 2.0
    hi_all = cen + ext / 2.0
    n_depth = vol.shape[depth_axis]
    for i in range(n):
        sl = []
        for a in range(3):
            s, e = _clip_slice(lo_all[i, a], hi_all[i, a], vol.shape[a])
            sl.append(slice(s, e))
        inside[i] = vol[tuple(sl)].mean(dtype=np.float64)

        ds, de = sl[depth_axis].start, sl[depth_axis].stop
        base[i] = float(depth_prof[ds:de].mean())

        length = max(1, int(round(distal_scale * (de - ds))))
        d0, d1 = de, min(n_depth, de + length)
        if d1 > d0:
            dsl = list(sl)
            dsl[depth_axis] = slice(d0, d1)
            distal[i] = (float(vol[tuple(dsl)].mean(dtype=np.float64))
                         - float(depth_prof[d0:d1].mean()))
    return {"inside": inside, "depth_baseline": base,
            "contrast": inside - base, "distal_contrast": distal}


# --------------------------------------------------------------------------
# Stage 2 — alignment / ray structure
# --------------------------------------------------------------------------

def spread_fractions(cen_mm: np.ndarray, extent_mm: Optional[Sequence[float]] = None) -> np.ndarray:
    """How the centroid cloud's spread divides between the three axes.

    With ``extent_mm`` (the volume's physical size per axis) each axis variance is first
    divided by the variance a UNIFORM cloud would have on that axis (``L**2 / 12``), so a
    cloud filling the volume scores (1/3, 1/3, 1/3) regardless of the volume's own shape.

    That normalisation is not cosmetic. An ABUS volume is ~173 x 50 x 168 mm, so raw
    millimetre variances are dominated by the two long axes and *every* cloud looks
    "flat in depth" — which would be a fact about the field of view, not about the
    candidates. Without ``extent_mm`` the raw millimetre fractions are returned.
    """
    cen_mm = np.asarray(cen_mm, dtype=np.float64)
    if len(cen_mm) < 2:
        return np.full(3, np.nan)
    v = cen_mm.var(axis=0)
    if extent_mm is not None:
        ref = np.asarray(extent_mm, dtype=np.float64) ** 2 / 12.0
        v = np.divide(v, ref, out=np.zeros_like(v), where=ref > 0)
    tot = float(v.sum())
    return v / tot if tot > 0 else np.full(3, np.nan)


def coronal_stacking(cen_mm: np.ndarray, depth_axis: int, cell_mm: float = 5.0) -> Dict[str, np.ndarray]:
    """How much do candidates stack along shared beam lines?

    Bins centroids into ``cell_mm`` cells of the CORONAL plane (the two non-depth
    axes) — the plane a beam-direction shadow ray collapses to a point in. Returns

    - ``cell_counts``: occupancy of every non-empty cell.
    - ``cell_depth_spread``: p90-p10 of the depth coordinate (mm) within each cell.

    Reported as distributions, not thresholded: "fraction in cells with >= k" is a
    read-off the caller can make at any k, and the companion figure shows the same
    binning, so the number and the picture check each other.
    """
    cen_mm = np.asarray(cen_mm, dtype=np.float64)
    if len(cen_mm) == 0:
        return {"cell_counts": np.zeros(0), "cell_depth_spread": np.zeros(0)}
    lat = [a for a in range(3) if a != depth_axis]
    keys = np.floor(cen_mm[:, lat] / float(cell_mm)).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    spread = np.full(len(uniq), np.nan)
    depth = cen_mm[:, depth_axis]
    for k in range(len(uniq)):
        d = depth[inv == k]
        if len(d) >= 2:
            spread[k] = float(np.percentile(d, 90) - np.percentile(d, 10))
    return {"cell_counts": counts, "cell_depth_spread": spread}


# --------------------------------------------------------------------------
# Stage 2 — banding / parallel stripes
# --------------------------------------------------------------------------

def power_spectrum_1d(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Period (in samples) vs normalised power of a 1-D profile, DC removed.

    Used two ways: on the lateral intensity profile of the distal depth band (regular
    parallel shadows = a peak at their spacing) and on the lateral positions of FP
    centroids (a matching peak means the detector fires on the bands). A peak at the
    detector's own grid period is an anchor-grid artefact, which is why the caller
    marks that period on the axis.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    if len(p) < 4 or not np.any(p):
        return np.zeros(0), np.zeros(0)
    amp = np.abs(np.fft.rfft(p))[1:]          # drop DC
    freqs = np.fft.rfftfreq(len(p))[1:]
    power = amp ** 2
    tot = float(power.sum())
    return 1.0 / freqs, (power / tot if tot > 0 else power)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1]: P(a > b) - P(a < b). NaNs dropped.

    Same definition as ``probe.pool_diag._cliffs_delta`` so effect sizes here are
    directly comparable to the pool-diagnostics tables; computed by a sort rather
    than the O(n*m) double loop because the candidate arrays here are larger.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    sb = np.sort(b)
    gt = int(np.searchsorted(sb, a, side="left").sum())    # pairs with a > b
    ge = int(np.searchsorted(sb, a, side="right").sum())   # pairs with a >= b
    lt = len(a) * len(b) - ge                              # pairs with a < b
    return float((gt - lt) / (len(a) * len(b)))
