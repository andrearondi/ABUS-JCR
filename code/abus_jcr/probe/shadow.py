"""Shadow-field estimation and shadow <-> candidate correspondence (torch-free core).

Motivation
----------
The Phase-3 pool diagnostics ([P3U2.PD]) characterise each candidate *in isolation*
(score stats, tube geometry, its own anisotropy). What they never test is whether the
candidate **cloud** is organised by the volume's *acoustic* structure. The per-set
projection figures show candidates lining up along an axis; the published ABUS-CAD
literature says that is exactly what should happen — in a 846-patient study **86% of CAD
false positives were pseudo-lesions, of which marginal shadowing (39.1%) and Cooper's
ligament shadowing (26.8%) were the two leading causes**. That is the geometry/localisation
signal a *joint* rescorer can exploit and a per-candidate feature cannot.

Acoustic model this module encodes
----------------------------------
A posterior acoustic shadow is a **ray**: an attenuator at depth ``j0`` on the beam line
``(lateral, sweep) = (i, k)`` removes energy from **every** voxel below it on that same
line. Three consequences drive every estimator here:

1. **Shadows are 1-D objects along the beam axis.** Projected onto the plane
   perpendicular to the beam — the ABUS **coronal / C-plane**, spanned by
   (lateral, sweep) — a shadow ray collapses to a **point**. The coronal plane is
   therefore the natural space in which to build a shadow map and to ask whether
   candidates sit on shadowed lines.
2. **A shadow is a *sustained* deficit, an isolated hypoechoic mass is not.** Both are
   "dark". Only the shadow keeps the tissue below it dark all the way to the far field.
   Separating them needs a *distal-persistence* term, not a brightness threshold.
3. **"Dark" must be judged relative to depth.** Time-gain compensation never perfectly
   cancels attenuation, so mean intensity falls monotonically with depth in every volume
   (measured: Spearman(index, mean) = -0.93 median over the 30 Validation volumes along
   the beam axis). An absolute threshold would label the whole far field a shadow. All
   estimators here work on a **depth-normalised residual**.

Axis convention (VERIFY, DO NOT ASSUME)
---------------------------------------
Every function takes ``beam_axis`` explicitly. :func:`identify_beam_axis` re-derives it
from image content alone — no spacing constant, no stored convention — so the caller can
assert the convention instead of inheriting it. See ``conventions.SPACING_STORAGE_MM``
and the axis-audit note in ``RESULTS`` for why that matters here.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------------- axis identification
def axis_attenuation_stats(vol: np.ndarray, sub: int = 4) -> Dict[int, dict]:
    """Per-axis attenuation signature of one volume — the spacing-free beam-axis test.

    For each axis, reduce the volume to a 1-D mean-intensity profile over that axis and
    measure how much it looks like an ultrasound depth profile: a bright near-field
    entrance that decays monotonically toward the far field.

    Returns ``{axis: {spearman, asym, peak_pos_frac, first_decile, last_decile}}`` where

    - ``spearman``      — rank correlation between index and mean intensity. The beam axis
      is strongly negative (attenuation); the two axes orthogonal to the beam are flat.
    - ``asym``          — first-decile mean minus last-decile mean (entrance vs far field).
    - ``peak_pos_frac`` — argmax position as a fraction of the axis length. The beam axis
      peaks within the first few percent (the skin / coupling entrance band).

    ``sub`` strides the two *reduced* axes for speed; the profile axis is never strided.
    """
    v = np.asarray(vol)
    out: Dict[int, dict] = {}
    for ax in (0, 1, 2):
        others = [a for a in (0, 1, 2) if a != ax]
        sl = [slice(None)] * 3
        for o in others:
            sl[o] = slice(None, None, sub)
        sub_v = np.asarray(v[tuple(sl)], dtype=np.float64)
        prof = sub_v.mean(axis=tuple(a for a in (0, 1, 2) if a != ax))
        n = len(prof)
        idx = np.arange(n, dtype=float)
        rx = np.argsort(np.argsort(idx)).astype(float)
        ry = np.argsort(np.argsort(prof)).astype(float)
        sp = float(np.corrcoef(rx, ry)[0, 1]) if n > 2 and prof.std() > 0 else float("nan")
        dec = max(1, n // 10)
        out[ax] = {"spearman": sp,
                   "first_decile": float(prof[:dec].mean()),
                   "last_decile": float(prof[-dec:].mean()),
                   "asym": float(prof[:dec].mean() - prof[-dec:].mean()),
                   "peak_pos_frac": float(np.argmax(prof) / max(n, 1)),
                   "profile": prof}
    return out


def identify_beam_axis(vol: np.ndarray, sub: int = 4) -> int:
    """The storage axis along which the ultrasound beam propagates, from image content only.

    Picks the axis with the most negative index-vs-mean-intensity Spearman correlation.
    This is a *measurement*, not a convention lookup — call it and assert against the
    declared convention rather than trusting the constant.
    """
    st = axis_attenuation_stats(vol, sub=sub)
    return int(min((0, 1, 2), key=lambda a: st[a]["spearman"]))


# --------------------------------------------------------------------------- shadow field
def _to_beam_first(a: np.ndarray, beam_axis: int) -> np.ndarray:
    """View of ``a`` with the beam axis moved to position 0 (cheap, no copy)."""
    return np.moveaxis(a, beam_axis, 0)


def _from_beam_first(a: np.ndarray, beam_axis: int) -> np.ndarray:
    """Inverse of :func:`_to_beam_first`."""
    return np.moveaxis(a, 0, beam_axis)


def depth_normalised_residual(vol: np.ndarray, beam_axis: int, sub: int = 4,
                              eps: float = 1e-3, tissue: np.ndarray | None = None) -> np.ndarray:
    """Robust z-score of intensity **within each depth plane** — the shadow substrate.

    ``Z[..., j, ...] = (V - median_j) / MAD_j`` where the median and the (normal-scaled)
    median-absolute-deviation are taken over the plane at beam index ``j``. This removes
    the global attenuation / TGC trend, so a negative ``Z`` means "darker than this depth
    normally is", which is what a shadow is — as opposed to "dark", which the whole far
    field is. ``sub`` strides the in-plane axes when estimating the per-depth statistics.

    ``tissue`` is a coronal-plane boolean mask of beam lines inside the breast. Supply it:
    ABUS volumes carry large zero-padded regions outside the transducer footprint, and
    including them drags every per-depth median down, so genuine tissue reads uniformly
    *bright* and genuine shadows fail to clear the threshold. Defaults to all-lines when
    omitted, which is correct only for phantoms with no padding.

    Returned as float32 with the caller's original axis order.
    """
    v = _to_beam_first(np.asarray(vol), beam_axis)
    n_depth = v.shape[0]
    est = v[:, ::sub, ::sub].astype(np.float32)
    if tissue is not None:
        keep = np.asarray(tissue, dtype=bool)[::sub, ::sub]
        flat = est[:, keep] if keep.any() else est.reshape(n_depth, -1)
    else:
        flat = est.reshape(n_depth, -1)
    med = np.median(flat, axis=1).astype(np.float32)
    mad = np.median(np.abs(flat - med[:, None]), axis=1).astype(np.float32) * 1.4826
    mad = np.maximum(mad, eps * max(float(np.max(med)), 1.0))
    z = (v.astype(np.float32) - med[:, None, None]) / mad[:, None, None]
    return _from_beam_first(z, beam_axis)


def tissue_line_mask(vol: np.ndarray, beam_axis: int, near_frac: float = 0.30,
                     rel: float = 0.35) -> np.ndarray:
    """Which beam lines actually carry signal — a 2-D mask over the **coronal** plane.

    ABUS volumes are padded outside the transducer footprint and fall away at the lateral
    edges of the field of view. Those regions are dark for a non-acoustic reason and will
    otherwise dominate any "dark == shadow" statistic — on real Validation volumes they
    swamped the coronal shadow map entirely.

    A line qualifies on its **near-field median** — the median over the first ``near_frac``
    of its depth — relative to the median of that quantity across all lines. Both halves of
    that choice matter. Judging on the *near field* is right because every valid beam line
    has strong echoes close to the transducer, whatever happens deeper. Judging on the
    *median* rather than the maximum or the mean is right because an out-of-field column is
    mostly dark yet still carries bright speckle outliers, which drag a max — and, if there
    are enough of them, a mean — above any sensible threshold.

    Lines with no near-field signal at all are excluded even if the cause is a total
    shadow rather than padding; they carry no recoverable information either way, and
    :func:`shadow_field` reports the excluded fraction so the loss stays visible.

    Returns a boolean array whose shape is ``vol.shape`` with ``beam_axis`` removed, in
    ascending order of the two remaining axes.
    """
    v = _to_beam_first(np.asarray(vol), beam_axis)
    n_near = max(1, int(round(v.shape[0] * near_frac)))
    nf = np.median(np.asarray(v[:n_near], dtype=np.float32), axis=0)
    pos = nf[nf > 0]
    ref = float(np.median(pos)) if pos.size else 0.0
    return nf >= rel * ref


def shadow_field(vol: np.ndarray, beam_axis: int, dark_z: float = -0.6,
                 tail_z: float = -0.4, far_frac: float = 0.30,
                 near_ok_z: float = -0.35, near_frac: float = 0.12,
                 near_smooth: float = 2.0, sub: int = 4) -> dict:
    """Per-voxel shadow field + the coronal-plane shadow map.

    A voxel is flagged **shadow** iff it is *locally* darker than its depth plane
    (``Z < dark_z``) **and** its beam line's **far field** is also darker than expected
    (``far_mean < tail_z``). The second condition is what separates an acoustic shadow —
    energy removed from the line, which never returns — from an isolated hypoechoic mass,
    which is dark but leaves the tissue below it normal. Voxels on non-tissue lines are
    never flagged.

    The persistence test is deliberately a **per-line far-field** statistic (the deepest
    ``far_frac`` of the line) and not a per-voxel running mean over some gap below the
    voxel. A gapped running mean needs the gap to exceed the depth extent of any real
    mass, or the mass satisfies the persistence test with *its own* dark voxels — and on
    this dataset's cache that requirement cannot be met. The iso depth axis carries 341
    samples over ~50 mm (0.146 mm/voxel), so the median GT lesion is ~98 voxels deep and
    the largest ~184; a gap big enough to clear them would leave almost no line left to
    measure. Anchoring persistence to the far field removes the length scale from the
    problem entirely: attenuation is cumulative, so a genuine shadow darkens the bottom of
    its line no matter where along the line it started.

    The residual confusion is a mass sitting *inside* the far field, which darkens the far
    field by occupying it. That population is small and, from the volume alone, genuinely
    ambiguous with a deep shadow.

    ``near_ok_z`` is the second load-bearing guard, and it encodes the other defining
    property of a *posterior* shadow: there is normal tissue above the attenuator. A line
    already depressed in its own near field is not shadowed — it is weakly coupled or
    outside the usable field, and dim at every depth for a non-acoustic reason. Because
    the residual is normalised per depth *plane* rather than per line, such a line
    otherwise reads negative at every depth and is flagged end to end. On the Validation
    split this was not a minor contamination: the dim lateral margin of the field of view
    (near-field level ~0.5x the volume median, falling further with depth) was flagged as
    one enormous shadow and dominated the entire coronal map. Lines failing this test are
    counted in ``frac_weak_lines`` rather than silently dropped.

    ``near_frac`` sets how much of the line counts as "above", and it must stay SHALLOW.
    The guard's job is to spot a line that never carried signal at all, which is already
    visible just below the skin entrance band (measured argmax at 2% of the axis). Set it
    too deep and the guard starts rejecting genuine shadows that happen to begin early,
    because the shadow itself then dominates the window being used to judge whether the
    line was ever healthy. ``near_smooth`` blurs the statistic across neighbouring lines
    before thresholding, since weak coupling is regional and per-line sampling noise is not.

    The cost is explicit: a shadow beginning at the skin itself — a trapped air bubble, a
    gross coupling failure — is excluded along with the field margin, because from the
    volume alone the two are the same observation. Such lines carry no recoverable tissue
    signal either way.

    Returns
    -------
    dict with
      ``z``              — depth-normalised residual, caller's axis order (float32)
      ``far``            — **coronal-plane** map: mean ``z`` over the deepest ``far_frac``
                           of each beam line (the persistence statistic itself)
      ``shadow``         — boolean shadow field, caller's axis order
      ``line_shadow``    — **coronal-plane** map: fraction of each beam line flagged
                           shadow. A shadow ray is a point here; the canonical shadow map.
      ``line_deficit``   — coronal-plane map: mean ``z`` over the distal half of each line
                           (a graded, threshold-free alternative to ``line_shadow``)
      ``tissue``         — coronal-plane boolean mask: lines inside the breast
      ``strong``         — coronal-plane boolean mask: tissue lines that also pass the
                           near-field guard, i.e. the lines eligible to be flagged
      ``shadow_frac``    — scalar: flagged fraction among eligible voxels
      ``n_tissue_lines`` / ``n_strong_lines`` / ``frac_weak_lines`` — line accounting
    """
    tissue = tissue_line_mask(vol, beam_axis)
    z = depth_normalised_residual(vol, beam_axis, sub=sub, tissue=tissue)
    zb = _to_beam_first(z, beam_axis)                       # (depth, u, v)
    n_depth = zb.shape[0]

    # persistence: the mean residual over the DEEPEST far_frac of each line. One number
    # per beam line, so it carries no length scale a real mass could defeat.
    n_far = max(1, int(round(n_depth * far_frac)))
    far = zb[n_depth - n_far:].mean(axis=0)

    # a posterior shadow needs NORMAL tissue above it; a line already depressed in its own
    # near field is weakly coupled / out of field, not shadowed (see the docstring)
    n_near = max(1, int(round(n_depth * near_frac)))
    near_z = zb[:n_near].mean(axis=0)
    # Smooth ACROSS beam lines before thresholding. Poor coupling and the field margin are
    # regional — they affect a patch of neighbouring lines, never one line in isolation.
    # Per-line, near_z is a mean over only n_near samples (sd ~ 1/sqrt(n_near) ~ 0.2 here)
    # against a threshold of near_ok_z, so a few percent of perfectly healthy lines fail by
    # chance and punch holes in otherwise solid shadows. Smoothing removes that noise
    # without blurring the regional signal the guard is actually looking for.
    if near_smooth > 0:
        near_z = _gauss2d(near_z.astype(float), near_smooth).astype(np.float32)
    strong = tissue & (near_z > near_ok_z)

    shadow = (zb < dark_z) & (far < tail_z)[None, :, :] & strong[None, :, :]

    n_tissue_lines = int(tissue.sum())
    frac_weak = float((tissue & ~strong).sum()) / max(n_tissue_lines, 1)
    line_shadow = shadow.mean(axis=0).astype(np.float32)
    line_shadow[~strong] = np.nan
    half = n_depth // 2
    line_deficit = zb[half:].mean(axis=0).astype(np.float32)
    line_deficit[~strong] = np.nan

    far_map = far.astype(np.float32).copy()
    far_map[~strong] = np.nan
    denom = max(int(strong.sum()) * n_depth, 1)
    return {"z": _from_beam_first(zb, beam_axis),
            "far": far_map,
            "shadow": _from_beam_first(shadow, beam_axis),
            "line_shadow": line_shadow,
            "line_deficit": line_deficit,
            "tissue": tissue,
            "strong": strong,
            "shadow_frac": float(shadow.sum()) / denom,
            "n_tissue_lines": n_tissue_lines,
            "n_strong_lines": int(strong.sum()),
            "frac_weak_lines": frac_weak}


# --------------------------------------------------------------------------- marginals & planes
def axis_marginals(field: dict, vol: np.ndarray, beam_axis: int, n_bins: int = 60) -> Dict[int, dict]:
    """Per-axis 1-D profiles of mean intensity and shadow mass, binned to ``n_bins``.

    These are the histogram partners of the candidate-centroid marginals: plotting
    ``shadow`` and ``candidate density`` on the same normalised axis is the direct visual
    test of whether the two distributions track each other.
    """
    sh = field["shadow"]
    v = np.asarray(vol)
    out: Dict[int, dict] = {}
    for ax in (0, 1, 2):
        red = tuple(a for a in (0, 1, 2) if a != ax)
        inten = v.mean(axis=red).astype(np.float64)
        shad = sh.mean(axis=red).astype(np.float64)
        out[ax] = {"intensity": _rebin(inten, n_bins), "shadow": _rebin(shad, n_bins),
                   "n": int(v.shape[ax]), "is_beam": ax == beam_axis}
    return out


def _rebin(profile: np.ndarray, n_bins: int) -> np.ndarray:
    """Average a 1-D profile into ``n_bins`` equal-width bins over its normalised extent."""
    p = np.asarray(profile, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(n_bins)
    edges = np.linspace(0, n, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, np.arange(n), side="right") - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=p, minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins).astype(float)
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)


def plane_shadow_maps(field: dict, beam_axis: int) -> Dict[tuple, np.ndarray]:
    """Shadow mass projected onto each of the three orthogonal planes.

    Keys are the ``(axis_a, axis_b)`` storage-axis pairs in ascending order. The plane
    that excludes ``beam_axis`` is the **coronal / C-plane**, where a shadow ray is a
    point — that map is the one with real physical meaning; the other two integrate
    along the beam and are included for the side-by-side comparison the caller wants.
    """
    sh = np.asarray(field["shadow"], dtype=np.float32)
    out = {}
    for drop in (0, 1, 2):
        keep = tuple(a for a in (0, 1, 2) if a != drop)
        out[keep] = sh.mean(axis=drop)
    return out


# --------------------------------------------------------------------------- candidate features
def candidate_shadow_features(field: dict, boxes: np.ndarray, beam_axis: int,
                              distal_depth: int = 24) -> Dict[str, np.ndarray]:
    """Per-candidate shadow descriptors — the block a rescorer could actually consume.

    ``boxes`` is ``(n, 6)`` of iso storage boxes ``(min_d0, min_d1, min_d2, max_d0,
    max_d1, max_d2)``; coordinates are clipped to the volume and may be fractional.

    Returns arrays of length ``n``:

    - ``shadow_frac``   — fraction of the candidate's voxels flagged shadow. A candidate
      that *is* a shadow scores high; a solid mass scores low.
    - ``z_mean``        — mean depth-normalised residual inside the box (how dark it is
      relative to its depth, threshold-free).
    - ``distal_z``      — mean residual in a slab of ``distal_depth`` voxels immediately
      **below** the box. Genuine posterior shadowing (a real malignant mass sign) and a
      pure attenuation artifact both darken this; combined with ``shadow_frac`` it
      separates "casts a shadow" from "is a shadow".
    - ``proximal_z``    — mean residual in the slab immediately **above** the box. A true
      shadow has *normal* tissue above it (the attenuator) whereas a hypoechoic mass sits
      in normal tissue on both sides.
    - ``line_shadow``   — mean coronal-plane line-shadow value over the beam lines the
      box occupies: "how shadowed is the column this candidate lives in", integrated over
      the whole line rather than just the box.
    - ``tissue_frac``   — fraction of the box's beam lines that are inside tissue. Low
      values flag candidates sitting in padding / outside the breast.
    - ``weak_line_frac``— fraction of the box's tissue lines that are **weakly coupled**
      (depressed already in the near field, so excluded from shadow flagging). A candidate
      in the dim margin of the field of view scores high here and near zero on
      ``shadow_frac``; without this column those two very different situations are
      indistinguishable in the output.
    """
    z = _to_beam_first(np.asarray(field["z"]), beam_axis)
    sh = _to_beam_first(np.asarray(field["shadow"]), beam_axis)
    line_shadow = field["line_shadow"]
    tissue = field["tissue"]
    strong = field.get("strong", tissue)
    n_depth, n_u, n_v = z.shape

    # storage-axis order of the beam-first view: (beam_axis, then the two others ascending)
    other = [a for a in (0, 1, 2) if a != beam_axis]
    keys = ["shadow_frac", "z_mean", "distal_z", "proximal_z", "line_shadow",
            "tissue_frac", "weak_line_frac"]
    out = {k: np.full(len(boxes), np.nan, dtype=float) for k in keys}

    for n, b in enumerate(np.asarray(boxes, dtype=float)):
        lo = [b[0], b[1], b[2]]
        hi = [b[3], b[4], b[5]]
        d0 = int(np.clip(np.floor(lo[beam_axis]), 0, n_depth - 1))
        d1 = int(np.clip(np.ceil(hi[beam_axis]), d0 + 1, n_depth))
        u0 = int(np.clip(np.floor(lo[other[0]]), 0, n_u - 1))
        u1 = int(np.clip(np.ceil(hi[other[0]]), u0 + 1, n_u))
        v0 = int(np.clip(np.floor(lo[other[1]]), 0, n_v - 1))
        v1 = int(np.clip(np.ceil(hi[other[1]]), v0 + 1, n_v))

        sub_sh = sh[d0:d1, u0:u1, v0:v1]
        sub_z = z[d0:d1, u0:u1, v0:v1]
        out["shadow_frac"][n] = float(sub_sh.mean()) if sub_sh.size else np.nan
        out["z_mean"][n] = float(sub_z.mean()) if sub_z.size else np.nan

        dd0, dd1 = d1, min(n_depth, d1 + distal_depth)
        if dd1 > dd0:
            out["distal_z"][n] = float(z[dd0:dd1, u0:u1, v0:v1].mean())
        pp1, pp0 = d0, max(0, d0 - distal_depth)
        if pp1 > pp0:
            out["proximal_z"][n] = float(z[pp0:pp1, u0:u1, v0:v1].mean())

        ls = line_shadow[u0:u1, v0:v1]
        out["line_shadow"][n] = float(np.nanmean(ls)) if np.isfinite(ls).any() else np.nan
        tt = tissue[u0:u1, v0:v1]
        out["tissue_frac"][n] = float(tt.mean()) if tt.size else np.nan
        st = strong[u0:u1, v0:v1]
        n_tis = int(tt.sum()) if tt.size else 0
        out["weak_line_frac"][n] = float((tt & ~st).sum()) / n_tis if n_tis else np.nan
    return out


# --------------------------------------------------------------------------- structure tests
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1]: ``P(a > b) - P(a < b)``. NaNs dropped."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    bs = np.sort(b)
    lt = np.searchsorted(bs, a, side="left").sum()
    gt = (len(bs) - np.searchsorted(bs, a, side="right")).sum()
    return float((lt - gt) / (len(a) * len(b)))


def ray_colinearity(centres: np.ndarray, beam_axis: int, coronal_radius: float,
                    min_depth_sep: float, n_perm: int = 50, seed: int = 0) -> dict:
    """Are candidates stacked **along the beam** — i.e. strung out on shared shadow rays?

    Counts ordered pairs that are close in the **coronal** plane (perpendicular distance
    ``< coronal_radius``) yet far apart in depth (``> min_depth_sep``). That is the exact
    footprint of several candidates fired at different depths on one shadow ray.

    The raw count means nothing on its own, so it is referred to a permutation null. The
    null permutes **each coordinate column independently**, which preserves all three
    per-axis marginals while destroying the *joint* structure — two candidates that
    shared a coronal position no longer do, because their lateral and sweep coordinates
    are re-paired at random. Permuting depth alone, the obvious first choice, is useless
    here: it leaves every coronal distance untouched, so the statistic is invariant and
    the enrichment is identically 1.

    ``enrichment = observed / null_mean`` > 1 means genuine ray structure.
    """
    c = np.asarray(centres, dtype=float)
    if len(c) < 2:
        return {"observed": 0, "null_mean": float("nan"), "enrichment": float("nan"), "n": len(c)}
    other = [a for a in (0, 1, 2) if a != beam_axis]

    def _count(arr):
        cor = arr[:, other]
        dep = arr[:, beam_axis]
        dc = np.linalg.norm(cor[:, None, :] - cor[None, :, :], axis=2)
        dd = np.abs(dep[:, None] - dep[None, :])
        m = (dc < coronal_radius) & (dd > min_depth_sep)
        np.fill_diagonal(m, False)
        return int(m.sum())

    obs = _count(c)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        shuf = np.column_stack([rng.permutation(c[:, a]) for a in (0, 1, 2)])
        null.append(_count(shuf))
    nm = float(np.mean(null))
    return {"observed": obs, "null_mean": nm, "null_std": float(np.std(null)),
            "enrichment": (obs / nm) if nm > 0 else float("nan"), "n": len(c)}


def knn_direction_anisotropy(centres: np.ndarray, shape: Sequence[int],
                             k: int = 3) -> Dict[int, float]:
    """Along which axis is the candidate cloud locally elongated? — null-free, sums to 1.

    Coordinates are first divided by the volume's extent along each axis, so every axis
    spans ``[0, 1]``. That is deliberately the *only* normalisation used: it depends on
    nothing but the volume's own shape, and therefore stays valid even where the physical
    voxel spacing is uncertain. For each candidate the ``k`` nearest neighbours in that
    normalised space are taken, and the squared direction cosines of the displacement are
    accumulated per axis.

    The three returned fractions sum to 1. Isotropic scatter gives ``1/3`` each; a cloud
    strung out along the beam — several candidates fired at different depths on one shadow
    ray — pushes the beam-axis fraction well above ``1/3``. This is the direct, quantitative
    form of "the candidates visibly line up along an axis".
    """
    c = np.asarray(centres, dtype=float)
    if len(c) < 2:
        return {0: float("nan"), 1: float("nan"), 2: float("nan")}
    ext = np.array([max(float(shape[a]), 1.0) for a in (0, 1, 2)])
    p = c / ext
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    kk = int(min(max(k, 1), len(c) - 1))
    acc = np.zeros(3)
    for i in range(len(p)):
        nn = np.argpartition(d[i], kk - 1)[:kk]
        for j in nn:
            v = p[j] - p[i]
            n2 = float(v @ v)
            if n2 > 0:
                acc += (v * v) / n2
    tot = acc.sum()
    if tot <= 0:
        return {0: float("nan"), 1: float("nan"), 2: float("nan")}
    return {a: float(acc[a] / tot) for a in (0, 1, 2)}


def gini(x: Sequence[float]) -> float:
    """Gini coefficient of a non-negative vector. 0 = perfectly uniform, ->1 = concentrated."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.sum() <= 0:
        return float("nan")
    v = np.sort(np.maximum(v, 0))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def slice_concentration(centres: np.ndarray, axis: int, n_slices: int,
                        shadow_profile: np.ndarray | None = None) -> dict:
    """Do candidates pile onto a few slices of ``axis``, and are those the shadowed ones?

    Returns the Gini coefficient of the per-slice candidate count (over-dispersion vs a
    uniform spread), the share carried by the busiest 10% of slices, and — when a
    ``shadow_profile`` over the same axis is supplied — the Spearman correlation between
    per-slice candidate count and per-slice shadow mass.
    """
    c = np.asarray(centres, dtype=float)
    if len(c) == 0 or n_slices <= 0:
        return {"gini": float("nan"), "top10pct_share": float("nan"),
                "spearman_vs_shadow": float("nan"), "n": 0}
    idx = np.clip(c[:, axis].astype(int), 0, n_slices - 1)
    counts = np.bincount(idx, minlength=n_slices).astype(float)
    k = max(1, int(round(0.10 * n_slices)))
    top = float(np.sort(counts)[::-1][:k].sum() / max(counts.sum(), 1))
    sp = float("nan")
    if shadow_profile is not None and len(shadow_profile) == n_slices:
        s = np.asarray(shadow_profile, dtype=float)
        ok = np.isfinite(s) & np.isfinite(counts)
        if ok.sum() > 2 and counts[ok].std() > 0 and s[ok].std() > 0:
            rx = np.argsort(np.argsort(counts[ok])).astype(float)
            ry = np.argsort(np.argsort(s[ok])).astype(float)
            sp = float(np.corrcoef(rx, ry)[0, 1])
    return {"gini": gini(counts), "top10pct_share": top, "spearman_vs_shadow": sp,
            "n": int(len(c)), "counts": counts}


def dominant_period(profile: np.ndarray, min_lag: int = 3, max_lag: int | None = None) -> dict:
    """Strongest periodicity in a 1-D profile — the 'multiple parallel dark stripes' test.

    Autocorrelates the mean-removed profile and reports the lag of the strongest **local
    maximum** in ``[min_lag, max_lag]`` together with its normalised correlation. Ribs,
    Cooper's-ligament shadow families and reverberation bands all produce quasi-periodic
    dark banding; a clear peak here says the banding is regular rather than incidental.

    Requiring a local maximum, not merely the arg-max over the window, is what makes the
    result mean anything. Any smooth profile has a monotonically decaying autocorrelation,
    so an arg-max search returns ``min_lag`` with a near-1 correlation for *every* input
    and manufactures a "period" out of ordinary smoothness. When no interior peak exists
    the honest answer is NaN, and that is what this returns.
    """
    p = np.asarray(profile, dtype=float)
    p = p[np.isfinite(p)]
    n = len(p)
    if n < 8:
        return {"period": float("nan"), "strength": float("nan"), "n": n}
    p = p - p.mean()
    if p.std() <= 0:
        return {"period": float("nan"), "strength": float("nan"), "n": n}
    ac = np.correlate(p, p, mode="full")[n - 1:]
    ac = ac / ac[0]
    hi = min(max_lag if max_lag is not None else n // 2, n - 1)
    if hi <= min_lag + 1:
        return {"period": float("nan"), "strength": float("nan"), "n": n}
    lags = np.arange(min_lag, hi)
    interior = lags[(lags > 0) & (lags < len(ac) - 1)]
    peaks = [l for l in interior if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1] and ac[l] > 0]
    if not peaks:
        return {"period": float("nan"), "strength": float("nan"), "n": n,
                "autocorr": ac[:hi], "reason": "no interior autocorrelation peak"}
    best = max(peaks, key=lambda l: ac[l])
    return {"period": int(best), "strength": float(ac[best]), "n": n, "autocorr": ac[:hi]}


def map_correlation(map_a: np.ndarray, map_b: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Spearman + Pearson correlation between two co-registered 2-D maps.

    Used to ask "does candidate density track shadow density in this plane?". ``valid``
    restricts the comparison to meaningful cells (e.g. the tissue-line mask); cells that
    are NaN in either map are always dropped.
    """
    a = np.asarray(map_a, dtype=float).ravel()
    b = np.asarray(map_b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool).ravel()
    if ok.sum() < 3 or a[ok].std() <= 0 or b[ok].std() <= 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(ok.sum())}
    aa, bb = a[ok], b[ok]
    rx = np.argsort(np.argsort(aa)).astype(float)
    ry = np.argsort(np.argsort(bb)).astype(float)
    return {"pearson": float(np.corrcoef(aa, bb)[0, 1]),
            "spearman": float(np.corrcoef(rx, ry)[0, 1]), "n": int(ok.sum())}


def centroid_density_map(centres: np.ndarray, drop_axis: int, shape: Sequence[int],
                         out_shape: Sequence[int], sigma_bins: float = 1.5) -> np.ndarray:
    """Candidate-centroid density projected onto the plane that drops ``drop_axis``.

    Binned to ``out_shape`` (so it is directly comparable to a downsampled shadow map)
    and lightly smoothed, because centroid counts are sparse and an unsmoothed histogram
    correlates with nothing.
    """
    c = np.asarray(centres, dtype=float)
    keep = [a for a in (0, 1, 2) if a != drop_axis]
    h = np.zeros(tuple(out_shape), dtype=float)
    if len(c) == 0:
        return h
    for n in range(len(c)):
        i = int(np.clip(c[n, keep[0]] / max(shape[keep[0]], 1) * out_shape[0], 0, out_shape[0] - 1))
        j = int(np.clip(c[n, keep[1]] / max(shape[keep[1]], 1) * out_shape[1], 0, out_shape[1] - 1))
        h[i, j] += 1.0
    if sigma_bins > 0:
        h = _gauss2d(h, sigma_bins)
    return h


def _gauss2d(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding — avoids a scipy dependency here."""
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = a
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="reflect")
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), ax, p)
    return out


def downsample_map(m: np.ndarray, out_shape: Sequence[int]) -> np.ndarray:
    """Block-mean a 2-D map to ``out_shape``, ignoring NaNs."""
    a = np.asarray(m, dtype=float)
    oi, oj = int(out_shape[0]), int(out_shape[1])
    ei = np.linspace(0, a.shape[0], oi + 1).astype(int)
    ej = np.linspace(0, a.shape[1], oj + 1).astype(int)
    out = np.full((oi, oj), np.nan)
    for i in range(oi):
        for j in range(oj):
            blk = a[ei[i]:max(ei[i + 1], ei[i] + 1), ej[j]:max(ej[j + 1], ej[j] + 1)]
            if blk.size and np.isfinite(blk).any():
                out[i, j] = float(np.nanmean(blk))
    return out
