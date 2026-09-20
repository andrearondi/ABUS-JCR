"""Artifact-signature audit of the frozen candidate pools (iso substrate). READ-ONLY, torch-free.

Replaces the shape-only feasibility audit (clustering + beam elongation of the box) with three
blocks, each answering one question the thesis actually needs answered before the geometry term
is given any prior:

**Q1 — premise, per mechanism.** For every candidate, the intensity along the beam through the
candidate is compared with two lateral flanks at the same depth and sweep (the operational
shadow test of the shadow-detection literature: a bright rupture followed by a low-signal column
that continues to the bottom of the scan line). From that profile and the candidate's position
in the field, each candidate is assigned to ONE mechanism of the ABUS artifact taxonomy
(skin, marginal, deep/rib, reverberation, narrow column = Cooper's-like, broad column =
nipple/rib-like, shadowing blob, bounded blob, other). All thresholds are explicit ``Knobs``
and are swept by the caller. Every FP-vs-TP contrast is reported per volume first and within
size bins, because FPs are ~2x smaller than TPs and a scale-free ratio hides that.

**Q2 — arrangement.** Per (detector, volume), FP centroids in mm are tested for the three
arrangements a pairwise relative-geometry term could exploit — same-column stacking, sweep-
direction chains, regular lateral spacing inside a depth band — each against a permutation
null that keeps the per-axis marginals and breaks only the coupling under test.

**Q3 — consequence.** A grouped-by-volume out-of-fold logistic reader of the per-candidate
features (base token blocks, + signatures, + label-free context counts) is scored by the
official evaluator, so that "the premise holds / does not hold" is paired with "and it would /
would not move the metric".

No verdict is computed here or in the script. Numbers and figures are produced; the reading
is written by hand.

Axes: storage ``(d0, d1, d2) = (lateral, depth/beam, sweep)`` — ``conventions.LATERAL_AXIS``,
``DEPTH_AXIS``, ``SWEEP_AXIS``; the cache is 0.4 mm cubed on the ``measured`` profile, so one
voxel is one length on every axis and all outputs are in millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C

assert (C.LATERAL_AXIS, C.DEPTH_AXIS, C.SWEEP_AXIS) == (0, 1, 2), "audit assumes (lat, depth, sweep) storage order"

MECHANISMS = ("skin", "marginal", "deep", "reverberation", "narrow_column", "broad_column",
              "shadowing_blob", "bounded_blob", "other")

SIGNATURE_FEATURES = ("inside_contrast", "prox_contrast", "distal_contrast", "cap_bright",
                      "shadow_run_mm", "run_above_mm", "stripe_width_mm", "periodicity",
                      "abs_intensity", "depth_mm", "depth_frac", "contact_edge_mm", "sweep_edge_mm")
SIZE_FEATURES = ("diag_mm", "vol_mm3")
CONTEXT_FEATURES = ("n_band", "n_column", "n_chain", "set_size")
BASE_FEATURES = ("score_max", "score_mean", "score_std", "log_ext_d0", "log_ext_d1", "log_ext_d2", "rank_norm")


@dataclass(frozen=True)
class Knobs:
    """Every threshold the audit uses. Defaults are the values reported; the script sweeps them."""
    iso_mm: float = float(C.ISO_SPACING_MM)
    tau: float = 0.02               # contrast (on the [0,1] cache scale) below which a row counts as dark
    prox_mm: float = 4.0            # slab above the box for prox_contrast
    distal_mm: float = 8.0          # slab below the box for distal_contrast
    cap_mm: float = 2.0             # window above the dark run searched for the bright boundary
    flank_gap_mm: float = 1.2       # clearance between the box edge and the flank window
    dark_above_mm: float = 1.2      # dark run above the box top that makes the candidate a column segment
    run_mm: float = 6.0             # shadow run that counts as "casts a shadow"
    stripe_search_mm: float = 30.0  # half-width of the lateral search for the stripe edges
    stripe_smooth_mm: float = 1.2   # moving-average width on the lateral profile
    stripe_narrow_mm: float = 4.0   # narrow (Cooper's-like) vs broad column
    skin_mm: float = 5.0            # candidate CENTRE shallower than this -> skin (a big box's top reaches the skin)
    marginal_mm: float = 5.0        # box edge closer than this to a no-contact column or the field edge
    deep_frac: float = 0.75         # centre deeper than this fraction of the tissue floor -> deep/rib
    reverb_depth_mm: float = 10.0   # candidate centre shallower than this qualifies for the reverberation test
    reverb_range_mm: float = 12.0   # column profile length used for periodicity
    reverb_lag_mm: tuple = (0.8, 4.0)
    periodicity_thr: float = 0.5
    floor_frac: float = 0.25        # tissue floor: last row with mean > floor_frac * near-field peak
    floor_peak_mm: float = 12.0     # near-field peak searched in the first floor_peak_mm
    contact_frac: float = 0.20      # column mean below contact_frac * frame median -> no contact
    contact_depth_mm: tuple = (2.0, 20.0)
    reaches_frac: float = 0.90      # run covering this fraction of the rows to the floor -> reaches_floor
    band_mm: float = 3.0            # |d depth| for "same band"
    col_mm: float = 3.0             # |d lateral|, |d sweep| for "same column"
    stack_depth_mm: float = 5.0     # |d depth| beyond which two same-column points are "stacked"
    chain_mm: float = 5.0           # |d sweep| beyond which two same-in-plane points are a "chain"
    band_sweep_mm: float = 5.0      # |d sweep| within which band pairs are compared laterally
    offset_range_mm: tuple = (2.0, 40.0)
    min_pairs: int = 8


# --------------------------------------------------------------------------- small helpers

def _vox(mm: float, k: Knobs) -> int:
    return max(1, int(round(mm / k.iso_mm)))


def _clip(lo: float, hi: float, n: int):
    lo_i, hi_i = int(round(lo)), int(round(hi))
    lo_i = max(0, min(n - 1, lo_i))
    hi_i = max(lo_i + 1, min(n, hi_i))
    return lo_i, hi_i


def _run_length(x: np.ndarray, thr: float) -> int:
    """Number of leading entries of ``x`` strictly below ``thr`` (NaN stops the run)."""
    n = 0
    for v in x:
        if np.isfinite(v) and v < thr:
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------- Q1: signatures

def tissue_floor_row(vol: np.ndarray, k: Knobs) -> int:
    """Last depth row whose volume-mean intensity exceeds ``floor_frac`` x the near-field peak."""
    prof = np.asarray(vol).mean(axis=(0, 2))
    peak = float(prof[: _vox(k.floor_peak_mm, k)].max())
    above = np.flatnonzero(prof > k.floor_frac * peak)
    return int(above[-1]) if above.size else int(len(prof) - 1)


def beam_profile(vol: np.ndarray, cen: Sequence[float], ext: Sequence[float], k: Knobs) -> Dict:
    """Depth profile of the candidate's central column and of two lateral flanks, plus contrast.

    Column footprint = central half of the box in lateral and sweep. Flanks = the same footprint
    shifted laterally by +-(half box lateral extent + half footprint + ``flank_gap_mm``), same
    sweep. ``contrast[d] = col[d] - mean(flanks)[d]`` for every depth row ``d``.
    """
    D0, D1, D2 = vol.shape
    c0, c1, c2 = (float(x) for x in cen)
    e0, e1, e2 = (float(x) for x in ext)
    l0, l1 = _clip(c0 - e0 / 4, c0 + e0 / 4, D0)
    s0, s1 = _clip(c2 - e2 / 4, c2 + e2 / 4, D2)
    top, bot = _clip(c1 - e1 / 2, c1 + e1 / 2, D1)
    w = l1 - l0
    off = int(round(e0 / 2 + w / 2 + _vox(k.flank_gap_mm, k)))
    col = np.asarray(vol[l0:l1, :, s0:s1], dtype=np.float64).mean(axis=(0, 2))
    flanks = []
    for a, b in ((l0 - off, l1 - off), (l0 + off, l1 + off)):
        if a >= 0 and b <= D0:
            flanks.append(np.asarray(vol[a:b, :, s0:s1], dtype=np.float64).mean(axis=(0, 2)))
    if flanks:
        flank = np.mean(flanks, axis=0)
        contrast = col - flank
    else:
        flank = np.full(D1, np.nan)
        contrast = np.full(D1, np.nan)
    return {"col": col, "flank": flank, "contrast": contrast, "top": top, "bot": bot,
            "l0": l0, "l1": l1, "s0": s0, "s1": s1, "n_flanks": len(flanks)}


def periodicity(profile: np.ndarray, k: Knobs) -> float:
    """Max normalised autocorrelation of the linearly detrended profile at lags 0.8-4 mm, clipped at 0."""
    x = np.asarray(profile, dtype=np.float64)
    n = len(x)
    if n < 6:
        return 0.0
    t = np.arange(n)
    coef = np.polyfit(t, x, 1)
    x = x - np.polyval(coef, t)
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        return 0.0
    lo, hi = _vox(k.reverb_lag_mm[0], k), _vox(k.reverb_lag_mm[1], k)
    best = 0.0
    for lag in range(max(1, lo), min(hi, n - 2) + 1):
        r = float(np.sum(x[:-lag] * x[lag:]) / denom)
        best = max(best, r)
    return best


def stripe_width_mm(vol: np.ndarray, cen: Sequence[float], ext: Sequence[float], k: Knobs) -> float:
    """Width of the dark lateral run containing the box centre, in the row through the centre.

    Reference = median of the full lateral profile at that depth row (sweep footprint = central
    half of the box); a column is dark when the smoothed profile is below the midpoint between
    that reference and the centre value. Returns 0 when the centre itself is not dark.
    """
    D0, D1, D2 = vol.shape
    c0, c1, c2 = (float(x) for x in cen)
    e2 = float(ext[2])
    row = int(min(D1 - 1, max(0, round(c1))))
    s0, s1 = _clip(c2 - e2 / 4, c2 + e2 / 4, D2)
    prof = np.asarray(vol[:, row, s0:s1], dtype=np.float64).mean(axis=1)
    sm = _vox(k.stripe_smooth_mm, k)
    if sm > 1:
        kern = np.ones(sm) / sm
        prof = np.convolve(prof, kern, mode="same")
    ref = float(np.median(prof))
    ci = int(min(D0 - 1, max(0, round(c0))))
    thr = 0.5 * (ref + prof[ci])
    if not prof[ci] < thr:
        return 0.0
    half = _vox(k.stripe_search_mm, k)
    lo = ci
    while lo - 1 >= max(0, ci - half) and prof[lo - 1] < thr:
        lo -= 1
    hi = ci
    while hi + 1 < min(D0, ci + half + 1) and prof[hi + 1] < thr:
        hi += 1
    return float((hi - lo + 1) * k.iso_mm)


def contact_edge_mm(vol: np.ndarray, cen: Sequence[float], ext: Sequence[float], k: Knobs) -> float:
    """Distance from the box's lateral edges to the nearest no-contact column or field edge.

    A lateral column has no contact when its mean intensity over 2-20 mm depth (sweep footprint
    = central half of the box) is below ``contact_frac`` x the median column of that frame.
    """
    D0, D1, D2 = vol.shape
    c0, c1, c2 = (float(x) for x in cen)
    e0, e2 = float(ext[0]), float(ext[2])
    r0, r1 = _clip(_vox(k.contact_depth_mm[0], k), _vox(k.contact_depth_mm[1], k), D1)
    s0, s1 = _clip(c2 - e2 / 4, c2 + e2 / 4, D2)
    colmean = np.asarray(vol[:, r0:r1, s0:s1], dtype=np.float64).mean(axis=(1, 2))
    ref = float(np.median(colmean))
    nocontact = np.flatnonzero(colmean < k.contact_frac * ref)
    bl, br = _clip(c0 - e0 / 2, c0 + e0 / 2, D0)
    left = nocontact[nocontact < bl]
    right = nocontact[nocontact >= br]
    d_left = (bl - int(left[-1])) if left.size else bl
    d_right = (int(right[0]) - br) if right.size else (D0 - br)
    return float(min(d_left, d_right) * k.iso_mm)


def candidate_signature(vol: np.ndarray, cen: Sequence[float], ext: Sequence[float],
                        floor_row: int, k: Knobs) -> Dict:
    """All Q1 features of one candidate (mm units, cache intensity scale)."""
    D0, D1, D2 = vol.shape
    bp = beam_profile(vol, cen, ext, k)
    ct, top, bot = bp["contrast"], bp["top"], bp["bot"]
    floor_row = int(min(D1 - 1, max(bot, floor_row)))
    iso = k.iso_mm

    inside = float(np.nanmean(ct[top:bot])) if bot > top else float("nan")
    p_lo = max(0, top - _vox(k.prox_mm, k))
    prox = float(np.nanmean(ct[p_lo:top])) if top > p_lo else float("nan")
    d_hi = min(D1, bot + _vox(k.distal_mm, k))
    distal = float(np.nanmean(ct[bot:d_hi])) if d_hi > bot else float("nan")

    run_below = _run_length(ct[bot:floor_row + 1], -k.tau)
    run_above = _run_length(ct[:top][::-1], -k.tau)
    rows_to_floor = floor_row - bot + 1
    reaches = bool(rows_to_floor > 0 and run_below >= k.reaches_frac * rows_to_floor)
    dark_above = bool(run_above * iso >= k.dark_above_mm)

    cap_hi = top - run_above
    cap_lo = max(0, cap_hi - _vox(k.cap_mm, k))
    cap = float(np.nanmax(ct[cap_lo:cap_hi])) if cap_hi > cap_lo else float("nan")

    col = bp["col"]
    per = periodicity(col[: _vox(k.reverb_range_mm, k)], k)
    stripe = stripe_width_mm(vol, cen, ext, k)
    contact = contact_edge_mm(vol, cen, ext, k)
    abs_int = float(np.asarray(vol[bp["l0"]:bp["l1"], top:bot, bp["s0"]:bp["s1"]], dtype=np.float64).mean())

    c1, c2 = float(cen[1]), float(cen[2])
    e = np.asarray(ext, dtype=np.float64)
    return {
        "inside_contrast": inside, "prox_contrast": prox, "distal_contrast": distal, "cap_bright": cap,
        "shadow_run_mm": float(run_below * iso), "run_above_mm": float(run_above * iso),
        "dark_above": dark_above, "reaches_floor": reaches,
        "stripe_width_mm": stripe, "periodicity": float(per), "abs_intensity": abs_int,
        "depth_mm": c1 * iso, "top_mm": float(top * iso), "floor_mm": float(floor_row * iso),
        "depth_frac": float(c1 / max(1, floor_row)),
        "contact_edge_mm": contact, "sweep_edge_mm": float(min(c2, D2 - c2) * iso),
        "diag_mm": float(np.sqrt(np.sum(e ** 2)) * iso), "vol_mm3": float(np.prod(e) * iso ** 3),
        "n_flanks": int(bp["n_flanks"]),
    }


def assign_mechanism(f: Dict, k: Knobs) -> str:
    """Mutually exclusive mechanism label, applied in the order of the taxonomy."""
    if f["depth_mm"] < k.skin_mm:
        return "skin"
    if f["contact_edge_mm"] < k.marginal_mm:
        return "marginal"
    if f["depth_frac"] > k.deep_frac:
        return "deep"
    if f["depth_mm"] < k.reverb_depth_mm and f["periodicity"] > k.periodicity_thr:
        return "reverberation"
    casts = f["shadow_run_mm"] >= k.run_mm
    if f["dark_above"] and casts:
        return "narrow_column" if f["stripe_width_mm"] <= k.stripe_narrow_mm else "broad_column"
    dark_inside = np.isfinite(f["inside_contrast"]) and f["inside_contrast"] < -k.tau
    if dark_inside and casts:
        return "shadowing_blob"
    if dark_inside:
        return "bounded_blob"
    return "other"


def signature_table(load_vol: Callable[[int], np.ndarray], df: pd.DataFrame, k: Knobs,
                    progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Signatures + mechanism for every row of a candidate record (any labels)."""
    rows: List[Dict] = []
    for vid, g in df.groupby("public_id", sort=True):
        vol = load_vol(int(vid))
        floor = tissue_floor_row(vol, k)
        for idx, r in g.iterrows():
            f = candidate_signature(vol, (r.cen_d0, r.cen_d1, r.cen_d2), (r.ext_d0, r.ext_d1, r.ext_d2), floor, k)
            f["mechanism"] = assign_mechanism(f, k)
            f["_idx"] = idx
            rows.append(f)
        if progress:
            progress(f"vol {vid}: {len(g)} candidates")
    sig = pd.DataFrame(rows).set_index("_idx").sort_index()
    return df.join(sig)


# --------------------------------------------------------------------------- statistics

def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b), NaN-safe, NaN if either side is empty."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = np.sum(a[:, None] > b[None, :]); lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (a.size * b.size))


def per_volume_delta(df: pd.DataFrame, feat: str, min_fp: int = 3, min_tp: int = 1) -> np.ndarray:
    """Cliff's delta (FP vs TP) inside each (detector, volume) set with enough of both."""
    out = []
    for _, g in df.groupby(["detector_of_origin", "public_id"], sort=False):
        a = g.loc[g["label"] == "neg", feat]; b = g.loc[g["label"] == "pos", feat]
        if len(a) >= min_fp and len(b) >= min_tp:
            d = cliffs_delta(a, b)
            if np.isfinite(d):
                out.append(d)
    return np.asarray(out, dtype=float)


def size_binned_delta(df: pd.DataFrame, feat: str, nbins: int = 4) -> Dict:
    """FP-vs-TP delta within diagonal quantile bins; weighted by min(n_tp, n_fp) per bin."""
    g = df[df["label"].isin(["pos", "neg"])]
    try:
        bins = pd.qcut(g["diag_mm"], nbins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(g), dtype=int), index=g.index)
    edges, deltas, n_tp, n_fp = [], [], [], []
    for b in sorted(bins.dropna().unique()):
        gb = g[bins == b]
        tp = gb.loc[gb["label"] == "pos", feat]; fp = gb.loc[gb["label"] == "neg", feat]
        edges.append((float(gb["diag_mm"].min()), float(gb["diag_mm"].max())))
        deltas.append(cliffs_delta(fp, tp)); n_tp.append(int(len(tp))); n_fp.append(int(len(fp)))
    w = np.array([min(a, b) for a, b in zip(n_tp, n_fp)], dtype=float)
    d = np.array(deltas, dtype=float)
    ok = np.isfinite(d) & (w > 0)
    weighted = float(np.sum(w[ok] * d[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
    return {"bins": edges, "delta": deltas, "n_tp": n_tp, "n_fp": n_fp, "weighted": weighted}


def contrast_report(df: pd.DataFrame, feats: Sequence[str]) -> pd.DataFrame:
    """One row per feature: medians, pooled delta, per-volume delta + sign, size-matched delta."""
    g = df[df["label"].isin(["pos", "neg"])]
    tp = g[g["label"] == "pos"]; fp = g[g["label"] == "neg"]
    rows = []
    for f in feats:
        pv = per_volume_delta(g, f)
        sb = size_binned_delta(g, f)
        rows.append({
            "feature": f, "tp_median": float(np.nanmedian(tp[f])) if len(tp) else np.nan,
            "fp_median": float(np.nanmedian(fp[f])) if len(fp) else np.nan,
            "delta_pooled": cliffs_delta(fp[f], tp[f]),
            "delta_pervol_median": float(np.median(pv)) if pv.size else np.nan,
            "sign_frac": float(np.mean(pv > 0)) if pv.size else np.nan, "n_vol": int(pv.size),
            "delta_size_matched": sb["weighted"],
            "delta_by_size_bin": sb["delta"],
        })
    return pd.DataFrame(rows)


def partition_report(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of candidates per mechanism, per (detector, label): pooled and per-volume median."""
    g = df[df["label"].isin(["pos", "neg"])]
    rows = []
    for (det, lab), gg in g.groupby(["detector_of_origin", "label"], sort=True):
        pooled = gg["mechanism"].value_counts(normalize=True)
        pv = gg.groupby("public_id")["mechanism"].value_counts(normalize=True).unstack(fill_value=0.0)
        for m in MECHANISMS:
            rows.append({"detector": det, "label": lab, "mechanism": m, "n": int((gg["mechanism"] == m).sum()),
                         "frac_pooled": float(pooled.get(m, 0.0)),
                         "frac_pervol_median": float(pv[m].median()) if m in pv else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Q2: arrangement

def _pair_diffs(p: np.ndarray):
    iu = np.triu_indices(len(p), 1)
    d = np.abs(p[iu[0]] - p[iu[1]])
    return d[:, 0], d[:, 1], d[:, 2]          # |d lat|, |d depth|, |d sweep|


def pair_counts(p_mm: np.ndarray, k: Knobs) -> Dict:
    """Raw arrangement statistics of one point cloud (mm): column stacks, sweep chains, band regularity."""
    p = np.asarray(p_mm, dtype=float)
    if len(p) < 2:
        return {"same_column": 0.0, "sweep_chain": 0.0, "band_regularity": float("nan"), "n_band_pairs": 0}
    dl, dd, ds = _pair_diffs(p)
    same_col = float(np.sum((dl < k.col_mm) & (ds < k.col_mm) & (dd > k.stack_depth_mm)))
    chain = float(np.sum((dl < k.col_mm) & (dd < k.band_mm) & (ds > k.chain_mm)))
    band = (dd < k.band_mm) & (ds < k.band_sweep_mm) & (dl >= k.offset_range_mm[0]) & (dl < k.offset_range_mm[1])
    offs = dl[band]
    if offs.size >= k.min_pairs:
        # Regular bands at spacing s put the pairwise lateral offsets at multiples of s, so the
        # 1 mm offset histogram is periodic with period s. Statistic = peak normalised
        # autocorrelation of the (centred) histogram at lags 3..19 mm. (A spectral peak ratio was
        # tried first and failed the planted-band phantom: the triangular envelope of a finite
        # comb dominates its spectrum.)
        edges = np.arange(k.offset_range_mm[0], k.offset_range_mm[1] + 1e-9, 1.0)
        h, _ = np.histogram(offs, bins=edges)
        x = h - h.mean()
        den = float(np.sum(x * x))
        if den > 0:
            reg = max(float(np.sum(x[:-lag] * x[lag:]) / den) for lag in range(3, min(20, len(x) - 1)))
        else:
            reg = float("nan")
    else:
        reg = float("nan")
    return {"same_column": same_col, "sweep_chain": chain, "band_regularity": reg, "n_band_pairs": int(offs.size)}


def arrangement_null(p_mm: np.ndarray, k: Knobs, rng: np.random.Generator, n_perm: int = 200) -> Dict:
    """Observed statistics against permutation nulls that keep the marginals.

    ``same_column`` and ``sweep_chain``: lateral coordinates permuted among the points (in-plane
    coincidence broken, depth bands kept). ``band_regularity``: lateral coordinates redrawn
    uniformly inside the observed lateral range (spacing randomised, band membership kept).
    """
    p = np.asarray(p_mm, dtype=float)
    obs = pair_counts(p, k)
    nulls = {"same_column": [], "sweep_chain": [], "band_regularity": []}
    lo, hi = (p[:, 0].min(), p[:, 0].max()) if len(p) else (0.0, 1.0)
    for _ in range(n_perm):
        q = p.copy(); q[:, 0] = rng.permutation(p[:, 0])
        pc = pair_counts(q, k)
        nulls["same_column"].append(pc["same_column"]); nulls["sweep_chain"].append(pc["sweep_chain"])
        u = p.copy(); u[:, 0] = rng.uniform(lo, hi, size=len(p))
        nulls["band_regularity"].append(pair_counts(u, k)["band_regularity"])
    out = {"n_points": int(len(p)), "n_band_pairs": obs["n_band_pairs"]}
    for name in ("same_column", "sweep_chain", "band_regularity"):
        arr = np.asarray(nulls[name], dtype=float); arr = arr[np.isfinite(arr)]
        o = obs[name]
        if arr.size == 0 or not np.isfinite(o):
            out[name] = {"obs": o, "null_mean": np.nan, "null_sd": np.nan, "null_p95": np.nan, "z": np.nan, "exceeds": False}
            continue
        mean, sd, p95 = float(arr.mean()), float(arr.std(ddof=0)), float(np.percentile(arr, 95))
        if sd > 0:
            z = float(np.clip((o - mean) / sd, -99.0, 99.0))
        else:
            z = 99.0 if o > mean else 0.0
        out[name] = {"obs": float(o), "null_mean": mean, "null_sd": sd, "null_p95": p95, "z": z, "exceeds": bool(o > p95)}
    return out


def context_counts(p_mm: np.ndarray, k: Knobs = Knobs()) -> np.ndarray:
    """Label-free per-point counts: (n_band, n_column, n_chain, set_size)."""
    p = np.asarray(p_mm, dtype=float)
    n = len(p)
    out = np.zeros((n, 4), dtype=float)
    if n == 0:
        return out
    d = np.abs(p[:, None, :] - p[None, :, :])
    np.fill_diagonal(d[:, :, 0], np.inf)          # a point never pairs with itself
    dl, dd, ds = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    out[:, 0] = ((dd < k.band_mm) & np.isfinite(dl)).sum(axis=1)
    out[:, 1] = ((dl < k.col_mm) & (ds < k.col_mm)).sum(axis=1)
    out[:, 2] = ((dl < k.col_mm) & (dd < k.band_mm) & (ds > k.chain_mm)).sum(axis=1)
    out[:, 3] = n
    return out


def hit_vs_field(p_fp_mm: np.ndarray, hit_mm: np.ndarray, k: Knobs = Knobs()) -> Dict:
    """Counts of the best hit against the FP field vs the median FP's counts against the other FPs."""
    fp = np.asarray(p_fp_mm, dtype=float)
    out = {"hit_band": np.nan, "hit_column": np.nan, "hit_chain": np.nan,
           "fp_band_med": np.nan, "fp_column_med": np.nan, "fp_chain_med": np.nan}
    if len(fp) == 0:
        return out
    h = np.asarray(hit_mm, dtype=float).reshape(1, 3)
    d = np.abs(fp - h)
    out["hit_band"] = float(np.sum(d[:, 1] < k.band_mm))
    out["hit_column"] = float(np.sum((d[:, 0] < k.col_mm) & (d[:, 2] < k.col_mm)))
    out["hit_chain"] = float(np.sum((d[:, 0] < k.col_mm) & (d[:, 1] < k.band_mm) & (d[:, 2] > k.chain_mm)))
    if len(fp) >= 2:
        cc = context_counts(fp, k)
        out["fp_band_med"] = float(np.median(cc[:, 0])); out["fp_column_med"] = float(np.median(cc[:, 1]))
        out["fp_chain_med"] = float(np.median(cc[:, 2]))
    return out


# --------------------------------------------------------------------------- Q3: ranking information

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def irls_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50):
    """L2-regularised logistic regression by Newton/IRLS; returns (w, b) in the ORIGINAL feature scale."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    A = np.hstack([Xs, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2); reg[-1] = 0.0
    for _ in range(iters):
        p = _sigmoid(A @ w)
        g = A.T @ (p - y) + reg * w
        W = p * (1 - p) + 1e-9
        H = (A * W[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    w_raw = w[:-1] / sd
    b_raw = float(w[-1] - np.sum(w[:-1] * mu / sd))
    return w_raw, b_raw


def grouped_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per row; every group (volume) lands entirely in one fold. ``k <= 0`` = one fold per group
    (leave-one-volume-out), which keeps the pooled out-of-fold scores on one calibration — pooling the
    outputs of a few dissimilar fold models is itself a cross-volume miscalibration the metric punishes."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if k <= 0:
        k = len(uniq)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[i]: int(j % k) for j, i in enumerate(perm)}
    return np.asarray([fold_of[g] for g in groups], dtype=int)


def grouped_oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, fit_mask: np.ndarray,
                       k: int = 5, seed: int = 0, l2: float = 1.0) -> np.ndarray:
    """Out-of-fold probabilities in [0, 1) for EVERY row; only ``fit_mask`` rows are fitted on."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    folds = grouped_folds(groups, k, seed)
    oof = np.zeros(len(X))
    for f in range(int(folds.max()) + 1):
        tr = (folds != f) & fit_mask; te = folds == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        w, b = irls_logistic(X[tr], y[tr], l2=l2)
        oof[te] = _sigmoid(X[te] @ w + b)
    return np.clip(oof, 0.0, 1.0 - 1e-6)


def auc(scores, y) -> float:
    """Rank-based AUC (Mann-Whitney), ties counted half."""
    s = np.asarray(scores, dtype=float); y = np.asarray(y).astype(bool)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = np.sum(pos[:, None] > neg[None, :]); eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (pos.size * neg.size))
