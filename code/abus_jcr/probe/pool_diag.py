"""[P3U2.PD] Frozen candidate-pool deep diagnostics — torch-free core.

Reads only the frozen record columns (no detector / iso-cache / torch). Four analysis blocks that
inform Phase 4:

1. ``feature_discriminability`` — which per-candidate features separate TP from FP (→ which Phase-4
   feature *blocks* carry signal): Cliff's delta + best single-threshold balanced accuracy per feature.
2. ``ranking_headroom`` — where the TPs sit in the baseline ``score_max`` ranking, and an approximate
   recall-at-FP/vol curve (→ how much a re-ranker can win).
3. ``pairwise_geometry`` — the 6-D relative-log-geometry ``g(m,n)`` (the exact Axis-A descriptor) for
   TP-TP / TP-FP / FP-FP pairs (→ the DIRECT pairwise test the per-candidate FP-probe could not do).
4. ``set_structure`` — per-volume candidate counts, TP/FP/ignore mix, and spatial-cluster redundancy
   (clustered in ISO-CACHE voxels — see the units note in :func:`set_structure`; it clustered native
   voxel indices until 2026-08-27 and those recorded values are not comparable to any other probe).

TP := ``label == 'pos'`` (iou_gt > 0.30); FP := ``label == 'neg'``; the ignore band is excluded from
TP-vs-FP contrasts (Inv. 11).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C
from ..conventions import KEY_FP

EPS = 1e-6

# The per-candidate features present in the frozen record (+ two derived below).
POOL_FEATURES: List[str] = [
    "score_max", "score_mean", "score_std", "score_min",
    "slice_count", "z_span", "fill_ratio",
    "centroid_jitter", "area_cv", "rank_norm",
    "box_diag", "anisotropy",
]


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the two derived features: ``box_diag`` (official 3D diagonal) and
    ``anisotropy`` = ``ext_d0 / mean(ext_d1, ext_d2)``, matching the Phase-0b probe exactly.

    AXIS, corrected 2026-08-09 (the value and every recorded number are unchanged): ``d0`` is
    the MEASURED **lateral** axis, so this is lateral elongation, not depth elongation as the
    name suggests. See ``probe.fp_structure._anisotropy`` and ``[I.6b]`` for the full note and
    for the beam-axis measurement this one does NOT make.

    The numerator axis is ``conventions.FP_PROBE_ANISO_DEPTH_AXIS`` (2026-08-09): **0** on the
    ``legacy`` profile, so every recorded table reproduces byte-for-byte, and **1** on
    ``measured``, where it is the true beam axis on a genuinely cubic voxel. Kept identical to
    ``probe.fp_structure._anisotropy`` by construction — the two must never drift apart."""
    out = df.copy()
    out["box_diag"] = np.sqrt(out["x_length"].to_numpy(float) ** 2
                              + out["y_length"].to_numpy(float) ** 2
                              + out["z_length"].to_numpy(float) ** 2)
    a = int(C.FP_PROBE_ANISO_DEPTH_AXIS)
    others = [out[f"ext_d{k}"].to_numpy(float) for k in range(3) if k != a]
    lat = (others[0] + others[1]) / 2.0
    num = out[f"ext_d{a}"].to_numpy(float)
    out["anisotropy"] = np.where(lat > 0, num / np.where(lat > 0, lat, 1.0), np.nan)
    return out


def _cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1]: P(a>b) - P(a<b). O((na+nb) log nb). +1 => a all above b."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    bs = np.sort(b)
    n_b_lt_a = np.searchsorted(bs, a, side="left").sum()     # #(b < a)
    n_b_gt_a = (nb - np.searchsorted(bs, a, side="right")).sum()  # #(b > a)
    return float((n_b_lt_a - n_b_gt_a) / (na * nb))


def _set_groups(df: pd.DataFrame):
    """Yield (detector_of_origin, public_id, sub-df) per RESCORER SET. The rescorer runs on one
    detector's candidates for one volume at a time (Inv. 7), so the set — not the bare volume — is the
    unit for ranking/geometry/structure. Falls back to volume-only if the column is absent."""
    keys = ["detector_of_origin", "public_id"] if "detector_of_origin" in df.columns else ["public_id"]
    for k, g in df.groupby(keys, sort=False):
        det, pid = (k if isinstance(k, tuple) else ("ALL", k))
        yield det, pid, g


def best_balacc(pos: Sequence[float], neg: Sequence[float]) -> tuple:
    """Best single-threshold balanced accuracy (direction-agnostic) + the threshold. O(u*(np+nn)).

    Public since 2026-08-09: Phase-4's [4.3] reports B1's balanced accuracy next to its CPM,
    so a B1 that fails to clear B0 can be attributed (broken pipeline vs strong B0) against
    this pool's own single-feature ceiling. One implementation, so the two are comparable.
    """
    pos = np.asarray(pos, float); pos = pos[np.isfinite(pos)]
    neg = np.asarray(neg, float); neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    cands = np.unique(np.concatenate([pos, neg]))
    if len(cands) > 512:                                 # cap for speed on large pools
        cands = np.quantile(cands, np.linspace(0, 1, 512))
    best_ba, best_t = -1.0, float("nan")
    for t in cands:
        for hi in (True, False):
            tpr = float((pos >= t).mean()) if hi else float((pos < t).mean())
            tnr = float((neg < t).mean()) if hi else float((neg >= t).mean())
            ba = 0.5 * (tpr + tnr)
            if ba > best_ba:
                best_ba, best_t = ba, float(t)
    return best_ba, best_t


def feature_discriminability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature TP-vs-FP separation, sorted by |Cliff's delta| desc. One row per POOL_FEATURES."""
    d = augment(df)
    tp = d[d["label"] == "pos"]
    fp = d[d["label"] == "neg"]
    rows = []
    for f in POOL_FEATURES:
        if f not in d.columns:
            continue
        a, b = tp[f].to_numpy(float), fp[f].to_numpy(float)
        delta = _cliffs_delta(a, b)
        ba, thr = best_balacc(a, b)
        rows.append({"feature": f, "n_tp": int(np.isfinite(a).sum()), "n_fp": int(np.isfinite(b).sum()),
                     "tp_median": float(np.nanmedian(a)) if len(a) else float("nan"),
                     "fp_median": float(np.nanmedian(b)) if len(b) else float("nan"),
                     "cliffs_delta": delta, "balacc": ba, "best_thresh": thr})
    out = pd.DataFrame(rows)
    return out.reindex(out["cliffs_delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def ranking_headroom(df: pd.DataFrame, fp_budgets: Sequence[float] = KEY_FP) -> dict:
    """Where TPs rank by ``score_max`` per volume + an approximate recall-at-FP/vol curve.

    ``best_tp_rank`` = 1-indexed rank of the highest-scoring TP in its volume (NaN if none). The
    recall-at-FP curve is computed from labels directly (denominator = volumes present in ``df``; the
    official CPM is B0'). Returns per-vol records, the fraction of TP-bearing volumes whose best TP is
    not already rank-1, a rank histogram, and ``recall_at_fp``.
    """
    per_vol = []
    for det, pid, g in _set_groups(df):
        g = g.sort_values("score_max", ascending=False, kind="stable").reset_index(drop=True)
        pos_pos = np.where(g["label"].to_numpy() == "pos")[0]
        best_rank = int(pos_pos.min() + 1) if len(pos_pos) else None
        per_vol.append({"detector_of_origin": det, "public_id": pid, "n_cands": int(len(g)),
                        "n_tp": int((g["label"] == "pos").sum()), "best_tp_rank": best_rank})
    tp_ranks = [r["best_tp_rank"] for r in per_vol if r["best_tp_rank"] is not None]
    frac_not_rank1 = float(np.mean([r > 1 for r in tp_ranks])) if tp_ranks else float("nan")
    hist = {int(k): int(v) for k, v in zip(*np.unique(np.clip(tp_ranks, 1, 10), return_counts=True))} if tp_ranks else {}

    n_sets = len(per_vol)                                  # rescorer sets (det,vol), not bare volumes
    neg_scores = np.sort(df[df["label"] == "neg"]["score_max"].to_numpy(float))[::-1]  # desc
    pos = df[df["label"] == "pos"]
    n_pos_sets = len({(r["detector_of_origin"], r["public_id"]) for r in per_vol if r["n_tp"] > 0})
    recall_at_fp = {}
    for b in fp_budgets:
        # allow k = floor(b * n_sets) FPs above the threshold; thr = the (k+1)-th highest neg score, so
        # #(neg > thr) == k. As b grows, k grows, thr falls -> recall is non-decreasing (a sound curve).
        k = int(np.floor(b * n_sets))
        thr = float(neg_scores[k]) if k < len(neg_scores) else -np.inf
        hit_pos = pos[pos["score_max"].to_numpy(float) > thr]
        hit = hit_pos.groupby(["detector_of_origin", "public_id"]).ngroups if "detector_of_origin" in df.columns \
            else hit_pos["public_id"].nunique()
        recall_at_fp[float(b)] = hit / n_sets if n_sets else float("nan")
    return {"per_vol": per_vol, "frac_best_tp_not_rank1": frac_not_rank1,
            "best_tp_rank_hist": hist, "recall_at_fp": recall_at_fp, "n_sets": n_sets, "n_pos_sets": n_pos_sets}


def relative_geometry(cx_m, cy_m, cz_m, w_m, h_m, d_m, cx_n, cy_n, cz_n, w_n, h_n, d_n) -> np.ndarray:
    """The 6-D relative-log-geometry g(m,n) (Relation-Networks / iRPE descriptor; PHASE_4 §1.4).

    ``[log(|dx|/w_m+eps), log(|dy|/h_m+eps), log(|dz|/d_m+eps), log(w_n/w_m), log(h_n/h_m), log(d_n/d_m)]``.
    """
    return np.array([
        np.log(abs(cx_m - cx_n) / (w_m + EPS) + EPS),
        np.log(abs(cy_m - cy_n) / (h_m + EPS) + EPS),
        np.log(abs(cz_m - cz_n) / (d_m + EPS) + EPS),
        np.log((w_n + EPS) / (w_m + EPS)),
        np.log((h_n + EPS) / (h_m + EPS)),
        np.log((d_n + EPS) / (d_m + EPS)),
    ], dtype=float)


def pairwise_geometry(df: pd.DataFrame, max_fpfp_per_vol: int = 200) -> dict:
    """Within each volume, g(m,n) for TP-TP / TP-FP / FP-FP ordered pairs (Axis-A direct test).

    Keeps ALL TP-involving pairs (TPs are few) + a deterministic stride-sample of FP-FP pairs (cap
    ``max_fpfp_per_vol``). Returns per-type g arrays, per-component medians, and ``separability_per_component``
    = |Cliff's delta| of each g-component between TP-FP and FP-FP pairs (does a TP-involving pair have
    distinguishable relative geometry from a random FP-FP pair?).
    """
    g_by = {"TP-TP": [], "TP-FP": [], "FP-FP": []}
    counts = {"TP-TP": 0, "TP-FP": 0, "FP-FP": 0}
    for _det, _pid, grp in _set_groups(df):
        grp = grp[grp["label"].isin(["pos", "neg"])].sort_values("candidate_id" if "candidate_id" in grp
                                                                  else "score_max").reset_index(drop=True)
        n = len(grp)
        if n < 2:
            continue
        cx = grp["coordX"].to_numpy(float); cy = grp["coordY"].to_numpy(float); cz = grp["coordZ"].to_numpy(float)
        w = grp["x_length"].to_numpy(float); h = grp["y_length"].to_numpy(float); d = grp["z_length"].to_numpy(float)
        is_pos = (grp["label"].to_numpy() == "pos")
        fpfp_kept = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kind = ("TP-TP" if is_pos[i] and is_pos[j] else
                        "FP-FP" if not is_pos[i] and not is_pos[j] else "TP-FP")
                if kind == "FP-FP":
                    if fpfp_kept >= max_fpfp_per_vol or ((i * n + j) % max(1, (n * n) // max(max_fpfp_per_vol, 1))):
                        continue
                    fpfp_kept += 1
                counts[kind] += 1
                g_by[kind].append(relative_geometry(cx[i], cy[i], cz[i], w[i], h[i], d[i],
                                                    cx[j], cy[j], cz[j], w[j], h[j], d[j]))
    g_arrays = {k: (np.asarray(v, float) if v else np.zeros((0, 6))) for k, v in g_by.items()}
    med = {k: (np.median(a, axis=0).tolist() if len(a) else [float("nan")] * 6) for k, a in g_arrays.items()}

    def _sep(kind_a, kind_b):
        out = []
        for c in range(6):
            a = g_arrays[kind_a][:, c] if len(g_arrays[kind_a]) else np.zeros(0)
            b = g_arrays[kind_b][:, c] if len(g_arrays[kind_b]) else np.zeros(0)
            out.append(abs(_cliffs_delta(a, b)) if len(a) and len(b) else float("nan"))
        return out

    # TP-FP vs FP-FP: can a TP-involving CROSS pair be told from a random FP-FP pair? (FP-suppression view)
    # TP-TP vs TP-FP: from a TP's viewpoint, is its geometry to a co-located TP PEER distinguishable from its
    #   geometry to an FP? (the density/consensus signal — "am I in a tight cluster of real peers?").
    return {"counts": counts, "g": {k: a.tolist() for k, a in g_arrays.items()},
            "median_per_component": med,
            "separability_per_component": _sep("TP-FP", "FP-FP"),
            "separability_tptp_vs_tpfp": _sep("TP-TP", "TP-FP")}


def confidence_iou_stats(df: pd.DataFrame, k: int = 10) -> dict:
    """Detector-confidence (``score_max``) vs localization (``iou_gt``) relationship.

    Reports Pearson + Spearman correlation over all candidates, and — for the **top-``k`` by score per
    SET** (the candidates a low-FP operating point actually keeps) — the mean IoU, the fraction that are
    TP, and the top-1 mean IoU. A weak score↔IoU correlation means confidence does not track localization
    quality (rescoring by a learned score can then help). Returns a scatter payload for plotting.
    """
    sm = df["score_max"].to_numpy(float)
    iou = df["iou_gt"].to_numpy(float)
    ok = np.isfinite(sm) & np.isfinite(iou)
    pear = float(np.corrcoef(sm[ok], iou[ok])[0, 1]) if ok.sum() > 1 else float("nan")
    spear = float(np.corrcoef(df["score_max"].rank().to_numpy()[ok],
                              df["iou_gt"].rank().to_numpy()[ok])[0, 1]) if ok.sum() > 1 else float("nan")
    topk_iou, topk_is_tp, top1_iou = [], [], []
    for _det, _pid, g in _set_groups(df):
        gg = g.sort_values("score_max", ascending=False, kind="stable").head(k)
        topk_iou += gg["iou_gt"].tolist()
        topk_is_tp += (gg["label"] == "pos").tolist()
        top1_iou.append(float(gg["iou_gt"].iloc[0]))
    return {"k": k, "pearson_score_iou": pear, "spearman_score_iou": spear,
            "topk_mean_iou": float(np.mean(topk_iou)) if topk_iou else float("nan"),
            "topk_frac_tp": float(np.mean(topk_is_tp)) if topk_is_tp else float("nan"),
            "top1_mean_iou": float(np.mean(top1_iou)) if top1_iou else float("nan"),
            "scatter": {"score_max": sm[ok].tolist(), "iou_gt": iou[ok].tolist(),
                        "is_tp": (df["label"].to_numpy()[ok] == "pos").tolist()}}


def top_k_candidates(gset: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The top-``k`` candidates of ONE set by ``score_max`` desc (for the box viz + IoU side table)."""
    return gset.sort_values("score_max", ascending=False, kind="stable").head(k).reset_index(drop=True)


def iso_box_of_candidate(row) -> tuple:
    """A record row's ISO storage box ``(min_d0,min_d1,min_d2,max_d0,max_d1,max_d2)``.

    Rebuilt from the frozen ``cen_d*``/``ext_d*`` columns, which
    ``link.reconstruct.iso_centre_of_tube`` / ``iso_extents_of_tube`` wrote as
    ``centre = (min+max)/2`` and ``extent = max-min`` — so this recovers the tube's
    iso hull exactly. This is the box the [P3U2.PD] viz DRAWS; the record's official
    ``coordX..z_length`` is the same box after ``iso -> native -> ITK``.
    """
    return (row.cen_d0 - row.ext_d0 / 2.0, row.cen_d1 - row.ext_d1 / 2.0, row.cen_d2 - row.ext_d2 / 2.0,
            row.cen_d0 + row.ext_d0 / 2.0, row.cen_d1 + row.ext_d1 / 2.0, row.cen_d2 + row.ext_d2 / 2.0)


def box_audit(top: pd.DataFrame, gt_iso) -> pd.DataFrame:
    """Iso-space audit of drawn candidate boxes against the drawn GT box.

    Answers "the candidate visibly covers the lesion — why is its IoU 0.07?" numerically,
    and cross-checks the picture against the record:

    - ``iou_iso``    — IoU between the DRAWN boxes (iso space, ``geometry.iou_storage``).
    - ``iou_gt``     — the RECORDED official-space IoU (native voxels, vendored ``iou_3d``).
    - ``iou_resid``  — ``|iou_iso - iou_gt|``; must be small (the two spaces differ only by
      re-gridding quantization — IoU is invariant under per-axis scaling). A LARGE residual
      means the viz and the record disagree, i.e. a coordinate bug.
    - ``size_ratio`` — candidate box volume / GT box volume. For a candidate that CONTAINS
      the GT, IoU is exactly ``1 / size_ratio``: 14x oversize => IoU 0.07, no matter how
      well-centred it is.
    - ``centre_dist``— iso-voxel distance between box centres.
    - ``contains_gt``— True iff the candidate encloses the whole GT box.

    ``gt_iso`` is a storage box; returns an empty frame if it is None.
    """
    from ..geometry import box_volume_storage, iou_storage
    rows = []
    if gt_iso is None:
        return pd.DataFrame(columns=["rank", "iou_iso", "iou_gt", "iou_resid", "size_ratio",
                                     "centre_dist", "contains_gt"])
    gt_vol = box_volume_storage(gt_iso)
    gt_c = [(gt_iso[i] + gt_iso[i + 3]) / 2.0 for i in range(3)]
    for i, r in top.reset_index(drop=True).iterrows():
        b = iso_box_of_candidate(r)
        iou_iso = iou_storage(b, gt_iso)
        rec = float(r.iou_gt) if "iou_gt" in top.columns else float("nan")
        c = [(b[j] + b[j + 3]) / 2.0 for j in range(3)]
        rows.append({
            "rank": int(i) + 1,
            "iou_iso": iou_iso,
            "iou_gt": rec,
            "iou_resid": abs(iou_iso - rec) if np.isfinite(rec) else float("nan"),
            "size_ratio": (box_volume_storage(b) / gt_vol) if gt_vol > 0 else float("nan"),
            "centre_dist": float(np.linalg.norm(np.subtract(c, gt_c))),
            "contains_gt": bool(all(b[j] <= gt_iso[j] for j in range(3))
                                and all(b[j + 3] >= gt_iso[j + 3] for j in range(3))),
        })
    return pd.DataFrame(rows)


def localization_quality(df: pd.DataFrame, gt_by_pid: Dict[int, tuple]) -> dict:
    """How well-SIZED and well-CENTRED are the candidate boxes, pool-wide? (official space)

    The pool-level form of the per-PNG "box size / GT" column. Motivation: a candidate that visibly
    covers the lesion can still score IoU 0.07 — for a box that CONTAINS the GT, IoU is exactly
    ``1 / size_ratio``, so a box 2.4x too large per axis is already at 0.07 no matter how well
    centred. This separates the two ways the pool loses CPM:

    - **over-coverage** (``size_ratio`` >> 1, ``centre_mm`` small) — the reconstructed union hull is
      too big; the fix is in the linker/reconstruction, not in ranking.
    - **mis-location** (``centre_mm`` large) — the candidate is somewhere else entirely.

    Reported for (a) ALL candidates, (b) the **best-IoU** candidate per set (the pool's localization
    ceiling), and (c) the **top-10 by score** per set (what a low-FP operating point actually keeps).
    ``size_ratio`` is a volume ratio, invariant under the anisotropic voxel spacing; ``centre_mm``
    converts the official ITK-order voxel offset to millimetres via ``SPACING_STORAGE_MM`` reversed.
    ``gt_by_pid`` maps ``public_id -> (cx, cy, cz, lx, ly, lz)``; returns ``{}`` if it is empty.
    """
    from ..conventions import PERM_STORAGE_TO_ITK, SPACING_STORAGE_MM
    if not gt_by_pid:
        return {}
    sp_itk = np.array([SPACING_STORAGE_MM[PERM_STORAGE_TO_ITK[i]] for i in range(3)], dtype=float)

    def _stats(sub: pd.DataFrame) -> dict:
        if not len(sub):
            return {"n": 0}
        gt = np.array([gt_by_pid[int(p)] for p in sub["public_id"]], dtype=float)
        cand_v = (sub["x_length"].to_numpy(float) * sub["y_length"].to_numpy(float)
                  * sub["z_length"].to_numpy(float))
        gt_v = gt[:, 3] * gt[:, 4] * gt[:, 5]
        ratio = np.where(gt_v > 0, cand_v / np.where(gt_v > 0, gt_v, 1.0), np.nan)
        d = (sub[["coordX", "coordY", "coordZ"]].to_numpy(float) - gt[:, :3]) * sp_itk
        centre_mm = np.linalg.norm(d, axis=1)
        return {"n": int(len(sub)),
                "size_ratio_p10": float(np.nanpercentile(ratio, 10)),
                "size_ratio_med": float(np.nanmedian(ratio)),
                "size_ratio_p90": float(np.nanpercentile(ratio, 90)),
                "centre_mm_med": float(np.nanmedian(centre_mm)),
                "iou_med": float(np.nanmedian(sub["iou_gt"].to_numpy(float)))}

    d = df[df["public_id"].isin(gt_by_pid)]
    best, top10 = [], []
    for _det, _pid, g in _set_groups(d):
        if not len(g):
            continue
        best.append(g.loc[g["iou_gt"].idxmax()])
        top10.append(g.sort_values("score_max", ascending=False, kind="stable").head(10))
    out = {"all": _stats(d),
           "best_iou_per_set": _stats(pd.DataFrame(best)) if best else {"n": 0},
           "top10_by_score": _stats(pd.concat(top10, ignore_index=True)) if top10 else {"n": 0}}
    return out


def set_structure(df: pd.DataFrame, cluster_radius: float = C.FP_PROBE_CLUSTER_RADIUS) -> dict:
    """Per-volume counts (n, pos, neg, ignore, pos:neg) + spatial-cluster redundancy; plus aggregates.

    **UNITS — corrected 2026-08-27; the pre-correction numbers are not comparable to anything.**
    Clustering runs on ``cen_d0/cen_d1/cen_d2`` (ISO-CACHE voxels) at ``cluster_radius`` iso voxels,
    i.e. the same space, the same constant and the same single-linkage rule as
    :mod:`abus_jcr.probe.fp_structure` and ``scripts/phase3_tube_stats.py``. On the ``measured``
    profile that radius is a genuine ``radius * ISO_SPACING_MM`` sphere.

    Until 2026-08-27 it clustered ``coordX/coordY/coordZ`` — **native voxel indices** — with the same
    numeric radius, which on this dataset is a ``2.0 x 0.73 x 4.76`` mm sliver, not a 4 mm sphere. It
    therefore split single objects into many clusters and under-reported redundancy (recorded val
    medians ~1.7-1.8 against ``phase3_tube_stats``' 7.7 at the same nominal radius). The defect was
    **descriptive only** — no gate, frozen constant or deployed artefact ever read ``redundancy`` or
    ``clusters`` — but any recorded pre-2026-08-27 value of either must be re-run before it is
    quoted, and must never be placed beside an iso-space cluster statistic.
    """
    from .candidate_diag import cluster_counts
    missing = [c for c in ("cen_d0", "cen_d1", "cen_d2") if c not in df.columns]
    if missing:
        raise KeyError(f"set_structure clusters in iso-cache voxels and needs {missing}; the frozen "
                       "candidate record carries them (candidates/record.py). Do NOT substitute "
                       "coordX/Y/Z — those are native voxel indices and the radius would be a "
                       "mixed-unit sliver (see this function's docstring).")
    per_vol = []
    for det, pid, g in _set_groups(df):
        pos = int((g["label"] == "pos").sum()); neg = int((g["label"] == "neg").sum())
        ign = int((g["label"] == "ignore").sum())
        centres = g[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(float)
        ncl, npt, redund = cluster_counts(centres, cluster_radius)
        per_vol.append({"detector_of_origin": det, "public_id": pid, "n": int(len(g)), "pos": pos,
                        "neg": neg, "ignore": ign, "pos_to_neg": (pos / neg) if neg else float("nan"),
                        "clusters": ncl, "redundancy": redund})
    n_vol = len(per_vol)
    agg = {
        "n_volumes": n_vol,
        "cands_per_vol_median": float(np.median([v["n"] for v in per_vol])) if n_vol else float("nan"),
        "pos_per_vol_median": float(np.median([v["pos"] for v in per_vol])) if n_vol else float("nan"),
        "neg_per_vol_median": float(np.median([v["neg"] for v in per_vol])) if n_vol else float("nan"),
        "redundancy_median": float(np.nanmedian([v["redundancy"] for v in per_vol])) if n_vol else float("nan"),
        "total_pos": int(sum(v["pos"] for v in per_vol)), "total_neg": int(sum(v["neg"] for v in per_vol)),
        # Units travel with the number, so an output can be attributed from its own log.
        "cluster_space": "iso_cache_voxels (cen_d0,cen_d1,cen_d2)",
        "cluster_radius_iso_vox": float(cluster_radius),
        "cluster_radius_mm": float(cluster_radius * C.ISO_SPACING_MM),
    }
    return {"per_vol": per_vol, "aggregate": agg}
