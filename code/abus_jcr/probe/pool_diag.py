"""[P3U2.PD] Frozen candidate-pool deep diagnostics — torch-free core.

Reads only the frozen record columns (no detector / iso-cache / torch). Four analysis blocks that
inform Phase 4:

1. ``feature_discriminability`` — which per-candidate features separate TP from FP (→ which Phase-4
   feature *blocks* carry signal): Cliff's delta + best single-threshold balanced accuracy per feature.
2. ``ranking_headroom`` — where the TPs sit in the baseline ``score_max`` ranking, and an approximate
   recall-at-FP/vol curve (→ how much a re-ranker can win).
3. ``pairwise_geometry`` — the 6-D relative-log-geometry ``g(m,n)`` (the exact Axis-A descriptor) for
   TP-TP / TP-FP / FP-FP pairs (→ the DIRECT pairwise test the per-candidate FP-probe could not do).
4. ``set_structure`` — per-volume candidate counts, TP/FP/ignore mix, and spatial-cluster redundancy.

TP := ``label == 'pos'`` (iou_gt > 0.30); FP := ``label == 'neg'``; the ignore band is excluded from
TP-vs-FP contrasts (Inv. 11).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

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
    ``anisotropy`` (iso depth-extent / mean lateral extent, matching the Phase-0b probe)."""
    out = df.copy()
    out["box_diag"] = np.sqrt(out["x_length"].to_numpy(float) ** 2
                              + out["y_length"].to_numpy(float) ** 2
                              + out["z_length"].to_numpy(float) ** 2)
    lat = (out["ext_d1"].to_numpy(float) + out["ext_d2"].to_numpy(float)) / 2.0
    out["anisotropy"] = np.where(lat > 0, out["ext_d0"].to_numpy(float) / np.where(lat > 0, lat, 1.0), np.nan)
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


def _best_balacc(pos: Sequence[float], neg: Sequence[float]) -> tuple:
    """Best single-threshold balanced accuracy (direction-agnostic) + the threshold. O(u*(np+nn))."""
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
        ba, thr = _best_balacc(a, b)
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


def set_structure(df: pd.DataFrame, cluster_radius: float = 10.0) -> dict:
    """Per-volume counts (n, pos, neg, ignore, pos:neg) + spatial-cluster redundancy; plus aggregates."""
    from .candidate_diag import cluster_counts
    per_vol = []
    for det, pid, g in _set_groups(df):
        pos = int((g["label"] == "pos").sum()); neg = int((g["label"] == "neg").sum())
        ign = int((g["label"] == "ignore").sum())
        centres = g[["coordX", "coordY", "coordZ"]].to_numpy(float)
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
    }
    return {"per_vol": per_vol, "aggregate": agg}
