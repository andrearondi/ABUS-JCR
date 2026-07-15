"""Phase-0b FP-structure probe on the RetinaNet candidate pool (iso space).

Does the false-positive geometry carry structure the Phase-4 rescorer can exploit?
Two signatures, measured on the frozen candidate record:

- **Clustering** — FPs recur in tight spatial clumps (small nearest-neighbour distance
  among FP centroids, and >1 FP cluster per volume by single-linkage at a fixed
  radius), whereas TPs are isolated.
- **Anisotropy** — after isotropic resampling, real posterior-shadow FPs stay elongated
  along the depth/beam axis ``d0`` (``ext_d0 / mean(ext_d1, ext_d2)`` large), whereas a
  compact lesion TP is roughly isotropic.

Verdict: structure PRESENT iff FPs are BOTH more clustered (smaller NN distance AND
>1 cluster/vol) AND more depth-elongated than TPs. PRESENT -> Phase 4's geometry term
is "relational"; ABSENT -> "set-level contextual calibration". NN distances pool
across volumes (the split is single-lesion-dominant, so per-volume TP NN is undefined);
cluster counts are per-volume.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .. import conventions as C


def _anisotropy(df: pd.DataFrame) -> np.ndarray:
    """``ext_d0 / mean(ext_d1, ext_d2)`` per candidate (d0 = depth/beam)."""
    if len(df) == 0:
        return np.zeros(0)
    e0 = df["ext_d0"].to_numpy(dtype=float)
    e1 = df["ext_d1"].to_numpy(dtype=float)
    e2 = df["ext_d2"].to_numpy(dtype=float)
    denom = (e1 + e2) / 2.0
    denom = np.where(denom <= 0, np.nan, denom)
    return e0 / denom


def _centroids(df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        return np.zeros((0, 3))
    return df[["cen_d0", "cen_d1", "cen_d2"]].to_numpy(dtype=float)


def _nn_distances(points: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance for each point among the given centroids.

    Needs >= 2 points; returns an empty array otherwise. O(n^2) — the candidate
    counts per split are small.
    """
    n = len(points)
    if n < 2:
        return np.zeros(0)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _single_linkage_clusters(points: np.ndarray, radius: float) -> int:
    """Number of connected components when points within ``radius`` are linked."""
    n = len(points)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= radius:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def _iqr(a: np.ndarray) -> List[float]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 25)), float(np.percentile(a, 75))]


def _median(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-based effect size in [-1, 1]: P(a>b) - P(a<b). NaN if either empty."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (a.size * b.size))


def _group_summary(df: pd.DataFrame, radius: float) -> Dict:
    aniso = _anisotropy(df)
    # NN distances pooled across all centroids of the group (see module docstring).
    nn = _nn_distances(_centroids(df))
    # cluster count per volume (FP clumping is a within-volume signal)
    per_vol = []
    for _, grp in df.groupby("public_id", sort=False):
        per_vol.append(_single_linkage_clusters(_centroids(grp), radius))
    per_vol = np.asarray(per_vol, dtype=float)
    return {
        "n": int(len(df)),
        "n_volumes": int(df["public_id"].nunique()) if len(df) else 0,
        "anisotropy_median": _median(aniso),
        "anisotropy_iqr": _iqr(aniso),
        "nn_dist_median": _median(nn),
        "nn_dist_iqr": _iqr(nn),
        "clusters_per_vol_median": _median(per_vol),
        "clusters_per_vol_iqr": _iqr(per_vol),
    }


def fp_structure_probe(record_df: pd.DataFrame, split_filter: str = "val",
                       cluster_radius: float = C.FP_PROBE_CLUSTER_RADIUS) -> Dict:
    """FP-vs-TP clustering + anisotropy distributions, effect sizes, and a verdict.

    Separates FP (``label == 'neg'``) and TP (``label == 'pos'``) candidates of the
    ``split_filter`` split. Returns per-group medians/IQRs, two-sample Cliff's-delta
    effect sizes (FP vs TP for anisotropy and NN distance), and a verdict dict.
    """
    df = record_df[record_df["split"] == split_filter]
    fp = df[df["label"] == "neg"]
    tp = df[df["label"] == "pos"]

    fp_sum = _group_summary(fp, cluster_radius)
    tp_sum = _group_summary(tp, cluster_radius)

    fp_aniso, tp_aniso = _anisotropy(fp), _anisotropy(tp)
    fp_nn, tp_nn = _nn_distances(_centroids(fp)), _nn_distances(_centroids(tp))

    more_elongated = bool(fp_sum["anisotropy_median"] > tp_sum["anisotropy_median"])
    # "more clustered" = tighter FP neighbours than TP AND FPs form >1 clump/volume
    tighter_nn = bool(np.isfinite(fp_sum["nn_dist_median"]) and np.isfinite(tp_sum["nn_dist_median"])
                      and fp_sum["nn_dist_median"] < tp_sum["nn_dist_median"])
    multi_cluster = bool(np.isfinite(fp_sum["clusters_per_vol_median"])
                         and fp_sum["clusters_per_vol_median"] > 1)
    more_clustered = tighter_nn and multi_cluster
    structure_present = bool(more_elongated and more_clustered)

    return {
        "split": split_filter,
        "fp": fp_sum,
        "tp": tp_sum,
        "effect": {
            "anisotropy_cliffs_delta": _cliffs_delta(fp_aniso, tp_aniso),
            "nn_dist_cliffs_delta": _cliffs_delta(fp_nn, tp_nn),
        },
        "verdict": {
            "structure_present": structure_present,
            "more_elongated": more_elongated,
            "more_clustered": more_clustered,
            "tighter_nn": tighter_nn,
            "multi_cluster_per_vol": multi_cluster,
            "claim_scope": ("relational" if structure_present
                            else "set-level contextual calibration"),
        },
    }
