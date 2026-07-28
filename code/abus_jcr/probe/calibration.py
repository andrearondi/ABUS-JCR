"""[P3U2.CAL] Headroom decomposition — how much of the B0'→ceiling gap is CROSS-VOLUME calibration?

The official metric sweeps ONE GLOBAL probability threshold across every volume
(``_official_det_score._froc_single_thresh``). So a rescorer's CPM depends on two separable things:

1. **within-set ordering** — are this volume's hitting candidates above its non-hitting ones?
2. **cross-volume level** — is this volume's score scale comparable to every other volume's?

PHASE_4 §2 asserts FROC improves "iff, within a volume, the hitting candidates are ranked above the
non-hitting" — necessary, but NOT sufficient: perfect per-volume ordering still loses recall if a real
lesion's score in a quiet volume sits below a confident artefact's score in a noisy one. This module
measures how much that costs, by scoring the SAME frozen pool (Inv. 8 — re-ranking only, never a new
candidate) under three probability assignments through the official ``evaluate()``:

+-------------------+-----------------------------------------------------------------------------+
| ``score_max``     | B0' — the deployed baseline.                                                |
| ``volume_neutral``| every candidate's score replaced by ``f(within-set rank)`` with the SAME     |
|                   | ``f`` for every set. Keeps each set's own ordering, DISCARDS all            |
|                   | cross-volume confidence. **Not an upper bound** — it is the "trust every    |
|                   | volume equally" point. If it BEATS B0', the detector's cross-volume         |
|                   | confidence is actively harmful.                                             |
| ``per_vol_oracle``| the exact **upper bound** over all per-volume monotone rescalings, holding  |
|                   | within-set ordering fixed. Uses labels ⇒ a HEADROOM BOUND, never a result.  |
+-------------------+-----------------------------------------------------------------------------+

``per_vol_oracle`` is a knapsack. A monotone per-volume map can keep this set's top-``k`` for any
``k``, and a single global threshold can realise any ``(k_1..k_V)`` simultaneously. To surface set
``v``'s first hitting candidate you must accept every candidate ranked above it — all of which are
non-hitting by construction — so set ``v`` costs exactly ``fp_cost_v = best_tp_rank_v - 1`` false
positives and yields one lesion. Every set is worth the same (+1 lesion), so at any FP budget the
optimum is the cheapest sets: **sort by ``fp_cost`` ascending**. Sets with no hitting candidate are
unbuyable at any price (Inv. 8) and cost nothing.

Assumes **one GT box per volume** (true for this dataset: ``bbx_labels.csv`` is one row per case), so
"the set is hit" == "one lesion recovered". :func:`per_set_cost` reports the assumption's inputs so a
multi-lesion split would be visible rather than silently mis-counted.

Probabilities are placed ON the official threshold grid (``np.arange(0, 1, 0.005)``) so consecutive
operating points land in distinct grid cells and none is lost to quantization. NOTE the asymmetry this
creates: the synthetic assignments use the grid ideally, while raw ``score_max`` is compressed into a
narrow band (val seed0: FP median 0.045, max 0.528) and so shares grid cells. Part of any measured gap
is therefore score SPREAD, not only cross-volume level — report it as such.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

GRID = 0.005      # the official det_score threshold step: np.arange(0, 1, 0.005)
TOP = 0.995       # highest grid value < 1 (to_official_pred_csv requires probability in [0, 1))


def _set_keys(df: pd.DataFrame):
    return ["detector_of_origin", "public_id"] if "detector_of_origin" in df.columns else ["public_id"]


def _rank_within_set(g: pd.DataFrame) -> np.ndarray:
    """1-indexed within-set rank by ``score_max`` desc, stable ties.

    Recomputed here rather than read from the record's ``rank`` column so the diagnostic stays valid
    on any subset (the record's rank is relative to the whole (detector, volume) pool).
    """
    order = (-g["score_max"].to_numpy(float)).argsort(kind="stable")
    r = np.empty(len(g), dtype=np.int64)
    r[order] = np.arange(1, len(g) + 1)
    return r


def _grid_prob(level) -> np.ndarray:
    """Integer level (0 = best) -> a probability sitting exactly on the official threshold grid."""
    return np.clip(TOP - np.asarray(level, dtype=float) * GRID, 0.0, TOP)


def per_set_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Per (detector, volume) set: size, best hitting-candidate rank, and its FP cost.

    ``fp_cost`` = ``best_tp_rank - 1`` = how many non-hitting candidates a global threshold must also
    admit in order to surface this set's first hit. ``NaN`` when the set has no hitting candidate
    (below the Inv.-8 recall ceiling — unbuyable at any FP budget).
    """
    rows = []
    for k, g in df.groupby(_set_keys(df), sort=False):
        det, pid = (k if isinstance(k, tuple) else ("ALL", k))
        r = _rank_within_set(g)
        is_pos = (g["label"].to_numpy() == "pos")
        best = int(r[is_pos].min()) if is_pos.any() else None
        rows.append({"detector_of_origin": det, "public_id": pid, "n": int(len(g)),
                     "n_pos": int(is_pos.sum()),
                     "best_tp_rank": best, "fp_cost": (best - 1) if best is not None else np.nan})
    return pd.DataFrame(rows)


def volume_neutral_probability(df: pd.DataFrame) -> pd.Series:
    """``p = f(within-set rank)``, the SAME ``f`` for every set — cross-volume confidence discarded.

    Thresholding at ``f(k)`` keeps exactly the top-``k`` of EVERY set, so the swept curve is the
    top-k-per-volume curve.
    """
    p = pd.Series(0.0, index=df.index, dtype=float)
    for _k, g in df.groupby(_set_keys(df), sort=False):
        p.loc[g.index] = _grid_prob(_rank_within_set(g) - 1)
    return p


def per_volume_oracle_probability(df: pd.DataFrame) -> pd.Series:
    """The per-volume-monotone-rescaling UPPER BOUND (uses labels — a bound, never a result).

    Buys sets in ascending ``fp_cost``; set ``v``'s purchase admits its ranks ``1..best_tp_rank_v``.
    Everything not bought (a bought set's tail, and every set with no hit) goes one level lower — by
    then recall is already the Inv.-8 ceiling, so their order cannot affect any key-FP recall.

    **One grid level per distinct CUMULATIVE FP total**, not one per purchase. All zero-cost sets
    therefore share level 0. This is both semantically right — at any budget ≥ 0 you take every free
    set, so the intermediate "only 3 of the 9 free sets" points are operating points nobody would
    choose — and numerically necessary: the official ``_get_key_recall`` does
    ``sort_values("fp")`` with pandas' DEFAULT **quicksort**, which is NOT stable, so when several
    thresholds share one ``fp`` the interpolator's ``values[-1]`` picks an arbitrary member of the tie
    block. Emitting one level per distinct ``fp`` removes the tie and makes the bound deterministic.
    """
    cost = per_set_cost(df)
    buyable = cost[cost["fp_cost"].notna()].sort_values(["fp_cost", "public_id"], kind="stable")
    p = pd.Series(np.nan, index=df.index, dtype=float)
    groups = {(det, pid): g for (det, pid), g in
              ((k if isinstance(k, tuple) else ("ALL", k), g) for k, g in df.groupby(_set_keys(df), sort=False))}
    level, cum, prev_cum = -1, 0, None
    for _, row in buyable.iterrows():
        cum += int(row["fp_cost"])
        if cum != prev_cum:                          # a new distinct FP total -> a new operating point
            level += 1
            prev_cum = cum
        g = groups[(row["detector_of_origin"], row["public_id"])]
        take = _rank_within_set(g) <= int(row["best_tp_rank"])
        p.loc[g.index[take]] = float(_grid_prob(level))
    n_levels = level + 2                             # + the tail level
    if n_levels > int(TOP / GRID):
        raise ValueError(f"per_volume_oracle needs {n_levels} threshold levels but the official grid "
                         f"has only {int(TOP / GRID)}; the bound would be truncated")
    p[p.isna()] = float(_grid_prob(level + 1))       # the unbought tail, strictly below every purchase
    return p


def headroom_curve(df: pd.DataFrame) -> dict:
    """The knapsack curve itself (FP/vol, recall) + the per-set cost distribution — torch-free.

    Descriptive companion to the official three-way CPM comparison: it shows WHERE the cost sits
    (a few very expensive sets, or a broad tail?) without re-deriving any metric.
    """
    cost = per_set_cost(df)
    n_sets = int(len(cost))
    buy = np.sort(cost["fp_cost"].dropna().to_numpy(float))
    curve = [{"n_sets_bought": 0, "fp_per_vol": 0.0, "recall": 0.0}]
    for j, c in enumerate(np.cumsum(buy), start=1):
        curve.append({"n_sets_bought": j, "fp_per_vol": float(c / n_sets) if n_sets else float("nan"),
                      "recall": float(j / n_sets) if n_sets else float("nan")})
    free = int((cost["fp_cost"] == 0).sum())
    return {
        "n_sets": n_sets,
        "n_sets_with_hit": int(cost["fp_cost"].notna().sum()),
        "n_sets_free": free,                      # best hit already rank-1 => costs 0 FP
        "frac_sets_free": (free / n_sets) if n_sets else float("nan"),
        "fp_cost_p50": float(np.nanmedian(cost["fp_cost"])) if n_sets else float("nan"),
        "fp_cost_p90": float(np.nanpercentile(cost["fp_cost"].dropna(), 90)) if buy.size else float("nan"),
        "fp_cost_max": float(np.nanmax(cost["fp_cost"])) if buy.size else float("nan"),
        "curve": curve,
        "per_set": cost.to_dict(orient="records"),
    }


def assignments(df: pd.DataFrame, extra: Optional[dict] = None) -> dict:
    """``{name: probability Series}`` for the three-way comparison; ``score_max`` first (B0')."""
    out = {"score_max": df["score_max"].astype(float),
           "volume_neutral": volume_neutral_probability(df),
           "per_vol_oracle": per_volume_oracle_probability(df)}
    if extra:
        out.update(extra)
    return out
