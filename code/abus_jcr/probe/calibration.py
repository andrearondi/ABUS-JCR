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

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..conventions import KEY_FP

GRID = 0.005      # the official det_score threshold step: np.arange(0, 1, 0.005)
TOP = 0.995       # highest grid value < 1 (to_official_pred_csv requires probability in [0, 1))
N_LEVELS = int(TOP / GRID)   # usable grid cells: any assignment needing more would be truncated


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


def volume_neutral_probability(df: pd.DataFrame, anchored=False) -> pd.Series:
    """``p = f(within-set rank)``, the SAME ``f`` for every set — cross-volume confidence discarded.

    Thresholding at ``f(k)`` keeps exactly the top-``k`` of EVERY set, so the swept curve is the
    top-k-per-volume curve.

    ``anchored`` — ``False`` (**default: the behaviour every recorded number was measured under**),
    ``True``, or ``"auto"``. It shifts every level one grid cell down so the top cell is empty, and
    it matters more than it looks: ``_official_det_score._interpolate_recall_at_fp`` **returns 0**
    for any key FP below the smallest achievable one, and a real probability column always supplies
    the empty-set point at ``fp ~ 0`` (no real score saturates 0.995), whereas this assignment's
    level 0 does not. Its cheapest operating point is "every set's rank-1 candidate", which costs one
    FP per set whose rank-1 is not a hit — where that exceeds ``0.5 * n_vol`` FPs the three lowest
    key rates are forced to zero and the reported CPM is a **floor**, not what the official evaluator
    would give the same ranking written as a real prediction file.

    ``"auto"`` applies the same well-posedness rule as :func:`global_monotone_probability`: anchor
    unless level 0 is already free (every set's rank-1 is a hit), because then the anchor would land
    on the same ``fp`` as level 0 and ``_get_key_recall`` breaks that tie with a non-stable sort.

    Left off by default so no recorded value moves silently;
    ``scripts/phase3_calibration_quantisation.py`` reports both readings, so the size of the artefact
    is measured rather than argued, and changing the default is a decision for the user.
    """
    p = pd.Series(0.0, index=df.index, dtype=float)
    if anchored == "auto":
        rank1_is_hit = all((g["label"].to_numpy() == "pos")[_rank_within_set(g) == 1].all()
                           for _k, g in df.groupby(_set_keys(df), sort=False))
        base = 0 if rank1_is_hit else 1
    else:
        base = 1 if anchored else 0
    for _k, g in df.groupby(_set_keys(df), sort=False):
        p.loc[g.index] = _grid_prob(_rank_within_set(g) - 1 + base)
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


def pooled_prefix_scan(df: pd.DataFrame) -> dict:
    """Walk the POOLED ``score_max`` ranking once; return cumulative FP and unique-hit counts.

    Mirrors ``_official_det_score._froc_single_thresh`` exactly, under the same one-GT-per-volume
    assumption :func:`per_set_cost` documents: a prediction is a **false positive unless it hits**
    (so the Inv.-11 ``ignore`` band counts as FP, exactly as ``max_iou <= 0.3`` does there), and a
    volume's SECOND hitting candidate is neither TP nor FP (the evaluator counts *unique* hit GT
    labels). Ties in ``score_max`` are kept adjacent so a prefix can be snapped to a tie boundary.
    """
    order = (-df["score_max"].to_numpy(float)).argsort(kind="stable")
    s = df["score_max"].to_numpy(float)[order]
    is_hit = (df["label"].to_numpy() == "pos")[order]
    vol = df["public_id"].to_numpy()[order]
    seen, cum_fp, cum_hit, fp, hits = set(), [], [], 0, 0
    for i in range(len(order)):
        if is_hit[i]:
            if vol[i] not in seen:      # first hit in this volume -> +1 recovered lesion
                seen.add(vol[i]); hits += 1
        else:
            fp += 1                     # neg OR ignore: both fail max_iou > 0.3
        cum_fp.append(fp); cum_hit.append(hits)
    return {"order": order, "score_sorted": s,
            "cum_fp": np.asarray(cum_fp, dtype=np.int64),
            "cum_hits": np.asarray(cum_hit, dtype=np.int64)}


def global_monotone_cuts(df: pd.DataFrame, n_vol: int, key_fp: Sequence[float] = KEY_FP) -> list:
    """The prefixes of the pooled ranking that BRACKET each key FP rate, snapped to tie boundaries.

    ``n_vol`` MUST be the number of volumes the evaluator divides by (the GT row count for the
    split), not the number of volumes present in the pool — a volume with zero candidates still
    counts in ``fp = total_fp / len(df_list)`` (``_official_det_score.evaluate`` builds one frame per
    GT ``public_id``). Passing the wrong one shifts every FP rate.

    **Two cuts per key rate, not one.** ``_interpolate_recall_at_fp`` reads each key FP off the chord
    between the achievable points on either side of it, so a single cut below the budget leaves the
    interpolation to reach across everything above it — which loses more than the grid gains, and on
    a pool whose top is one big tie block the below-cut can collapse to the empty prefix entirely.
    Emitting the longest prefix at or under the budget AND the shortest one over it makes every key
    rate's chord as tight as the ordering allows. Plus the exact zero-FP prefix, which is the natural
    left anchor for the lowest key rate. At most ``2 * len(key_fp) + 1`` levels — far inside the
    grid's ``N_LEVELS``.

    Cuts are snapped to ``score_max`` tie boundaries (the below-cut down, the above-cut up), so equal
    scores are never split. That is what keeps the assignment a genuine function of ``score_max``.
    """
    scan = pooled_prefix_scan(df)
    cum_fp, cum_hits, s = scan["cum_fp"], scan["cum_hits"], scan["score_sorted"]
    n = len(s)

    def _snap_down(k: int) -> int:
        while 0 < k < n and s[k - 1] == s[k]:
            k -= 1
        return k

    def _snap_up(k: int) -> int:
        while 0 < k < n and s[k - 1] == s[k]:
            k += 1
        return k

    def _rec(f: float, side: str, k: int) -> dict:
        return {"key_fp": float(f), "side": side, "prefix": int(k),
                "fp": int(cum_fp[k - 1]) if k else 0,
                "fp_per_vol": (float(cum_fp[k - 1]) / n_vol) if k else 0.0,
                "hits": int(cum_hits[k - 1]) if k else 0}

    out = [_rec(0.0, "below", _snap_down(int(np.searchsorted(cum_fp, 0.0, side="right"))))]
    for f in key_fp:
        budget = f * float(n_vol)
        k = int(np.searchsorted(cum_fp, budget, side="right"))   # longest prefix with cum_fp <= budget
        out.append(_rec(f, "below", _snap_down(k)))
        if k < n:
            out.append(_rec(f, "above", _snap_up(k + 1)))
    keep = _undominated_prefixes(out)
    for c in out:
        c["kept"] = c["prefix"] in keep
    return out


def _undominated_prefixes(cuts: Sequence[dict]) -> set:
    """The Pareto frontier of the candidate cuts: cheapest prefix for each achievable recall.

    A monotone map is free NOT to emit an operating point, and it should not emit a dominated one.
    Two cuts at the same FP total differ only in how many hits they include, so only the longest
    survives; a cut that costs more FPs without recovering another lesion is dropped outright.

    This is not cosmetic. The official ``_get_key_recall`` does ``sort_values("fp")`` with pandas'
    default **non-stable** quicksort and then reads ``values[-1]``, so when two operating points
    share one FP total the recall it reports is an arbitrary member of the tie block. Emitting a
    dominated point therefore does not merely waste a grid level — it can hand the interpolator the
    *worse* of two curves at the same cost. Dropping it removes the tie instead of gambling on it.
    """
    best: dict = {}
    for c in cuts:
        if c["prefix"] == 0:
            continue
        cur = best.get(c["fp"])
        if cur is None or c["prefix"] > cur["prefix"]:
            best[c["fp"]] = c
    keep, top = set(), -1
    for f in sorted(best):
        if best[f]["hits"] > top:
            keep.add(best[f]["prefix"])
            top = best[f]["hits"]
    return keep


def global_monotone_probability(df: pd.DataFrame, n_vol: int,
                                key_fp: Sequence[float] = KEY_FP) -> pd.Series:
    """A global monotone rescaling of ``score_max`` that puts each key-FP operating point on its own
    grid cell — the **pure threshold-quantisation** term.

    The pooled ordering of ``score_max`` is preserved exactly and ties are preserved as ties, so this
    map carries **no information** the baseline does not already have: ``p = phi(score_max)`` for a
    non-decreasing ``phi``. All it does is relocate the operating points the official sweep can
    actually read — see :func:`global_monotone_cuts` for the two-sided bracketing that does it.
    ``CPM(global_monotone) - B0'`` is therefore attributable to how the official code *reads* a
    swept curve — the fixed ``np.arange(0, 1, 0.005)`` grid, ``score_max``'s compression into a
    narrow band, and the interpolation across tied FP values — and **not** to cross-volume
    calibration. Subtracting it from ``per_vol_oracle - B0'`` leaves the part of the calibration
    headroom that no monotone transform of the detector's own score can reach.

    Two honest limits, both to be reported rather than smoothed:

    * It is a **lower** bound on what global monotone maps achieve. Cuts land at ``fp <= key_fp``
      (FP counts are integers), and ``_interpolate_recall_at_fp`` then reads the chord to the next
      operating point, which sits below the achievable curve where that curve is concave.
    * It can come out **below** B0'. That would mean the baseline's own coarse curve is being
      interpolated *across* a gap in a way that flatters it. Record it; do not discard it.
    """
    recs = global_monotone_cuts(df, n_vol, key_fp)
    kept = [c for c in recs if c["kept"]]
    cuts = sorted({c["prefix"] for c in kept})
    # Leave the TOP grid cell empty unless the cheapest operating point is already free.
    # `_interpolate_recall_at_fp` returns 0 below the smallest achievable FP, so a curve whose first
    # point costs FPs needs the empty-set anchor a real probability column always has. When a kept
    # cut sits at fp == 0 the anchor would instead land on the SAME fp as that cut, and the official
    # `_get_key_recall` breaks an fp tie with a non-stable sort — so there the unanchored placement
    # is both sufficient and deterministic. The two cases are mutually exclusive; a monotone map is
    # free to choose either placement, so taking the one that is well-posed is not a free parameter.
    base = 0 if any(c["fp"] == 0 for c in kept) else 1
    n_needed = len(cuts) + 1 + base
    if n_needed > N_LEVELS:
        raise ValueError(f"global_monotone needs {n_needed} grid levels, the official grid has "
                         f"{N_LEVELS}")
    order = pooled_prefix_scan(df)["order"]
    level = np.searchsorted(np.asarray(cuts, dtype=np.int64), np.arange(len(order)), side="right")
    p = pd.Series(0.0, index=df.index, dtype=float)
    p.iloc[order] = _grid_prob(level + base)
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
