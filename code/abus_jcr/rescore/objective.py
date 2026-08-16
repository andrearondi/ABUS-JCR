"""[4.2b] Objective-alignment study — the factorial grid and its target/weight builders.

**Why this module exists.** `[4.2]` measured the encoder's B1 at val CPM **0.5166** against a
measured B0 of **0.6673** (seed 0, `[F.8]` denominator), while its per-candidate balanced
accuracy reached **0.9035** — above the pool's 0.811 single-feature ceiling. A model whose
input vector *contains* ``score_max`` (``SCORE_STAT_COLUMNS[0]``, so B0's ranking function is
literally one of its 160 features) trained to a worse ranking than that one feature alone.
The hypothesis class contains B0; therefore the features are not the constraint and the
**objective** is.

Four misalignments between ``rescorer_loss`` and ``eval/froc``, each measured, each a factor
in the grid below. None of them is about the label being binary — plain BCE's population
optimum is the posterior, and thresholding the posterior is Neyman-Pearson optimal at every
FP budget, which is exactly what a global-threshold FROC sweep does:

1. **``gamma``** — focal loss is deliberately *not* a proper scoring rule. That is correct for
   dense detection (only within-image order matters before NMS) and wrong here, where `[F.8]`
   puts **78.7 %** of the headroom in cross-volume *calibration*, and where the saturated
   score band collides with the oracle's fixed ``np.arange(0, 1, 0.005)`` grid
   (:func:`threshold_occupancy`).
2. **``soft``** — Inv. 11's ignore band is a hole in the supervision that the oracle scores
   as an FP (:func:`losses.soft_quality_target`).
3. **``per_lesion``** — duplicates are free to the metric and cost ~15.6x in the loss
   (:func:`losses.per_lesion_weights`).
4. **``alpha``** — 0.25 down-weights positives to 0.25 against negatives at 0.75, a RetinaNet
   default for ~1:1000 dense anchors, applied to a pool that is already 1:6.2. The `[4.6]`
   ladder sweeps it; the `[4.3]` encoder pretraining does not.

**The grid always contains the deployed cell** (``is_deployed``), so the study reads as a
controlled comparison against what actually ran, not as a fresh search.

Torch-free: everything here builds arrays that are handed to the loss, never gradients.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C
from .datasets import group_sets
from .losses import encode_labels, soft_quality_target

__all__ = ["OBJECTIVE_FACTORS", "DEPLOYED_CELL", "CONFIRMATION_CELLS", "objective_grid",
           "record_targets", "record_lesion_weights", "threshold_occupancy",
           "ignore_band_audit", "select_ci_cells", "estimate_oracle_calls",
           "ORACLE_SECONDS_PER_CALL"]

#: Measured on the promoted val seed-0 pool (2225 candidates / 30 volumes): the seed-0
#: `[4.2b]` run issued ~1428 `evaluate()` calls and took ~3 h. The Phase-4 runbook's ~2 s
#: figure is from a smaller pool; do not plan `[4.2b/c]` with it.
ORACLE_SECONDS_PER_CALL = 7.6

#: The factorial axes. Order fixes the variant name, so a name is readable without a legend.
#: Values are LITERAL, not read from ``conventions`` — the grid must keep spanning both arms
#: after a promotion moves the deployed value onto one of them (2026-08-15: gamma 2.0 -> 0.0,
#: which would otherwise have collapsed this axis to (0.0, 0.0)).
OBJECTIVE_FACTORS: Dict[str, Tuple] = {
    "gamma": (2.0, 0.0),                  # focal vs plain BCE = a proper scoring rule
    "alpha": (0.25, 0.50),                # deployed vs positives-not-down-weighted
    "soft": (False, True),                # ignore-band hole vs ramp
    "per_lesion": (False, True),          # per-candidate vs metric's own unit of credit
}

#: The cell `[4.3]`/`[4.6]` actually ship — the control the study is read against. Tracks
#: ``conventions`` so it follows a promotion automatically. **Was** ``g2_a0.25_hard_cand``; since
#: `[4.2c]` (2026-08-15) it is ``g0_a0.25_hard_lesion``, so any future re-run compares against
#: what is deployed *then*, not against a stale label.
DEPLOYED_CELL: Dict = {"gamma": C.RESC_FOCAL_GAMMA, "alpha": 0.25,
                       "soft": False, "per_lesion": bool(C.RESC_PER_LESION_WEIGHTS)}

#: `[4.2c]`'s 2x2: the two factors that moved on seed 0, at the deployed alpha, hard labels.
#: `soft` is dropped (its clean read inside the per-candidate column is +0.0023 = nil, and its
#: `per_lesion` quadrant was void before the 2026-08-13 weight fix); `alpha` is dropped (+0.001
#: overall, and its apparent +0.04 sits only in the corner the winner was picked from, so it is
#: confounded with the selection). Narrowing the grid is the ONLY cheap lever: the per-epoch
#: table was 67 % of the seed-0 run, and shortening the schedule would reshape the cosine
#: anneal and make the result incomparable to seed 0.
CONFIRMATION_CELLS = ("g0_a0.25_hard_lesion", "g2_a0.25_hard_lesion",
                      "g0_a0.25_hard_cand", "g2_a0.25_hard_cand")


def _variant_name(cell: Dict) -> str:
    return (f"g{cell['gamma']:g}_a{cell['alpha']:g}"
            f"_{'soft' if cell['soft'] else 'hard'}"
            f"_{'lesion' if cell['per_lesion'] else 'cand'}")


def objective_grid() -> List[Dict]:
    """Every cell of the factorial, each with a unique ``name`` and an ``is_deployed`` flag."""
    keys = list(OBJECTIVE_FACTORS)
    out: List[Dict] = [{}]
    for k in keys:
        out = [{**cell, k: v} for cell in out for v in OBJECTIVE_FACTORS[k]]
    for cell in out:
        cell["name"] = _variant_name(cell)
        cell["is_deployed"] = all(cell[k] == v for k, v in DEPLOYED_CELL.items())
    return out


def select_ci_cells(results: Sequence[Dict], k: int) -> List[Dict]:
    """The ``k`` cells worth paying a bootstrap for: best of ``raw``/``spread`` CPM, desc.

    Ranked on the better of the two because a cell that only wins after the rank-preserving
    remap is still the cell to promote — the remap changes no ordering.
    """
    ranked = sorted(results, key=lambda r: -max(r["raw"]["cpm"], r["spread"]["cpm"]))
    return ranked[:max(int(k), 0)]


def estimate_oracle_calls(n_cells: int, epochs: int, n_boot_b0: int, n_boot_paired: int,
                          ci_top_k: int, boot_b0_spread: bool = False) -> int:
    """Total ``evaluate()`` calls a `[4.2b/c]` run will issue. ~7.6 s each on this pool.

    Breakdown, and why it is worth printing before the run: the **per-epoch selection table**
    is ``n_cells x epochs`` and was **67 %** of the seed-0 run. Narrowing the grid is therefore
    the only cheap lever — shortening the schedule would reshape the cosine anneal and make the
    result incomparable to seed 0, so it is not one.
    """
    per_epoch = int(n_cells) * int(epochs)
    per_cell_final = int(n_cells) * 4          # raw + 2 headroom + spread
    b0 = 1 + int(n_boot_b0) + 2                # point + CI + 2 headroom
    b0_spread = 1 + (int(n_boot_b0) if boot_b0_spread else 0)
    paired = 2 * int(n_boot_paired) * max(int(ci_top_k), 0)   # a paired draw scores BOTH arms
    return per_epoch + per_cell_final + b0 + b0_spread + paired


def record_targets(record_df: pd.DataFrame, soft: bool) -> np.ndarray:
    """The per-row loss target: the hard ``+1/0/-1`` code, or the ``iou_gt`` ramp.

    The hard path is ``encode_labels`` verbatim, so the deployed variant consumes exactly the
    array `[4.6]` builds. The soft path returns values in ``[0, 1]`` and **never** the ignore
    code, so no row is masked — ``collate_sets`` still pads with ``-1``, which ``focal_bce(
    soft=True)`` reads as "excluded", so padding stays distinguishable from a true negative.
    """
    if not soft:
        return encode_labels(record_df["label"].to_numpy())
    return soft_quality_target(record_df["iou_gt"].to_numpy(float))


def record_lesion_weights(record_df: pd.DataFrame, targets=None) -> np.ndarray:
    """Per-row weights: a SET's **positive target mass** sums to 1.0; negatives stay at 1.0.

    Grouped by ``(detector_of_origin, public_id)`` — the Inv.-7 set key — so one volume seen
    by three detectors is three lesions' worth of credit, which is what the oracle scores
    (each seed pool is evaluated separately, Inv. 14). One-lesion-per-set approximation: the
    record carries ``iou_gt`` but no GT lesion id, and Phase 0a's single-lesion dominance
    holds 99/100 Train.

    With ``t_i`` the row's target in ``[0, 1]`` and ``S = sum(t)`` over the set::

        w_i = 1 / S   if t_i > 0   else   1.0

    so ``sum(w_i * t_i) == 1`` **exactly**: one set contributes one lesion's worth of positive
    credit, spread over everything that partially hits it. A pure negative keeps unit weight,
    and for hard labels ``S = n_pos``, reducing this **exactly** to ``1/n_pos`` / ``1.0``.

    A single scalar per row cannot scale a partial row's positive and negative halves
    differently, so this discounts an ignore-band row's *negative* contribution by ``1/S`` too.
    That is a real distortion and it is the better one available: the hard path does not
    discount those rows, it **deletes** them (Inv. 11 masks the band out of the loss entirely),
    so this strictly dominates the baseline on that axis.

    **CORRECTED 2026-08-13.** The first version read the hard label even when the caller was
    training on the ramp, so a true positive was cut to ``1/n_pos`` (~1/15.6 on the train
    pool) while an ignore-band row kept weight **1.0** *and* carried a partial positive
    target — the model was trained to prefer near misses over hits. That is what produced the
    four worst cells of the seed-0 `[4.2b]` grid (``soft_lesion``, mean −0.142, three of four
    peaking at **epoch 0**). Those four cells are **void**, not evidence about soft targets;
    the clean read of the ramp is the ``soft`` vs ``hard`` contrast inside the per-candidate
    column, which is ``+0.0023`` — nil.
    """
    t = np.clip(np.asarray(
        encode_labels(record_df["label"].to_numpy()) if targets is None else targets,
        dtype=float), 0.0, 1.0)
    w = np.ones(len(record_df), dtype=float)
    for idx in group_sets(record_df).values():
        s = float(t[idx].sum())
        if s > 0.0:
            w[idx] = np.where(t[idx] > 0.0, 1.0 / s, 1.0)
    return w


def threshold_occupancy(prob, sweep: Sequence[float] = None) -> int:
    """How many of the oracle's 200 sweep thresholds actually separate two candidates.

    ``froc()`` sweeps ``np.arange(0, 1, 0.005)`` and can resolve nothing finer. A focal-BCE
    classifier drives negatives to ~0 and positives to ~1, so its whole ranking can collapse
    into a handful of bins — and CPM's four tightest key points (0.125-1 FP/volume, i.e. the
    top ~4 to ~30 entries of a 2225-long pool) are then read off an interpolation across a
    gap the sweep never sampled. This counts the bins that are genuinely occupied, so the
    quantisation artefact is measured rather than assumed.
    """
    sweep = np.arange(0.0, 1.0, 0.005) if sweep is None else np.asarray(sweep, dtype=float)
    p = np.asarray(prob, dtype=float)
    counts = np.array([(p >= t).sum() for t in sweep])
    return int(len(np.unique(counts[counts > 0])))


def ignore_band_audit(record_df: pd.DataFrame, prob, top_k: int = None) -> Dict:
    """How much of the top of a ranking is Inv.-11 ignore band — invisible to the loss, an FP
    to the oracle.

    ``top_k`` defaults to the pool's own 1-FP-per-volume budget, which is the CPM key point
    where a handful of near-misses costs the most.
    """
    lab = record_df["label"].to_numpy()
    p = np.asarray(prob, dtype=float)
    n_vol = int(record_df["public_id"].nunique())
    top_k = int(n_vol if top_k is None else top_k)
    order = np.argsort(-p, kind="stable")
    top = order[:top_k]
    is_pos = lab[order] == "pos"
    first_tp = int(np.argmax(is_pos)) if is_pos.any() else -1
    return {
        "top_k": top_k,
        "ignore_in_top_k": int((lab[top] == "ignore").sum()),
        "pos_in_top_k": int((lab[top] == "pos").sum()),
        "neg_in_top_k": int((lab[top] == "neg").sum()),
        "rank_of_first_tp": first_tp,
        "ignore_above_first_tp": int((lab[order[:max(first_tp, 0)]] == "ignore").sum()),
        "neg_above_first_tp": int((lab[order[:max(first_tp, 0)]] == "neg").sum()),
        "n_ignore_total": int((lab == "ignore").sum()),
    }
