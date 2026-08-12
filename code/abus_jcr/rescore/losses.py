"""[4.6] Axis C — the rescorer's two loss terms.

``L = w_rank · smooth_ap_loss + λ · focal_bce``

* **smooth-AP** (listwise, per SET) is the *ranking* term. It depends only on WITHIN-set
  score differences, so it is invariant to adding any per-set constant and therefore
  **cannot set the cross-volume level**.
* **focal BCE** (per candidate, batch-mean) is the *calibration* term — the only one that
  supplies cross-volume gradient. That matters because the headroom decomposition on the
  PROMOTED pool ([F.8]) puts **+0.1754 of the +0.2229 total headroom (78.7 %)** in
  cross-volume calibration and only +0.0475 (21.3 %) in within-set ranking, a **3.69 : 1**
  ratio. Every ranking rung therefore carries the BCE term at a swept λ, so A2/FULL vs B2
  compares *ranking + BCE* against *BCE alone* and the calibration confound is removed.

  The DIRECTION of this argument survived the substrate promotion unchanged (it was 3.46 : 1
  / 77.6 % on the archived pool), but the absolute prize shrank: ``score_max`` now captures
  **39.8 %** of the distance from ``volume_neutral`` to ``per_vol_oracle``, versus 18.8 %
  before, so ~60 % of the volume-trust signal is unexploited rather than ~81 %. Do not quote
  the old +0.2405.

**RS-loss is deliberately NOT offered.** Phase-0a single-lesion dominance (99/100 Train,
29/30 Val) plus frozen localisation (the rescorer regresses no boxes) plus ``det_score.py``
collapsing duplicate hits make RS's "sort positives by IoU" term inert here. Recorded as a
decided flip condition, not an open choice.

**Inv. 11:** ``label == "ignore"`` (IoU 0.1–0.3) is dropped from BOTH terms while staying
in the set for attention. **Sets with no positive** (**19 of 100** train sets on the promoted
pool — 81/100 are TP-bearing, [F.9] §4; was 25/97) are KEPT: they contribute to ``focal_bce``
only, as the all-negative calibration anchors the level term needs.

Labels are encoded numerically: ``+1`` positive, ``0`` negative, ``-1`` ignore.
Everything here is array-agnostic (numpy on the laptop, torch under autograd on the server).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .. import conventions as C

__all__ = ["POS_LABEL", "NEG_LABEL", "IGNORE_LABEL", "encode_labels",
           "smooth_ap_loss", "focal_bce", "rescorer_loss", "to_probability",
           "soft_quality_target", "per_lesion_weights"]

POS_LABEL, NEG_LABEL, IGNORE_LABEL = 1.0, 0.0, -1.0

_LABEL_CODE = {"pos": POS_LABEL, "neg": NEG_LABEL, "ignore": IGNORE_LABEL}


def encode_labels(labels) -> np.ndarray:
    """Map the record's string ``label`` column to the numeric code used here."""
    return np.array([_LABEL_CODE[str(x)] for x in np.asarray(labels).ravel()],
                    dtype=float).reshape(np.asarray(labels).shape)


# ----------------------------------------------------------------------------- dispatch
def _xp(x):
    if type(x).__module__.split(".")[0] == "torch":
        import torch
        return torch
    return np


def _sigmoid(xp, x):
    """Numerically stable logistic, identical in numpy and torch."""
    if xp is not np:
        return xp.sigmoid(x)
    out = np.empty_like(np.asarray(x, dtype=float))
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softplus(xp, x):
    """``log(1 + exp(x))`` without overflow: ``max(x,0) + log1p(exp(-|x|))``."""
    if xp is not np:
        import torch
        return torch.nn.functional.softplus(x)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _clip01(xp, x):
    return np.clip(x, 0.0, 1.0) if xp is np else xp.clamp(x, 0.0, 1.0)


def _as_mask(xp, set_mask, like):
    if set_mask is None:
        return xp.ones_like(like)
    if xp is not np:
        return set_mask.to(like.dtype)
    return np.asarray(set_mask, dtype=float)


# ----------------------------------------------------------------------------- ranking term
def smooth_ap_loss(logits, labels, set_mask=None, tau: Optional[float] = None):
    """``1 - mean_S AP_S`` over the sets containing at least one positive.

    Per set ``S`` with positives ``P`` and negatives ``N`` (ignore EXCLUDED, Inv. 11)::

        AP_S = (1/|P|) * sum_{i in P} [1 + sum_{j in P\\{i}}      G(s_j - s_i)]
                                     / [1 + sum_{j in (P u N)\\{i}} G(s_j - s_i)]
        G(x) = sigmoid(x / tau)

    Autograd-differentiable end to end. Sets with no positive contribute nothing (they are
    the ``focal_bce`` anchors); a batch with no positive anywhere returns 0.
    """
    tau = C.RESC_SMOOTH_AP_TAU if tau is None else float(tau)
    xp = _xp(logits)
    mask = _as_mask(xp, set_mask, logits)

    is_pos = (labels > 0.5) if xp is np else (labels > 0.5)
    is_neg = (labels > -0.5) & (labels < 0.5)
    p = (is_pos.astype(float) if xp is np else is_pos.to(logits.dtype)) * mask
    v = ((is_pos | is_neg).astype(float) if xp is np else (is_pos | is_neg).to(logits.dtype)) * mask

    # G[b,i,j] = sigmoid((s_j - s_i) / tau)
    diff = (logits[:, None, :] - logits[:, :, None]) / tau
    g = _sigmoid(xp, diff)

    # exclude j == i: G(0) = 0.5, subtracted once from each running sum
    num = 1.0 + (g * p[:, None, :]).sum(-1) - 0.5
    den = 1.0 + (g * v[:, None, :]).sum(-1) - 0.5
    ap_terms = p * (num / den)                       # only i in P counts

    n_pos = p.sum(-1)                                # (B,)
    has_pos = (n_pos > 0.5)
    safe = xp.where(has_pos, n_pos, xp.ones_like(n_pos))
    ap = ap_terms.sum(-1) / safe                     # (B,)
    hp = (has_pos.astype(float) if xp is np else has_pos.to(logits.dtype))
    n_sets = hp.sum()
    if float(n_sets) == 0.0:
        return xp.zeros_like(logits.sum())
    return 1.0 - (ap * hp).sum() / n_sets


# ----------------------------------------------------------------------------- calibration term
def focal_bce(logits, labels, alpha: float, gamma: Optional[float] = None, set_mask=None,
              soft: bool = False, weights=None):
    """Per-candidate focal BCE over pos+neg (ignore excluded), averaged over ALL such
    candidates in the batch — **including** the candidates of all-negative sets.

    This is the only term supplying cross-volume calibration gradient: the quantity the
    official single global FROC threshold actually reads.

    Two OPT-IN alignment knobs, both added 2026-08-11 for the ``[4.2b]`` study and both
    inert at their defaults, so the deployed loss is bit-identical unless asked otherwise:

    ``soft``
        Read ``labels`` as a **continuous target in [0, 1]** (see :func:`soft_quality_target`)
        instead of the hard ``+1/0/-1`` code; anything negative is still the ignore code, so
        the two conventions can be mixed. With ``gamma = 0`` this is a proper scoring rule
        whose optimum sits exactly at the target.
    ``weights``
        Per-candidate weights, broadcast over ``logits``, applied as a **weighted mean** (the
        normaliser is ``sum(valid * weight)``, not the count). :func:`per_lesion_weights`
        supplies the metric-aligned choice: ``det_score`` collapses duplicate hits on one GT
        to a single TP, so a lesion covered by 40 candidates must not out-weight one covered
        by 1.
    """
    gamma = C.RESC_FOCAL_GAMMA if gamma is None else float(gamma)
    xp = _xp(logits)
    mask = _as_mask(xp, set_mask, logits)

    if soft:
        y = _clip01(xp, labels)
        valid = labels > -0.5
        v = (valid.astype(float) if xp is np else valid.to(logits.dtype)) * mask
    else:
        is_pos = labels > 0.5
        is_neg = (labels > -0.5) & (labels < 0.5)
        y = (is_pos.astype(float) if xp is np else is_pos.to(logits.dtype))
        v = ((is_pos | is_neg).astype(float) if xp is np
             else (is_pos | is_neg).to(logits.dtype)) * mask
    if weights is not None:
        # device AND dtype: a caller handing numpy weights to CUDA logits must not blow up on
        # a cross-device multiply, and `.to(dtype)` alone does not move a tensor.
        v = v * (weights if xp is np
                 else xp.as_tensor(weights).to(device=logits.device, dtype=logits.dtype))

    # log p_t without overflow: log sigmoid(x) = -softplus(-x); log(1-sigmoid(x)) = -softplus(x)
    log_pt = -(y * _softplus(xp, -logits) + (1.0 - y) * _softplus(xp, logits))
    pt = xp.exp(log_pt)
    w = alpha * y + (1.0 - alpha) * (1.0 - y)
    per = -w * ((1.0 - pt) ** gamma) * log_pt

    n = v.sum()
    if float(n) == 0.0:
        return xp.zeros_like(logits.sum())
    return (per * v).sum() / n


# ------------------------------------------------------- [4.2b] alignment primitives
def soft_quality_target(iou, lo: Optional[float] = None, hi: Optional[float] = None):
    """``clip((IoU - LABEL_NEG_IOU) / (LABEL_POS_IOU - LABEL_NEG_IOU), 0, 1)``.

    Inv. 11's ignore band turned from a **hole** into a **ramp**. The band exists because a
    candidate at IoU 0.2 is ambiguous *for a detector*, which can still move its box. The
    rescorer cannot — the boxes are frozen (Inv. 8) — and the oracle has no ambiguity about
    those rows either: at ``iou_thresh = 0.3`` it scores every one of them as a false
    positive. Masking them (1544/10662 train rows, 1039/7822 val) hides from the model
    precisely the population that is most expensive at low FP budgets.

    Full credit at and above the oracle's own hit threshold, so **no capacity is spent
    distinguishing IoU 0.6 from 0.9**, where the metric is a flat step. (The oracle's test is
    strict, ``max_iou > iou_thresh``; the ramp reaching 1.0 exactly at 0.30 is a measure-zero
    difference and is deliberate — the target is "as good as a hit", not "is a hit".)

    Returns a plain array: this builds the target, it is not part of the autograd graph.
    """
    lo = float(C.LABEL_NEG_IOU if lo is None else lo)
    hi = float(C.LABEL_POS_IOU if hi is None else hi)
    return np.clip((np.asarray(iou, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def per_lesion_weights(labels, set_mask=None):
    """Positive weights that sum to ONE LESION per set; negatives and anchors stay at 1.0.

    Phase 0a established that ``det_score`` collapses duplicate candidates hitting one GT
    into a single TP — *duplicates are free*. The loss does not: the promoted train pool
    carries **1265 positives over 81 TP-bearing volumes** (mean 15.6, long-tailed), so under
    a plain mean a lesion covered by 40 candidates contributes 40x the positive gradient of
    one covered by 1, while contributing **identically** to CPM.

    **Approximation, stated rather than hidden:** one lesion per SET. The record has no GT
    lesion id (``iou_gt`` is the max IoU only), and Phase 0a's single-lesion dominance holds
    in 99/100 Train and 29/30 Val volumes, so "the positives of this set" and "the positives
    on this lesion" coincide except in the handful of multi-lesion volumes, where this
    under-weights by the lesion count.

    Sets with no positive are left at unit weight — they are the all-negative cross-volume
    calibration anchors ``focal_bce`` needs, and zeroing them would delete the level term.
    """
    lab = np.asarray(labels, dtype=float)
    m = np.ones_like(lab) if set_mask is None else np.asarray(set_mask, dtype=float)
    is_pos = (lab > 0.5) & (m > 0.5)
    n_pos = is_pos.sum(axis=-1, keepdims=True)
    share = 1.0 / np.maximum(n_pos, 1.0)
    return np.where(is_pos, share, 1.0) * m


# ----------------------------------------------------------------------------- combination
def to_probability(logits, eps: Optional[float] = None):
    """``sigmoid(logit)`` clamped to ``[0, 1 - RESC_PROB_EPS]`` — the det_score contract.

    The official FROC sweep is ``np.arange(0, 1, 0.005)`` and ``write_pred_csv`` rejects
    anything outside ``[0, 1)``, so a saturated sigmoid returning exactly 1.0 would abort
    the whole evaluation. Array-agnostic, so it is used identically by the torch forward
    and by the numpy scoring path.
    """
    eps = C.RESC_PROB_EPS if eps is None else float(eps)
    xp = _xp(logits)
    p = _sigmoid(xp, logits)
    hi = 1.0 - eps
    if xp is np:
        return np.clip(p, 0.0, hi)
    return xp.clamp(p, min=0.0, max=hi)


def rescorer_loss(logits, labels, set_mask, w_rank: float, lam: float, alpha: float,
                  tau: Optional[float] = None, gamma: Optional[float] = None,
                  soft: bool = False, bce_weights=None) -> Tuple:
    """``(total, {"rank": ..., "bce": ...})`` — the per-variant weighted sum (§4.7).

    ``w_rank = 0`` is the CE-only rung (B1/B2); ``lam = RESC_LAMBDA_DIAGNOSTIC = 0`` is the
    pure-ranking diagnostic endpoint (reported, excluded from selection).

    ``soft`` / ``bce_weights`` are the `[4.2b]` alignment knobs; both are inert at their
    defaults, so every rung trained before 2026-08-11 is reproduced bit-for-bit.

    **``soft`` is refused with a ranking term.** ``smooth_ap_loss`` reads ``labels > 0.5`` as
    the positive set, which on the ``iou_gt`` ramp means IoU > 0.20 — not the oracle's hit
    test (IoU > 0.30). Ranking against the wrong positive set would be a silent defect of
    exactly the kind Inv. 13 cost this project twice, so it raises instead. Aligning the
    RANKING term with the metric is a separate, larger change (the ranking term is per-set
    and provably shift-invariant, while the metric thresholds once GLOBALLY across volumes)
    and is deliberately not smuggled in here.
    """
    if soft and float(w_rank) != 0.0:
        raise ValueError(
            "soft targets are only defined for the CE rungs (w_rank = 0): smooth_ap_loss "
            "would read `labels > 0.5` as IoU > 0.20, not the oracle's IoU > 0.30 hit test. "
            "A metric-aligned ranking term is a separate change.")
    rank = smooth_ap_loss(logits, labels, set_mask, tau=tau)
    bce = focal_bce(logits, labels, alpha=alpha, gamma=gamma, set_mask=set_mask,
                    soft=soft, weights=bce_weights)
    return w_rank * rank + lam * bce, {"rank": rank, "bce": bce}
