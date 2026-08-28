"""[4.6] Axis C — the pooled, duplicate-aware FROC surrogate.

The claim under test is narrow and checkable. Written out, the metric is

    CPM = (1/L) * sum_lesions (1/7) * sum_k  1[ cost_lesion / n_vol <= f_k ]

where ``cost`` is the number of NON-HITTING candidates scored above a lesion's best hitting
candidate, counted across every volume in the batch, and ``L`` counts every lesion including
the ones no candidate reaches. Three smoothings make that differentiable. At low temperature
the surrogate must return the step definition exactly, and it must be sensitive to the
per-volume score offsets that ``smooth_ap_loss`` is provably blind to.
"""

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.losses import (POS_LABEL, NEG_LABEL, IGNORE_LABEL, froc_surrogate_loss,
                                     rescorer_loss, smooth_ap_loss)

TINY = dict(tau=1e-3, beta=1e-3, tau_max=1e-3)


def _sets(spec):
    """``spec = [[(label, score), ...], ...]`` -> padded ``(logits, labels, mask)``."""
    n = max(len(s) for s in spec)
    logits = np.zeros((len(spec), n)); labels = np.zeros((len(spec), n)); mask = np.zeros((len(spec), n))
    for b, s in enumerate(spec):
        for i, (lab, sc) in enumerate(s):
            logits[b, i] = sc; labels[b, i] = lab; mask[b, i] = 1.0
    return logits, labels, mask


def _cost_pool():
    """Four volumes with lesion costs 0, 3, 5 and one unreachable lesion.

    None of those costs coincides with a budget (``f_k * 4`` = 0.5, 1, 2, 4, 8, 16, 32), so the
    limit is not read on a sigmoid's midpoint. Sensitivity is 7/7, 4/7, 3/7 and 0/7, so the step
    CPM is 14/28 = 0.5 exactly.
    """
    return _sets([
        [(POS_LABEL, 1.00)],
        [(POS_LABEL, 0.80), (NEG_LABEL, 0.95), (NEG_LABEL, 0.90)],
        [(POS_LABEL, 0.65), (NEG_LABEL, 0.85), (NEG_LABEL, 0.75), (NEG_LABEL, 0.70)],
        [(NEG_LABEL, 0.60), (NEG_LABEL, 0.55)],
    ])


def test_zero_temperature_limit_is_the_step_definition_of_cpm():
    logits, labels, mask = _cost_pool()
    loss = froc_surrogate_loss(logits, labels, mask, **TINY)
    assert float(loss) == pytest.approx(0.5, abs=1e-6)


def test_agrees_with_the_official_evaluator_within_its_tie_ambiguity():
    """The evaluator reads a swept curve, and where two operating points share one FP rate its
    ``_get_key_recall`` picks between them with a non-stable sort. That ambiguity is worth at
    most one lesion per rate, which is the tolerance asserted here; anything larger would mean
    the surrogate is not counting what the evaluator counts.
    """
    from abus_jcr.eval.froc import evaluate_froc, cpm
    from abus_jcr.candidates.record import to_official_pred_csv  # noqa: F401  (schema reference)
    import pandas as pd

    n_vol, hit, miss = 20, dict(x=100.0), dict(x=400.0)
    spec, rows, gt, score = [], [], [], 0.98
    for v in range(n_vol):                       # lesion v sits behind exactly v false positives
        gt.append({"public_id": v, "coordX": 100.0, "coordY": 100.0, "coordZ": 100.0,
                   "x_length": 20.0, "y_length": 20.0, "z_length": 20.0})
        s = []
        if v:                                    # one FP above each lesion after the first
            s.append((NEG_LABEL, score)); rows.append((v, miss, score)); score -= 0.02
        s.append((POS_LABEL, score)); rows.append((v, hit, score)); score -= 0.02
        spec.append(s)
    logits, labels, mask = _sets(spec)
    surrogate_cpm = 1.0 - float(froc_surrogate_loss(logits, labels, mask, **TINY))

    pred = pd.DataFrame([{"public_id": v, "coordX": g["x"], "coordY": 100.0 if g is hit else 400.0,
                          "coordZ": 100.0 if g is hit else 400.0, "x_length": 20.0,
                          "y_length": 20.0, "z_length": 20.0, "probability": p}
                         for v, g, p in rows])
    official = cpm(evaluate_froc(pd.DataFrame(gt)[C.GT_COLUMNS], pred[C.PRED_COLUMNS]))
    assert surrogate_cpm == pytest.approx(official, abs=1.0 / n_vol)


def test_is_invariant_to_a_global_score_shift_exactly_as_the_metric_is():
    logits, labels, mask = _cost_pool()
    a = float(froc_surrogate_loss(logits, labels, mask, **TINY))
    b = float(froc_surrogate_loss(logits + 3.7, labels, mask, **TINY))
    assert b == pytest.approx(a, abs=1e-9)


def test_is_sensitive_to_a_per_volume_shift_where_smooth_ap_is_not():
    """The pathology the whole design turns on, and the one the headroom decomposition measured.

    Volume 1's scores are lifted bodily by a constant. Its own ordering is untouched, so its
    lesion is no easier to find inside it, but its two artifacts now outrank the lesions of every
    other volume, and a single global threshold pays for that. ``smooth_ap_loss`` depends only on
    within-set differences and cannot see it; asserting both is what makes the test distinguish
    the two objectives rather than merely exercise one.

    ``n_vol = 7`` keeps every rate off a key value, so nothing is read on a sigmoid's midpoint.
    """
    logits, labels, mask = _cost_pool()
    lifted = logits.copy()
    lifted[1] += 0.5
    assert float(froc_surrogate_loss(lifted, labels, mask, n_vol=7, **TINY)) > \
           float(froc_surrogate_loss(logits, labels, mask, n_vol=7, **TINY)) + 1e-6
    assert float(smooth_ap_loss(lifted, labels, mask)) == \
           pytest.approx(float(smooth_ap_loss(logits, labels, mask)), abs=1e-9)


def test_counts_ignore_band_candidates_as_false_positives():
    """``det_score`` charges anything that fails the hit test, whatever its overlap, so an
    ignore-band candidate above a lesion costs exactly as much as a negative would."""
    base = _sets([[(POS_LABEL, 0.50)], [(NEG_LABEL, 0.10)]])
    with_ign = _sets([[(POS_LABEL, 0.50)], [(IGNORE_LABEL, 0.90)]])
    with_neg = _sets([[(POS_LABEL, 0.50)], [(NEG_LABEL, 0.90)]])
    l0 = float(froc_surrogate_loss(*base, **TINY))
    li = float(froc_surrogate_loss(*with_ign, **TINY))
    ln = float(froc_surrogate_loss(*with_neg, **TINY))
    assert li > l0 + 1e-6
    assert li == pytest.approx(ln, abs=1e-9)


def test_unreachable_lesions_stay_in_the_denominator():
    """The evaluator divides recall by every ground-truth lesion, so a volume whose pool holds
    no hit must drag the surrogate down rather than being skipped."""
    reachable = _sets([[(POS_LABEL, 0.9)], [(POS_LABEL, 0.8)]])
    one_lost = _sets([[(POS_LABEL, 0.9)], [(NEG_LABEL, 0.8)]])
    assert float(froc_surrogate_loss(*reachable, **TINY)) == pytest.approx(0.0, abs=1e-6)
    assert float(froc_surrogate_loss(*one_lost, **TINY)) == pytest.approx(0.5, abs=1e-6)


def test_reference_population_adds_false_positives_without_adding_lesions():
    """The reference table stands in for the volumes outside the batch: its candidates raise
    every lesion's cost, and its volumes enter the rate, but it contributes no lesion of its own."""
    logits, labels, mask = _sets([[(POS_LABEL, 0.50), (NEG_LABEL, 0.40)]])
    alone = float(froc_surrogate_loss(logits, labels, mask, n_vol=4, **TINY))
    with_ref = float(froc_surrogate_loss(logits, labels, mask, n_vol=4,
                                         ref_logits=np.array([0.9, 0.8, 0.7]),
                                         ref_labels=np.array([NEG_LABEL] * 3), **TINY))
    assert alone == pytest.approx(0.0, abs=1e-6)       # cost 0 -> under every budget
    assert with_ref > alone + 1e-6                     # 3 FPs above it -> 0.75/vol, misses two rates


def test_gradient_reaches_candidate_scores_but_never_the_ignore_band():
    torch = pytest.importorskip("torch")
    logits, labels, mask = _cost_pool()
    labels[2, 1] = IGNORE_LABEL                        # one of volume 2's false positives
    t = torch.tensor(logits, requires_grad=True)
    loss = froc_surrogate_loss(t, torch.tensor(labels), torch.tensor(mask), tau=0.1, beta=0.1)
    loss.backward()
    g = t.grad.numpy()
    assert abs(g[2, 1]) == 0.0                         # Inv. 11: the ambiguous middle is not taught
    assert abs(g[2, 0]) > 0.0                          # its lesion is
    assert abs(g[1, 1]) > 0.0                          # a plain negative above a lesion is


def test_default_temperatures_come_from_conventions():
    logits, labels, mask = _cost_pool()
    a = float(froc_surrogate_loss(logits, labels, mask))
    b = float(froc_surrogate_loss(logits, labels, mask, tau=C.RESC_FROC_TAU,
                                  beta=C.RESC_FROC_BETA, tau_max=C.RESC_FROC_MAX_TAU))
    assert a == pytest.approx(b, abs=1e-12)


# ---- wiring: the three added rungs and the loss selector ----------------------
#
# The six pre-registered rungs are NOT touched. The pooled surrogate arrives as three extra
# conditions, exactly as thesis.tex §3.1.3 describes them, so that "fixed in advance" keeps
# meaning what it says.

def test_the_pre_registered_ladder_is_untouched_by_the_addition():
    from abus_jcr.rescore.variants import LADDER, COMPARISONS
    assert LADDER == ("B0", "B1", "B2", "A1", "A2", "FULL")
    assert ("FULL", "B2") == COMPARISONS[0]
    assert all("-P" not in a and "-P" not in b for a, b in COMPARISONS)


def test_the_pooled_rungs_mirror_their_per_volume_twins():
    from abus_jcr.rescore.variants import VARIANTS, LADDER, LADDER_POOLED
    assert LADDER_POOLED == ("B1-P", "A2-P", "FULL-P")
    for pooled, twin in (("A2-P", "A2"), ("FULL-P", "FULL")):
        for k in ("module", "geometry", "w_rank", "search"):
            assert VARIANTS[pooled][k] == VARIANTS[twin][k], f"{pooled}.{k} must match {twin}"
    # B1-P is the independent classifier trained against the surrogate: no cross-candidate path,
    # but a ranking term, which is what separates "the objective" from "the set module".
    assert VARIANTS["B1-P"]["module"] == "mlp" and VARIANTS["B1-P"]["w_rank"] == 1.0
    assert all(VARIANTS[v]["rank_loss"] == "froc" for v in LADDER_POOLED)
    assert all(VARIANTS[v].get("rank_loss", "smooth_ap") == "smooth_ap" for v in LADDER[1:])


def test_the_pooled_comparisons_attribute_the_effect():
    from abus_jcr.rescore.variants import COMPARISONS_POOLED
    assert ("FULL-P", "FULL") in COMPARISONS_POOLED     # the objective, architecture held fixed
    assert ("FULL-P", "B1-P") in COMPARISONS_POOLED     # the set module, objective held fixed


def test_rescorer_loss_dispatches_to_the_surrogate_and_leaves_the_default_alone():
    logits, labels, mask = _cost_pool()
    smooth, _ = rescorer_loss(logits, labels, mask, w_rank=1.0, lam=0.0, alpha=0.25)
    froc, parts = rescorer_loss(logits, labels, mask, w_rank=1.0, lam=0.0, alpha=0.25,
                                rank_loss="froc", n_vol=7, tau=1e-3)
    assert float(smooth) == pytest.approx(float(smooth_ap_loss(logits, labels, mask)), abs=1e-9)
    assert float(parts["rank"]) == pytest.approx(
        float(froc_surrogate_loss(logits, labels, mask, n_vol=7, tau=1e-3)), abs=1e-9)
    assert float(froc) != pytest.approx(float(smooth), abs=1e-6)
