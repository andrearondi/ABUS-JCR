"""[4.6] Axis C — the two loss terms and the property that makes BOTH necessary.

The headline fact this module pins: ``smooth_ap_loss`` depends only on WITHIN-set score
differences, so it is invariant to adding any per-set constant — it **cannot** set the
cross-volume level. The [P3U2.12] decomposition puts **77.6 %** of the total ranking
headroom in exactly that cross-volume level, which is why every ranking rung also carries
the ``λ·focal_bce`` term (and why A2/FULL vs B2 is a fair comparison rather than a
calibration confound).

The losses are array-agnostic, so their VALUE semantics are tested locally with numpy;
autograd is additionally checked wherever torch is installed.
"""

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.losses import focal_bce, rescorer_loss, smooth_ap_loss

POS, NEG, IGN = 1, 0, -1


def _one_set(logits, labels):
    return np.array([logits], dtype=float), np.array([labels], dtype=float)


# --------------------------------------------------------------------------- smooth AP
def test_perfect_ranking_gives_zero_loss():
    lg, lb = _one_set([5.0, 4.0, -1.0, -2.0, -3.0], [POS, POS, NEG, NEG, NEG])
    assert float(smooth_ap_loss(lg, lb)) < 1e-3


def test_worst_ranking_gives_a_large_loss():
    lg, lb = _one_set([-3.0, -2.0, 1.0, 2.0, 3.0], [POS, POS, NEG, NEG, NEG])
    assert float(smooth_ap_loss(lg, lb)) > 0.5


def test_swapping_a_positive_below_a_negative_increases_the_loss():
    good, lb = _one_set([5.0, 1.0, 0.0, -1.0], [POS, NEG, NEG, NEG])
    bad, _ = _one_set([-0.5, 1.0, 0.0, -1.0], [POS, NEG, NEG, NEG])
    assert float(smooth_ap_loss(bad, lb)) > float(smooth_ap_loss(good, lb)) + 0.1


def test_loss_degrades_monotonically_as_the_positive_sinks():
    lb = np.array([[POS, NEG, NEG, NEG, NEG]], dtype=float)
    losses = []
    for pos_score in (5.0, 0.5, -0.5, -1.5, -2.5):
        lg = np.array([[pos_score, 2.0, 1.0, 0.0, -1.0]], dtype=float)
        losses.append(float(smooth_ap_loss(lg, lb)))
    assert all(b >= a - 1e-9 for a, b in zip(losses, losses[1:])), losses


def test_smooth_ap_is_invariant_to_a_PER_SET_constant_shift():
    """THE property: the ranking term cannot set the cross-volume level."""
    lg = np.array([[3.0, 1.0, 0.0, -1.0], [2.0, 0.5, -0.5, -2.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG], [POS, NEG, NEG, NEG]], dtype=float)
    shifted = lg.copy()
    shifted[0] += 7.3                      # shift ONE set only
    assert float(smooth_ap_loss(shifted, lb)) == pytest.approx(float(smooth_ap_loss(lg, lb)), abs=1e-9)


def test_focal_bce_is_NOT_invariant_to_that_same_shift():
    """...which is exactly why the lambda*BCE term is required (spec §4.6)."""
    lg = np.array([[3.0, 1.0, 0.0, -1.0], [2.0, 0.5, -0.5, -2.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG], [POS, NEG, NEG, NEG]], dtype=float)
    shifted = lg.copy()
    shifted[0] += 7.3
    base = float(focal_bce(lg, lb, alpha=0.25))
    assert abs(float(focal_bce(shifted, lb, alpha=0.25)) - base) > 1e-3


def test_ignore_band_candidates_enter_neither_loss_term():
    """Inv. 11: IoU 0.1-0.3 stays in the set (for attention) but out of both losses."""
    lg = np.array([[3.0, 1.0, 0.0, -1.0]], dtype=float)
    lb = np.array([[POS, IGN, NEG, NEG]], dtype=float)
    moved = lg.copy()
    moved[0, 1] = 99.0                     # an ignore candidate scored absurdly high
    assert float(smooth_ap_loss(moved, lb)) == pytest.approx(float(smooth_ap_loss(lg, lb)), abs=1e-12)
    assert float(focal_bce(moved, lb, alpha=0.25)) == pytest.approx(
        float(focal_bce(lg, lb, alpha=0.25)), abs=1e-12)


def test_all_negative_sets_contribute_zero_to_the_ranking_term():
    """25/97 train sets carry no positive; they are KEPT as calibration anchors, but the
    listwise term must ignore them rather than divide by zero."""
    lg = np.array([[3.0, 1.0, 0.0], [2.0, 1.0, 0.5]], dtype=float)
    lb = np.array([[POS, NEG, NEG], [NEG, NEG, NEG]], dtype=float)
    with_allneg = float(smooth_ap_loss(lg, lb))
    only_pos_set = float(smooth_ap_loss(lg[:1], lb[:1]))
    assert with_allneg == pytest.approx(only_pos_set, abs=1e-12)
    assert np.isfinite(with_allneg)


def test_all_negative_sets_DO_contribute_to_the_calibration_term():
    lg = np.array([[3.0, 1.0, 0.0], [2.0, 1.0, 0.5]], dtype=float)
    lb = np.array([[POS, NEG, NEG], [NEG, NEG, NEG]], dtype=float)
    assert float(focal_bce(lg, lb, alpha=0.25)) != pytest.approx(
        float(focal_bce(lg[:1], lb[:1], alpha=0.25)), abs=1e-6)


def test_a_batch_with_no_positive_anywhere_gives_zero_ranking_loss():
    lg = np.array([[1.0, 0.0, -1.0]], dtype=float)
    lb = np.array([[NEG, NEG, NEG]], dtype=float)
    assert float(smooth_ap_loss(lg, lb)) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- padding
def test_padded_slots_change_neither_loss():
    lg = np.array([[3.0, 1.0, 0.0, 0.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG]], dtype=float)
    mask = np.array([[1.0, 1.0, 1.0, 0.0]])
    lg_junk = lg.copy()
    lg_junk[0, 3] = 500.0
    lb_junk = lb.copy()
    lb_junk[0, 3] = POS
    assert float(smooth_ap_loss(lg_junk, lb_junk, mask)) == pytest.approx(
        float(smooth_ap_loss(lg, lb, mask)), abs=1e-12)
    assert float(focal_bce(lg_junk, lb_junk, alpha=0.25, set_mask=mask)) == pytest.approx(
        float(focal_bce(lg, lb, alpha=0.25, set_mask=mask)), abs=1e-12)


def test_masked_batch_equals_the_unpadded_batch():
    lg = np.array([[3.0, 1.0, 0.0, -7.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG]], dtype=float)
    mask = np.array([[1.0, 1.0, 1.0, 0.0]])
    assert float(smooth_ap_loss(lg, lb, mask)) == pytest.approx(
        float(smooth_ap_loss(lg[:, :3], lb[:, :3])), abs=1e-9)


# --------------------------------------------------------------------------- focal BCE
def test_focal_bce_is_lower_for_confidently_correct_predictions():
    lb = np.array([[POS, NEG, NEG]], dtype=float)
    confident = np.array([[6.0, -6.0, -6.0]], dtype=float)
    wrong = np.array([[-6.0, 6.0, 6.0]], dtype=float)
    assert float(focal_bce(confident, lb, alpha=0.25)) < float(focal_bce(wrong, lb, alpha=0.25))


def test_focal_bce_is_finite_at_saturating_logits():
    lb = np.array([[POS, NEG]], dtype=float)
    for x in (500.0, -500.0):
        assert np.isfinite(float(focal_bce(np.array([[x, -x]]), lb, alpha=0.25)))


def test_focal_gamma_zero_reduces_to_alpha_weighted_bce():
    lg = np.array([[1.0, -1.0]], dtype=float)
    lb = np.array([[POS, NEG]], dtype=float)
    p = 1.0 / (1.0 + np.exp(-lg))
    expected = -(0.25 * np.log(p[0, 0]) + 0.75 * np.log(1 - p[0, 1])) / 2.0
    assert float(focal_bce(lg, lb, alpha=0.25, gamma=0.0)) == pytest.approx(expected, abs=1e-9)


def test_alpha_shifts_the_positive_negative_balance():
    """On the IMBALANCED sets this pool actually has (~1:7.5 pos:neg), alpha moves the loss;
    on a perfectly balanced 1:1 set it provably cannot, so the test uses 1 pos + 2 neg."""
    lg = np.array([[0.0, 0.0, 0.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG]], dtype=float)
    a25 = float(focal_bce(lg, lb, alpha=0.25))
    a50 = float(focal_bce(lg, lb, alpha=0.50))
    assert a25 > a50 + 1e-6, (a25, a50)     # alpha 0.25 up-weights the majority negatives


# --------------------------------------------------------------------------- combination
def test_rescorer_loss_is_the_declared_weighted_sum():
    lg = np.array([[3.0, 1.0, 0.0, -1.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG]], dtype=float)
    total, parts = rescorer_loss(lg, lb, None, w_rank=1.0, lam=0.3, alpha=0.25)
    assert float(total) == pytest.approx(
        1.0 * float(parts["rank"]) + 0.3 * float(parts["bce"]), abs=1e-12)


def test_w_rank_zero_is_the_pure_calibration_rung_B1_B2():
    lg = np.array([[3.0, 1.0, 0.0, -1.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG]], dtype=float)
    total, parts = rescorer_loss(lg, lb, None, w_rank=0.0, lam=1.0, alpha=0.25)
    assert float(total) == pytest.approx(float(parts["bce"]), abs=1e-12)


def test_lambda_zero_is_the_pure_ranking_diagnostic_endpoint():
    """RESC_LAMBDA_DIAGNOSTIC — reported, excluded from selection (spec §4.7)."""
    lg = np.array([[3.0, 1.0, 0.0, -1.0]], dtype=float)
    lb = np.array([[POS, NEG, NEG, NEG]], dtype=float)
    total, parts = rescorer_loss(lg, lb, None, w_rank=1.0,
                                 lam=C.RESC_LAMBDA_DIAGNOSTIC, alpha=0.25)
    assert float(total) == pytest.approx(float(parts["rank"]), abs=1e-12)


def test_rs_loss_is_not_offered(monkeypatch):
    """The flip condition is DECIDED, not open: Phase-0a single-lesion dominance makes RS's
    'sort positives by IoU' term inert (spec §4.6)."""
    import abus_jcr.rescore.losses as L
    assert not hasattr(L, "rs_loss")
    assert C.RESC_LOSS_RANK == "smooth_ap"


# --------------------------------------------------------------------------- autograd
def test_torch_losses_produce_finite_gradients():
    torch = pytest.importorskip("torch")
    lg = torch.tensor([[3.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.5, -0.5]],
                      dtype=torch.float64, requires_grad=True)
    lb = torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    total, _ = rescorer_loss(lg, lb, None, w_rank=1.0, lam=1.0, alpha=0.25)
    total.backward()
    assert torch.isfinite(lg.grad).all()
    assert float(lg.grad.abs().sum()) > 0.0


def test_torch_and_numpy_paths_agree():
    torch = pytest.importorskip("torch")
    lg = np.array([[3.0, 1.0, 0.0, -1.0]], dtype=float)
    lb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    t_total, _ = rescorer_loss(torch.as_tensor(lg, dtype=torch.float64),
                               torch.as_tensor(lb, dtype=torch.float64),
                               None, w_rank=1.0, lam=1.0, alpha=0.25)
    n_total, _ = rescorer_loss(lg, lb, None, w_rank=1.0, lam=1.0, alpha=0.25)
    assert float(t_total) == pytest.approx(float(n_total), abs=1e-9)
