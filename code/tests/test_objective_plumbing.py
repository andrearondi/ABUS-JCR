"""[4.2b] Wiring the objective variants through the production loss and batcher.

The study must exercise the DEPLOYED code path, not a parallel copy of it — otherwise a
promoted variant would be re-implemented on the way in. So the knobs live in
``rescorer_loss``/``collate_sets`` and are inert at their defaults.
"""

import numpy as np
import pytest

from abus_jcr.rescore.datasets import batch_row_weights
from abus_jcr.rescore.losses import rescorer_loss


# ----------------------------------------------------------------- rescorer_loss pass-through
def test_defaults_leave_the_deployed_loss_bit_identical():
    logits = np.array([[0.4, -1.2, 2.0]])
    labels = np.array([[1.0, 0.0, -1.0]])
    mask = np.ones_like(labels)
    a, _ = rescorer_loss(logits, labels, mask, w_rank=0.0, lam=1.0, alpha=0.25)
    b, _ = rescorer_loss(logits, labels, mask, w_rank=0.0, lam=1.0, alpha=0.25,
                         soft=False, bce_weights=None)
    assert float(a) == float(b)


def test_bce_weights_reach_the_calibration_term():
    logits = np.array([[0.4, 0.4]])
    labels = np.array([[1.0, 0.0]])
    mask = np.ones_like(labels)
    flat, _ = rescorer_loss(logits, labels, mask, w_rank=0.0, lam=1.0, alpha=0.5)
    tilted, _ = rescorer_loss(logits, labels, mask, w_rank=0.0, lam=1.0, alpha=0.5,
                              bce_weights=np.array([[9.0, 1.0]]))
    assert float(flat) != float(tilted)


def test_a_fractional_target_is_invisible_to_the_hard_path_and_consumed_by_the_soft_one():
    """The ramp's whole point: an IoU-0.20 row scores 0.5, which the hard code reads as
    NEITHER pos nor neg — it vanishes from the loss, exactly as Inv. 11 intends and exactly
    as the oracle does not."""
    logits = np.array([[0.0]])
    labels = np.array([[0.5]])
    mask = np.ones((1, 1))
    hard, _ = rescorer_loss(logits, labels, mask, w_rank=0.0, lam=1.0, alpha=0.5)
    soft, _ = rescorer_loss(logits, labels, mask, w_rank=0.0, lam=1.0, alpha=0.5, soft=True)
    assert float(hard) == 0.0, "a mid-band row contributes nothing under the hard code"
    assert float(soft) > 0.0, "the ramp must give it a gradient"


def test_soft_targets_with_a_ranking_term_are_refused_not_silently_misread():
    """smooth_ap reads `labels > 0.5` as positive; on a ramp that means IoU > 0.20, which is
    NOT the oracle's hit test. Rather than quietly rank against the wrong set, refuse."""
    with pytest.raises(ValueError, match="soft"):
        rescorer_loss(np.array([[0.1, 0.2]]), np.array([[1.0, 0.5]]), np.ones((1, 2)),
                      w_rank=1.0, lam=1.0, alpha=0.5, soft=True)


# ----------------------------------------------------------------- batch weight lookup
def test_padded_slots_get_zero_weight():
    rows = np.array([[0, 2, -1]])
    w = batch_row_weights(rows, np.array([0.5, 9.9, 0.25]))
    assert w.ravel() == pytest.approx([0.5, 0.25, 0.0])


def test_shape_follows_the_padded_batch():
    rows = np.full((3, 7), -1, dtype=np.int64)
    rows[0, :2] = [1, 0]
    w = batch_row_weights(rows, np.array([1.0, 0.5]))
    assert w.shape == (3, 7)
    assert w[0, 0] == pytest.approx(0.5) and w[0, 1] == pytest.approx(1.0)
    assert w[1].sum() == 0.0


def test_none_means_no_weighting():
    assert batch_row_weights(np.array([[0, -1]]), None) is None
