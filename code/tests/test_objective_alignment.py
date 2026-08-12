"""[4.2b] The objective-alignment primitives — the Stage-A study's loss variants.

Four measured misalignments between the [4.3]/[4.6] objective and the official metric
motivate these; each primitive attacks exactly one and each is OFF by default, so the
deployed loss is bit-identical until a variant is promoted:

* ``soft_quality_target`` — Inv. 11's ignore band (IoU 0.10-0.30) is a hole in the
  supervision that the oracle scores as a false positive. 1544/10662 train rows, 1039/7822
  val. The ramp turns the hole into a graded target without spending capacity above the
  metric's 0.30 hit threshold, where the metric is flat.
* ``focal_bce(soft=True)`` — consume that target.
* ``focal_bce(weights=...)`` + ``per_lesion_weights`` — ``det_score`` collapses duplicate
  hits on one GT to ONE TP, but the train pool carries a mean 15.6 positives per TP-bearing
  volume (1265 over 81), so the loss over-weights well-covered lesions by their duplicate
  count while the metric weights every lesion equally.
* ``gamma=0`` — already supported by the existing signature; pinned here because plain BCE
  is a proper scoring rule and focal (gamma=2) is deliberately not, and [F.8] puts 78.7 % of
  the headroom in calibration.
"""

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore.losses import (focal_bce, per_lesion_weights, soft_quality_target)


# ----------------------------------------------------------------- the ramp target
def test_ramp_is_zero_below_the_negative_band_and_one_above_the_hit_threshold():
    t = soft_quality_target(np.array([0.0, 0.05, 0.10, 0.30, 0.55, 1.0]))
    assert t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0
    # LABEL_POS_IOU is the oracle's own hit threshold: full credit at and above it, and no
    # gradient spent distinguishing 0.55 from 1.0, where the metric is a flat step.
    assert t[3] == 1.0 and t[4] == 1.0 and t[5] == 1.0


def test_ramp_is_linear_across_the_old_ignore_band():
    mid = 0.5 * (C.LABEL_NEG_IOU + C.LABEL_POS_IOU)          # 0.20
    assert soft_quality_target(np.array([mid]))[0] == pytest.approx(0.5)


def test_ramp_never_returns_the_ignore_code():
    """No row is masked out any more — that is the whole point of the ramp."""
    t = soft_quality_target(np.linspace(0.0, 1.0, 101))
    assert (t >= 0.0).all()


# ----------------------------------------------------------------- soft-target BCE
def test_soft_target_bce_is_minimised_at_the_target_probability():
    """A proper scoring rule (gamma=0) must put its optimum exactly at the target."""
    target = np.array([0.25])
    logits = np.log(np.array([0.05, 0.15, 0.25, 0.40, 0.70]) / (
        1.0 - np.array([0.05, 0.15, 0.25, 0.40, 0.70])))
    losses = [float(focal_bce(np.array([l]), target, alpha=0.5, gamma=0.0, soft=True))
              for l in logits]
    assert int(np.argmin(losses)) == 2, f"optimum not at p=target: {losses}"


def test_hard_labels_are_untouched_when_soft_is_off():
    """Back-compatibility: the deployed path must not move by a single ulp."""
    logits = np.array([[-2.0, 0.3, 1.7, 0.9]])
    labels = np.array([[0.0, 1.0, -1.0, 0.0]])          # includes an ignore
    before = float(focal_bce(logits, labels, alpha=0.25))
    after = float(focal_bce(logits, labels, alpha=0.25, soft=False, weights=None))
    assert before == after


def test_soft_target_still_honours_the_ignore_code():
    """A negative target means 'excluded', so a caller can mix the two conventions."""
    logits = np.array([[0.5, 0.5]])
    kept = float(focal_bce(logits, np.array([[1.0, -1.0]]), alpha=0.5, gamma=0.0, soft=True))
    alone = float(focal_bce(np.array([[0.5]]), np.array([[1.0]]), alpha=0.5, gamma=0.0, soft=True))
    assert kept == pytest.approx(alone)


# ----------------------------------------------------------------- per-example weights
def test_weights_rescale_a_candidates_contribution():
    logits = np.array([[0.4, 0.4]])
    labels = np.array([[1.0, 0.0]])
    w = np.array([[3.0, 1.0]])
    weighted = float(focal_bce(logits, labels, alpha=0.5, weights=w))
    # weighted mean, not a plain mean: the positive counts three times on both sides
    pos = float(focal_bce(np.array([[0.4]]), np.array([[1.0]]), alpha=0.5))
    neg = float(focal_bce(np.array([[0.4]]), np.array([[0.0]]), alpha=0.5))
    assert weighted == pytest.approx((3.0 * pos + 1.0 * neg) / 4.0)


def test_uniform_weights_change_nothing():
    logits = np.array([[-1.0, 0.2, 2.0]])
    labels = np.array([[0.0, 1.0, 0.0]])
    assert float(focal_bce(logits, labels, alpha=0.25, weights=np.ones((1, 3)))) == pytest.approx(
        float(focal_bce(logits, labels, alpha=0.25)))


# ----------------------------------------------------------------- per-lesion weights
def test_positives_in_a_set_share_one_lesions_worth_of_weight():
    """Four duplicate hits on one lesion must total the same as one lone hit elsewhere."""
    labels = np.array([[1.0, 1.0, 1.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0, 0.0]])
    mask = np.ones_like(labels)
    w = per_lesion_weights(labels, mask)
    assert w[0, :4].sum() == pytest.approx(1.0)
    assert w[1, 0] == pytest.approx(1.0)


def test_negatives_keep_unit_weight():
    labels = np.array([[1.0, 1.0, 0.0, 0.0]])
    w = per_lesion_weights(labels, np.ones_like(labels))
    assert w[0, 2] == pytest.approx(1.0) and w[0, 3] == pytest.approx(1.0)


def test_padded_entries_get_no_weight():
    labels = np.array([[1.0, 1.0, 0.0, -1.0]])
    mask = np.array([[1.0, 1.0, 1.0, 0.0]])
    w = per_lesion_weights(labels, mask)
    assert w[0, 3] == 0.0
    assert w[0, 0] == pytest.approx(0.5) and w[0, 1] == pytest.approx(0.5)


def test_a_set_with_no_positive_is_all_unit_weight():
    """The all-negative calibration anchors must not be silently zeroed."""
    labels = np.array([[0.0, 0.0, 0.0]])
    w = per_lesion_weights(labels, np.ones_like(labels))
    assert w == pytest.approx(np.ones((1, 3)))
