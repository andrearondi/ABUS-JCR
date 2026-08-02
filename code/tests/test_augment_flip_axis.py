"""Tests for the selectable mirror-flip axis (the RB_AUG_FLIP_AB.md experimental arm).

The load-bearing property is the FIRST test: the default must be byte-identical to what
produced every recorded detector. An A/B whose control arm has silently drifted measures
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from abus_jcr.augment import TRAIN_AUGMENT, hflip_stack
from abus_jcr.detect import augment_ops as AO


def _stack(rng, c=3, h=8, w=10):
    return rng.random((c, h, w)).astype(np.float32)


def test_default_is_still_d1_so_the_control_arm_is_unchanged():
    """The deployed default must not move — every recorded number depends on it."""
    assert TRAIN_AUGMENT["flip_stack_axis"] == 1


def test_hflip_stack_default_matches_the_pre_change_implementation():
    rng = np.random.default_rng(0)
    s = _stack(rng)
    np.testing.assert_array_equal(hflip_stack(s), s[:, :, ::-1])
    np.testing.assert_array_equal(hflip_stack(s, axis=1), s[:, :, ::-1])
    np.testing.assert_array_equal(hflip_stack(s, axis=0), s[:, ::-1, :])


def test_hflip_stack_is_its_own_inverse_on_both_axes():
    rng = np.random.default_rng(1)
    s = _stack(rng)
    for a in (0, 1):
        np.testing.assert_array_equal(hflip_stack(hflip_stack(s, axis=a), axis=a), s)


def test_hflip_stack_rejects_a_bad_axis():
    with pytest.raises(ValueError):
        hflip_stack(_stack(np.random.default_rng(2)), axis=2)


@pytest.mark.parametrize("axis", [0, 1])
def test_boxes_follow_the_image_under_either_flip(axis):
    """A bright square must still be inside its box after the flip.

    Checking the box arithmetic against the pixels — rather than against a formula — is what
    catches a mirrored image paired with unmirrored boxes, which trains the detector on
    systematically wrong targets while every unit test on the formula still passes.
    """
    h, w = 12, 16
    stack = np.zeros((3, h, w), dtype=np.float32)
    stack[:, 2:5, 3:7] = 1.0                       # square at rows 2:5, cols 3:7
    boxes = np.array([[3.0, 2.0, 7.0, 5.0]])       # (x1, y1, x2, y2)

    policy = dict(TRAIN_AUGMENT, flip_stack_axis=axis, horizontal_flip_p=1.0,
                  small_translation=False, scale_zoom=False, rotation=False,
                  intensity_jitter=False, gaussian_blur=False, gaussian_noise=False)
    out_stack, out_boxes = AO.apply_train_augment(stack, boxes, np.random.default_rng(0), policy)

    assert out_stack.sum() == pytest.approx(stack.sum())          # nothing lost
    x1, y1, x2, y2 = out_boxes[0].astype(int)
    inside = out_stack[0, y1:y2, x1:x2]
    assert inside.size == 3 * 4                                   # box kept its shape
    assert inside.min() == 1.0                                    # and it covers the square
    assert out_stack[0].sum() == inside.sum()                     # nothing bright outside it


def test_the_two_axes_produce_different_images():
    h, w = 12, 16
    stack = np.zeros((1, h, w), dtype=np.float32)
    stack[:, 1:3, 1:4] = 1.0
    boxes = np.array([[1.0, 1.0, 4.0, 3.0]])
    base = dict(TRAIN_AUGMENT, horizontal_flip_p=1.0, small_translation=False,
                scale_zoom=False, rotation=False, intensity_jitter=False,
                gaussian_blur=False, gaussian_noise=False)
    a0, b0 = AO.apply_train_augment(stack, boxes, np.random.default_rng(0),
                                    dict(base, flip_stack_axis=0))
    a1, b1 = AO.apply_train_augment(stack, boxes, np.random.default_rng(0),
                                    dict(base, flip_stack_axis=1))
    assert not np.array_equal(a0, a1)
    assert not np.array_equal(b0, b1)


def test_bad_flip_axis_raises_rather_than_silently_defaulting():
    policy = dict(TRAIN_AUGMENT, flip_stack_axis=7, horizontal_flip_p=1.0)
    with pytest.raises(ValueError):
        AO.apply_train_augment(np.zeros((1, 4, 4), np.float32), np.zeros((0, 4)),
                               np.random.default_rng(0), policy)


def test_vflip_boxes_is_the_row_mirror_of_hflip_boxes():
    boxes = np.array([[1.0, 2.0, 5.0, 7.0], [0.0, 0.0, 3.0, 3.0]])
    h = 10
    out = AO._vflip_boxes(boxes, h)
    np.testing.assert_allclose(out[:, 1], h - boxes[:, 3])
    np.testing.assert_allclose(out[:, 3], h - boxes[:, 1])
    np.testing.assert_allclose(out[:, [0, 2]], boxes[:, [0, 2]])   # x untouched
    np.testing.assert_allclose(AO._vflip_boxes(out, h), boxes)     # involution


def test_forbidden_flags_still_raise_regardless_of_flip_axis():
    """Changing the flip axis must not create a hole in the Inv.-13 guard."""
    for axis in (0, 1):
        policy = dict(TRAIN_AUGMENT, flip_stack_axis=axis, mosaic=True)
        with pytest.raises(ValueError, match="Inv. 13"):
            AO.apply_train_augment(np.zeros((1, 4, 4), np.float32), np.zeros((0, 4)),
                                   np.random.default_rng(0), policy)
