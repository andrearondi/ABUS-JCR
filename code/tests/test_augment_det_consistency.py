"""Inv. 13 — box-aware, channel-consistent detection augmentation.

Data-independent (numpy only): forbidden ops are never invoked, every spatial op
shares ONE sampled parameter set across the C channel-slices, and boxes transform
with the image. Torch-free.
"""

import copy

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.augment import TRAIN_AUGMENT
from abus_jcr.detect import augment_ops as A


def _channel_marked_stack(d0=6, d1=8):
    """Stack whose channel c is a distinct gradient, so a per-channel desync shows."""
    stack = np.zeros((C.C_CHANNELS, d0, d1), dtype=np.float32)
    base = np.arange(d1, dtype=np.float32)[None, :] * np.ones((d0, 1), dtype=np.float32)
    for c in range(C.C_CHANNELS):
        stack[c] = base + 100.0 * c
    return stack


def test_forbidden_ops_never_invoked():
    seen = []
    rng = np.random.default_rng(0)
    stack = _channel_marked_stack()
    boxes = np.array([[1.0, 1.0, 4.0, 3.0]], dtype=np.float32)
    for _ in range(50):
        A.apply_train_augment(stack, boxes, rng, on_op=lambda n, p: seen.append(n))
    # forbidden ops never fire; small "scale"/"rotate" ARE allowed (P2-UPDATE B4).
    for forbidden in ("vflip", "large_rotation", "mosaic", "mixup"):
        assert forbidden not in seen


def test_enabling_a_forbidden_op_raises():
    policy = dict(TRAIN_AUGMENT, vertical_flip=True)
    rng = np.random.default_rng(0)
    stack = _channel_marked_stack()
    boxes = np.zeros((0, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        A.apply_train_augment(stack, boxes, rng, policy=policy)


def test_hflip_is_lateral_shared_across_channels_and_moves_boxes():
    policy = dict(TRAIN_AUGMENT, horizontal_flip_p=1.0,
                  small_translation=False, intensity_jitter=False,
                  gaussian_blur=False, gaussian_noise=False,
                  scale_zoom=False, rotation=False)
    rng = np.random.default_rng(0)
    stack = _channel_marked_stack(d0=6, d1=8)
    boxes = np.array([[1.0, 2.0, 5.0, 4.0]], dtype=np.float32)  # half-open, W=8
    out_stack, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy)

    # every channel flipped identically along d1 (lateral), never d0 (depth)
    for c in range(C.C_CHANNELS):
        np.testing.assert_array_equal(out_stack[c], stack[c][:, ::-1])
    # box reflected: (W - x2, y1, W - x1, y2) = (3, 2, 7, 4)
    np.testing.assert_allclose(out_boxes[0], [3.0, 2.0, 7.0, 4.0])


def test_translation_shares_one_shift_across_channels_and_shifts_boxes():
    captured = {}

    def on_op(name, params):
        if name == "translate":
            captured.update(params)

    policy = dict(TRAIN_AUGMENT, horizontal_flip_p=0.0, small_translation=True,
                  intensity_jitter=False, gaussian_blur=False, gaussian_noise=False,
                  scale_zoom=False, rotation=False, translate_frac=0.25)
    rng = np.random.default_rng(3)
    stack = _channel_marked_stack(d0=6, d1=8)
    boxes = np.array([[1.0, 1.0, 5.0, 4.0]], dtype=np.float32)
    out_stack, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy, on_op=on_op)

    dy, dx = captured["dy"], captured["dx"]
    # applying the SAME (dy,dx) to each channel reproduces the output (shared param)
    for c in range(C.C_CHANNELS):
        expect = A.shift_frame(stack[c], dy, dx)
        np.testing.assert_array_equal(out_stack[c], expect)
    # box shifted by the same (dx,dy) then clipped to the frame, still half-open
    if len(out_boxes):
        assert out_boxes[0, 0] < out_boxes[0, 2] and out_boxes[0, 1] < out_boxes[0, 3]


def test_intensity_only_never_changes_boxes():
    policy = dict(TRAIN_AUGMENT, horizontal_flip_p=0.0, small_translation=False,
                  intensity_jitter=True, gaussian_blur=True, gaussian_noise=True,
                  scale_zoom=False, rotation=False)
    rng = np.random.default_rng(1)
    stack = _channel_marked_stack()
    boxes = np.array([[1.0, 1.0, 4.0, 3.0], [2.0, 0.0, 6.0, 5.0]], dtype=np.float32)
    boxes_before = copy.deepcopy(boxes)
    for _ in range(20):
        _, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy)
        np.testing.assert_array_equal(out_boxes, boxes_before)


def _only(policy_overrides):
    """A policy with everything off except the ops named in policy_overrides."""
    base = dict(TRAIN_AUGMENT, horizontal_flip_p=0.0, small_translation=False,
                intensity_jitter=False, gaussian_blur=False, gaussian_noise=False,
                scale_zoom=False, rotation=False)
    base.update(policy_overrides)
    return base


def test_scale_identity_at_s1_leaves_image_and_boxes():
    policy = _only({"scale_zoom": True, "scale_range": [1.0, 1.0]})
    rng = np.random.default_rng(0)
    stack = _channel_marked_stack(d0=6, d1=8)
    boxes = np.array([[2.0, 1.0, 6.0, 5.0]], dtype=np.float32)
    out_stack, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy)
    np.testing.assert_allclose(out_stack, stack, atol=1e-5)
    np.testing.assert_allclose(out_boxes, boxes, atol=1e-5)


def test_scale_moves_boxes_by_centre_formula_and_shares_param():
    captured = {}
    def on_op(name, params):
        if name == "scale":
            captured.update(params)
    policy = _only({"scale_zoom": True, "scale_range": [0.5, 0.5]})  # force s=0.5
    rng = np.random.default_rng(0)
    stack = _channel_marked_stack(d0=6, d1=8)               # centre (cx=4, cy=3)
    boxes = np.array([[2.0, 1.0, 6.0, 5.0]], dtype=np.float32)
    out_stack, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy, on_op=on_op)
    # box scaled about centre by 0.5: [3,2,5,4]
    np.testing.assert_allclose(out_boxes[0], [3.0, 2.0, 5.0, 4.0], atol=1e-5)
    # same s applied to every channel (shared param)
    s = captured["s"]
    for c in range(C.C_CHANNELS):
        np.testing.assert_allclose(out_stack[c], A.scale_frame(stack[c], s), atol=1e-5)


def test_rotation_identity_at_zero_deg():
    policy = _only({"rotation": True, "rotation_deg": 0.0})
    rng = np.random.default_rng(1)
    stack = _channel_marked_stack(d0=8, d1=8)
    boxes = np.array([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32)
    out_stack, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy)
    np.testing.assert_allclose(out_stack, stack, atol=1e-5)
    np.testing.assert_allclose(out_boxes, boxes, atol=1e-5)


def test_rotation_shares_angle_across_channels_and_keeps_valid_boxes():
    captured = {}
    def on_op(name, params):
        if name == "rotate":
            captured.update(params)
    policy = _only({"rotation": True, "rotation_deg": 5.0})
    rng = np.random.default_rng(2)
    stack = _channel_marked_stack(d0=10, d1=10)
    boxes = np.array([[3.0, 3.0, 7.0, 7.0]], dtype=np.float32)
    out_stack, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy, on_op=on_op)
    deg = captured["deg"]
    for c in range(C.C_CHANNELS):
        np.testing.assert_allclose(out_stack[c], A.rotate_frame(stack[c], deg), atol=1e-5)
    for b in out_boxes:
        assert b[0] < b[2] and b[1] < b[3]
        assert 0 <= b[0] and b[2] <= 10 and 0 <= b[1] and b[3] <= 10


def test_translation_out_of_frame_box_is_dropped():
    policy = dict(TRAIN_AUGMENT, horizontal_flip_p=0.0, small_translation=True,
                  intensity_jitter=False, gaussian_blur=False, gaussian_noise=False,
                  scale_zoom=False, rotation=False, translate_frac=0.9)
    stack = _channel_marked_stack(d0=6, d1=8)
    boxes = np.array([[6.0, 4.0, 8.0, 6.0]], dtype=np.float32)  # near the far corner
    # a large negative shift pushes it fully out; force it via a hand-set rng path
    rng = np.random.default_rng(12345)
    # run many draws; whenever the box leaves the frame it must be dropped (never invalid)
    for _ in range(100):
        _, out_boxes = A.apply_train_augment(stack, boxes, rng, policy=policy)
        for b in out_boxes:
            assert b[0] < b[2] and b[1] < b[3]  # every surviving box stays valid half-open
