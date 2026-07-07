"""ABUS-physics augmentation policy (Inv. 13), written now, exercised in Phase 2.

Depth axis d0 is special (skin at top, shadows downward) => NO vertical flip, NO
large rotation, NO mosaic/mixup. Horizontal flip (lateral d1) is allowed and must
be applied identically across the C channel-slices or the stack desynchronises.
"""

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.augment import TRAIN_AUGMENT, hflip_stack


def test_policy_forbidden_ops_off():
    assert TRAIN_AUGMENT["vertical_flip"] is False
    assert TRAIN_AUGMENT["mosaic"] is False
    assert TRAIN_AUGMENT["mixup"] is False
    assert TRAIN_AUGMENT["large_rotation"] is False
    assert TRAIN_AUGMENT["tta"] is False


def test_policy_allowed_ops_on():
    assert TRAIN_AUGMENT["horizontal_flip"] is True
    assert TRAIN_AUGMENT["intensity_jitter"] is True


def test_hflip_is_lateral_and_channel_consistent():
    rng = np.random.default_rng(0)
    stack = rng.random((C.C_CHANNELS, 6, 8), dtype=np.float64).astype(np.float32)
    flipped = hflip_stack(stack)
    # flips along d1 (col = last axis), i.e. lateral, not depth
    for c in range(C.C_CHANNELS):
        np.testing.assert_array_equal(flipped[c], stack[c][:, ::-1])
    # identical transform across channels => involution restores the original
    np.testing.assert_array_equal(hflip_stack(flipped), stack)
