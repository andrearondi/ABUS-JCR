"""ABUS-physics augmentation policy (Inv. 13), written now, exercised in Phase 2.

The DEPTH/BEAM axis is special (skin at top, shadows downward) => NO flip along it, NO
large rotation, NO mosaic/mixup. The mirror flip along the LATERAL axis is allowed and must
be applied identically across the C channel-slices or the stack desynchronises.

**Axis roles corrected 2026-08-08.** This file previously said "depth axis d0 / lateral d1".
That is backwards: **`d1` is depth/beam and `d0` is lateral**, measured four independent ways
on 129/130 volumes (`results/AXIS_CHECK.md`). The flip test below asserted `d1` and *called it
lateral*, so it would have PASSED the Inv.-13-violating implementation — the same defect
already found and fixed in `test_rescore_crop_aug.py`. It now asserts `d0` and explicitly
rejects the `d1` flip.
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


def test_hflip_is_lateral_d0_and_channel_consistent():
    """The default flip must mirror d0 (MEASURED lateral), never d1 (measured depth/beam)."""
    rng = np.random.default_rng(0)
    stack = rng.random((C.C_CHANNELS, 6, 8), dtype=np.float64).astype(np.float32)
    flipped = hflip_stack(stack)
    for c in range(C.C_CHANNELS):
        # d0 = row axis = the measured LATERAL axis
        np.testing.assert_array_equal(flipped[c], stack[c][::-1, :])
        # and NOT d1 — a depth flip is acoustically impossible (Inv. 13). This assertion is
        # what makes the test able to fail if the defect is ever re-introduced.
        assert not np.array_equal(flipped[c], stack[c][:, ::-1])
    # identical transform across channels => involution restores the original
    np.testing.assert_array_equal(hflip_stack(flipped), stack)
