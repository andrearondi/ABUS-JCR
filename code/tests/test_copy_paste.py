"""Shadow-aware lesion copy-paste (Inv. 13 amended, P2-UPDATE, default-OFF).

Data-independent (numpy). Pins the load-bearing property: a pasted lesion travels
WITH its posterior-shadow column and keeps its DEPTH band (never moved up/down
relative to its shadow) — the amendment's condition for permitting copy-paste. A
new box is added; channels stay consistent.
"""

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.augment import TRAIN_AUGMENT
from abus_jcr.detect.copy_paste import extract_lesion_crop, paste_lesion


def _scene():
    # lesion (value 1.0) at rows 3..6, cols 5..8; its shadow (0.5) directly below.
    stack = np.zeros((C.C_CHANNELS, 20, 20), dtype=np.float32)
    stack[:, 3:7, 5:9] = 1.0
    stack[:, 7:20, 5:9] = 0.5     # posterior shadow column
    box = np.array([5.0, 3.0, 9.0, 7.0], dtype=np.float32)  # half-open (x1,y1,x2,y2)
    return stack, box


def test_extract_captures_lesion_plus_shadow_column():
    stack, box = _scene()
    crop = extract_lesion_crop(stack, box)
    assert crop["y0"] == 3 and crop["lesion_h"] == 4 and crop["w"] == 4
    # crop spans the lesion top down to the frame bottom (lesion + shadow)
    assert crop["crop"].shape == (C.C_CHANNELS, 17, 4)


def test_paste_preserves_depth_band_and_adds_box():
    stack, box = _scene()
    crop = extract_lesion_crop(stack, box)
    boxes = np.zeros((0, 4), dtype=np.float32)
    out_stack, out_boxes = paste_lesion(stack, boxes, crop, np.random.default_rng(0), x_offset=12)

    assert len(out_boxes) == 1
    x1, y1, x2, y2 = out_boxes[0]
    # DEPTH band preserved: same y-range as the source lesion (shadow-aware).
    assert y1 == 3.0 and y2 == 7.0
    # moved laterally to the requested column.
    assert x1 == 12.0 and x2 == 16.0
    # lesion content pasted (1.0) with its shadow (0.5) below, on every channel.
    for c in range(C.C_CHANNELS):
        assert np.allclose(out_stack[c, 3:7, 12:16], 1.0)
        assert np.allclose(out_stack[c, 7:20, 12:16], 0.5)


def test_paste_never_lifts_lesion_off_its_shadow():
    # Whatever the lateral offset, the pasted box top equals the source box top.
    stack, box = _scene()
    crop = extract_lesion_crop(stack, box)
    for xo in (0, 3, 10, 16):
        _, out_boxes = paste_lesion(stack, np.zeros((0, 4), np.float32), crop,
                                    np.random.default_rng(1), x_offset=xo)
        assert out_boxes[0, 1] == crop["y0"]              # top row unchanged
        assert out_boxes[0, 3] == crop["y0"] + crop["lesion_h"]


def test_copy_paste_is_off_by_default():
    assert TRAIN_AUGMENT.get("lesion_copy_paste", False) is False
