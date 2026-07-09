"""Degenerate-box filtering in inference (torch-free).

torchvision postprocessing at a permissive score threshold can emit zero-area
boxes clipped to the image border (``x2==x1`` or ``y2==y1``). Those are not
candidates and must be dropped before the strict schema validation, not crash it.
"""

import numpy as np

from abus_jcr.detect import infer


def test_drops_zero_width_height_and_inverted_boxes():
    boxes = np.array([
        [1.0, 1.0, 5.0, 4.0],   # valid -> keep
        [2.0, 2.0, 2.0, 6.0],   # x2 == x1 -> drop
        [0.0, 3.0, 4.0, 3.0],   # y2 == y1 -> drop
        [3.0, 3.0, 1.0, 7.0],   # x2 <  x1 -> drop
    ], dtype=np.float32)
    scores = np.array([0.9, 0.5, 0.4, 0.3], dtype=np.float32)
    b, s = infer.drop_degenerate_boxes(boxes, scores)
    assert b.shape == (1, 4)
    np.testing.assert_array_equal(b[0], [1.0, 1.0, 5.0, 4.0])
    np.testing.assert_allclose(s, [0.9], rtol=1e-6)


def test_empty_passthrough():
    b, s = infer.drop_degenerate_boxes(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
    assert b.shape == (0, 4) and s.shape == (0,)


def test_all_valid_unchanged():
    boxes = np.array([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 5.0]], dtype=np.float32)
    scores = np.array([0.2, 0.8], dtype=np.float32)
    b, s = infer.drop_degenerate_boxes(boxes, scores)
    assert b.shape == (2, 4)
    np.testing.assert_allclose(s, scores, rtol=1e-6)
