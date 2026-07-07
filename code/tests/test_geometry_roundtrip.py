"""Geometry conversions invert exactly, and IoU is invariant under per-axis
(anisotropic) rescaling.

The IoU-invariance fact is what legitimises scoring isotropic-space candidates
after mapping them back to native voxel space (Inv. 6).
"""

import numpy as np
import pytest

from abus_jcr.geometry import (
    mask_to_box_storage,
    storage_box_to_official,
    official_box_to_storage,
    mask_to_official_box,
    iou_official,
)
from abus_jcr.eval.froc import iou_3d


def test_mask_to_box_storage_inclusive_minmax():
    # a 3x3x3 mask with two set voxels; box is the tight inclusive hull
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2, 3, 4] = 1
    mask[6, 7, 8] = 1
    assert mask_to_box_storage(mask) == (2, 3, 4, 6, 7, 8)


def test_mask_to_box_storage_raises_on_empty():
    with pytest.raises(ValueError):
        mask_to_box_storage(np.zeros((4, 4, 4), dtype=np.uint8))


def test_storage_box_to_official_case100_numbers():
    # from DATA_INFO.md reference case 100 (verified 0-residual round-trip)
    box = (163, 58, 153, 465, 426, 225)  # storage min/max inclusive
    off = storage_box_to_official(box)
    coordX, coordY, coordZ, lx, ly, lz = off
    assert (coordX, coordY, coordZ) == (189.0, 242.0, 314.0)
    assert (lx, ly, lz) == (72.0, 368.0, 302.0)


def test_storage_official_roundtrip_inverts():
    box = (163, 58, 153, 465, 426, 225)
    assert official_box_to_storage(storage_box_to_official(box)) == box


def test_mask_to_official_box_composition():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:10, 3:8, 2:6] = 1  # inclusive max = (9, 7, 5)
    off = mask_to_official_box(mask)
    # storage box = (5,3,2, 9,7,5) -> official permute (2,1,0)
    # x from d2: min2=2 max5 -> coordX=3.5 len=3 ; y from d1: 3..7 -> 5.0,4 ; z from d0:5..9 ->7.0,4
    assert off == (3.5, 5.0, 7.0, 3.0, 4.0, 4.0)


def test_iou_official_delegates_to_vendored_iou_3d():
    a = (10.0, 10.0, 10.0, 4.0, 4.0, 4.0)
    b = (12.0, 10.0, 10.0, 4.0, 4.0, 4.0)
    assert iou_official(a, b) == iou_3d(a, b)


def test_iou_invariant_under_anisotropic_scaling():
    # scaling both boxes by per-axis (a,b,c) multiplies intersection, both
    # volumes, and union all by abc -> IoU unchanged.
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = tuple(rng.uniform(-5, 5, 3)) + tuple(rng.uniform(0.5, 6, 3))
        b = tuple(rng.uniform(-5, 5, 3)) + tuple(rng.uniform(0.5, 6, 3))
        base = iou_3d(a, b)
        sx, sy, sz = rng.uniform(0.1, 10, 3)
        a_s = (a[0] * sx, a[1] * sy, a[2] * sz, a[3] * sx, a[4] * sy, a[5] * sz)
        b_s = (b[0] * sx, b[1] * sy, b[2] * sz, b[3] * sx, b[4] * sy, b[5] * sz)
        scaled = iou_3d(a_s, b_s)
        assert scaled == pytest.approx(base, abs=1e-9)
