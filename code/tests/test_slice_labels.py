"""2D GT boxes derived from the MASK per slice — never the 3D-box projection.

This encodes the Inv.-11 asymmetry and the phase exit check: the set of slices
carrying a box equals the set of non-empty-mask slices exactly (a 3D-box
projection would wrongly box every slice in the z-span), each lesion's box
slices form a contiguous z-run equal to its z-extent, and a slice with two
disjoint blobs yields one box per component.
"""

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.slice_labels import boxes_for_slice, build_slice_labels


def test_boxes_for_slice_inclusive_and_per_component():
    sl = np.zeros((20, 20), dtype=np.uint8)
    sl[2:5, 3:6] = 1          # blob A: rows 2-4, cols 3-5
    sl[10:14, 12:15] = 1      # blob B: rows 10-13, cols 12-14 (disjoint)
    boxes = boxes_for_slice(sl)
    assert len(boxes) == 2
    assert (2, 3, 4, 5) in boxes      # inclusive min/max (r0,c0,r1,c1)
    assert (10, 12, 13, 14) in boxes


def test_empty_slice_yields_no_boxes():
    assert boxes_for_slice(np.zeros((8, 8), dtype=np.uint8)) == []


def test_build_slice_labels_box_set_equals_mask_set_and_contiguous():
    # SLICE_AXIS is d2. Lesion occupies z = 5..9 (contiguous), empty elsewhere.
    mask = np.zeros((30, 30, 20), dtype=np.uint8)
    for z in range(5, 10):
        mask[8:13, 6:11, z] = 1
    # add a second disjoint blob on a single slice z=7 -> 2 components there
    mask[20:24, 20:24, 7] = 1

    rows = build_slice_labels(volume_id=42, mask_iso=mask)
    box_slices = sorted({r["slice_z"] for r in rows})
    mask_slices = sorted(
        z for z in range(mask.shape[C.SLICE_AXIS]) if mask[:, :, z].any()
    )
    # box-set == mask-set, exactly
    assert box_slices == mask_slices == [5, 6, 7, 8, 9]

    # contiguous z-run equal to the lesion z-extent
    assert box_slices == list(range(min(box_slices), max(box_slices) + 1))

    # z=7 carries two components (two rows); others carry one
    per_z = {}
    for r in rows:
        per_z.setdefault(r["slice_z"], []).append(r)
        assert r["volume_id"] == 42
        assert r["r0"] <= r["r1"] and r["c0"] <= r["c1"]
    assert len(per_z[7]) == 2
    assert all(len(per_z[z]) == 1 for z in (5, 6, 8, 9))
