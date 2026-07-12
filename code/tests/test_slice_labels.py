"""2D GT boxes derived from the MASK per slice — never the 3D-box projection.

Phase-2-UPDATE contract (Inv. 11 amended): one box per **lesion**, i.e. mask
components are proximity-clustered — fragments of one lesion split by speckle /
shadow (gap <= DET_LABEL_MERGE_GAP) are unioned into ONE tight box, while
genuinely separate foci (gap > DET_LABEL_MERGE_GAP) stay as distinct boxes. The
Inv.-11 asymmetry still holds: the set of slices carrying a box equals the set of
non-empty-mask slices exactly (a 3D-box projection would box the whole z-span).
"""

import numpy as np

from abus_jcr import conventions as C
from abus_jcr.slice_labels import boxes_for_slice, build_slice_labels


def test_near_fragments_merge_into_one_tight_box():
    # Two blobs 2 empty cols apart (speckle-split fragments of one lesion).
    sl = np.zeros((20, 20), dtype=np.uint8)
    sl[2:5, 2:5] = 1          # rows 2-4, cols 2-4
    sl[2:5, 7:10] = 1         # rows 2-4, cols 7-9  (col gap = 7-4-1 = 2)
    boxes = boxes_for_slice(sl, merge_gap=8)
    assert len(boxes) == 1
    assert boxes[0] == (2, 2, 4, 9)   # inclusive union hull (r0,c0,r1,c1)


def test_far_foci_stay_separate():
    # Two blobs 15 empty cols apart (distinct lesions) -> NOT merged.
    sl = np.zeros((20, 30), dtype=np.uint8)
    sl[2:5, 2:5] = 1          # rows 2-4, cols 2-4
    sl[2:5, 20:23] = 1        # rows 2-4, cols 20-22 (col gap = 20-4-1 = 15 > 8)
    boxes = boxes_for_slice(sl, merge_gap=8)
    assert len(boxes) == 2
    assert (2, 2, 4, 4) in boxes
    assert (2, 20, 4, 22) in boxes


def test_diagonal_touch_is_one_component():
    # 8-connectivity: diagonally-adjacent pixels are the same component.
    sl = np.zeros((10, 10), dtype=np.uint8)
    sl[2, 2] = 1
    sl[3, 3] = 1              # touches (2,2) only diagonally
    boxes = boxes_for_slice(sl, merge_gap=0)   # gap 0 => rely on 8-conn labelling
    assert len(boxes) == 1
    assert boxes[0] == (2, 2, 3, 3)


def test_empty_slice_yields_no_boxes():
    assert boxes_for_slice(np.zeros((8, 8), dtype=np.uint8), merge_gap=8) == []


def test_merge_gap_zero_recovers_per_component():
    # gap=0 with orthogonally-separated blobs => the old per-component behaviour.
    sl = np.zeros((20, 20), dtype=np.uint8)
    sl[2:5, 2:5] = 1
    sl[2:5, 7:10] = 1         # 2-col gap, but merge_gap=0 keeps them apart
    boxes = boxes_for_slice(sl, merge_gap=0)
    assert len(boxes) == 2


def test_build_slice_labels_box_set_equals_mask_set_and_contiguous():
    # SLICE_AXIS is d2. Lesion occupies z = 5..9 (contiguous), empty elsewhere.
    mask = np.zeros((30, 30, 20), dtype=np.uint8)
    for z in range(5, 10):
        mask[8:13, 6:11, z] = 1
    rows = build_slice_labels(volume_id=42, mask_iso=mask, merge_gap=8)
    box_slices = sorted({r["slice_z"] for r in rows})
    mask_slices = sorted(z for z in range(mask.shape[C.SLICE_AXIS]) if mask[:, :, z].any())
    assert box_slices == mask_slices == [5, 6, 7, 8, 9]
    assert box_slices == list(range(min(box_slices), max(box_slices) + 1))
    # single-lesion slice -> exactly one box per slice
    per_z = {}
    for r in rows:
        per_z.setdefault(r["slice_z"], []).append(r)
        assert r["volume_id"] == 42 and r["r0"] <= r["r1"] and r["c0"] <= r["c1"]
    assert all(len(per_z[z]) == 1 for z in (5, 6, 7, 8, 9))


def test_build_slice_labels_multifocal_slice_keeps_two_boxes():
    # A slice with two FAR-apart lesions keeps two boxes (multifocal safety).
    mask = np.zeros((40, 40, 6), dtype=np.uint8)
    mask[5:9, 5:9, 3] = 1          # focus A
    mask[5:9, 30:34, 3] = 1        # focus B (col gap 30-8-1 = 21 > 8)
    rows = build_slice_labels(volume_id=93, mask_iso=mask, merge_gap=8)
    z3 = [r for r in rows if r["slice_z"] == 3]
    assert len(z3) == 2
