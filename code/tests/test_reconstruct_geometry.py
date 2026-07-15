"""Tube -> iso storage box -> native -> official reconstruction (Phase 3, coord-critical).

Part A (pure): a hand tube maps to the expected inclusive iso storage box and the
iso centre/extents are right (half-open -> inclusive: d0=[round min y1, round max
y2 -1], d1=[round min x1, round max x2 -1], d2=[min z, max z]).

Part B (data, local Val): the exact Phase-3 GT reconstruction path
(``gt_reconstruction_consistency``) on real cases lands in the Phase-1-measured
fidelity band, never near zero (a coord/axis bug tanks it). Built on the fly via
``resample_case`` — no iso cache needed on the laptop. Skips if local Val absent.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.geometry import mask_to_official_box, iou_official
from abus_jcr.link.reconstruct import (
    tube_to_iso_storage_box,
    iso_centre_of_tube,
    iso_extents_of_tube,
    iso_tube_to_official,
    gt_reconstruction_consistency,
)


# ---- Part A: pure geometry -------------------------------------------------

def test_tube_to_iso_storage_box_halfopen_to_inclusive():
    # two slices; half-open boxes (x1,y1,x2,y2). Union: x1in[2,4], x2in[12,14],
    # y1in[3], y2in[13]; z in {5,7}.
    tube = [
        (5, (2.0, 3.0, 12.0, 13.0), 0.9),
        (7, (4.0, 3.0, 14.0, 13.0), 0.8),
    ]
    box = tube_to_iso_storage_box(tube)
    # d0 (row=y): [round(min y1)=3, round(max y2)-1=12]
    # d1 (col=x): [round(min x1)=2, round(max x2)-1=13]
    # d2 (slice): [5, 7]
    assert box == (3, 2, 5, 12, 13, 7)


def test_iso_centre_and_extents():
    tube = [(5, (2.0, 3.0, 12.0, 13.0), 0.9), (7, (4.0, 3.0, 14.0, 13.0), 0.8)]
    box = tube_to_iso_storage_box(tube)  # (3,2,5,12,13,7)
    cen = iso_centre_of_tube(tube)
    ext = iso_extents_of_tube(tube)
    # centre = (min+max)/2 per storage axis; extent = max-min (full extent)
    assert cen == pytest.approx(((3 + 12) / 2, (2 + 13) / 2, (5 + 7) / 2))
    assert ext == pytest.approx((12 - 3, 13 - 2, 7 - 5))


def test_iso_tube_to_official_matches_manual_composition():
    from abus_jcr.geometry import iso_storage_to_native_storage, storage_box_to_official
    from abus_jcr.preprocess import zoom_factors

    tube = [(5, (2.0, 3.0, 12.0, 13.0), 0.9), (10, (4.0, 3.0, 14.0, 13.0), 0.8)]
    meta = {"zoom_factors": list(zoom_factors())}
    got = iso_tube_to_official(tube, meta)
    expect = storage_box_to_official(
        iso_storage_to_native_storage(tube_to_iso_storage_box(tube), meta))
    assert got == pytest.approx(expect)


# ---- Part B: GT reconstruction consistency on local Val --------------------

def _val_root() -> Path | None:
    env = os.environ.get("ABUS_SPLIT_ROOT")
    if env:
        return Path(env)
    default = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")
    return default if default.exists() else None


@pytest.mark.parametrize("case_id,lo", [(100, 0.85), (104, C.RESAMPLE_IOU_FLOOR), (116, 0.85)])
def test_gt_reconstruction_consistency_local_val(case_id, lo):
    root = _val_root()
    if root is None or not root.exists():
        pytest.skip("local Validation split not available")
    from abus_jcr.io_nrrd import discover_cases, load_array
    from abus_jcr.preprocess import resample_case

    cases = discover_cases(root)
    if case_id not in cases:
        pytest.skip(f"case {case_id} not in split")

    vol, _ = load_array(cases[case_id].data)
    mask, _ = load_array(cases[case_id].mask)
    mask = (np.asarray(mask) > 0).astype(np.uint8)

    gt_official = mask_to_official_box(mask)  # native official GT (== bbx; residual 0)
    _, mask_iso, meta = resample_case(vol.astype(np.uint8), mask)

    iou = gt_reconstruction_consistency(mask_iso, gt_official, meta)
    assert iou >= lo, f"case {case_id}: recon IoU {iou:.3f} < {lo}"
    assert iou <= 1.0
