"""iso -> native -> official round-trip fidelity (the Phase-3-critical check).

Part A (pure unit): the recorded inverse affine maps a known iso box back to the
expected native voxel box.

Part B (data, Val): for every available case, the GT mask box round-tripped
through iso space and mapped back scores IoU >= RESAMPLE_IOU_FLOOR against the
native official box. The floor is a FROC-hit SAFETY MARGIN (0.50, well above the
0.3 hit threshold), not a fidelity target — the small-lesion quantization tail is
characterised (not asserted) by scripts/phase1_resample_fidelity.py. This test is
the regression tripwire: a coordinate/affine bug tanks it far below 0.50.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.geometry import (
    mask_to_box_storage,
    mask_to_official_box,
    storage_box_to_official,
    iou_official,
    iso_storage_to_native_storage,
)
from abus_jcr.preprocess import resample_case, zoom_factors


# ---- Part A: pure affine unit test ----------------------------------------

def test_iso_to_native_affine_known_values():
    f = zoom_factors()
    meta = {"zoom_factors": list(f), "native_shape": [500, 400, 300]}
    # native = (iso + 0.5)/f - 0.5, applied to inclusive min and max
    box_iso = (10, 20, 30, 40, 50, 60)
    got = iso_storage_to_native_storage(box_iso, meta)
    exp = tuple(
        int(round((box_iso[i] + 0.5) / f[i % 3] - 0.5))
        for i in (0, 1, 2, 3, 4, 5)
    )
    # index i uses axis i%3 (min triplet then max triplet)
    exp = (
        int(round((box_iso[0] + 0.5) / f[0] - 0.5)),
        int(round((box_iso[1] + 0.5) / f[1] - 0.5)),
        int(round((box_iso[2] + 0.5) / f[2] - 0.5)),
        int(round((box_iso[3] + 0.5) / f[0] - 0.5)),
        int(round((box_iso[4] + 0.5) / f[1] - 0.5)),
        int(round((box_iso[5] + 0.5) / f[2] - 0.5)),
    )
    assert got == exp


def test_iso_to_native_recovers_original_native_box_within_quantization():
    # A native box, pushed to iso via the forward map, then pulled back, lands
    # within the theoretical quantization bound: nearest-iso-index rounding costs
    # up to half an iso-voxel, which in native voxels is 0.5/f (large on the
    # heavily-downsampled depth axis f=0.1825 at 0.4 mm, tiny on the sweep axis).
    f = zoom_factors()
    meta = {"zoom_factors": list(f)}
    native = (100, 80, 40, 260, 300, 200)
    box_iso = tuple(
        int(round((native[i] + 0.5) * f[i % 3] - 0.5)) for i in range(6)
    )
    back = iso_storage_to_native_storage(box_iso, meta)
    for i in range(6):
        bound = 0.5 / f[i % 3] + 1.0  # half-iso-voxel + integer rounding slack
        assert abs(back[i] - native[i]) <= bound


# ---- Part B: end-to-end resampling fidelity on Val ------------------------

def _split_root() -> Path | None:
    # ABUS_SPLIT_ROOT (server: Train/Validation) overrides; else local Val default.
    env = os.environ.get("ABUS_SPLIT_ROOT")
    if env:
        return Path(env)
    default = Path("/Users/andrearondi/Desktop/KTH/Tesi/Dataset/Validation")
    return default if default.exists() else None


def _case_ids() -> list[int]:
    """Cases to exercise: ALL discovered when ABUS_SPLIT_ROOT is set (server run
    covers every Train case, exit-check #2); else the three local Val cases."""
    root = _split_root()
    if root is None or not root.exists():
        return []
    if os.environ.get("ABUS_SPLIT_ROOT"):
        from abus_jcr.io_nrrd import discover_cases
        return sorted(discover_cases(root))
    return [100, 104, 116]


@pytest.mark.parametrize("case_id", _case_ids() or [pytest.param(-1, marks=pytest.mark.skip(reason="no split available"))])
def test_iso_native_official_iou_above_floor(case_id):
    root = _split_root()
    if root is None or not root.exists():
        pytest.skip("split not available")
    from abus_jcr.io_nrrd import discover_cases, load_array

    cases = discover_cases(root)
    if case_id not in cases:
        pytest.skip(f"case {case_id} not in split")

    vol, _ = load_array(cases[case_id].data)
    mask, _ = load_array(cases[case_id].mask)
    mask = (np.asarray(mask) > 0).astype(np.uint8)

    # native official GT box (== bbx GT; Phase-0 residual 0)
    official_native = mask_to_official_box(mask)

    # resample, take the iso mask box, map it back to native, score
    _, mask_iso, meta = resample_case(vol.astype(np.uint8), mask)
    box_iso = mask_to_box_storage(mask_iso)
    box_native = iso_storage_to_native_storage(box_iso, meta)
    official_roundtrip = storage_box_to_official(box_native)

    iou = iou_official(official_roundtrip, official_native)
    assert iou >= C.RESAMPLE_IOU_FLOOR, f"case {case_id}: IoU {iou:.3f} < floor"
