"""Per-slice 2D GT boxes derived from the isotropic MASK (Inv. 11).

Boxes are the tight per-slice hull of each 2D connected mask component, **never**
the projection of the 3D box onto slices. The deliberate consequence: the set of
slices with a box equals the set of non-empty-mask slices (a 3D-box projection
would box every slice across the full z-span, inventing lesion presence on empty
slices). The IoU ignore-band (Inv. 11) is applied later at 3D-candidate labeling
(Phase 3), not here.

Coordinate convention (pinned): boxes are in **iso-voxel ``(d0, d1)``**,
inclusive integer min/max, on the ``(row=d0, col=d1)`` frame indexed by
``SLICE_AXIS = d2``. The image ``(x, y)`` mapping is ``x = d1 = col``,
``y = d0 = row`` (torchvision convention); Phase 2's dataloader converts
inclusive -> half-open pixels.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from . import conventions as C

Box2D = Tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive, iso-voxel

# 4-connectivity (cross) structuring element for 2D component labeling.
_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def boxes_for_slice(mask_slice_2d: np.ndarray) -> List[Box2D]:
    """Inclusive integer boxes of each 4-connected component of ``mask > 0``.

    One box per component on the ``(d0=row, d1=col)`` frame. ``MIN_2D_BOX_AREA``
    (default 0) keeps every non-empty component, so the box set equals the
    non-empty-mask slice set exactly.
    """
    labeled, n = ndimage.label(np.asarray(mask_slice_2d) > 0, structure=_STRUCT_4)
    boxes: List[Box2D] = []
    if n == 0:
        return boxes
    for comp in range(1, n + 1):
        idx = np.argwhere(labeled == comp)
        if idx.shape[0] < C.MIN_2D_BOX_AREA:
            continue
        r0, c0 = idx.min(axis=0)
        r1, c1 = idx.max(axis=0)
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def build_slice_labels(volume_id: int, mask_iso: np.ndarray) -> List[Dict]:
    """Iterate ``z`` over ``SLICE_AXIS`` and emit one row per 2D component.

    Slices with an empty mask emit **no row** (they are background/negative
    slices). Returned rows have keys ``volume_id, slice_z, r0, c0, r1, c1,
    component_id`` — ready to freeze to Parquet/CSV.
    """
    mask_iso = np.asarray(mask_iso)
    if mask_iso.ndim != 3:
        raise ValueError(f"mask_iso must be 3D, got shape {mask_iso.shape}")
    n_slices = mask_iso.shape[C.SLICE_AXIS]

    rows: List[Dict] = []
    for z in range(n_slices):
        # SLICE_AXIS == 2 for this dataset; index the (d0, d1) frame.
        sl = np.take(mask_iso, z, axis=C.SLICE_AXIS)
        boxes = boxes_for_slice(sl)
        for comp_id, (r0, c0, r1, c1) in enumerate(boxes):
            rows.append({
                "volume_id": int(volume_id),
                "slice_z": int(z),
                "r0": r0, "c0": c0, "r1": r1, "c1": c1,
                "component_id": comp_id,
            })
    return rows
