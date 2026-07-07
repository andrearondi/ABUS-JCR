"""Box representations and the conversions between them.

Two explicit representations; conversion functions are the *only* place the
axis order or parameterization changes, so a coordinate bug has exactly one
place to hide.

- ``BoxStorage`` = ``(min_d0, min_d1, min_d2, max_d0, max_d1, max_d2)`` — int,
  **inclusive** max, voxel, storage order. Internal representation.
- ``OfficialBox`` = ``(coordX, coordY, coordZ, x_length, y_length, z_length)``
  — float, centre + **full** extent, ITK order, native voxel. Scoring space.

``mask_to_official_box`` is the recorded ``mask -> box`` transform that Phase 3
reuses to reconstruct GT boxes for the consistency check.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .conventions import PERM_STORAGE_TO_ITK
from .eval.froc import iou_3d

BoxStorage = Tuple[int, int, int, int, int, int]
OfficialBox = Tuple[float, float, float, float, float, float]


def mask_to_box_storage(mask: np.ndarray) -> BoxStorage:
    """Tight inclusive hull of all set voxels, in storage order.

    Whole-mask union: every tumour voxel goes into one box, matching the
    one-row-per-case GT even when the mask has multiple components. Raises on
    an empty mask.
    """
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        raise ValueError("mask_to_box_storage: mask has no set voxels")
    mn = idx.min(axis=0)
    mx = idx.max(axis=0)
    return (int(mn[0]), int(mn[1]), int(mn[2]), int(mx[0]), int(mx[1]), int(mx[2]))


def storage_box_to_official(b: BoxStorage) -> OfficialBox:
    """Storage inclusive-min/max box -> official centre + full-extent box.

    Permute storage (d0, d1, d2) to ITK (x, y, z) via PERM_STORAGE_TO_ITK =
    (2, 1, 0); centre = (min + max) / 2; length = max - min (full extent, NOT
    max - min + 1 — the inclusive voxel *count* is deliberately unused because
    ``iou_3d`` consumes continuous lengths).
    """
    mn = (b[0], b[1], b[2])
    mx = (b[3], b[4], b[5])
    p = PERM_STORAGE_TO_ITK
    mn_itk = (mn[p[0]], mn[p[1]], mn[p[2]])
    mx_itk = (mx[p[0]], mx[p[1]], mx[p[2]])
    coord = tuple((mn_itk[i] + mx_itk[i]) / 2.0 for i in range(3))
    length = tuple(float(mx_itk[i] - mn_itk[i]) for i in range(3))
    return (coord[0], coord[1], coord[2], length[0], length[1], length[2])


def official_box_to_storage(o: OfficialBox) -> BoxStorage:
    """Inverse of :func:`storage_box_to_official` (reporting-time mapping).

    Recovers ITK inclusive min/max from centre + full extent, then applies the
    self-inverse permutation back to storage order. Endpoints are integers for
    any box that originated from a voxel mask.
    """
    coord = (o[0], o[1], o[2])
    length = (o[3], o[4], o[5])
    mn_itk = tuple(coord[i] - length[i] / 2.0 for i in range(3))
    mx_itk = tuple(coord[i] + length[i] / 2.0 for i in range(3))
    p = PERM_STORAGE_TO_ITK  # self-inverse: ITK -> storage uses the same tuple
    mn_st = tuple(mn_itk[p[i]] for i in range(3))
    mx_st = tuple(mx_itk[p[i]] for i in range(3))
    return (
        int(round(mn_st[0])), int(round(mn_st[1])), int(round(mn_st[2])),
        int(round(mx_st[0])), int(round(mx_st[1])), int(round(mx_st[2])),
    )


def mask_to_official_box(mask: np.ndarray) -> OfficialBox:
    """The recorded ``mask -> box`` transform: composition of the two above."""
    return storage_box_to_official(mask_to_box_storage(mask))


def iou_official(a: OfficialBox, b: OfficialBox) -> float:
    """3D IoU in official space — a thin delegate to the vendored ``iou_3d`` so
    Phase 3's candidate-labeling IoU is byte-identical to the scoring IoU."""
    return iou_3d(a, b)
