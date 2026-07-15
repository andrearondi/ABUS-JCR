"""Tube -> 3D box, iso -> official; and the detector-independent GT geometry gate.

A reconstructed candidate lives in iso-voxel storage order, is mapped back to native
voxel indices by the recorded inverse affine (``geometry.iso_storage_to_native_storage``),
then to the official centre+extent scoring box (``geometry.storage_box_to_official``) —
exactly the path LOCAL-VALIDATED in the spec (IoU 0.92–0.97 on Val).

``gt_reconstruction_consistency`` runs that same path on the GT mask (per-slice union
boxes from ``slice_labels.boxes_for_slice``, one iso tube), isolating a linking/coord
bug from detector error. Its tolerance is the Phase-1 measured per-case fidelity, NOT
≈1.0 (the resampling boundary costs a few IoU points — a true quantization effect).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import conventions as C
from ..geometry import (
    BoxStorage,
    OfficialBox,
    iso_storage_to_native_storage,
    storage_box_to_official,
)
from ..slice_labels import boxes_for_slice
from .tubes import Tube


def tube_to_iso_storage_box(tube: Tube) -> BoxStorage:
    """Union the tube's per-slice half-open boxes into one inclusive iso-voxel box.

    ``d0`` (row = y): ``[round(min y1), round(max y2) - 1]``;
    ``d1`` (col = x): ``[round(min x1), round(max x2) - 1]``;
    ``d2`` (slice)  : ``[min slice_z, max slice_z]`` (inclusive z-span).
    Returns ``BoxStorage = (min_d0, min_d1, min_d2, max_d0, max_d1, max_d2)``.
    """
    if not tube:
        raise ValueError("tube_to_iso_storage_box: empty tube")
    zs = [z for z, _, _ in tube]
    x1 = min(b[0] for _, b, _ in tube)
    y1 = min(b[1] for _, b, _ in tube)
    x2 = max(b[2] for _, b, _ in tube)
    y2 = max(b[3] for _, b, _ in tube)
    min_d0 = int(round(y1))
    max_d0 = int(round(y2)) - 1
    min_d1 = int(round(x1))
    max_d1 = int(round(x2)) - 1
    min_d2 = int(min(zs))
    max_d2 = int(max(zs))
    return (min_d0, min_d1, min_d2, max_d0, max_d1, max_d2)


def iso_extents_of_tube(tube: Tube) -> Tuple[float, float, float]:
    """Iso-voxel full extents ``(ext_d0, ext_d1, ext_d2) = max - min`` per axis."""
    b = tube_to_iso_storage_box(tube)
    return (float(b[3] - b[0]), float(b[4] - b[1]), float(b[5] - b[2]))


def iso_centre_of_tube(tube: Tube) -> Tuple[float, float, float]:
    """Iso-voxel centre ``(cen_d0, cen_d1, cen_d2) = (min + max) / 2`` — Phase-4 crop centre."""
    b = tube_to_iso_storage_box(tube)
    return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)


def iso_tube_to_official(tube: Tube, meta: dict) -> OfficialBox:
    """``tube_to_iso_storage_box`` -> ``iso_storage_to_native_storage`` -> ``storage_box_to_official``."""
    box_iso = tube_to_iso_storage_box(tube)
    box_native = iso_storage_to_native_storage(box_iso, meta)
    return storage_box_to_official(box_native)


def _mask_to_gt_tube(mask_iso: np.ndarray) -> Tube:
    """Build ONE synthetic tube from every per-slice GT lesion box of the iso mask.

    Uses ``slice_labels.boxes_for_slice`` (the frozen mask->box rule) on each
    ``SLICE_AXIS`` slice, converts inclusive ``(r0, c0, r1, c1)`` to the half-open
    schema box ``(c0, r0, c1+1, r1+1)`` (``x = d1``, ``y = d0``), score = 1.0. The
    union over all boxes (multiple foci or slices) reconstructs the whole-mask hull
    that the official single GT box also encloses.
    """
    mask_iso = np.asarray(mask_iso)
    n = mask_iso.shape[C.SLICE_AXIS]
    tube: Tube = []
    for z in range(n):
        sl = np.take(mask_iso, z, axis=C.SLICE_AXIS)
        for (r0, c0, r1, c1) in boxes_for_slice(sl):
            tube.append((int(z), (float(c0), float(r0), float(c1 + 1), float(r1 + 1)), 1.0))
    return tube


def gt_reconstruction_consistency(mask_iso: np.ndarray, gt_official: OfficialBox, meta: dict) -> float:
    """GEOMETRY GATE (detector-independent): recon-through-iso IoU vs the official GT box.

    Builds one iso tube from the mask's per-slice union boxes, maps it to official
    space, and returns ``geometry.iou_official`` against ``gt_official``. Isolates a
    linking/coord bug (would tank the IoU) from detector error. Raises if the mask is
    empty (no GT box to reconstruct).
    """
    from ..geometry import iou_official

    tube = _mask_to_gt_tube(mask_iso)
    if not tube:
        raise ValueError("gt_reconstruction_consistency: mask has no set voxels")
    recon_official = iso_tube_to_official(tube, meta)
    return iou_official(recon_official, gt_official)
