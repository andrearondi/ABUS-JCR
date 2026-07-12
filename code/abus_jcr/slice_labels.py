"""Per-slice 2D GT boxes derived from the isotropic MASK (Inv. 11, amended P2-UPDATE).

One box per **lesion**: 8-connected mask components are proximity-clustered — those
whose bounding boxes lie within ``merge_gap`` iso px are unioned into one tight box
(speckle/shadow fragments of a single lesion), while genuinely separate foci stay
distinct. This replaces the old one-box-per-connected-component behaviour (which
fragmented one tumour into many tiny boxes and starved the detector of clean
positives — see PHASE_2_UPDATE.md B1). ``merge_gap=inf`` is global union;
``merge_gap=0`` recovers per-component.

The deliberate Inv.-11 consequence is unchanged: the set of slices with a box
equals the set of non-empty-mask slices (a 3D-box projection would box every slice
across the full z-span, inventing lesion presence on empty slices). The IoU
ignore-band (Inv. 11) is applied later at 3D-candidate labeling (Phase 3), not here.

Coordinate convention (pinned): boxes are in **iso-voxel ``(d0, d1)``**, inclusive
integer min/max, on the ``(row=d0, col=d1)`` frame indexed by ``SLICE_AXIS = d2``.
The image ``(x, y)`` mapping is ``x = d1 = col``, ``y = d0 = row`` (torchvision
convention); Phase 2's dataloader converts inclusive -> half-open pixels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from . import conventions as C

Box2D = Tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive, iso-voxel

# 8-connectivity: diagonally-adjacent lesion pixels belong to the same component
# (the old 4-connectivity over-fragmented speckle-textured lesions).
_STRUCT_8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)


def _box_gap(a: Box2D, b: Box2D) -> float:
    """Euclidean gap (iso px) between two inclusive boxes; 0 if they touch/overlap.

    Per axis, the separation is the count of empty cells between the boxes (0 when
    they overlap on that axis); the gap is the hypot of the two separations.
    """
    r_sep = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    c_sep = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return math.hypot(r_sep, c_sep)


def _cluster(boxes: List[Box2D], merge_gap: float) -> List[List[int]]:
    """Union-find over boxes: same cluster iff some chain of pairwise gaps <= merge_gap."""
    n = len(boxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _box_gap(boxes[i], boxes[j]) <= merge_gap:
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def boxes_for_slice(
    mask_slice_2d: np.ndarray,
    merge_gap: float = C.DET_LABEL_MERGE_GAP,
    min_area: int = C.MIN_2D_BOX_AREA,
) -> List[Box2D]:
    """Inclusive integer boxes of each **lesion** (proximity-clustered components).

    Labels 8-connected components of ``mask > 0``, computes each component's tight
    inclusive box, agglomeratively merges components within ``merge_gap`` iso px,
    and returns one union box per cluster. Clusters whose total foreground pixel
    count is ``< min_area`` are dropped (``min_area=0`` keeps every cluster, so the
    box set still equals the non-empty-mask slice set).
    """
    labeled, n = ndimage.label(np.asarray(mask_slice_2d) > 0, structure=_STRUCT_8)
    if n == 0:
        return []
    comp_boxes: List[Box2D] = []
    comp_area: List[int] = []
    for comp in range(1, n + 1):
        idx = np.argwhere(labeled == comp)
        r0, c0 = idx.min(axis=0)
        r1, c1 = idx.max(axis=0)
        comp_boxes.append((int(r0), int(c0), int(r1), int(c1)))
        comp_area.append(int(idx.shape[0]))

    out: List[Box2D] = []
    for members in _cluster(comp_boxes, merge_gap):
        area = sum(comp_area[m] for m in members)
        if area < min_area:
            continue
        r0 = min(comp_boxes[m][0] for m in members)
        c0 = min(comp_boxes[m][1] for m in members)
        r1 = max(comp_boxes[m][2] for m in members)
        c1 = max(comp_boxes[m][3] for m in members)
        out.append((r0, c0, r1, c1))
    return out


def build_slice_labels(
    volume_id: int,
    mask_iso: np.ndarray,
    merge_gap: float = C.DET_LABEL_MERGE_GAP,
    min_area: int = C.MIN_2D_BOX_AREA,
) -> List[Dict]:
    """Iterate ``z`` over ``SLICE_AXIS`` and emit one row per per-slice lesion box.

    Slices with an empty mask emit **no row** (background/negative slices). Returned
    rows have keys ``volume_id, slice_z, r0, c0, r1, c1, component_id`` — ready to
    freeze to Parquet/CSV. ``component_id`` is now the per-slice **lesion-cluster
    index** (kept under the old column name for schema stability).
    """
    mask_iso = np.asarray(mask_iso)
    if mask_iso.ndim != 3:
        raise ValueError(f"mask_iso must be 3D, got shape {mask_iso.shape}")
    n_slices = mask_iso.shape[C.SLICE_AXIS]

    rows: List[Dict] = []
    for z in range(n_slices):
        sl = np.take(mask_iso, z, axis=C.SLICE_AXIS)
        boxes = boxes_for_slice(sl, merge_gap=merge_gap, min_area=min_area)
        for comp_id, (r0, c0, r1, c1) in enumerate(boxes):
            rows.append({
                "volume_id": int(volume_id),
                "slice_z": int(z),
                "r0": r0, "c0": c0, "r1": r1, "c1": c1,
                "component_id": comp_id,
            })
    return rows
