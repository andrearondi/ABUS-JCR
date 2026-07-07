"""26-connectivity connected-component audit of tumour masks.

Gates the relational framing: the plan assumes one dominant lesion per volume
(``labels.csv`` is one row per case). This module counts components so that
assumption is *confirmed on the data*, never asserted. The ``min_voxels`` floor
is descriptive only — it separates genuine lesions from sub-voxel mask specks
and affects no GT box, model, or label. Raw counts are always reported too.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage

from .conventions import LESION_MIN_VOXELS

# full 3x3x3 structuring element = 26-connectivity
_STRUCT_26 = np.ones((3, 3, 3), dtype=int)


def component_sizes(mask: np.ndarray) -> List[int]:
    """Voxel counts of each 26-connected component of ``mask > 0``."""
    labeled, n = ndimage.label(np.asarray(mask) > 0, structure=_STRUCT_26)
    if n == 0:
        return []
    counts = np.bincount(labeled.ravel())[1:]  # drop background label 0
    return [int(c) for c in counts]


def audit_mask(mask: np.ndarray, min_voxels: int = LESION_MIN_VOXELS) -> Dict[str, object]:
    """Raw and floored component statistics for one mask.

    Returns ``n_components_raw``, the per-component ``sizes``, and
    ``n_components_effective`` = components with ``size >= min_voxels``.
    """
    sizes = component_sizes(mask)
    effective = [s for s in sizes if s >= min_voxels]
    return {
        "n_components_raw": len(sizes),
        "sizes": sizes,
        "n_components_effective": len(effective),
        "min_voxels": int(min_voxels),
    }
