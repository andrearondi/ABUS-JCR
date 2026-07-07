"""The 2.5D sample index and C-channel stack assembly (Inv. 1, 6).

Enumerates every ``(volume_id, slice_z)`` in a split — both lesion-bearing and
background slices; Phase 2 owns the sampling/negative ratio. ``get_stack``
gathers the ``C_CHANNELS`` adjacent slices centred on ``slice_z`` along
``SLICE_AXIS`` from the memmapped cache, clamps boundary indices
(``EDGE_SLICE_POLICY``), orders channels near->far, and returns ``(C, d0, d1)``.

Augmentation is NOT applied here (Inv. 13: candidate generation and rescorer
feature extraction run without augmentation); Phase 2's training dataloader wraps
this and applies the shared-across-channels transforms from :mod:`augment`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from . import conventions as C
from . import cache as K

# canonical split-name normalisation (mirror manifest.py)
_SPLIT_CANON = {
    "train": "train", "Train": "train",
    "val": "val", "validation": "val", "Validation": "val",
    "test": "test", "Test": "test",
}


class SliceIndex:
    """Flat list of ``(volume_id, slice_z)`` samples for one split."""

    def __init__(self, manifest: pd.DataFrame, split: str, cache_root):
        split = _SPLIT_CANON.get(split, split)
        self.split = split
        self.cache_root = cache_root
        vids = sorted(int(v) for v in manifest.loc[manifest["split"] == split, "volume_id"])
        samples: List[Tuple[int, int]] = []
        self.n_slices = {}
        for vid in vids:
            n = int(K.read_meta(cache_root, vid)["iso_shape"][C.SLICE_AXIS])
            self.n_slices[vid] = n
            samples.extend((vid, z) for z in range(n))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[int, int]:
        return self.samples[i]


def get_stack(cache_root, volume_id: int, slice_z: int) -> np.ndarray:
    """Assemble the C-channel 2.5D stack centred on ``slice_z`` -> ``(C, d0, d1)``.

    Channels span ``slice_z - half .. slice_z + half`` (``half = C//2``) along
    ``SLICE_AXIS``, ordered near->far (increasing z). Out-of-range indices are
    clamped to ``[0, n-1]`` (``EDGE_SLICE_POLICY == "clamp"``), replicating the
    boundary slice.
    """
    vol = K.open_vol(cache_root, volume_id)
    n = vol.shape[C.SLICE_AXIS]
    half = C.C_CHANNELS // 2
    channels = []
    for off in range(-half, C.C_CHANNELS - half):
        z = min(max(slice_z + off, 0), n - 1)  # clamp
        frame = np.take(vol, z, axis=C.SLICE_AXIS)  # (d0, d1)
        channels.append(np.asarray(frame, dtype=np.float32))
    return np.stack(channels, axis=0)  # (C, d0, d1)
