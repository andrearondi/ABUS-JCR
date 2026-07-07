"""The 2.5D sample index and C=3 stack assembly (Inv. 1, 6).

SliceIndex enumerates every (volume_id, slice_z) for a split; get_stack gathers
the centre +/- 1 slices along SLICE_AXIS from the memmapped cache, clamps
boundary indices (EDGE_SLICE_POLICY), orders channels near->far, and returns
(C, d0, d1).
"""

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr import cache as K
from abus_jcr.slice_dataset import SliceIndex, get_stack


def _write_synth_volume(cache_root, vid, d0=6, d1=5, nz=4):
    # each slice z filled with the constant value z (distinguishable channels)
    vol = np.zeros((d0, d1, nz), dtype=np.float32)
    for z in range(nz):
        vol[:, :, z] = float(z)
    mask = np.zeros_like(vol, dtype=np.uint8)
    meta = {"iso_shape": [d0, d1, nz], "native_shape": [d0, d1, nz],
            "zoom_factors": [1.0, 1.0, 1.0]}
    K.write_case(cache_root, vid, vol, mask, meta)
    return d0, d1, nz


def test_slice_index_enumerates_all_slices(tmp_path):
    d0, d1, nz = _write_synth_volume(tmp_path, 100)
    manifest = pd.DataFrame([{"volume_id": 100, "split": "val", "fold": -1, "label": "B"}])
    idx = SliceIndex(manifest, "val", tmp_path)
    assert len(idx) == nz
    assert list(idx.samples) == [(100, z) for z in range(nz)]


def test_get_stack_shape_channels_and_edge_clamp(tmp_path):
    d0, d1, nz = _write_synth_volume(tmp_path, 100)

    mid = get_stack(tmp_path, 100, 2)
    assert mid.shape == (C.C_CHANNELS, d0, d1)
    # near->far ordering: channel c holds slice z-1+c
    assert mid[0].mean() == 1.0 and mid[1].mean() == 2.0 and mid[2].mean() == 3.0

    # z=0: z-1 clamps to 0 -> channels 0 and 1 identical (both slice 0)
    lo = get_stack(tmp_path, 100, 0)
    np.testing.assert_array_equal(lo[0], lo[1])
    assert lo[1].mean() == 0.0 and lo[2].mean() == 1.0

    # z=last: z+1 clamps to last -> channels 1 and 2 identical
    hi = get_stack(tmp_path, 100, nz - 1)
    np.testing.assert_array_equal(hi[1], hi[2])
    assert hi[1].mean() == float(nz - 1)
