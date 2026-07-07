"""26-connectivity lesion audit: component count and size distribution, with a
descriptive speck floor.
"""

import numpy as np

from abus_jcr.lesions import component_sizes, audit_mask


def test_two_diagonally_touching_voxels_are_one_component_under_26conn():
    # diagonal neighbours are connected only under 26-connectivity
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    assert component_sizes(mask) == [2]


def test_separated_blobs_counted_with_sizes():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1   # 8 voxels
    mask[7:9, 7:9, 7:9] = 1   # 8 voxels, far away
    mask[5, 5, 5] = 1         # single speck, not touching either
    sizes = sorted(component_sizes(mask))
    assert sizes == [1, 8, 8]


def test_audit_mask_raw_and_effective_counts():
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[0:4, 0:4, 0:4] = 1   # 64-voxel lesion
    mask[10, 10, 10] = 1      # 1-voxel speck
    out = audit_mask(mask, min_voxels=10)
    assert out["n_components_raw"] == 2
    assert sorted(out["sizes"]) == [1, 64]
    assert out["n_components_effective"] == 1  # speck dropped by the floor


def test_audit_mask_empty_is_zero_components():
    out = audit_mask(np.zeros((4, 4, 4), dtype=np.uint8), min_voxels=10)
    assert out["n_components_raw"] == 0
    assert out["n_components_effective"] == 0
    assert out["sizes"] == []
