"""Isotropic cache: round-trip, memmap access, and hash guard (Inv. 6).

The cache directory is named by the preprocess hash; reading a cache whose
recorded hash differs from the current config must be refused, never silently
reused.
"""

import json

import numpy as np
import pytest

from abus_jcr import cache as K
from abus_jcr.preprocess import resample_case, preprocess_hash


def _case():
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 256, size=(24, 20, 16), dtype=np.uint8)
    mask = np.zeros((24, 20, 16), dtype=np.uint8)
    mask[6:14, 5:13, 4:11] = 1
    return resample_case(vol, mask)


def test_cache_roundtrip_and_memmap(tmp_path):
    vol_iso, mask_iso, meta = _case()
    K.write_case(tmp_path, 100, vol_iso, mask_iso, meta)

    got_vol = K.open_vol(tmp_path, 100)
    got_mask = K.open_mask(tmp_path, 100)
    assert isinstance(got_vol, np.memmap)
    np.testing.assert_array_equal(np.asarray(got_vol), vol_iso)
    np.testing.assert_array_equal(np.asarray(got_mask), mask_iso)

    got_meta = K.read_meta(tmp_path, 100)
    assert tuple(got_meta["iso_shape"]) == vol_iso.shape

    cache_dir = K.cache_dir(tmp_path)
    assert cache_dir.name == preprocess_hash()
    cm = json.loads((cache_dir / "CACHE_META.json").read_text())
    assert cm["preprocess_hash"] == preprocess_hash()


def test_assert_hash_passes_and_rejects(tmp_path):
    vol_iso, mask_iso, meta = _case()
    K.write_case(tmp_path, 100, vol_iso, mask_iso, meta)

    K.assert_hash(tmp_path)  # matching config: no raise

    # tamper the recorded hash -> read must be refused
    cm_path = K.cache_dir(tmp_path) / "CACHE_META.json"
    cm = json.loads(cm_path.read_text())
    cm["preprocess_hash"] = "deadbeef"
    cm_path.write_text(json.dumps(cm))
    with pytest.raises(Exception):
        K.assert_hash(tmp_path)
