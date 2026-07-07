"""Isotropic resampling geometry, dtype/range invariants, and hash stability.

The resample is the substrate every later phase stands on (Inv. 6): the shape
formula must be exact, the mask must survive as {0,1}, the image must land in
float32 [0,1], and the preprocess hash must be deterministic and sensitive to
every cache-invalidating input.
"""

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.preprocess import (
    zoom_factors,
    iso_shape,
    resample_case,
    preprocess_hash,
)


def test_zoom_factors_match_spacing_ratio():
    f = zoom_factors()
    expected = tuple(C.SPACING_STORAGE_MM[a] / C.ISO_SPACING_MM for a in range(3))
    assert f == pytest.approx(expected, abs=1e-12)
    # sanity: (0.146, 0.400, 0.951348) at 0.5 mm
    assert f == pytest.approx((0.146, 0.400, 0.951348), abs=1e-6)


def test_iso_shape_is_round_of_native_times_f():
    native = (500, 400, 300)
    f = zoom_factors()
    expected = tuple(int(round(native[a] * f[a])) for a in range(3))
    assert iso_shape(native) == expected


def test_resample_case_shapes_dtype_range_and_mask_binary():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 256, size=(40, 30, 20), dtype=np.uint8)
    mask = np.zeros((40, 30, 20), dtype=np.uint8)
    mask[10:25, 8:20, 5:15] = 1

    vol_iso, mask_iso, meta = resample_case(vol, mask)

    exp_shape = iso_shape(vol.shape)
    assert vol_iso.shape == exp_shape
    assert mask_iso.shape == exp_shape

    assert vol_iso.dtype == np.float32
    assert float(vol_iso.min()) >= 0.0
    assert float(vol_iso.max()) <= 1.0

    assert mask_iso.dtype == np.uint8
    assert set(np.unique(mask_iso)).issubset({0, 1})
    # a chunky lesion must not be resampled out of existence
    assert mask_iso.sum() > 0

    # meta carries exactly what Phase 3 needs to invert the transform
    assert tuple(meta["native_shape"]) == vol.shape
    assert tuple(meta["iso_shape"]) == exp_shape
    assert tuple(meta["zoom_factors"]) == pytest.approx(zoom_factors(), abs=1e-12)
    assert meta["iso_spacing_mm"] == C.ISO_SPACING_MM


def test_preprocess_hash_deterministic_and_sensitive():
    h1 = preprocess_hash()
    h2 = preprocess_hash()
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64  # sha256 hexdigest

    # changing a cache-invalidating input changes the hash
    h_other = preprocess_hash(iso_spacing_mm=0.4)
    assert h_other != h1
