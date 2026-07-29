"""[4.1] Crop augmentation is ENCODER-PRETRAINING-ONLY and ABUS-physics-safe (Inv. 13).

d0 = depth/beam (skin at the top, shadows extend downward) => NO flip along d0, ever.
d1 = lateral => horizontal flip is allowed. Centre jitter is the only other op.
No rotation, no scale jitter, no mosaic/mixup, no copy-paste. Torch-free.
"""

import numpy as np
import pytest

from abus_jcr import conventions as C
from abus_jcr.rescore import crop_aug


def _asym():
    """A crop that is asymmetric along every axis, so a flip is detectable on each."""
    a = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    return a / a.max()


def test_flip_is_along_d1_the_lateral_axis():
    crop = _asym()
    out = crop_aug.maybe_flip_d1(crop, np.random.default_rng(0), p=1.0)
    np.testing.assert_array_equal(out, crop[:, ::-1, :])


def test_flip_is_never_along_d0_the_depth_beam_axis():
    """The Inv.-13 red line: replicating/mirroring the beam axis is acoustically impossible."""
    crop = _asym()
    for seed in range(20):
        out = crop_aug.maybe_flip_d1(crop, np.random.default_rng(seed), p=1.0)
        assert not np.array_equal(out, crop[::-1, :, :])
        assert not np.array_equal(out, crop[:, :, ::-1])


def test_flip_probability_zero_is_identity():
    crop = _asym()
    np.testing.assert_array_equal(crop_aug.maybe_flip_d1(crop, np.random.default_rng(1), p=0.0), crop)


def test_flip_axis_constant_matches_the_conventions_lateral_axis():
    assert crop_aug.FLIP_AXIS == C.IN_PLANE_COL_AXIS == 1


def test_centre_jitter_stays_within_the_declared_fraction_of_the_roi_side():
    rng = np.random.default_rng(0)
    cen = (100.0, 200.0, 300.0)
    side = 80.0
    frac = C.RESC_ENC_AUG["centre_jitter_frac"]
    for _ in range(500):
        j = crop_aug.jitter_centre(cen, side, rng)
        for a in range(3):
            assert abs(j[a] - cen[a]) <= frac * side + 1e-9


def test_centre_jitter_is_zero_mean_and_actually_moves_the_centre():
    rng = np.random.default_rng(0)
    cen = (50.0, 50.0, 50.0)
    draws = np.array([crop_aug.jitter_centre(cen, 60.0, rng) for _ in range(4000)])
    assert np.abs(draws.mean(axis=0) - 50.0).max() < 0.5
    assert draws.std(axis=0).min() > 0.5


def test_augmentation_is_reproducible_under_a_seeded_rng():
    cen = (10.0, 20.0, 30.0)
    a = crop_aug.jitter_centre(cen, 48.0, np.random.default_rng(7))
    b = crop_aug.jitter_centre(cen, 48.0, np.random.default_rng(7))
    assert a == b


def test_augment_params_reports_exactly_the_two_sanctioned_ops():
    p = crop_aug.augment_params()
    assert set(p) == {"hflip_d1_p", "centre_jitter_frac"}
    assert p["hflip_d1_p"] == C.RESC_ENC_AUG["hflip_d1_p"]


def test_disabled_augmentation_is_the_identity_path():
    """Feature extraction, set-module training and every evaluation run un-augmented (Inv. 13)."""
    crop = _asym()
    cen = (10.0, 20.0, 30.0)
    rng = np.random.default_rng(0)
    assert crop_aug.jitter_centre(cen, 48.0, rng, frac=0.0) == cen
    np.testing.assert_array_equal(crop_aug.maybe_flip_d1(crop, rng, p=0.0), crop)


def test_augmented_crop_reduces_to_extract_crop_when_disabled():
    from abus_jcr.rescore.crops import extract_crop
    rng = np.random.default_rng(0)
    vol = np.random.default_rng(3).random((40, 40, 40)).astype(np.float32)
    plain = extract_crop(vol, 20.0, 20.0, 20.0, 12.0, 12.0, 12.0)
    aug = crop_aug.augmented_crop(vol, 20.0, 20.0, 20.0, 12.0, 12.0, 12.0,
                                  rng=rng, hflip_p=0.0, jitter_frac=0.0)
    np.testing.assert_allclose(aug, plain)


def test_augmented_crop_flip_and_jitter_change_the_output():
    from abus_jcr.rescore.crops import extract_crop
    vol = np.random.default_rng(3).random((60, 60, 60)).astype(np.float32)
    plain = extract_crop(vol, 30.0, 30.0, 30.0, 12.0, 12.0, 12.0)
    aug = crop_aug.augmented_crop(vol, 30.0, 30.0, 30.0, 12.0, 12.0, 12.0,
                                  rng=np.random.default_rng(0), hflip_p=1.0, jitter_frac=0.1)
    assert not np.allclose(aug, plain)
    assert aug.shape == plain.shape
    assert 0.0 <= float(aug.min()) and float(aug.max()) <= 1.0
