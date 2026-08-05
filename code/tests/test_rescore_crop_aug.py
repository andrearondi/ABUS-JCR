"""[4.1] Crop augmentation is ENCODER-PRETRAINING-ONLY and ABUS-physics-safe (Inv. 13).

**MEASURED axis roles** (results/AXIS_CHECK.md, 129/130 volumes, four independent lines) —
NOT the inverted roles the `conventions` names still declare:

    d0 = LATERAL     => the mirror flip is allowed here, and ONLY here.
    d1 = DEPTH/BEAM  => skin at the top, shadows extend downward. NO flip, ever (Inv. 13).
    d2 = sweep       => NO flip.

These tests previously pinned the flip to d1 and so would have PASSED the defective
implementation corrected on 2026-08-04. They now assert against measured pixels on both
sides: the lateral flip must happen, and the depth flip must never happen.
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


def test_flip_is_along_d0_the_measured_lateral_axis():
    crop = _asym()
    out = crop_aug.maybe_flip_lateral(crop, np.random.default_rng(0), p=1.0)
    np.testing.assert_array_equal(out, crop[::-1, :, :])


def test_flip_is_never_along_d1_the_measured_depth_beam_axis():
    """The Inv.-13 red line: mirroring the beam axis is acoustically impossible.

    This is the assertion that was inverted before 2026-08-04 and let the defect through.
    """
    crop = _asym()
    for seed in range(20):
        out = crop_aug.maybe_flip_lateral(crop, np.random.default_rng(seed), p=1.0)
        assert not np.array_equal(out, crop[:, ::-1, :])   # d1 = depth/beam — FORBIDDEN
        assert not np.array_equal(out, crop[:, :, ::-1])   # d2 = sweep — FORBIDDEN


def test_deprecated_alias_still_works_and_flips_laterally():
    crop = _asym()
    a = crop_aug.maybe_flip_d1(crop, np.random.default_rng(0), p=1.0)
    b = crop_aug.maybe_flip_lateral(crop, np.random.default_rng(0), p=1.0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, crop[::-1, :, :])


def test_flip_probability_zero_is_identity():
    crop = _asym()
    np.testing.assert_array_equal(crop_aug.maybe_flip_lateral(crop, np.random.default_rng(1), p=0.0), crop)


def test_flip_axis_is_the_measured_lateral_axis_not_the_declared_one():
    """Regression pin for the 2026-08-04 Inv.-13 correction.

    `C.IN_PLANE_COL_AXIS` (= 1) is DECLARED "lateral" but MEASURED depth/beam, so the flip
    must NOT resolve through it. See results/AXIS_CHECK.md and the crop_aug module docstring.
    """
    assert crop_aug.FLIP_AXIS == 0                       # d0 = measured lateral
    assert crop_aug.FLIP_AXIS != C.IN_PLANE_COL_AXIS     # the defect this replaced


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
    np.testing.assert_array_equal(crop_aug.maybe_flip_lateral(crop, rng, p=0.0), crop)


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
