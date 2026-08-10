"""Tests for the corrected-anisotropy recomputation.

Every case has an answer known by construction, so a surprising number on the real record
is a fact about the pool rather than a bug in the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.probe import anisotropy as A


MEASURED = (0.200, 0.073, 0.475674)      # results/AXIS_CHECK.md — the TRUE spacing
LEGACY_DECLARED = (0.073, 0.200, 0.475674)   # what the DEPLOYED cache was built with

# These tests are about the *mechanism* "declared != true distorts the cache", so every call
# below passes its declared map EXPLICITLY rather than inheriting `C.SPACING_STORAGE_MM`.
# Until 2026-08-09 they inherited it, which silently assumed the legacy profile was the only
# one — the same class of substrate-pinning that made five flip-axis tests pass a defect
# (HISTORY.md §8). Under `ABUS_AXIS_PROFILE=measured` the inherited version asserted that a
# genuinely isotropic cache is 1.096 x 0.146 x 0.400 mm, i.e. it asserted the defect.


def test_iso_voxel_mm_is_isotropic_only_when_the_declared_spacing_is_right():
    # declared == true -> truly isotropic. This is the `measured` profile's cache.
    assert np.allclose(A.iso_voxel_mm(MEASURED, declared_spacing=MEASURED), C.ISO_SPACING_MM)

    # declared != true -> the deployed distortion, 1.0959 x 0.1460 x 0.4000 mm
    v = A.iso_voxel_mm(MEASURED, declared_spacing=LEGACY_DECLARED)
    assert np.allclose(v, [0.4 * 0.200 / 0.073, 0.4 * 0.073 / 0.200, 0.4])
    assert v[0] > 1.0 and v[1] < 0.2

    # and it must agree with whatever this process's profile actually declares
    assert np.allclose(A.iso_voxel_mm(MEASURED), A.iso_voxel_mm(MEASURED, C.SPACING_STORAGE_MM))


def test_cubic_reference_is_one_only_for_an_isotropic_cache():
    assert A.cubic_reference([0.4, 0.4, 0.4]) == pytest.approx(1.0)
    ref = A.cubic_reference(A.iso_voxel_mm(MEASURED, declared_spacing=LEGACY_DECLARED))
    assert ref == pytest.approx(0.195, abs=0.005)
    # the point of the number: the recorded medians (~0.39-0.41) sit ABOVE this reference,
    # so those candidates are d0-extended, not "sub-unity therefore compressed"
    assert 0.40 > ref


def test_elongation_ratios_are_one_for_a_cube_and_scale_as_expected():
    ext = np.array([[10.0, 10.0, 10.0], [30.0, 10.0, 10.0]])
    r = A.elongation_ratios(ext)
    assert r["elong_d0"][0] == pytest.approx(1.0)
    assert r["elong_d1"][0] == pytest.approx(1.0)
    assert r["elong_d2"][0] == pytest.approx(1.0)
    assert r["elong_d0"][1] == pytest.approx(3.0)        # 30 / mean(10,10)
    assert r["elong_d1"][1] == pytest.approx(10.0 / 20.0)


def test_a_beam_aligned_ray_is_only_visible_on_the_corrected_ratio():
    """A candidate elongated along DEPTH must score high on elong_d1, not on the deployed feature.

    This is the whole claim in one test: the same box looks unremarkable through the shipped
    lateral ratio and obviously ray-shaped through the depth ratio.
    """
    # Explicitly the DEPLOYED cache — this test is about what the shipped feature misses THERE.
    # On the `measured` profile the shipped feature is no longer the d0 ratio (it reads
    # FP_PROBE_ANISO_DEPTH_AXIS = 1), so the claim below is specific to the legacy substrate.
    iso_mm = A.iso_voxel_mm(MEASURED, declared_spacing=LEGACY_DECLARED)
    # a 4 x 40 x 4 mm box (long along d1 = depth), expressed in cache voxels
    ext_vox = np.array([[4.0 / iso_mm[0], 40.0 / iso_mm[1], 4.0 / iso_mm[2]]])
    df = pd.DataFrame({"ext_d0": ext_vox[:, 0], "ext_d1": ext_vox[:, 1], "ext_d2": ext_vox[:, 2]})

    dep = A.deployed_anisotropy(df)[0]
    mm = ext_vox * iso_mm[None, :]
    r = A.elongation_ratios(mm)
    assert r["elong_d1"][0] == pytest.approx(10.0, rel=1e-6)   # 40 / mean(4,4)
    assert dep < 0.05                                          # shipped feature: near zero
    assert dep < A.cubic_reference(iso_mm)                     # and BELOW its own cubic point


def test_native_and_iso_routes_to_millimetres_agree():
    """The two independent conversions must land on the same physical extents."""
    iso_mm = A.iso_voxel_mm(MEASURED)
    native = np.array([[120.0, 300.0, 40.0]])                  # d0, d1, d2 native voxels
    df = pd.DataFrame({
        "z_length": native[:, 0], "y_length": native[:, 1], "x_length": native[:, 2],
        "ext_d0": native[:, 0] * C.SPACING_STORAGE_MM[0] / C.ISO_SPACING_MM,
        "ext_d1": native[:, 1] * C.SPACING_STORAGE_MM[1] / C.ISO_SPACING_MM,
        "ext_d2": native[:, 2] * C.SPACING_STORAGE_MM[2] / C.ISO_SPACING_MM,
    })
    e = A.extents_mm(df, MEASURED, iso_mm)
    assert np.allclose(e["native_mm"], e["iso_mm"], rtol=1e-9)
    assert np.allclose(e["native_mm"][0], native[0] * np.asarray(MEASURED))


def test_the_three_anisotropy_definitions_agree():
    """``anisotropy`` is NOT a frozen record column — it is derived, in three places, and the
    recorded effect sizes come from them. One definition, or the recorded numbers stop being
    comparable across probes.

    (Until 2026-08-09 this compared against a FOURTH copy inside the Phase-4 token. That dim
    was removed — see ``rescore/tokens`` — so the model no longer consumes it at all; the
    diagnostics below are unchanged and keep every recorded number valid.)"""
    from abus_jcr.probe.fp_structure import _anisotropy
    from abus_jcr.probe.pool_diag import augment

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ext_d0": rng.uniform(1, 50, 40),
                       "ext_d1": rng.uniform(1, 50, 40),
                       "ext_d2": rng.uniform(1, 50, 40)})

    # THE profile-independent invariant: the two live probes must never drift apart, whichever
    # axis `FP_PROBE_ANISO_DEPTH_AXIS` names. Both read that constant (since 2026-08-09), so a
    # hand-edit to one of them shows up here.
    live = _anisotropy(df)
    diag = augment(df.assign(x_length=1.0, y_length=1.0, z_length=1.0))["anisotropy"].to_numpy()
    np.testing.assert_allclose(live, diag)

    # `A.deployed_anisotropy` is frozen to the d0 ratio by design — it is the definition every
    # RECORDED number was computed with, and it must not follow the profile, or the record would
    # stop being reproducible. On `legacy` it therefore equals the live probes; on `measured` it
    # deliberately does NOT, and the two are not comparable (a physically cubic candidate reads
    # 0.195 there and 1.0 here).
    if C.AXIS_PROFILE == "legacy":
        assert C.FP_PROBE_ANISO_DEPTH_AXIS == 0
        np.testing.assert_allclose(A.deployed_anisotropy(df), live)
    else:
        assert C.FP_PROBE_ANISO_DEPTH_AXIS == C.DEPTH_AXIS == 1
        assert not np.allclose(A.deployed_anisotropy(df), live)


def test_the_phase4_token_no_longer_consumes_anisotropy():
    """Removed 2026-08-09: wrong axis (d0 is lateral), uninterpretable units (a physically
    cubic candidate reads 0.195), the weakest of all 12 pool features (val delta 0.097), a
    closed-negative motivating hypothesis, and ~95% recoverable from the log1p extents that
    remain. Pinned so it cannot drift back in."""
    from abus_jcr.rescore.tokens import BLOCK_DIMS, _block_abs_geom

    df = pd.DataFrame({"ext_d0": [10.0, 20.0], "ext_d1": [30.0, 5.0], "ext_d2": [7.0, 9.0],
                       "public_id": [0, 0], "cen_d0": 1.0, "cen_d1": 1.0, "cen_d2": 1.0})
    x, names = _block_abs_geom(df, {0: (100, 100, 100)})
    assert BLOCK_DIMS["abs_geom"] == 6 and x.shape[1] == 6 and len(names) == 6
    ref = A.deployed_anisotropy(df)
    assert not any(np.allclose(x[:, j], ref) for j in range(x.shape[1]))


def test_per_volume_delta_skips_underpowered_volumes_and_reports_sign():
    rng = np.random.default_rng(1)
    vals, tp, vid = [], [], []
    for v in range(6):
        vals += list(rng.normal(1.0, 0.1, 10)) + list(rng.normal(0.0, 0.1, 10))
        tp += [True] * 10 + [False] * 10
        vid += [v] * 20
    # a 7th volume with only 1 TP -> must be skipped, not counted as a delta
    vals += [5.0] + list(rng.normal(0.0, 0.1, 10))
    tp += [True] + [False] * 10
    vid += [6] * 11
    med, sign, n = A.per_volume_delta(np.array(vals), np.array(tp), np.array(vid))
    assert n == 6
    assert med == pytest.approx(1.0)
    assert sign == pytest.approx(1.0)


def test_per_volume_delta_is_nan_when_nothing_is_powered():
    med, sign, n = A.per_volume_delta(np.array([1.0, 2.0]), np.array([True, False]),
                                      np.array([0, 0]))
    assert n == 0 and np.isnan(med)
