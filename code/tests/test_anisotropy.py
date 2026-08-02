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


MEASURED = (0.200, 0.073, 0.475674)      # results/AXIS_CHECK.md


def test_iso_voxel_mm_is_isotropic_only_when_the_declared_spacing_is_right():
    same = A.iso_voxel_mm(C.SPACING_STORAGE_MM)
    assert np.allclose(same, C.ISO_SPACING_MM)          # declared == "true" -> truly isotropic

    v = A.iso_voxel_mm(MEASURED)
    assert np.allclose(v, [0.4 * 0.200 / 0.073, 0.4 * 0.073 / 0.200, 0.4])
    assert v[0] > 1.0 and v[1] < 0.2                     # 1.096 x 0.146 x 0.400 mm


def test_cubic_reference_is_one_only_for_an_isotropic_cache():
    assert A.cubic_reference([0.4, 0.4, 0.4]) == pytest.approx(1.0)
    ref = A.cubic_reference(A.iso_voxel_mm(MEASURED))
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
    iso_mm = A.iso_voxel_mm(MEASURED)
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


def test_deployed_anisotropy_matches_the_shipped_token_block():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ext_d0": rng.uniform(1, 50, 40),
                       "ext_d1": rng.uniform(1, 50, 40),
                       "ext_d2": rng.uniform(1, 50, 40)})
    from abus_jcr.rescore.tokens import _block_abs_geom
    df2 = df.assign(public_id=0, cen_d0=1.0, cen_d1=1.0, cen_d2=1.0)
    x, names = _block_abs_geom(df2, {0: (100, 100, 100)})
    shipped = x[:, names.index("anisotropy")]
    assert np.allclose(A.deployed_anisotropy(df), shipped)


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
