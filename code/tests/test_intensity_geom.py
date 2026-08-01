"""Synthetic-phantom tests for the intensity <-> candidate probe.

Every test builds a volume whose answer is known by construction, so the code is
validated without appealing to intuition about real ABUS data. The point is that
if these pass, a surprising result on real data is a fact about the data, not a
bug in the measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abus_jcr.probe import intensity_geom as ig


# ---------------------------------------------------------------- helpers

def attenuating_volume(shape=(40, 50, 60), beam_axis=1, floor=20.0, peak=200.0, seed=0):
    """A volume that decays exponentially away from index 0 along ``beam_axis``.

    Everything else is uniform noise, so the ONLY systematic gradient is the one
    we planted.
    """
    rng = np.random.default_rng(seed)
    n = shape[beam_axis]
    prof = floor + (peak - floor) * np.exp(-3.0 * np.arange(n) / n)
    sh = [1, 1, 1]
    sh[beam_axis] = n
    vol = np.broadcast_to(prof.reshape(sh), shape).astype(np.float64).copy()
    vol += rng.normal(0.0, 2.0, size=shape)
    return vol


# ---------------------------------------------------------------- Stage 1

@pytest.mark.parametrize("beam_axis", [0, 1, 2])
def test_beam_axis_recovered_for_every_placement(beam_axis):
    """The vote must find the planted attenuation axis wherever it is put.

    Parametrising over all three placements is what makes this a real test: a
    hard-coded 'the answer is d1' would pass on the data and prove nothing.
    """
    vol = attenuating_volume(beam_axis=beam_axis)
    axis, stats = ig.beam_axis_vote(vol)
    assert axis == beam_axis
    assert stats[beam_axis]["spearman"] < -0.9
    assert stats[beam_axis]["peak_pos"] < 0.1        # bright near-field entrance
    assert stats[beam_axis]["decile_ratio"] > 1.5    # starts bright, ends dark
    for other in [a for a in range(3) if a != beam_axis]:
        assert abs(stats[other]["spearman"]) < 0.5


def test_axis_profile_is_spacing_free_and_matches_manual_mean():
    vol = attenuating_volume()
    assert np.allclose(ig.axis_profile(vol, 1), vol.mean(axis=(0, 2)))


def test_uniform_volume_gives_no_strong_beam_axis():
    """A volume with no gradient must not produce a confident answer."""
    rng = np.random.default_rng(1)
    vol = rng.normal(100.0, 5.0, size=(30, 30, 30))
    _axis, stats = ig.beam_axis_vote(vol)
    assert all(abs(stats[a]["spearman"]) < 0.4 for a in range(3))


# ---------------------------------------------------------------- Stage 2: planes

def test_dark_stripe_correspondence_is_positive_on_it_and_not_off_it():
    """Points placed ON a planted dark column correlate with darkness; points off it do not.

    The stripe runs along the beam axis (a shadow ray), so in the projection that
    DROPS the beam axis it is a dark point, and candidates sitting on it must show
    up as co-located with low intensity.
    """
    shape = (40, 50, 60)
    beam = 1
    vol = np.full(shape, 150.0)
    vol[10:14, :, 30:34] = 20.0                     # a dark column spanning all depth

    out_shape = (20, 30)                            # coarse grid over (d0, d2)
    inten = ig.plane_intensity_map(vol, drop_axis=beam, out_shape=out_shape)

    on = np.column_stack([np.full(40, 12.0), np.full(40, 32.0)])       # inside the stripe
    off = np.column_stack([np.linspace(20, 35, 40), np.linspace(5, 25, 40)])  # elsewhere

    rho_on = ig.map_spearman(-inten, ig.plane_count_map(on, (shape[0], shape[2]), out_shape))
    rho_off = ig.map_spearman(-inten, ig.plane_count_map(off, (shape[0], shape[2]), out_shape))
    assert rho_on > 0.15
    assert rho_on > rho_off


def test_map_spearman_returns_nan_when_a_map_is_constant():
    a = np.ones((5, 5))
    b = np.arange(25.0).reshape(5, 5)
    assert np.isnan(ig.map_spearman(a, b))
    assert np.isnan(ig.map_spearman(b, np.zeros((5, 5))))   # no candidates at all


def test_block_mean_preserves_the_overall_mean_and_shape():
    rng = np.random.default_rng(2)
    img = rng.normal(size=(37, 41))
    out = ig.block_mean(img, (6, 7))
    assert out.shape == (6, 7)
    assert np.isfinite(out).all()
    assert abs(np.nanmean(out) - img.mean()) < 0.5


def test_plane_count_map_counts_every_point_once():
    pts = np.array([[0.0, 0.0], [9.9, 9.9], [5.0, 5.0]])
    cm = ig.plane_count_map(pts, (10, 10), (4, 4))
    assert cm.sum() == 3


# ---------------------------------------------------------------- Stage 2: 3-D boxes

def test_storage_boxes_map_itk_columns_to_the_right_storage_axes():
    df = pd.DataFrame({"coordX": [1.0], "coordY": [2.0], "coordZ": [3.0],
                       "x_length": [10.0], "y_length": [20.0], "z_length": [30.0]})
    b = ig.storage_boxes(df, spacing=(1.0, 2.0, 4.0))
    assert np.allclose(b["cen"][0], [3.0, 2.0, 1.0])     # coordZ -> d0, coordX -> d2
    assert np.allclose(b["ext"][0], [30.0, 20.0, 10.0])
    assert np.allclose(b["cen_mm"][0], [3.0, 4.0, 4.0])


def test_contrast_is_negative_for_a_dark_box_and_zero_for_a_typical_one():
    """A box on a planted dark blob must read darker than its own depth band."""
    shape = (40, 50, 60)
    beam = 1
    vol = attenuating_volume(shape=shape, beam_axis=beam, seed=3)
    # generously larger than the 6-voxel box below, so the inclusive-max box
    # convention (min = c-len/2 .. max = c+len/2 INCLUSIVE, per DATA_INFO.md) cannot
    # pull bright rim voxels into the readout and weaken the planted contrast
    vol[16:27, 18:29, 26:37] = 5.0                  # a very dark blob

    cen = np.array([[21.0, 23.0, 31.0], [8.0, 23.0, 10.0]])
    ext = np.array([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
    st = ig.box_intensity_stats(vol, cen, ext, depth_axis=beam)
    assert st["contrast"][0] < -50.0                 # the dark blob
    assert abs(st["contrast"][1]) < 20.0             # ordinary tissue at the same depth


def test_depth_baseline_removes_the_attenuation_confound():
    """Two identical-contrast boxes at different depths must get the same contrast.

    Without the depth-matched baseline the deeper one would look much darker purely
    because of attenuation — which is exactly the false 'shadow' finding this
    control exists to prevent.
    """
    shape = (30, 60, 30)
    beam = 1
    vol = attenuating_volume(shape=shape, beam_axis=beam, floor=20.0, peak=200.0, seed=4)
    prof = ig.axis_profile(vol, beam)
    for d in (10, 45):                                # shallow and deep
        vol[12:18, d - 3:d + 3, 12:18] = prof[d] - 30.0   # same 30-unit deficit

    cen = np.array([[15.0, 10.0, 15.0], [15.0, 45.0, 15.0]])
    ext = np.array([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
    st = ig.box_intensity_stats(vol, cen, ext, depth_axis=beam)
    assert abs(st["contrast"][0] - st["contrast"][1]) < 8.0
    # and the raw intensity WOULD have been misleading:
    assert st["inside"][0] - st["inside"][1] > 50.0


def test_distal_contrast_detects_a_shadow_cast_below_the_box():
    """A box with a dark tail immediately distal must score a negative distal contrast."""
    shape = (30, 60, 30)
    beam = 1
    vol = np.full(shape, 120.0)
    vol[12:18, 20:26, 12:18] = 60.0                  # the "lesion"
    vol[12:18, 26:40, 12:18] = 10.0                  # its posterior shadow

    cen = np.array([[15.0, 23.0, 15.0], [15.0, 23.0, 3.0]])
    ext = np.array([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
    st = ig.box_intensity_stats(vol, cen, ext, depth_axis=beam, distal_scale=1.0)
    assert st["distal_contrast"][0] < -30.0          # shadow caster
    assert st["distal_contrast"][1] > -10.0          # nothing below it


def test_boxes_hanging_off_the_volume_still_read_a_voxel():
    """A candidate whose box starts past the last index must not silently become NaN.

    An empty slice means a NaN mean, which then disappears into an effect size as a
    missing value instead of as an error — so out-of-range boxes are clamped, not dropped.
    """
    vol = np.full((10, 10, 10), 100.0)
    cen = np.array([[50.0, 5.0, 5.0], [-20.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    ext = np.array([[4.0, 4.0, 4.0]] * 3)
    st = ig.box_intensity_stats(vol, cen, ext, depth_axis=1)
    assert np.isfinite(st["inside"]).all()
    assert np.allclose(st["inside"], 100.0)


def test_clip_slice_is_never_empty():
    for lo, hi, n in [(-5.0, -1.0, 10), (50.0, 60.0, 10), (3.0, 3.0, 10), (0.0, 99.0, 10)]:
        a, b = ig._clip_slice(lo, hi, n)
        assert 0 <= a < b <= n, (lo, hi, n, a, b)


def test_distal_contrast_is_nan_at_the_far_edge():
    vol = np.full((10, 20, 10), 100.0)
    cen = np.array([[5.0, 19.0, 5.0]])
    ext = np.array([[4.0, 4.0, 4.0]])
    st = ig.box_intensity_stats(vol, cen, ext, depth_axis=1)
    assert np.isnan(st["distal_contrast"][0])


# ---------------------------------------------------------------- Stage 2: alignment

def test_spread_fractions_find_the_axis_a_cloud_is_strung_along():
    rng = np.random.default_rng(5)
    cen = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 20, 200), rng.normal(0, 1, 200)])
    f = ig.spread_fractions(cen)
    assert np.argmax(f) == 1
    assert f[1] > 0.9
    assert abs(f.sum() - 1.0) < 1e-9


def test_spread_fractions_are_balanced_for_an_isotropic_cloud():
    rng = np.random.default_rng(6)
    f = ig.spread_fractions(rng.normal(0, 5, size=(500, 3)))
    assert np.all(np.abs(f - 1 / 3) < 0.1)


def test_extent_normalisation_removes_the_field_of_view_shape():
    """A cloud filling a very anisotropic volume must score ~1/3 per axis, not follow the FOV.

    Uses ABUS-like proportions (173 x 50 x 168 mm): without the normalisation the depth axis
    would look nearly empty purely because the breast is thin.
    """
    rng = np.random.default_rng(9)
    extent = [173.0, 50.0, 168.0]
    cen = np.column_stack([rng.uniform(0, e, 4000) for e in extent])
    raw = ig.spread_fractions(cen)
    norm = ig.spread_fractions(cen, extent)
    assert raw[1] < 0.06                                  # depth looks empty on raw mm
    assert np.all(np.abs(norm - 1 / 3) < 0.05)            # and balanced once normalised


def test_map_spearman_where_restricts_the_cells_used():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 400.0]])
    where = np.array([[True, True], [True, False]])
    assert ig.map_spearman(a, b) == pytest.approx(1.0)
    assert ig.map_spearman(a, b, where=where) == pytest.approx(1.0)
    # a cell that reverses the ordering is excluded by the mask
    b2 = np.array([[3.0, 2.0], [1.0, 0.0]])
    assert ig.map_spearman(a, b2) < 0
    assert np.isnan(ig.map_spearman(a, b2, where=np.array([[True, False], [False, False]])))


def test_coronal_stacking_sees_a_beam_line_as_one_deep_cell():
    """Points sharing a lateral/sweep position but spread in depth = one tall cell."""
    depth = np.linspace(0, 40, 20)
    cen = np.column_stack([np.full(20, 10.0), depth, np.full(20, 10.0)])   # d1 = depth
    out = ig.coronal_stacking(cen, depth_axis=1, cell_mm=5.0)
    assert len(out["cell_counts"]) == 1
    assert out["cell_counts"][0] == 20
    assert out["cell_depth_spread"][0] > 25.0


def test_coronal_stacking_spreads_a_flat_sheet_over_many_cells():
    rng = np.random.default_rng(7)
    cen = np.column_stack([rng.uniform(0, 50, 60), np.full(60, 20.0), rng.uniform(0, 50, 60)])
    out = ig.coronal_stacking(cen, depth_axis=1, cell_mm=5.0)
    assert len(out["cell_counts"]) > 20
    assert np.nanmax(out["cell_depth_spread"]) < 1e-6 or np.all(np.isnan(out["cell_depth_spread"]))


# ---------------------------------------------------------------- Stage 2: banding

def test_power_spectrum_recovers_a_planted_period():
    n = 256
    period = 16.0
    prof = 100.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / period)
    periods, power = ig.power_spectrum_1d(prof)
    assert abs(periods[int(np.argmax(power))] - period) < 1.0


def test_power_spectrum_of_flat_signal_is_empty():
    periods, power = ig.power_spectrum_1d(np.full(64, 5.0))
    assert len(periods) == 0 and len(power) == 0


# ---------------------------------------------------------------- effect size

def test_cliffs_delta_matches_the_pool_diag_definition():
    from abus_jcr.probe.pool_diag import _cliffs_delta as reference
    rng = np.random.default_rng(8)
    a = rng.normal(0.5, 1.0, 60)
    b = rng.normal(0.0, 1.0, 80)
    assert abs(ig.cliffs_delta(a, b) - reference(a, b)) < 1e-9


def test_cliffs_delta_edges():
    assert ig.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert ig.cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert np.isnan(ig.cliffs_delta([], [1, 2]))
