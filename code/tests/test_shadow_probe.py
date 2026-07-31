"""Unit tests for ``abus_jcr.probe.shadow`` on synthetic acoustic phantoms.

The phantoms encode the one distinction the whole module exists to make: an acoustic
**shadow** (a deficit that persists distally on a beam line) versus a **hypoechoic mass**
(a deficit that is dark but leaves the tissue below it normal). Both are "dark"; only the
first is an artifact. A test suite that cannot tell them apart tests nothing.
"""

import numpy as np
import pytest

from abus_jcr.probe import shadow as SH


# ----------------------------------------------------------------------------- phantoms
def make_phantom(beam_axis: int, n_depth: int = 80, n_u: int = 50, n_v: int = 40,
                 seed: int = 0):
    """A beam-first phantom then moved so the beam lies on ``beam_axis``.

    Contents (all in beam-first coordinates ``(depth, u, v)``):
      - tissue speckle at ~120 with exponential depth attenuation (the TGC residual)
      - a SHADOW ray at ``(u, v) = (10, 10)``: everything from depth 25 down is 0.25x
      - a HYPOECHOIC MASS at ``(u, v) = (35, 30)``, depths 25..33: 0.25x, nothing below
      - a non-tissue (padding) block at ``u >= 45``: near zero everywhere

    Returns ``(volume, info)`` with the ground-truth coordinates already mapped into the
    caller's axis order.
    """
    rng = np.random.default_rng(seed)
    depth = np.arange(n_depth, dtype=np.float32)
    base = 120.0 * np.exp(-depth / (n_depth * 1.2))
    v = base[:, None, None] * (1.0 + 0.12 * rng.standard_normal((n_depth, n_u, n_v)).astype(np.float32))

    v[25:, 8:13, 8:13] *= 0.25          # shadow ray (a small bundle of lines)
    v[25:34, 33:38, 28:33] *= 0.25      # hypoechoic mass, normal tissue below
    v[:, 45:, :] = 2.0                  # padding / outside the breast

    v = np.clip(v, 0, 255)
    vol = np.moveaxis(v, 0, beam_axis)

    def _pt(d, u, w):
        c = [0, 0, 0]
        other = [a for a in (0, 1, 2) if a != beam_axis]
        c[beam_axis] = d
        c[other[0]] = u
        c[other[1]] = w
        return c

    return vol, {"shadow_centre": _pt(50, 10, 10), "mass_centre": _pt(29, 35, 30),
                 "pad_u": 45, "n_depth": n_depth, "point": _pt}


def _mass_and_ray_phantom(n_depth=200, mass_start=30, thickness=40, seed=2):
    """Beam-axis-1 phantom with a hypoechoic mass of controllable thickness at
    ``(u, v) ~ (17, 17)`` and a genuine full-depth shadow ray at ``(u, v) ~ (32, 32)``."""
    rng = np.random.default_rng(seed)
    depth = np.arange(n_depth, dtype=np.float32)
    base = 120.0 * np.exp(-depth / (n_depth * 1.2))
    v = base[:, None, None] * (1.0 + 0.10 * rng.standard_normal(
        (n_depth, 40, 40)).astype(np.float32))
    v[mass_start:mass_start + thickness, 15:20, 15:20] *= 0.25
    v[mass_start:, 30:34, 30:34] *= 0.25                 # ray: continues to the far field
    return np.moveaxis(np.clip(v, 0, 255), 0, 1)


def _box_around(centre, half, beam_axis, half_depth=None):
    """(min0,min1,min2,max0,max1,max2) box centred on ``centre``."""
    hd = half if half_depth is None else half_depth
    lo, hi = [], []
    for a in (0, 1, 2):
        h = hd if a == beam_axis else half
        lo.append(centre[a] - h)
        hi.append(centre[a] + h)
    return lo + hi


# ----------------------------------------------------------------------------- axis identification
@pytest.mark.parametrize("beam_axis", [0, 1, 2])
def test_identify_beam_axis_recovers_the_attenuation_axis(beam_axis):
    vol, _ = make_phantom(beam_axis)
    assert SH.identify_beam_axis(vol) == beam_axis


@pytest.mark.parametrize("beam_axis", [0, 1, 2])
def test_beam_axis_has_the_only_strongly_negative_spearman(beam_axis):
    vol, _ = make_phantom(beam_axis)
    st = SH.axis_attenuation_stats(vol)
    assert st[beam_axis]["spearman"] < -0.8
    for a in (0, 1, 2):
        if a != beam_axis:
            assert st[a]["spearman"] > -0.5
    # the beam axis is bright at the entrance and dark at the far field
    assert st[beam_axis]["asym"] > 0
    assert st[beam_axis]["peak_pos_frac"] < 0.2


# ----------------------------------------------------------------------------- residual
def test_depth_normalised_residual_is_centred_within_each_depth_plane():
    vol, _ = make_phantom(1)
    tissue = SH.tissue_line_mask(vol, beam_axis=1)
    z = SH.depth_normalised_residual(vol, beam_axis=1, tissue=tissue, sub=1)
    assert z.shape == vol.shape
    zb = np.moveaxis(z, 1, 0)
    # excluding the padding block, each depth plane is centred near zero
    for j in (10, 40, 70):
        plane = zb[j][:45, :]
        assert abs(float(np.median(plane))) < 0.1


def test_striding_the_estimator_costs_only_sampling_noise():
    """``sub`` trades accuracy of the per-depth baseline for speed; bound the cost."""
    vol, _ = make_phantom(1)
    tissue = SH.tissue_line_mask(vol, beam_axis=1)
    exact = np.moveaxis(SH.depth_normalised_residual(vol, 1, tissue=tissue, sub=1), 1, 0)
    fast = np.moveaxis(SH.depth_normalised_residual(vol, 1, tissue=tissue, sub=4), 1, 0)
    for j in (10, 40, 70):
        assert abs(float(np.median(fast[j][:45, :])) -
                   float(np.median(exact[j][:45, :]))) < 0.4


def test_padding_biases_the_baseline_unless_the_tissue_mask_is_passed():
    """Why ``tissue`` is not optional in practice: padding shifts every depth median."""
    vol, _ = make_phantom(1)
    tissue = SH.tissue_line_mask(vol, beam_axis=1)
    z_masked = np.moveaxis(SH.depth_normalised_residual(vol, 1, tissue=tissue), 1, 0)
    z_naive = np.moveaxis(SH.depth_normalised_residual(vol, 1), 1, 0)
    m_masked = abs(float(np.median(z_masked[40][:45, :])))
    m_naive = abs(float(np.median(z_naive[40][:45, :])))
    assert m_naive > m_masked


def test_residual_removes_the_global_attenuation_trend():
    """Raw intensity falls with depth; the residual must not."""
    vol, _ = make_phantom(1)
    z = SH.depth_normalised_residual(vol, beam_axis=1)
    zb = np.moveaxis(z, 1, 0)[:, :45, :]
    raw = np.moveaxis(vol, 1, 0)[:, :45, :].mean(axis=(1, 2))
    assert raw[5] > 2 * raw[-5]                       # the trend really is there
    prof = np.array([float(np.median(zb[j])) for j in range(zb.shape[0])])
    assert prof.max() - prof.min() < 1.0              # and the residual has flattened it


# ----------------------------------------------------------------------------- shadow vs mass
@pytest.mark.parametrize("beam_axis", [0, 1, 2])
def test_shadow_ray_is_flagged_and_hypoechoic_mass_is_not(beam_axis):
    """THE test: both regions are equally dark; only the ray persists distally."""
    vol, info = make_phantom(beam_axis)
    f = SH.shadow_field(vol, beam_axis=beam_axis)
    sh = np.moveaxis(f["shadow"], beam_axis, 0)

    ray = sh[40:60, 9:12, 9:12].mean()               # well below the ray's onset
    mass = sh[26:33, 34:37, 29:32].mean()            # inside the mass
    assert ray > 0.8, f"shadow ray not flagged (got {ray:.2f})"
    assert mass < 0.2, f"hypoechoic mass misfiled as shadow (got {mass:.2f})"
    assert ray > 4 * max(mass, 1e-3)


def test_persistence_test_has_no_length_scale_a_thick_mass_can_defeat():
    """The reason persistence is anchored to the far field rather than to a gap below each
    voxel: on the real cache the depth axis is finely sampled (341 voxels over ~50 mm), so
    the median GT lesion is ~98 voxels deep. A gapped running mean would need a gap larger
    than that. A far-field statistic is unaffected by how thick the mass is."""
    # far field = deepest 30% of 200 = depths 140..200, so a mass ending by 130 is clear of
    # it. Thickness then ranges over 6..100 voxels -- far beyond any gap that would fit.
    for thickness in (6, 40, 100):
        vol = _mass_and_ray_phantom(n_depth=200, mass_start=30, thickness=thickness)
        sh = np.moveaxis(SH.shadow_field(vol, beam_axis=1)["shadow"], 1, 0)
        mass = sh[32:30 + thickness - 2, 16:19, 16:19].mean()
        ray = sh[150:190, 31:33, 31:33].mean()
        assert mass < 0.2, f"mass of thickness {thickness} misfiled as shadow ({mass:.2f})"
        assert ray > 0.8, f"true ray lost at thickness {thickness} ({ray:.2f})"


def test_known_limitation_a_mass_inside_the_far_field_is_ambiguous():
    """Pins the one confusion the far-field design accepts, so it cannot regress silently.

    A mass that extends INTO the deepest 30% of its line darkens the far field by
    occupying it, and is flagged. From the volume alone that is genuinely ambiguous with a
    deep shadow, which is why it is documented rather than patched. It is also why §4 of
    the probe reports ``shadow_frac`` (is a shadow) next to ``distal_z`` (casts a shadow)
    instead of either one alone.
    """
    vol = _mass_and_ray_phantom(n_depth=200, mass_start=120, thickness=60)   # ends at 180
    sh = np.moveaxis(SH.shadow_field(vol, beam_axis=1)["shadow"], 1, 0)
    assert sh[125:175, 16:19, 16:19].mean() > 0.5


def test_padding_lines_are_never_flagged():
    vol, info = make_phantom(1)
    f = SH.shadow_field(vol, beam_axis=1)
    sh = np.moveaxis(f["shadow"], 1, 0)
    assert sh[:, info["pad_u"]:, :].sum() == 0
    assert not f["tissue"][info["pad_u"]:, :].any()
    assert f["tissue"][:40, :].all()


def test_tissue_mask_rejects_a_dim_edge_column_with_bright_speckle():
    """The failure that a max-based mask has on real volumes: an out-of-field column is
    mostly dark but carries bright outliers, so only a near-field MEAN test excludes it."""
    rng = np.random.default_rng(7)
    vol, _ = make_phantom(1)
    vb = np.moveaxis(vol, 1, 0).copy()
    edge = slice(40, 45)
    vb[:, edge, :] *= 0.06                                   # field falls away at the edge
    spec = rng.integers(0, vb.shape[0], 40)
    vb[spec, 42, 20] = 240.0                                 # bright speckle outliers
    v2 = np.moveaxis(vb, 0, 1)

    tissue = SH.tissue_line_mask(v2, beam_axis=1)
    assert not tissue[edge, :].any(), "dim edge column admitted as tissue"
    assert tissue[:38, :].all(), "genuine tissue lines rejected"
    assert np.moveaxis(SH.shadow_field(v2, beam_axis=1)["shadow"], 1, 0)[:, edge, :].sum() == 0


def test_uniformly_dim_lines_are_not_flagged_as_shadow():
    """The failure seen on real Validation volumes: the dim lateral margin of the field of
    view is darker than the depth median at EVERY depth, so a per-depth residual flags the
    whole column. A posterior shadow needs normal tissue above the attenuator; a column
    that is already depressed in its own near field is weakly coupled, not shadowed."""
    vol, _ = make_phantom(1)
    vb = np.moveaxis(vol, 1, 0).copy()
    dim = slice(20, 26)
    vb[:, dim, :] *= 0.45                      # uniformly dim from the skin down
    v2 = np.moveaxis(vb, 0, 1)

    f = SH.shadow_field(v2, beam_axis=1)
    sh = np.moveaxis(f["shadow"], 1, 0)
    assert sh[:, dim, :].mean() < 0.05, "uniformly dim column misfiled as shadow"
    assert f["frac_weak_lines"] > 0.0
    assert not f["strong"][dim, :].any()
    # the genuine ray, which IS normal near the surface, still survives the guard
    assert sh[40:60, 9:12, 9:12].mean() > 0.8


def test_weak_line_frac_separates_dim_margin_from_shadow():
    vol, info = make_phantom(1)
    vb = np.moveaxis(vol, 1, 0).copy()
    vb[:, 20:26, :] *= 0.45
    v2 = np.moveaxis(vb, 0, 1)
    f = SH.shadow_field(v2, beam_axis=1)

    dim_box = _box_around(info["point"](45, 23, 20), 2, beam_axis=1, half_depth=8)
    ray_box = _box_around(info["shadow_centre"], 2, beam_axis=1, half_depth=8)
    feats = SH.candidate_shadow_features(f, np.array([dim_box, ray_box]), beam_axis=1)
    assert feats["weak_line_frac"][0] > 0.9 and feats["shadow_frac"][0] < 0.1
    assert feats["weak_line_frac"][1] < 0.1 and feats["shadow_frac"][1] > 0.5


def test_line_shadow_map_is_coronal_and_peaks_on_the_ray():
    """A shadow ray collapses to a POINT in the plane perpendicular to the beam."""
    vol, _ = make_phantom(1)
    f = SH.shadow_field(vol, beam_axis=1)
    ls = f["line_shadow"]
    assert ls.shape == (vol.shape[0], vol.shape[2])   # beam axis 1 removed
    assert np.nanmax(ls[8:13, 8:13]) > 0.3
    assert np.nanmean(ls[20:30, 15:25]) < 0.05        # quiet tissue elsewhere


# ----------------------------------------------------------------------------- candidate features
def test_candidate_features_separate_a_shadow_box_from_a_mass_box():
    vol, info = make_phantom(1)
    f = SH.shadow_field(vol, beam_axis=1)
    shadow_box = _box_around(info["shadow_centre"], 2, beam_axis=1, half_depth=8)
    mass_box = _box_around(info["mass_centre"], 2, beam_axis=1, half_depth=4)
    quiet_box = _box_around(info["point"](50, 22, 20), 2, beam_axis=1, half_depth=8)

    feats = SH.candidate_shadow_features(f, np.array([shadow_box, mass_box, quiet_box]),
                                         beam_axis=1)
    assert feats["shadow_frac"][0] > 0.5              # the ray
    assert feats["shadow_frac"][1] < 0.3              # the mass
    assert feats["shadow_frac"][2] < 0.05             # quiet tissue
    # both dark regions have a negative in-box residual -- darkness alone does NOT separate
    assert feats["z_mean"][0] < -1.0 and feats["z_mean"][1] < -1.0
    # but only the ray keeps the tissue below it dark
    assert feats["distal_z"][0] < feats["distal_z"][1]
    assert feats["line_shadow"][0] > feats["line_shadow"][2]
    assert feats["tissue_frac"][0] == pytest.approx(1.0)


def test_candidate_features_flag_boxes_outside_tissue():
    vol, info = make_phantom(1)
    f = SH.shadow_field(vol, beam_axis=1)
    pad_box = _box_around(info["point"](40, 47, 20), 2, beam_axis=1)
    feats = SH.candidate_shadow_features(f, np.array([pad_box]), beam_axis=1)
    assert feats["tissue_frac"][0] == pytest.approx(0.0)


def test_candidate_features_clip_out_of_range_boxes():
    vol, _ = make_phantom(1)
    f = SH.shadow_field(vol, beam_axis=1)
    huge = [-50, -50, -50, 500, 500, 500]
    feats = SH.candidate_shadow_features(f, np.array([huge]), beam_axis=1)
    assert np.isfinite(feats["shadow_frac"][0])


# ----------------------------------------------------------------------------- structure tests
def test_ray_colinearity_detects_a_shared_ray_and_is_null_on_scatter():
    rng = np.random.default_rng(0)
    # 8 candidates stacked at different depths on ONE coronal position (beam axis 1),
    # embedded in a background of coronally-scattered candidates.
    on_ray = np.zeros((8, 3))
    on_ray[:, 0] = 30 + rng.normal(0, 0.5, 8)
    on_ray[:, 2] = 20 + rng.normal(0, 0.5, 8)
    on_ray[:, 1] = np.linspace(5, 75, 8)
    bg = np.column_stack([rng.uniform(0, 60, 30), rng.uniform(0, 80, 30),
                          rng.uniform(0, 60, 30)])
    mixed = np.vstack([on_ray, bg])
    enr = SH.ray_colinearity(mixed, beam_axis=1, coronal_radius=4, min_depth_sep=10,
                             n_perm=60)["enrichment"]
    assert enr > 1.5, f"stacked ray not detected (enrichment {enr})"

    e2 = SH.ray_colinearity(bg, beam_axis=1, coronal_radius=4, min_depth_sep=10,
                            n_perm=60)["enrichment"]
    assert 0.3 < e2 < 3.0, f"scatter should be near the null (enrichment {e2})"


def test_ray_colinearity_null_would_be_degenerate_if_it_permuted_depth_only():
    """Guards the design note in the docstring: depth-only permutation detects nothing."""
    rng = np.random.default_rng(0)
    on_ray = np.zeros((8, 3))
    on_ray[:, 0] = 30.0
    on_ray[:, 2] = 20.0
    on_ray[:, 1] = np.linspace(5, 75, 8)
    # every coronal distance is 0 regardless of how depth is shuffled, so a depth-only
    # null is invariant; the implemented column-wise null is what makes the test possible
    out = SH.ray_colinearity(on_ray, beam_axis=1, coronal_radius=4, min_depth_sep=10,
                             n_perm=20)
    assert out["observed"] > 0


def test_knn_direction_anisotropy_singles_out_the_ray_axis():
    shape = (60, 80, 60)
    # 6 well-separated coronal spots, each carrying 10 candidates strung along depth
    pts = []
    for u in (10, 30, 50):
        for v in (12, 48):
            for d in np.linspace(5, 75, 10):
                pts.append([u, d, v])
    frac = SH.knn_direction_anisotropy(np.array(pts), shape=shape, k=3)
    assert sum(frac.values()) == pytest.approx(1.0)
    assert frac[1] > 0.9, f"beam axis not singled out: {frac}"


def test_knn_direction_anisotropy_is_flat_for_isotropic_scatter():
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 60, (400, 3))
    frac = SH.knn_direction_anisotropy(pts, shape=(60, 60, 60), k=3)
    assert sum(frac.values()) == pytest.approx(1.0)
    for a in (0, 1, 2):
        assert 0.25 < frac[a] < 0.42, f"isotropic scatter should be ~1/3 each: {frac}"


def test_knn_direction_anisotropy_normalises_by_axis_extent():
    """A cloud isotropic in VOXELS on an anisotropic grid must not read as anisotropic."""
    rng = np.random.default_rng(5)
    # uniform over a slab-shaped volume: normalised coords are uniform in the unit cube
    pts = np.column_stack([rng.uniform(0, 400, 400), rng.uniform(0, 100, 400),
                           rng.uniform(0, 200, 400)])
    frac = SH.knn_direction_anisotropy(pts, shape=(400, 100, 200), k=3)
    for a in (0, 1, 2):
        assert 0.25 < frac[a] < 0.42, f"extent normalisation failed: {frac}"


def test_ray_colinearity_handles_degenerate_input():
    out = SH.ray_colinearity(np.zeros((1, 3)), beam_axis=1, coronal_radius=4,
                             min_depth_sep=10)
    assert out["observed"] == 0


def test_gini_is_zero_for_uniform_and_high_for_concentrated():
    assert SH.gini(np.ones(50)) == pytest.approx(0.0, abs=1e-9)
    spike = np.zeros(50); spike[0] = 100
    assert SH.gini(spike) > 0.9
    assert np.isnan(SH.gini(np.zeros(10)))


def test_slice_concentration_finds_piled_up_slices_and_matches_a_shadow_profile():
    # all candidates on slices 10-12 of axis 2
    c = np.zeros((30, 3))
    c[:, 2] = np.repeat([10, 11, 12], 10)
    prof = np.zeros(40); prof[10:13] = 1.0
    out = SH.slice_concentration(c, axis=2, n_slices=40, shadow_profile=prof)
    assert out["gini"] > 0.85
    assert out["top10pct_share"] > 0.9
    assert out["spearman_vs_shadow"] > 0.5

    uni = np.zeros((40, 3)); uni[:, 2] = np.arange(40)
    assert SH.slice_concentration(uni, axis=2, n_slices=40)["gini"] < 0.05


def test_dominant_period_recovers_a_known_stripe_spacing():
    x = np.arange(200)
    prof = np.sin(2 * np.pi * x / 17.0)
    out = SH.dominant_period(prof, min_lag=3)
    assert abs(out["period"] - 17) <= 1
    assert out["strength"] > 0.7
    flat = np.zeros(200)
    assert np.isnan(SH.dominant_period(flat)["period"])


def test_dominant_period_reports_nan_for_a_merely_smooth_profile():
    """A smooth ramp/bump has a decaying autocorrelation and NO period. An arg-max search
    would return min_lag with strength ~1 and invent a stripe pattern that is not there."""
    x = np.linspace(-3, 3, 300)
    smooth = np.exp(-x ** 2)
    out = SH.dominant_period(smooth, min_lag=3)
    assert np.isnan(out["period"]), f"smooth bump reported a period: {out}"

    ramp = np.linspace(0, 1, 300)
    assert np.isnan(SH.dominant_period(ramp, min_lag=3)["period"])


def test_dominant_period_survives_a_periodic_signal_on_a_smooth_trend():
    """Real shadow profiles are periodic banding ON TOP of a slow envelope."""
    x = np.arange(300)
    prof = np.exp(-((x - 150) / 120.0) ** 2) + 0.5 * np.sin(2 * np.pi * x / 23.0)
    out = SH.dominant_period(prof, min_lag=3)
    assert abs(out["period"] - 23) <= 2, f"period not recovered under a trend: {out}"


def test_map_correlation_endpoints():
    rng = np.random.default_rng(1)
    a = rng.random((20, 20))
    assert SH.map_correlation(a, a)["spearman"] == pytest.approx(1.0)
    assert SH.map_correlation(a, -a)["spearman"] == pytest.approx(-1.0)
    b = a.copy(); b[0, 0] = np.nan
    assert SH.map_correlation(a, b)["n"] == 399


def test_cliffs_delta_endpoints():
    assert SH.cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert SH.cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)
    assert abs(SH.cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-9


# ----------------------------------------------------------------------------- plumbing
def test_plane_maps_and_marginals_have_the_right_shapes():
    vol, _ = make_phantom(1)
    f = SH.shadow_field(vol, beam_axis=1)
    planes = SH.plane_shadow_maps(f, beam_axis=1)
    assert set(planes) == {(0, 1), (0, 2), (1, 2)}
    assert planes[(0, 2)].shape == (vol.shape[0], vol.shape[2])   # the coronal plane
    marg = SH.axis_marginals(f, vol, beam_axis=1, n_bins=25)
    assert marg[1]["is_beam"] and not marg[0]["is_beam"]
    for a in (0, 1, 2):
        assert len(marg[a]["shadow"]) == 25


def test_centroid_density_map_places_points_in_the_right_cell():
    c = np.array([[10.0, 5.0, 30.0]])
    h = SH.centroid_density_map(c, drop_axis=1, shape=(40, 80, 60), out_shape=(4, 6),
                                sigma_bins=0)
    assert h[1, 3] == 1.0 and h.sum() == 1.0


def test_downsample_map_block_means_and_ignores_nan():
    a = np.arange(16, dtype=float).reshape(4, 4)
    d = SH.downsample_map(a, (2, 2))
    assert d[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))
    a[0, 0] = np.nan
    assert np.isfinite(SH.downsample_map(a, (2, 2))[0, 0])


def test_rebin_preserves_level():
    p = np.full(100, 3.0)
    assert np.allclose(SH._rebin(p, 10), 3.0)
