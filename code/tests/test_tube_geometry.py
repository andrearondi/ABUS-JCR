"""[P3U2 3.D] Cross-slice tube-geometry features (torch-free).

A compact lesion's axial cross-section rises to a centred peak then falls (steady
centre, high area_cv, area_peak_pos ~0.5, area_monotonicity 1.0); a posterior shadow
keeps a ~constant footprint (low area_cv, unimodal/flat -> 1.0); an erratic FP has a
multi-peak area profile (area_monotonicity < 1.0). These are SOFT rescorer features,
never linker gates.
"""

import numpy as np
import pytest

from abus_jcr.link.aggregate import tube_geometry_stats, TUBE_GEOM_COLUMNS


def _sq(cx, cy, h, z, score=0.5):
    """A square box of half-size h centred at (cx, cy) on slice z -> tube member."""
    return (int(z), (cx - h, cy - h, cx + h, cy + h), float(score))


def test_columns_constant_matches_keys():
    tube = [_sq(5, 5, 1, 0), _sq(5, 5, 2, 1)]
    s = tube_geometry_stats(tube)
    assert set(s.keys()) == set(TUBE_GEOM_COLUMNS)
    assert TUBE_GEOM_COLUMNS == ["centroid_jitter", "area_cv", "area_peak_pos", "area_monotonicity"]


def test_compact_lesion_profile():
    # steady centre (5,5); half-sizes 1,2,3,2,1 -> areas 4,16,36,16,4 (unimodal, centred)
    tube = [_sq(5, 5, h, z) for z, h in enumerate([1, 2, 3, 2, 1])]
    s = tube_geometry_stats(tube)
    assert s["centroid_jitter"] == pytest.approx(0.0)          # centre never moves
    assert s["area_peak_pos"] == pytest.approx(0.5)            # peak at index 2 of 0..4
    assert s["area_monotonicity"] == pytest.approx(1.0)        # single peak
    assert s["area_cv"] > 0.5                                  # strong cross-slice size change


def test_shadow_constant_footprint():
    # steady centre, constant area -> low CV, unimodal/flat score 1.0
    tube = [_sq(5, 5, 2, z) for z in range(4)]
    s = tube_geometry_stats(tube)
    assert s["area_cv"] == pytest.approx(0.0)
    assert s["area_monotonicity"] == pytest.approx(1.0)
    assert s["centroid_jitter"] == pytest.approx(0.0)


def test_erratic_multipeak_lowers_monotonicity():
    # areas 4,36,4,36,4 -> multiple sign changes in the area profile
    tube = [_sq(5, 5, h, z) for z, h in enumerate([1, 3, 1, 3, 1])]
    s = tube_geometry_stats(tube)
    assert s["area_monotonicity"] < 1.0


def test_centroid_jitter_detects_wander():
    # same sizes, but the centre walks in-plane -> non-zero jitter
    steady = [_sq(5, 5, 2, z) for z in range(4)]
    wander = [_sq(5 + 3 * z, 5, 2, z) for z in range(4)]
    assert tube_geometry_stats(steady)["centroid_jitter"] == pytest.approx(0.0)
    assert tube_geometry_stats(wander)["centroid_jitter"] > tube_geometry_stats(steady)["centroid_jitter"]


def test_single_member_and_empty():
    s = tube_geometry_stats([_sq(5, 5, 2, 7)])
    assert s["centroid_jitter"] == pytest.approx(0.0)
    assert s["area_cv"] == pytest.approx(0.0)
    assert s["area_peak_pos"] == pytest.approx(0.5)
    assert s["area_monotonicity"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tube_geometry_stats([])
