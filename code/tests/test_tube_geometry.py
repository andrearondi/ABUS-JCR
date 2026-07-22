"""[P3U2 3.D] Cross-slice tube-geometry features (torch-free).

Two SOFT rescorer cues (never linker gates), kept after the [P3U2.diag] validation:
``centroid_jitter`` (steadier tube = lower; TP < FP) and ``area_cv`` (a lesion's cross-section
grows then shrinks, a shadow stays constant; TP > FP). ``area_peak_pos`` and ``area_monotonicity``
were PRUNED (no signal / length-confounded).
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
    assert TUBE_GEOM_COLUMNS == ["centroid_jitter", "area_cv"]


def test_compact_lesion_high_area_cv_steady_centre():
    # steady centre (5,5); half-sizes 1,2,3,2,1 -> areas 4,16,36,16,4 (lesion cross-section)
    tube = [_sq(5, 5, h, z) for z, h in enumerate([1, 2, 3, 2, 1])]
    s = tube_geometry_stats(tube)
    assert s["centroid_jitter"] == pytest.approx(0.0)          # centre never moves
    assert s["area_cv"] > 0.5                                  # strong cross-slice size change


def test_shadow_constant_footprint_low_area_cv():
    # steady centre, constant area -> low CV (shadow-like)
    tube = [_sq(5, 5, 2, z) for z in range(4)]
    s = tube_geometry_stats(tube)
    assert s["area_cv"] == pytest.approx(0.0)
    assert s["centroid_jitter"] == pytest.approx(0.0)


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
    assert set(s.keys()) == set(TUBE_GEOM_COLUMNS)
    with pytest.raises(ValueError):
        tube_geometry_stats([])
