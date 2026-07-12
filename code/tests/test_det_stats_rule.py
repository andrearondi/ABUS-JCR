"""The pinned [2.0] derivation rule (data-independent).

``derive_constants`` maps Train iso-space stats -> the data-dependent detector
constants (input size, normalisation, anchors). Pinned on synthetic stats so the
rule is verified without touching any split (the actual Train run reconciles the
provisional conventions.py (B) block against this same rule).
"""

import numpy as np

from abus_jcr.detect import det_stats as DS
from abus_jcr import conventions as C


def _wide_skewed_stats():
    """A wide-and-short lesion distribution (the ABUS iso-zoom consequence)."""
    return {
        "frame_d0_max": 158,      # -> round_up(.,32) = 160
        "frame_d1_max": 341,      # -> round_up(.,32) = 352
        "intensity_mean": 0.23,
        "intensity_std": 0.16,
        "diag_pct": {"1": 16.0, "99": 175.0},
        # h/w percentiles: wide-skewed. Snap to grid, union {1.0}, dedupe, sort.
        "aspect_pct": {"10": 0.24, "50": 0.5, "90": 0.9},
    }


def test_round_up_helper():
    assert DS.round_up(158, 32) == 160
    assert DS.round_up(341, 32) == 352
    assert DS.round_up(160, 32) == 160  # already a multiple -> unchanged
    assert DS.round_up(1, 32) == 32


def test_input_size_from_frame_maxima():
    d = DS.derive_constants(_wide_skewed_stats(), rule=C.DET_RULE)
    assert d["min_size"] == 160
    assert d["max_size"] == 352


def test_normalisation_passthrough():
    d = DS.derive_constants(_wide_skewed_stats(), rule=C.DET_RULE)
    assert d["image_mean"] == 0.23
    assert d["image_std"] == 0.16


def test_anchor_bases_geometric_ratio2_from_diag():
    d = DS.derive_constants(_wide_skewed_stats(), rule=C.DET_RULE)
    # lo=16 -> b0 = 2**round(log2 16) = 16; 5 levels ratio 2; top*2^(2/3) >= 175.
    assert d["anchor_base_sizes"] == (16, 32, 64, 128, 256)


def test_anchor_aspect_ratios_snapped_unioned_deduped_sorted():
    d = DS.derive_constants(_wide_skewed_stats(), rule=C.DET_RULE)
    # 0.24->0.25, 0.5->0.5, 0.9->1.0; union {1.0}; dedupe the duplicate 1.0; sort.
    assert d["anchor_aspect_ratios"] == (0.25, 0.5, 1.0)


def test_anchor_ladder_grows_to_cover_large_diag():
    stats = _wide_skewed_stats()
    stats["diag_pct"] = {"1": 16.0, "99": 500.0}  # top of fixed ladder (256) can't cover
    d = DS.derive_constants(stats, rule=C.DET_RULE)
    # ladder shifts up one octave so base_max * 2^(2/3) >= 500.
    assert d["anchor_base_sizes"] == (32, 64, 128, 256, 512)
    assert d["anchor_base_sizes"][-1] * 2 ** (2 / 3) >= 500.0


def test_anchor_min_base_floor_keeps_small_level():
    # [P2-UPDATE B3] After the union-box regen the fragment diag-p1 floor is gone, so
    # p1 rises (e.g. 40). The anchor_min_base=16 floor must keep a small base so the
    # small-lesion tail retains coverage.
    stats = _wide_skewed_stats()
    stats["diag_pct"] = {"1": 40.0, "99": 175.0}
    d = DS.derive_constants(stats, rule=C.DET_RULE)      # DET_RULE["anchor_min_base"] == 16
    assert d["anchor_base_sizes"] == (16, 32, 64, 128, 256)
    # without the floor, the small base is lost (b0 = 2**round(log2 40) = 32):
    no_floor = dict(C.DET_RULE, anchor_min_base=4096)
    assert DS.derive_constants(stats, rule=no_floor)["anchor_base_sizes"][0] == 32


def test_returns_exactly_the_six_pinned_keys():
    d = DS.derive_constants(_wide_skewed_stats(), rule=C.DET_RULE)
    assert set(d) == {
        "min_size", "max_size", "image_mean", "image_std",
        "anchor_base_sizes", "anchor_aspect_ratios",
    }
