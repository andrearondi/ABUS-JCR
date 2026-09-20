"""The factor grid's pairs move exactly ONE switch (rescore/factor_grid, 2026-09-20).

The pair this module exists to retire, FULL-P vs B1-P, was read as "the set module with the
objective fixed" while it also switches the geometry term on. The switches are read from
``VARIANTS``, never from a rung's name, so a mislabelled pair cannot pass.
"""

import numpy as np
import pytest

from abus_jcr.rescore import factor_grid as G
from abus_jcr.rescore.variants import VARIANTS


def test_grid_is_complete_and_made_of_trained_rungs():
    assert set(G.GRID) == {(a, o) for a in G.ARCH for o in G.OBJ}
    assert len(set(G.GRID.values())) == 6
    assert all(cid in VARIANTS for cid in G.GRID.values())
    assert all(cid not in VARIANTS for cid in G.REFERENCES.values())


def test_coordinates_match_the_switches():
    for (arch, obj), cid in G.GRID.items():
        s = G.switches(cid)
        assert s["jointness"] == (arch != "Independent")
        assert s["geometry"] == (arch == "Joint+Geo")
        assert s["objective"] == obj


@pytest.mark.parametrize("pair", G.FACTOR_PAIRS, ids=lambda p: f"{p['a']}-{p['b']}")
def test_every_factor_pair_moves_exactly_its_own_switch(pair):
    assert G.differing_switches(pair["a"], pair["b"]) == [pair["factor"]]


def test_factor_pairs_are_all_the_one_switch_pairs_of_the_grid():
    ids = list(G.GRID.values())
    one_switch = {frozenset((a, b)) for a in ids for b in ids
                  if a != b and len(G.differing_switches(a, b)) == 1}
    assert {frozenset((p["a"], p["b"])) for p in G.FACTOR_PAIRS} == one_switch


def test_the_retired_pair_is_not_a_jointness_pair():
    assert G.differing_switches("FULL-P", "B1-P") == ["jointness", "geometry"]
    assert ("FULL-P", "B1-P") in G.APPENDIX_PAIRS
    assert all((p["a"], p["b"]) != ("FULL-P", "B1-P") for p in G.FACTOR_PAIRS)


def test_boot_conditions_cover_every_reported_pair():
    need = {x for p in G.FACTOR_PAIRS + (G.OVERALL,) for x in (p["a"], p["b"])}
    assert need == set(G.BOOT_CONDITIONS)


def test_display_names_are_the_coordinates():
    assert G.DISPLAY["FULL-P"] == "Joint+Geo/pooled"
    assert G.DISPLAY["B1"] == "Independent/CE"
    assert G.DISPLAY["B0"] == "Detector" and G.DISPLAY["B0-rank"] == "Rank rule"


def test_pair_stats_envelope_brackets_the_official_read():
    rng = np.random.default_rng(0)
    a, b = rng.random(200), rng.random(200)
    w = 0.01
    ps = G.pair_stats(a, b, lo_a=a - w, hi_a=a + w, lo_b=b - w, hi_b=b + w)
    for k in ("lo", "hi", "frac_positive"):
        assert ps["envelope"][k][0] <= ps[k] <= ps["envelope"][k][1]
