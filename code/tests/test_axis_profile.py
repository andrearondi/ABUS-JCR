"""The ``ABUS_AXIS_PROFILE`` switch — two substrates, one repo (2026-08-09).

The whole isotropic-rebuild branch (``iso/RB_ISO_REBUILD.md``) rests on one property:
selecting the ``measured`` profile changes ``SPACING_STORAGE_MM`` and therefore
``preprocess_hash``, so the two caches land in different directories and
``cache.assert_hash`` refuses to read one under the other. If that property broke, a
forgotten ``export`` would train a detector on the wrong substrate *silently* — the
exact failure mode `HISTORY.md` §8 records twice ("a cache key that names a run does not
identify a model").

**The load-bearing assertion is the legacy hash.** ``ab1fdf28…`` names the DEPLOYED iso
cache and every artefact built on it. If introducing the profile switch had perturbed the
canonical config by so much as a key, that hash would move and the deployed cache would be
orphaned. Pinned here as a literal, not re-derived.

Profiles are read from the environment at import, so each case runs in a **subprocess**:
that both avoids polluting the rest of the suite and exercises the switch exactly as the
runbook uses it (``export`` then run), rather than a reload trick the runbook never does.
"""

import json
import subprocess
import sys
import os

import pytest

# The deployed cache directory name. Recorded in RESULTS_PHASE_1.md [1.2]/[1.3] and in
# RESULTS_PHASE_4.md [4.0b]. This literal is the tripwire; do not compute it.
DEPLOYED_PREPROCESS_HASH = "ab1fdf287c86bafabd24030f9cac23c7545d32bf999d29218fc6591da388e702"

_PROBE = r"""
import json
from abus_jcr import conventions as C
from abus_jcr.preprocess import preprocess_hash, zoom_factors, iso_shape
print(json.dumps({
    "profile": C.AXIS_PROFILE,
    "spacing": list(C.SPACING_STORAGE_MM),
    "hash": preprocess_hash(),
    "zoom": list(zoom_factors()),
    "iso_865_682_353": list(iso_shape((865, 682, 353))),
    "depth_axis": C.DEPTH_AXIS,
    "lateral_axis": C.LATERAL_AXIS,
    "slice_axis": C.SLICE_AXIS,
    "aniso_axis": C.FP_PROBE_ANISO_DEPTH_AXIS,
    "iso_spacing_mm": C.ISO_SPACING_MM,
    "min_size": C.DET_MIN_SIZE,
    "max_size": C.DET_MAX_SIZE,
    "aspects": list(C.DET_ANCHOR_ASPECT_RATIOS),
    "bases": list(C.DET_ANCHOR_BASE_SIZES),
    "merge_gap": C.DET_LABEL_MERGE_GAP,
    "min_tube_len": C.LINK_MIN_TUBE_LEN,
    "zspan_cap": C.LINK_MAX_TUBE_ZSPAN,
    "drift_cap": C.LINK_MAX_CENTROID_DRIFT,
    "floor": C.PREFILTER_SCORE_FLOOR,
    "op": C.LINK_OP_SCORE_THRESH,
    "cpm_tol": C.DET_SELECT_CPM_TOL,
    "pool_budget": C.RESCORER_POOL_BUDGET,
    "crop": [C.RESC_CROP_OUT, C.RESC_CROP_MIN_SIDE, C.RESC_CROP_MAX_SIDE],
}))
"""


def _probe(profile=None):
    """Import the package in a fresh process under ``profile`` and return its constants."""
    env = dict(os.environ)
    env.pop("ABUS_AXIS_PROFILE", None)
    if profile is not None:
        env["ABUS_AXIS_PROFILE"] = profile
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def legacy():
    return _probe(None)          # unset == the deployed default


@pytest.fixture(scope="module")
def measured():
    return _probe("measured")


# --- the deployed arm must be bit-for-bit what it always was -----------------

def test_default_profile_is_legacy(legacy):
    assert legacy["profile"] == "legacy"


def test_legacy_preprocess_hash_is_the_deployed_cache(legacy):
    """THE tripwire. This hash names the on-disk cache all 8 detectors trained from."""
    assert legacy["hash"] == DEPLOYED_PREPROCESS_HASH


def test_explicit_legacy_equals_the_default(legacy):
    assert _probe("legacy") == legacy


def test_legacy_keeps_the_inverted_spacing_map(legacy):
    # Deliberately NOT corrected: correcting it in place would invalidate the cache.
    assert tuple(legacy["spacing"]) == (0.073, 0.200, 0.475674)
    assert tuple(legacy["iso_865_682_353"]) == (158, 341, 420)


def test_legacy_detector_and_linker_constants_unchanged(legacy):
    assert (legacy["min_size"], legacy["max_size"]) == (160, 352)
    assert tuple(legacy["aspects"]) == (0.2, 0.25, 0.33, 1.0)
    assert tuple(legacy["bases"]) == (16, 32, 64, 128, 256)
    assert legacy["merge_gap"] == 8
    assert (legacy["min_tube_len"], legacy["zspan_cap"], legacy["drift_cap"]) == (8, 182, 342)
    assert (legacy["floor"], legacy["op"]) == (0.08, 0.03)


def test_legacy_anisotropy_axis_reproduces_every_recorded_number(legacy):
    """d0 — the axis every recorded `anisotropy` figure was computed on.

    It is the WRONG axis physically (`DEPTH_AXIS` is 1), and that is the point: the
    constant's only job on this profile is byte-reproduction of the record.
    """
    assert legacy["aniso_axis"] == 0
    assert legacy["depth_axis"] == 1        # the physical fact, stated even here


# --- the isotropic branch ----------------------------------------------------

def test_measured_uses_the_measured_spacing_map(measured):
    """results/AXIS_CHECK.md: d0 = lateral 0.200, d1 = depth/beam 0.073, d2 = sweep."""
    assert tuple(measured["spacing"]) == (0.200, 0.073, 0.475674)


#: The physical truth, measured on 129/130 volumes (results/AXIS_CHECK.md). What a cache
#: voxel actually spans is TRUE_SPACING[a] / zoom[a] — the zoom is applied with whatever map
#: the profile DECLARES, so declaring the wrong one does not change physics, only the grid.
TRUE_SPACING_MM = (0.200, 0.073, 0.475674)


def _voxel_mm(probe):
    return [t / f for t, f in zip(TRUE_SPACING_MM, probe["zoom"])]


def test_measured_cache_is_genuinely_isotropic(measured):
    """One cache voxel really is ISO_SPACING_MM on every axis — the Inv. 6 property the
    deployed cache does not have."""
    iso = measured["iso_spacing_mm"]
    assert _voxel_mm(measured) == pytest.approx([iso, iso, iso])


def test_legacy_cache_is_NOT_isotropic(legacy):
    """The defect, asserted rather than described, so it cannot be quietly forgotten.

    Note the direction: the deployed cache spends 0.146 mm on the 0.073 mm DEPTH axis and
    1.096 mm on the 0.200 mm LATERAL axis — it throws away lateral detail it had and keeps
    depth detail it did not need.
    """
    voxel_mm = _voxel_mm(legacy)
    assert voxel_mm == pytest.approx([1.09589, 0.14600, 0.40000], abs=1e-4)
    assert max(voxel_mm) / min(voxel_mm) == pytest.approx(7.5068, abs=1e-3)
    # the in-plane aspect error quoted throughout INVARIANTS.md Inv. 6 / AXIS_CHECK.md §5
    assert (0.200 / 0.073) ** 2 == pytest.approx(7.5068, abs=1e-3)


def test_measured_hash_differs_so_the_caches_cannot_collide(legacy, measured):
    assert measured["hash"] != legacy["hash"]


def test_measured_frame_shape_inverts_at_the_same_voxel_budget(legacy, measured):
    """432 x 124 x 420 against 158 x 341 x 420: 2.7x finer laterally for ~the same voxels,
    which is why the rebuild costs no extra detector training time."""
    assert tuple(measured["iso_865_682_353"]) == (432, 124, 420)
    n_leg = legacy["iso_865_682_353"][0] * legacy["iso_865_682_353"][1]
    n_new = measured["iso_865_682_353"][0] * measured["iso_865_682_353"][1]
    assert abs(n_new - n_leg) / n_leg < 0.02


def test_measured_slice_axis_and_sweep_are_untouched(legacy, measured):
    """d2 = sweep was always correct, so Inv. 1 (2.5D axial-only) is unaffected and the
    per-volume slice count does not move — the reason z-denominated linker constants are
    expected to re-derive unchanged."""
    assert measured["slice_axis"] == legacy["slice_axis"] == 2
    assert measured["spacing"][2] == legacy["spacing"][2]
    assert measured["iso_865_682_353"][2] == legacy["iso_865_682_353"][2] == 420


def test_measured_anisotropy_finally_measures_the_beam_axis(measured):
    assert measured["aniso_axis"] == measured["depth_axis"] == 1


def test_measured_input_size_follows_the_inverted_frame(measured):
    """Provisional, reconciled at the [I2.1] gate — but it must at least be self-consistent:
    min_size is the SHORT side (124 -> 128), max_size the LONG one (432 -> 448)."""
    assert measured["min_size"] < measured["max_size"]
    assert (measured["min_size"], measured["max_size"]) == (128, 448)


def test_measured_anchor_aspects_are_no_longer_wide_skewed(measured):
    """The deployed (0.2, 0.25, 0.33, 1.0) is the 7.5x in-plane distortion, not morphology."""
    assert min(measured["aspects"]) >= 1.0


# --- what must NOT differ between profiles -----------------------------------

@pytest.mark.parametrize("key", ["iso_spacing_mm", "cpm_tol", "pool_budget", "crop",
                                 "depth_axis", "lateral_axis", "slice_axis"])
def test_policy_and_physical_facts_are_profile_independent(legacy, measured, key):
    """ISO_SPACING_MM is held fixed so the spacing MAP is the only changed variable;
    DET_SELECT_CPM_TOL and RESCORER_POOL_BUDGET are selection policy (Inv. 2's "identical
    selection policy"); RESC_CROP_* are voxel counts whose MEANING changes while the number
    does not; and the axis roles are facts about the scanner, not about the cache."""
    assert legacy[key] == measured[key]


def test_unknown_profile_refuses_to_import():
    env = dict(os.environ, ABUS_AXIS_PROFILE="isotropic")   # a plausible typo
    r = subprocess.run([sys.executable, "-c", "import abus_jcr.conventions"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "ABUS_AXIS_PROFILE" in r.stderr


def test_profile_name_is_case_and_whitespace_insensitive():
    assert _probe(" Measured ")["profile"] == "measured"
