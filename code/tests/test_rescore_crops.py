"""[4.1] The Phase-4 3D crop — the classic silent coordinate bug (Inv. 5, 6).

Torch-free: the crop is pure numpy/scipy, so this whole module runs on the laptop.
The negative control (applying ``PERM_STORAGE_TO_ITK`` first) is what makes the
centring assertion discriminative rather than vacuous — it reproduces the spec's
"Check 1" table, where the permuted path returned an empty crop in 5/6 Val cases.
"""

import json

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.preprocess import preprocess_hash
from abus_jcr.rescore.crops import (
    build_crop_cache,
    crop_hash,
    extract_crop,
    open_crop_cache,
    pad_fraction,
    roi_side_iso,
    sample_grid,
)


# --------------------------------------------------------------------------- helpers
def _sphere_volume(shape=(60, 70, 80), centre=(30, 35, 40), radius=6.0):
    """A float32 [0,1] volume holding one bright sphere (storage order d0,d1,d2)."""
    d0, d1, d2 = np.ogrid[: shape[0], : shape[1], : shape[2]]
    r2 = (d0 - centre[0]) ** 2 + (d1 - centre[1]) ** 2 + (d2 - centre[2]) ** 2
    return (r2 <= radius**2).astype(np.float32)


def _centre_of_mass(crop):
    tot = float(crop.sum())
    if tot <= 0:
        return None
    idx = np.indices(crop.shape, dtype=float)
    return np.array([float((idx[a] * crop).sum() / tot) for a in range(3)])


# --------------------------------------------------------------------------- roi_side_iso
def test_roi_side_is_context_times_max_extent_between_the_clamps():
    # max ext 40 -> 1.5 * 40 = 60, inside [48, 160]
    assert roi_side_iso(10.0, 40.0, 20.0) == pytest.approx(60.0)


def test_roi_side_clamps_at_min_and_max():
    assert roi_side_iso(1.0, 2.0, 3.0) == pytest.approx(float(C.RESC_CROP_MIN_SIDE))
    assert roi_side_iso(300.0, 10.0, 10.0) == pytest.approx(float(C.RESC_CROP_MAX_SIDE))


def test_roi_side_is_cubic_not_per_axis():
    """A thin depth-column and a fat blob with the same max extent get the SAME side, so a
    shadow column still LOOKS like a shadow column in the crop (spec §4.1 rationale)."""
    assert roi_side_iso(80.0, 4.0, 4.0) == roi_side_iso(80.0, 80.0, 80.0)


# --------------------------------------------------------------------------- sample grid
def test_sample_grid_is_the_cell_centred_grid_over_the_roi():
    g = sample_grid(cen=10.0, side=4.0, out=4)          # lo = 8, step = 1 -> 8.5, 9.5, 10.5, 11.5
    np.testing.assert_allclose(g, [8.5, 9.5, 10.5, 11.5])
    assert g.mean() == pytest.approx(10.0)              # the grid is centred on `cen`


# --------------------------------------------------------------------------- the coordinate contract
def test_crop_centres_the_lesion_in_storage_order():
    vol = _sphere_volume()
    crop = extract_crop(vol, 30.0, 35.0, 40.0, 12.0, 12.0, 12.0)
    assert crop.shape == (C.RESC_CROP_OUT,) * 3
    com = _centre_of_mass(crop)
    assert com is not None, "correct crop must contain the sphere"
    grid_centre = (C.RESC_CROP_OUT - 1) / 2.0
    assert np.linalg.norm(com - grid_centre) < 2.0


def test_permuted_crop_fails_the_same_centring_assertion():
    """NEGATIVE CONTROL. Applying PERM_STORAGE_TO_ITK before indexing the iso volume is the
    silent bug this test exists to catch; it must NOT pass the assertion above."""
    vol = _sphere_volume()
    p = C.PERM_STORAGE_TO_ITK
    cen = (30.0, 35.0, 40.0)
    ext = (12.0, 12.0, 12.0)
    crop = extract_crop(vol, *[cen[p[a]] for a in range(3)], *[ext[p[a]] for a in range(3)])
    com = _centre_of_mass(crop)
    grid_centre = (C.RESC_CROP_OUT - 1) / 2.0
    offset = np.inf if com is None else float(np.linalg.norm(com - grid_centre))
    assert offset > 5.0, f"permuted crop must be badly off-centre, got {offset:.2f}"


def test_outside_the_volume_is_zero_padded_never_edge_replicated():
    vol = np.full((40, 40, 40), 0.7, dtype=np.float32)
    # centre the ROI far outside the volume: every sample lands outside -> all zeros
    crop = extract_crop(vol, -200.0, -200.0, -200.0, 10.0, 10.0, 10.0)
    assert float(crop.max()) == pytest.approx(C.RESC_CROP_PAD_VALUE)


def test_crop_values_stay_in_the_cache_contract_range():
    rng = np.random.default_rng(0)
    vol = rng.random((50, 50, 50)).astype(np.float32)
    crop = extract_crop(vol, 25.0, 25.0, 25.0, 30.0, 30.0, 30.0)
    assert crop.dtype == np.float32
    assert float(crop.min()) >= 0.0 and float(crop.max()) <= 1.0


def test_pad_fraction_is_zero_inside_and_one_outside():
    shape = (100, 100, 100)
    assert pad_fraction(shape, (50.0, 50.0, 50.0), 48.0, 48) == pytest.approx(0.0)
    assert pad_fraction(shape, (-500.0, -500.0, -500.0), 48.0, 48) == pytest.approx(1.0)


def test_pad_fraction_matches_the_measured_out_of_volume_share():
    """ROI deeper than the volume's d0 (the RESC_CROP_MAX_SIDE=160 vs 158-slice depth case)."""
    shape = (158, 300, 300)
    frac = pad_fraction(shape, (79.0, 150.0, 150.0), 160.0, 48)
    assert 0.0 < frac < 0.5


# --------------------------------------------------------------------------- crop_hash
def test_crop_hash_changes_with_every_crop_constant(monkeypatch):
    base = crop_hash()
    for name, new in [("RESC_CROP_OUT", 64), ("RESC_CROP_CONTEXT", 2.0),
                      ("RESC_CROP_MIN_SIDE", 32), ("RESC_CROP_MAX_SIDE", 192),
                      ("RESC_CROP_INTERP", 3), ("RESC_CROP_PAD_VALUE", 0.5)]:
        monkeypatch.setattr(C, name, new)
        assert crop_hash() != base, f"crop_hash ignored {name}"
        monkeypatch.undo()


def test_crop_hash_changes_with_the_preprocess_hash(monkeypatch):
    """A crop cut from a different iso cache must land in a different directory."""
    import abus_jcr.rescore.crops as crops_mod
    base = crop_hash()
    monkeypatch.setattr(crops_mod, "preprocess_hash", lambda *a, **k: "a-different-iso-cache")
    assert crop_hash() != base


def test_crop_hash_changes_when_a_real_cache_invalidating_constant_moves(monkeypatch):
    base = crop_hash()
    monkeypatch.setattr(C, "INTENSITY_NORM", {"method": "scale", "divisor": 1.0})
    assert crop_hash() != base


# --------------------------------------------------------------------------- the crop cache
def _record(n_by_pid, phash=None):
    phash = phash or preprocess_hash()
    rows = []
    for pid, n in n_by_pid.items():
        for k in range(n):
            rows.append({"public_id": pid, "candidate_id": f"fold0:{pid}:{k}",
                         "detector_of_origin": "fold0",
                         "cen_d0": 20.0 + k, "cen_d1": 25.0, "cen_d2": 30.0,
                         "ext_d0": 12.0, "ext_d1": 12.0, "ext_d2": 12.0,
                         "preprocess_hash": phash})
    return pd.DataFrame(rows)


def test_crop_cache_row_order_matches_the_record_row_order(tmp_path):
    rec = _record({7: 3, 9: 2})
    vols = {7: _sphere_volume(centre=(20, 25, 30)), 9: _sphere_volume(centre=(21, 25, 30))}
    stats = build_crop_cache(rec, cache_root=None, out_dir=tmp_path, split="train",
                             vol_loader=lambda _root, pid: vols[int(pid)], progress=False)
    crops, meta = open_crop_cache(tmp_path, "train")
    assert crops.shape == (len(rec), C.RESC_CROP_OUT, C.RESC_CROP_OUT, C.RESC_CROP_OUT)
    assert meta["crop_hash"] == crop_hash() and meta["preprocess_hash"] == preprocess_hash()
    assert meta["n_rows"] == len(rec) == stats["n_rows"]
    idx = pd.read_csv(tmp_path / crop_hash() / "train" / "index.csv")
    assert list(idx["candidate_id"]) == list(rec["candidate_id"])
    assert list(idx["row"]) == list(range(len(rec)))
    # row k really is candidate k: re-extract row 0 directly and compare
    direct = extract_crop(vols[7], 20.0, 25.0, 30.0, 12.0, 12.0, 12.0)
    np.testing.assert_allclose(np.asarray(crops[0], dtype=np.float32), direct.astype(np.float16), atol=1e-3)


def test_crop_cache_reports_max_set_size_and_roi_sides(tmp_path):
    rec = _record({7: 3, 9: 2})
    vols = {7: _sphere_volume(), 9: _sphere_volume()}
    stats = build_crop_cache(rec, cache_root=None, out_dir=tmp_path, split="train",
                             vol_loader=lambda _root, pid: vols[int(pid)], progress=False)
    assert stats["max_set_size"] == 3          # SET = (detector_of_origin, public_id)
    assert stats["roi_side_min"] == pytest.approx(float(C.RESC_CROP_MIN_SIDE))
    assert 0.0 <= stats["pad_frac_median"] <= 1.0


def test_crop_cache_refuses_a_record_from_a_different_preprocess_hash(tmp_path):
    rec = _record({7: 2}, phash="deadbeef")
    with pytest.raises(ValueError, match="preprocess_hash"):
        build_crop_cache(rec, cache_root=None, out_dir=tmp_path, split="train",
                         vol_loader=lambda _root, pid: _sphere_volume(), progress=False)


def test_crop_cache_meta_is_json_and_names_the_directory(tmp_path):
    rec = _record({7: 2})
    build_crop_cache(rec, cache_root=None, out_dir=tmp_path, split="train",
                     vol_loader=lambda _root, pid: _sphere_volume(), progress=False)
    p = tmp_path / crop_hash() / "train" / "CROP_META.json"
    assert p.exists()
    meta = json.loads(p.read_text())
    assert meta["config"]["RESC_CROP_OUT"] == C.RESC_CROP_OUT
