"""Phase 5 — the test-split seams (PHASE_5_SPEC §5.1/§5.2/§5.10; written test-first 2026-09-05).

What is pinned here, and why it matters:

* **Inv. 9 is enforced in code, not convention** — every path that can read the Test record
  refuses without ``--phase5-execute``, and the eval grid refuses BEFORE anything is loaded.
* **``reanchor``** — every ``variants/*.json`` records ``deployed.selected_ckpt``/``dir`` as
  ABSOLUTE paths of the machine that trained them (Berzelius ``/proj/...``). The §13-M MAIA
  route would have died on the first checkpoint load; re-anchoring at the one segment every
  Phase-4 artefact shares (``outputs_iso/phase4/``) fixes it, and the identity case pins that
  same-machine behaviour cannot change.
* **``eval_split`` threading** — ``load_variant_inputs`` keeps its historical ``rec_va``/``Zva``/
  ``gt_va`` keys whatever split it evaluates, so every consumer works unchanged on test.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _phase4_common as PC  # noqa: E402


def _rec(pids, det="full_seed0"):
    n = len(pids)
    return pd.DataFrame({
        "public_id": pids, "detector_of_origin": [det] * n, "label": ["pos"] * n,
        "score_max": [0.5] * n,
        "coordX": [1.0] * n, "coordY": [1.0] * n, "coordZ": [1.0] * n,
        "x_length": [2.0] * n, "y_length": [2.0] * n, "z_length": [2.0] * n,
    })


# ----------------------------------------------------------------- load_record guard
def test_load_record_refuses_test_without_the_flag():
    args = SimpleNamespace(phase3_out="/nowhere", record_suffix="")
    with pytest.raises(SystemExit, match="phase5-execute"):
        PC.load_record(args, "test")


def test_load_record_reads_test_with_the_flag(monkeypatch, tmp_path):
    seen = {}

    def fake_read(base):
        seen["base"] = str(base)
        return _rec([130, 131])

    monkeypatch.setattr(PC, "read_candidate_record", fake_read)
    args = SimpleNamespace(phase3_out=str(tmp_path), record_suffix="", phase5_execute=True)
    out = PC.load_record(args, "test")
    assert len(out) == 2
    assert seen["base"].endswith("candidates_test")


def test_load_record_still_refuses_an_unknown_split():
    args = SimpleNamespace(phase3_out="/nowhere", record_suffix="", phase5_execute=True)
    with pytest.raises(SystemExit, match="unknown split"):
        PC.load_record(args, "banana")


# ----------------------------------------------------------------- phase4_root / reanchor
def test_phase4_root_prefers_variants_root_and_falls_back_to_out_root():
    assert PC.phase4_root(SimpleNamespace(out_root="/o", variants_root="/v")) == Path("/v")
    assert PC.phase4_root(SimpleNamespace(out_root="/o", variants_root=None)) == Path("/o")
    assert PC.phase4_root(SimpleNamespace(out_root="/o")) == Path("/o")


def test_reanchor_rebases_a_foreign_absolute_path(tmp_path):
    root = tmp_path / "p4"
    ck = root / "variants" / "FULL_seed0_trial1" / "epoch02.pt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")
    args = SimpleNamespace(out_root="/nowhere", variants_root=str(root))
    recorded = ("/proj/berzbiomedicalimagingkth/users/x_andro/Andre2/"
                "outputs_iso/phase4/variants/FULL_seed0_trial1/epoch02.pt")
    assert PC.reanchor(args, recorded) == ck


def test_reanchor_is_the_identity_on_an_existing_path(tmp_path):
    p = tmp_path / "a.pt"
    p.write_bytes(b"x")
    args = SimpleNamespace(out_root="/nowhere", variants_root=str(tmp_path / "unused"))
    assert PC.reanchor(args, p) == p


def test_reanchor_refuses_when_neither_location_exists(tmp_path):
    args = SimpleNamespace(out_root="/nowhere", variants_root=str(tmp_path))
    with pytest.raises(SystemExit, match="re-anchored"):
        PC.reanchor(args, "/proj/x/outputs_iso/phase4/variants/missing.pt")
    with pytest.raises(SystemExit, match="outputs_iso/phase4"):
        PC.reanchor(args, "/some/unrelated/path.pt")


# ----------------------------------------------------------------- load_variant_inputs
def test_load_variant_inputs_threads_the_eval_split(monkeypatch):
    calls = []

    def fake_load_record(args, split):
        calls.append(split)
        return _rec([1, 2], det="fold0") if split == "train" else _rec([130, 131])

    monkeypatch.setattr(PC, "load_record", fake_load_record)
    monkeypatch.setattr(PC, "iso_shape_map",
                        lambda a, r: {int(p): (10, 10, 10) for p in r["public_id"].unique()})
    monkeypatch.setattr(PC, "load_gt",
                        lambda a, s: pd.DataFrame({"public_id": [130], "coordX": [1.0]}))
    monkeypatch.setattr(PC, "gt_for_pool", lambda gt, rec: gt)
    monkeypatch.setattr(
        PC, "build_features",
        lambda rec, emb, ub, shapes, stats=None: (np.zeros((len(rec), 3), np.float32),
                                                  ["a", "b", "c"], stats or {"fit": True}))
    args = SimpleNamespace(phase5_execute=True)
    out = PC.load_variant_inputs(args, 0, ("abs_geom",), eval_split="test")
    assert calls == ["train", "test"]
    assert sorted(out["rec_va"]["public_id"]) == [130, 131]   # the EVAL split, keys unchanged
    assert out["d_in"] == 3


def test_load_variant_inputs_refuses_appearance_off_val(monkeypatch):
    monkeypatch.setattr(PC, "load_record",
                        lambda a, s: _rec([1], det="fold0") if s == "train" else _rec([130]))
    args = SimpleNamespace(phase5_execute=True)
    with pytest.raises(SystemExit, match="train/val only"):
        PC.load_variant_inputs(args, 0, ("appearance", "rank"), eval_split="test")


# ----------------------------------------------------------------- the eval grid entry point
def test_eval_grid_refuses_test_without_the_flag(monkeypatch):
    import phase4_eval_grid as G

    monkeypatch.setattr(sys, "argv", ["phase4_eval_grid.py", "--eval-split", "test"])
    with pytest.raises(SystemExit, match="phase5-execute"):
        G.main()


def test_dump_pred_frames_one_csv_per_rung_same_pool_different_probability(tmp_path):
    import phase4_eval_grid as G

    base = pd.DataFrame({
        "public_id": [1, 1, 2], "coordX": [1.0, 2.0, 3.0], "coordY": [0.0] * 3,
        "coordZ": [0.0] * 3, "x_length": [1.0] * 3, "y_length": [1.0] * 3,
        "z_length": [1.0] * 3, "probability": [0.9, 0.5, 0.4]})
    other = base.copy()
    other["probability"] = [0.1, 0.2, 0.3]
    paths = G.dump_pred_frames({"B0": base, "B1": other}, seed=1, out_dir=tmp_path / "preds")
    assert sorted(p.name for p in paths) == ["pred_B0_seed1.csv", "pred_B1_seed1.csv"]
    a = pd.read_csv(tmp_path / "preds" / "pred_B0_seed1.csv")
    b = pd.read_csv(tmp_path / "preds" / "pred_B1_seed1.csv")
    assert (a[["public_id", "coordX", "x_length"]].values
            == b[["public_id", "coordX", "x_length"]].values).all()
    assert not (a["probability"].values == b["probability"].values).all()
