"""[4.2d] Two encoder checkpoints must be cacheable side by side.

`[4.2d.1]` selected **epoch 0** (val CPM 0.6312, the earliest within tol of the 0.6411 max at
ep2) while per-candidate **balanced accuracy peaks at epoch 14** (0.9049). Those are different
encoders, and the appearance test has to be able to hold both: judging an encoder by B1 val CPM
is judging it by a quantity dominated by cross-volume calibration, which an encoder cannot
control — so "appearance does not help" must not rest on one checkpoint chosen that way.

Without a suffix both epochs write to ``emb_{split}_seed{R}.npy`` and the second silently
overwrites the first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _phase4_common import add_phase4_paths, emb_path, emb_report_path  # noqa: E402


def _args(argv):
    ap = argparse.ArgumentParser()
    add_phase4_paths(ap)
    return ap.parse_args(argv)


def test_default_path_is_unchanged():
    a = _args(["--out-root", "/p4"])
    assert emb_path(a, "train", 0) == Path("/p4/embeddings/emb_train_seed0.npy")


def test_suffix_separates_two_checkpoints_of_the_same_seed():
    a = _args(["--out-root", "/p4", "--emb-suffix", "_ep14"])
    assert emb_path(a, "train", 0) == Path("/p4/embeddings/emb_train_seed0_ep14.npy")
    assert emb_path(a, "val", 0) == Path("/p4/embeddings/emb_val_seed0_ep14.npy")


def test_the_two_do_not_collide():
    base = _args(["--out-root", "/p4"])
    alt = _args(["--out-root", "/p4", "--emb-suffix", "_ep14"])
    assert emb_path(base, "val", 0) != emb_path(alt, "val", 0)


def test_an_explicit_argument_still_overrides_the_parsed_one():
    """`load_variant_inputs` passes no suffix; a caller may pass one directly."""
    a = _args(["--out-root", "/p4", "--emb-suffix", "_ep14"])
    assert emb_path(a, "val", 0, suffix="") == Path("/p4/embeddings/emb_val_seed0.npy")


# --------------------------------------------------------------------------- the JSON record
# Added 2026-08-31. The arrays carried the suffix and the report did not, so `[4.2d.2]`'s second
# checkpoint (epoch 12) overwrote the first's record and left the CANONICAL epoch-2 embeddings
# documented as `encoder_epoch = 12`. Nothing reads the file, so no measurement moved — but the
# whole point of a provenance record is that it names the right checkpoint.


def test_report_default_path_sits_beside_the_default_arrays():
    a = _args(["--out-root", "/p4"])
    assert emb_report_path(a, 0) == Path("/p4/embeddings/embeddings_seed0.json")


def test_report_carries_the_same_suffix_as_the_arrays_it_describes():
    a = _args(["--out-root", "/p4", "--emb-suffix", "_ep12"])
    assert emb_report_path(a, 0) == Path("/p4/embeddings/embeddings_seed0_ep12.json")


def test_two_checkpoints_of_one_seed_leave_two_reports():
    """THE regression. Both must exist, or the canonical arrays lose their provenance."""
    base = _args(["--out-root", "/p4"])
    alt = _args(["--out-root", "/p4", "--emb-suffix", "_ep12"])
    assert emb_report_path(base, 0) != emb_report_path(alt, 0)


def test_report_and_arrays_agree_on_the_suffix_for_every_case():
    """The two names are one decision; this is what stops them drifting apart again."""
    for sfx in ("", "_ep12", "_ep14"):
        a = _args(["--out-root", "/p4", "--emb-suffix", sfx])
        arr, rep = emb_path(a, "train", 0), emb_report_path(a, 0)
        assert arr.parent == rep.parent
        assert arr.stem.endswith(sfx) and rep.stem.endswith(sfx) if sfx else True
        assert rep.name == f"embeddings_seed0{sfx}.json"
