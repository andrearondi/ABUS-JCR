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
from _phase4_common import add_phase4_paths, emb_path  # noqa: E402


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
