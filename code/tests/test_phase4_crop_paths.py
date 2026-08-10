"""[4.2] The crop cache is a SHARED read-only artefact — its path must not follow ``--out-root``.

``phase4_select_crop_geometry.py`` redirects the child's ``--out-root`` per arm so the two
encoders land in ``crop_geometry/{adaptive,fixed}/``. Before the fix that also redirected
``crops_dir()``, so the **adaptive** arm looked for the [4.1] cache under
``crop_geometry/adaptive/crops/<crop_hash>/val`` and died with ``no CROP_META.json`` before
a single epoch ran. The fixed arm never noticed: it re-extracts at a forced side and touches
no cache, so the bug could only ever surface on the arm that is the DEFAULT winner.

Writes go to ``--out-root``; the crop cache is read from ``--crops-root``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _phase4_common import add_phase4_paths, crops_dir  # noqa: E402
import phase4_select_crop_geometry as sel  # noqa: E402


def _parsed(argv):
    ap = argparse.ArgumentParser()
    add_phase4_paths(ap)
    return ap.parse_args(argv)


def test_crops_dir_defaults_under_out_root():
    args = _parsed(["--out-root", "/p4"])
    assert crops_dir(args) == Path("/p4/crops")


def test_crops_dir_honours_explicit_override():
    args = _parsed(["--out-root", "/p4/crop_geometry/adaptive", "--crops-root", "/p4/crops"])
    assert crops_dir(args) == Path("/p4/crops")


def test_geometry_driver_points_both_arms_at_the_shared_crop_cache():
    """The child writes into its own arm dir but READS the one [4.1] cache."""
    args = _parsed(["--out-root", "/p4"])
    args.device, args.epochs = "cuda", 30

    for crop_side in (None, 128.0):
        arm = Path("/p4/crop_geometry") / ("adaptive" if crop_side is None else "fixed")
        cmd = sel.child_cmd(args, crop_side, arm)
        assert cmd[cmd.index("--out-root") + 1] == str(arm)
        assert cmd[cmd.index("--crops-root") + 1] == "/p4/crops"
