"""[RB_FOLD_FLIP] `run_suffix` must reach BOTH the checkpoint path AND the detection-cache tag.

The failure this pins down is silent and destructive: if the suffix reached only the checkpoint
path, generating the corrected-flip pool would re-use the DEPLOYED arm's cached detections and
produce a record that looks new but is the old arm's detections re-linked. Torch-free.
"""

import pandas as pd
import pytest

from abus_jcr.candidates.generate import plan_generation, run_suffix_tag


def _manifest():
    rows = []
    for v in range(6):
        rows.append({"volume_id": v, "split": "train", "fold": v % 3})
    for v in range(100, 103):
        rows.append({"volume_id": v, "split": "val", "fold": -1})
    return pd.DataFrame(rows)


def test_run_suffix_tag_shapes():
    assert run_suffix_tag("") == ""
    assert run_suffix_tag("latflip") == "_latflip"


def test_no_suffix_reads_the_deployed_checkpoints():
    jobs = plan_generation(_manifest(), "train", "/ckpt")
    names = sorted(j["checkpoint"].name for j in jobs)
    assert names == ["retinanet_fold0.pt", "retinanet_fold1.pt", "retinanet_fold2.pt"]


def test_suffix_redirects_every_fold_checkpoint_and_never_the_deployed_one():
    jobs = plan_generation(_manifest(), "train", "/ckpt", run_suffix="latflip")
    names = sorted(j["checkpoint"].name for j in jobs)
    assert names == ["retinanet_fold0_latflip.pt", "retinanet_fold1_latflip.pt",
                     "retinanet_fold2_latflip.pt"]
    assert not any(j["checkpoint"].name == f"retinanet_fold{f}.pt" for j in jobs for f in range(5))


def test_suffix_redirects_every_seed_checkpoint():
    jobs = plan_generation(_manifest(), "val", "/ckpt", run_suffix="latflip")
    for j in jobs:
        assert j["checkpoint"].name.endswith("_latflip.pt")
        assert j["checkpoint"].name.startswith("retinanet_full_seed")


def test_detector_of_origin_is_NOT_suffixed_so_the_two_records_stay_comparable():
    """The suffix must not leak into the record's bookkeeping column or pred-CSV names."""
    for split in ("train", "val"):
        a = sorted(j["detector_of_origin"] for j in plan_generation(_manifest(), split, "/ckpt"))
        b = sorted(j["detector_of_origin"]
                   for j in plan_generation(_manifest(), split, "/ckpt", run_suffix="latflip"))
        assert a == b


def test_cache_tag_differs_between_arms():
    """The tag generate_split builds is `{det}{suffix}_op{op}` — the two arms must not collide."""
    det, op = "fold0", 0.03
    deployed = f"{det}{run_suffix_tag('')}_op{op}"
    latflip = f"{det}{run_suffix_tag('latflip')}_op{op}"
    assert deployed == "fold0_op0.03"
    assert latflip == "fold0_latflip_op0.03"
    assert deployed != latflip


def test_epoch_override_composes_with_the_suffix():
    jobs = plan_generation(_manifest(), "train", "/ckpt",
                           epoch_overrides={"fold1": 7}, run_suffix="latflip")
    by_det = {j["detector_of_origin"]: j["checkpoint"] for j in jobs}
    assert by_det["fold1"].parent.name == "retinanet_fold1_latflip"
    assert by_det["fold1"].name == "epoch07.pt"
    assert by_det["fold0"].name == "retinanet_fold0_latflip.pt"
