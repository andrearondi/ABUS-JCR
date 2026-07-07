"""Case discovery handles both split layouts and keys on integer case_id.

The Train split nests DATA under shard dirs (DATA00_49/, DATA50_99/) with
3-digit zero-padded names, while Validation/Test are flat and unpadded. A
non-recursive glob silently finds nothing under Train (SERVER_LAYOUT.md), and
padding-based matching would mis-pair. This is exercised with synthetic
fixtures because Train is not on the laptop.
"""

import pytest

from abus_jcr.io_nrrd import discover_cases


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discover_flat_validation_layout(tmp_path):
    root = tmp_path / "Validation"
    for cid in (100, 101, 129):
        _touch(root / "DATA" / f"DATA_{cid}.nrrd")
        _touch(root / "MASK" / f"MASK_{cid}.nrrd")
    cases = discover_cases(root)
    assert set(cases.keys()) == {100, 101, 129}
    assert cases[100].data.name == "DATA_100.nrrd"
    assert cases[100].mask.name == "MASK_100.nrrd"


def test_discover_nested_train_shards_with_padding(tmp_path):
    root = tmp_path / "Train"
    _touch(root / "DATA" / "DATA00_49" / "DATA_000.nrrd")
    _touch(root / "DATA" / "DATA00_49" / "DATA_049.nrrd")
    _touch(root / "DATA" / "DATA50_99" / "DATA_050.nrrd")
    for cid in ("000", "049", "050"):
        _touch(root / "MASK" / f"MASK_{cid}.nrrd")
    cases = discover_cases(root)
    # keyed on parsed integer case_id, not padded string
    assert set(cases.keys()) == {0, 49, 50}
    assert cases[0].data.parent.name == "DATA00_49"


def test_discover_raises_on_unmatched_mask(tmp_path):
    root = tmp_path / "Validation"
    _touch(root / "DATA" / "DATA_100.nrrd")  # no matching MASK
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        discover_cases(root)
