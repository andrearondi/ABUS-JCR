"""Frozen split + stratified k-fold manifest (Inv. 9, 10).

The manifest is the single source of truth every later phase reads. It must be
deterministic (two builds byte-identical), respect the official split, and
partition every Train volume into k=5 B/M-stratified folds; val/test carry
fold = -1.
"""

import numpy as np
import pandas as pd
import pytest

from abus_jcr import conventions as C
from abus_jcr.manifest import build_manifest, stratified_folds


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _make_split(root, ids, labels):
    for cid, lab in zip(ids, labels):
        _touch(root / "DATA" / f"DATA_{cid}.nrrd")
        _touch(root / "MASK" / f"MASK_{cid}.nrrd")
    rows = ["case_id,label,data_path,mask_path"]
    for cid, lab in zip(ids, labels):
        rows.append(f"{cid},{lab},DATA/DATA_{cid}.nrrd,MASK/MASK_{cid}.nrrd")
    (root / "labels.csv").write_text("\n".join(rows) + "\n")


@pytest.fixture
def roots(tmp_path):
    train_ids = list(range(0, 100))
    train_labels = ["B" if i % 2 == 0 else "M" for i in train_ids]  # 50 B / 50 M
    val_ids = list(range(100, 130))
    val_labels = ["B" if i % 2 == 0 else "M" for i in val_ids]
    _make_split(tmp_path / "Train", train_ids, train_labels)
    _make_split(tmp_path / "Validation", val_ids, val_labels)
    return {"train": tmp_path / "Train", "val": tmp_path / "Validation"}


def test_stratified_folds_balanced_and_deterministic():
    ids = list(range(100))
    labels = ["B" if i % 2 == 0 else "M" for i in ids]
    f1 = stratified_folds(ids, labels, k=5, seed=0)
    f2 = stratified_folds(ids, labels, k=5, seed=0)
    assert f1 == f2  # deterministic
    counts = pd.Series(list(f1.values())).value_counts().to_dict()
    assert set(counts) == {0, 1, 2, 3, 4}
    assert all(v == 20 for v in counts.values())  # 100 / 5
    # each label split evenly: 10 B + 10 M per fold
    for fold in range(5):
        members = [i for i, fo in f1.items() if fo == fold]
        b = sum(1 for i in members if labels[i] == "B")
        m = sum(1 for i in members if labels[i] == "M")
        assert b == 10 and m == 10


def test_manifest_deterministic_and_correct(roots):
    m1 = build_manifest(roots)
    m2 = build_manifest(roots)
    assert m1.to_csv(index=False) == m2.to_csv(index=False)  # byte-identical

    assert list(m1["volume_id"]) == sorted(m1["volume_id"])  # sorted
    assert set(m1.columns) >= {"volume_id", "split", "fold", "label", "manifest_hash"}

    train = m1[m1["split"] == "train"]
    val = m1[m1["split"] == "val"]
    assert len(train) == 100 and len(val) == 30

    # official split respected: 0-99 train, 100-129 val
    assert set(train["volume_id"]) == set(range(100))
    assert set(val["volume_id"]) == set(range(100, 130))

    # 5 folds partition all 100 train volumes; val has fold -1
    fold_counts = train["fold"].value_counts().to_dict()
    assert set(fold_counts) == {0, 1, 2, 3, 4}
    assert sum(fold_counts.values()) == 100
    assert all(fo == -1 for fo in val["fold"])

    # manifest_hash constant across rows
    assert m1["manifest_hash"].nunique() == 1
