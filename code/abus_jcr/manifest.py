"""Frozen split + stratified k-fold manifest (Inv. 9, 10).

Lists every discovered case with its official ``split`` and, for Train volumes,
a ``fold in {0..K-1}`` from a seeded, B/M-stratified partition — the single
source of truth Phase 3 uses to pick, per Train volume, the fold detector that
did **not** see it. ``val``/``test`` rows carry ``fold = -1``. The build is
deterministic and hash-stable (a ``manifest_hash`` pins seed + members).

The manifest deliberately *lists* all provided splits (the official split is
public) but Phase 1 only materialises the Train+Val cache (Inv. 9); Test stays
closed until Phase 5.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from . import conventions as C
from .io_nrrd import discover_cases

# canonical split-name normalisation
_SPLIT_CANON = {
    "train": "train", "Train": "train",
    "val": "val", "validation": "val", "Validation": "val",
    "test": "test", "Test": "test",
}


def _read_labels(split_root: Path) -> Dict[int, str]:
    """Map ``case_id -> label`` (B/M) from ``labels.csv`` under a split root."""
    df = pd.read_csv(split_root / "labels.csv")
    return {int(r.case_id): str(r.label) for r in df.itertuples(index=False)}


def stratified_folds(
    volume_ids: Sequence[int],
    labels: Sequence[str],
    k: int = C.KFOLD_K,
    seed: int = C.KFOLD_SEED,
) -> Dict[int, int]:
    """Assign each volume a fold in ``{0..k-1}``, stratified by label, seeded.

    Within each label group the volumes are sorted by id, permuted with a fixed
    RNG, then round-robin assigned (position % k). This spreads every label as
    evenly as possible across folds and is fully determined by ``(seed, k,
    members)``.
    """
    rng = np.random.default_rng(seed)
    fold_of: Dict[int, int] = {}
    by_label: Dict[str, List[int]] = {}
    for vid, lab in zip(volume_ids, labels):
        by_label.setdefault(lab, []).append(int(vid))
    for lab in sorted(by_label):
        ids = sorted(by_label[lab])
        order = rng.permutation(len(ids))
        for pos, idx in enumerate(order):
            fold_of[ids[idx]] = pos % k
    return fold_of


def _manifest_hash(rows: List[dict]) -> str:
    payload = {
        "seed": C.KFOLD_SEED,
        "k": C.KFOLD_K,
        "stratify_by": C.KFOLD_STRATIFY_BY,
        "members": sorted((r["volume_id"], r["split"], r["fold"], r["label"]) for r in rows),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_manifest(data_roots: Mapping[str, Path]) -> pd.DataFrame:
    """Build the frozen manifest from a mapping ``split -> split_root``.

    ``split`` keys are normalised to ``{train, val, test}``. Only ``train``
    volumes receive folds; ``val``/``test`` get ``fold = -1``. Returns a
    DataFrame sorted by ``volume_id`` with columns
    ``[volume_id, split, fold, label, manifest_hash]``.
    """
    rows: List[dict] = []
    for raw_split, root in data_roots.items():
        split = _SPLIT_CANON.get(raw_split)
        if split is None:
            raise ValueError(f"unknown split name {raw_split!r}")
        root = Path(root)
        cases = discover_cases(root)
        labels = _read_labels(root)

        vids = sorted(cases)
        vlabels = [labels[v] for v in vids]
        if split == "train":
            folds = stratified_folds(vids, vlabels)
        else:
            folds = {v: -1 for v in vids}

        for v in vids:
            rows.append({
                "volume_id": int(v),
                "split": split,
                "fold": int(folds[v]),
                "label": labels[v],
            })

    mh = _manifest_hash(rows)
    for r in rows:
        r["manifest_hash"] = mh

    df = pd.DataFrame(rows, columns=["volume_id", "split", "fold", "label", "manifest_hash"])
    df = df.sort_values("volume_id", kind="stable").reset_index(drop=True)
    return df
