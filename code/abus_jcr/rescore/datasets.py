"""[4.8] Batching — the unit is a SET, not a candidate.

A **SET** is one ``(detector_of_origin, public_id)`` pair (Inv. 7): one detector's
candidates for one volume. A Val volume therefore contributes **three** independent sets,
one per full-train seed (Inv. 14) — never a bare volume, and never concatenated.

Two batchers:

* :class:`CropDataset` — the encoder-pretraining loader. With ``augment=True`` it
  **re-extracts** from the iso cache (centre jitter changes the sample grid, so an
  augmented crop cannot come from the cached 48³ tensor); with ``augment=False`` it reads
  the crop-cache memmap. Every non-pretraining consumer uses ``augment=False`` (Inv. 13:
  no augmentation and no TTA outside encoder pretraining).
* :func:`collate_sets` — pads a batch of sets to a common width with a key-padding mask.
  ``RESC_NEG_POS_RATIO is None``: **no negative subsampling**, so the train sets are
  exactly the eval sets.

  Production callers pass ``max_set_size=None``, i.e. pad to the **batch** max. Padding
  every batch to the global ``RESC_MAX_SET_SIZE`` (576 since the promotion, to admit the
  509-candidate train set) would make a batch of median 85-candidate sets ~45× larger than
  it needs to be, and it buys nothing: with batch-max padding a set can never exceed the
  width, so the truncate-a-frozen-pool failure the explicit width guards against becomes
  unreachable by construction. ``RESC_MAX_SET_SIZE`` keeps its real job — the [4.1]
  assertion over both whole pools — and remains available here as an explicit cap.

Numpy core (unit-tested on the laptop); torch is imported lazily only by
:meth:`CropDataset.torch_dataset`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C

__all__ = ["SET_KEYS", "group_sets", "set_sizes", "collate_sets", "CropDataset"]

#: The set key (Inv. 7). ``detector_of_origin`` is bookkeeping only — never a model input.
SET_KEYS = ("detector_of_origin", "public_id")


def group_sets(record_df: pd.DataFrame) -> Dict[Tuple, np.ndarray]:
    """``{(detector_of_origin, public_id): positional row indices}``, record order preserved."""
    pos = np.arange(len(record_df))
    out: Dict[Tuple, np.ndarray] = {}
    for key, idx in pd.Series(pos).groupby(
            [record_df[k].to_numpy() for k in SET_KEYS], sort=False):
        out[tuple(key) if isinstance(key, tuple) else (key,)] = idx.to_numpy()
    return out


def set_sizes(record_df: pd.DataFrame) -> np.ndarray:
    """Candidate count per set — [4.1] asserts ``max() <= RESC_MAX_SET_SIZE``."""
    return record_df.groupby(list(SET_KEYS), sort=False).size().to_numpy()


def collate_sets(index_lists: Sequence[np.ndarray], feats: np.ndarray, labels: np.ndarray,
                 coord: np.ndarray, length: np.ndarray,
                 max_set_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Pad a list of sets into ``(B, N, ·)`` tensors plus a boolean ``mask`` (True = real).

    ``max_set_size=None`` (what production callers use) pads to the **batch** max; an
    explicit value pads to that width and refuses any set larger than it.

    Padded ``length`` entries are **1.0**, not 0.0: the Axis-A descriptor takes
    ``log(w_n/w_m)``, and a zero would produce ``-inf`` inside the attention logits before
    the mask is applied. Padded ``labels`` are the ignore code, so they enter no loss term
    even if a mask were ever dropped — belt and braces on the Inv.-11 boundary.
    """
    if not len(index_lists):
        raise ValueError("collate_sets got no sets")
    sizes = [len(i) for i in index_lists]
    width = int(max(sizes)) if max_set_size is None else int(max_set_size)
    if max(sizes) > width:
        raise ValueError(f"set of size {max(sizes)} exceeds the pad width {width} "
                         f"(RESC_MAX_SET_SIZE = {C.RESC_MAX_SET_SIZE}); [4.1] should have "
                         f"caught this — do not silently truncate a frozen pool")

    b, d = len(index_lists), feats.shape[1]
    out = {
        "feats": np.zeros((b, width, d), dtype=np.float32),
        "labels": np.full((b, width), -1.0, dtype=np.float32),   # ignore code
        "coord": np.zeros((b, width, 3), dtype=np.float32),
        "length": np.ones((b, width, 3), dtype=np.float32),
        "mask": np.zeros((b, width), dtype=bool),
        "rows": np.full((b, width), -1, dtype=np.int64),
    }
    for i, idx in enumerate(index_lists):
        n = len(idx)
        out["feats"][i, :n] = feats[idx]
        out["labels"][i, :n] = labels[idx]
        out["coord"][i, :n] = coord[idx]
        out["length"][i, :n] = length[idx]
        out["mask"][i, :n] = True
        out["rows"][i, :n] = idx
    return out


class CropDataset:
    """Per-candidate 3D crops for encoder pretraining ([4.3]) and embedding caching ([4.4]).

    ``augment=True`` re-extracts from the iso cache with centre jitter + a mirror along
    **d0 = the MEASURED LATERAL axis** (pretraining only; never d1 = depth/beam, Inv. 13);
    ``augment=False`` reads the crop-cache memmap (everything else).
    Each item is ``(crop (1,48,48,48) float32, rest_features, label_code, row)``.
    """

    def __init__(self, record_df: pd.DataFrame, rest_feats: np.ndarray, label_codes: np.ndarray,
                 crops_memmap=None, cache_root=None, augment: bool = False, seed: int = 0,
                 vol_loader=None, source: Optional[str] = None,
                 fixed_side: Optional[float] = None):
        source = source or ("volume" if augment else "cache")
        if source not in ("cache", "volume"):
            raise ValueError(f"source must be 'cache' or 'volume', got {source!r}")
        if source == "volume" and cache_root is None:
            raise ValueError("source='volume' re-extracts from the iso cache; pass cache_root")
        if source == "cache":
            if crops_memmap is None:
                raise ValueError("source='cache' reads the crop cache; pass crops_memmap")
            if augment:
                raise ValueError("augmentation needs re-extraction (centre jitter changes the "
                                 "sample grid); use source='volume'")
            if fixed_side is not None:
                raise ValueError("the crop cache is built at the ADAPTIVE ROI; a fixed side "
                                 "needs source='volume'")
        self.record = record_df.reset_index(drop=True)
        self.rest = np.asarray(rest_feats, dtype=np.float32)
        self.labels = np.asarray(label_codes, dtype=np.float32)
        self.crops = crops_memmap
        self.cache_root = cache_root
        self.augment = bool(augment)
        self.source = source
        self.fixed_side = None if fixed_side is None else float(fixed_side)
        self.seed = int(seed)
        self._vol_loader = vol_loader
        self._vol_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.record)

    def _volume(self, pid: int):
        if pid not in self._vol_cache:
            if self._vol_loader is not None:
                self._vol_cache[pid] = np.asarray(self._vol_loader(self.cache_root, pid))
            else:
                from ..cache import open_vol
                self._vol_cache[pid] = np.asarray(open_vol(self.cache_root, pid))
        return self._vol_cache[pid]

    def crop(self, i: int) -> np.ndarray:
        if self.source == "cache":
            return np.asarray(self.crops[i], dtype=np.float32)
        r = self.record.iloc[i]
        vol = self._volume(int(r["public_id"]))
        geom = (float(r["cen_d0"]), float(r["cen_d1"]), float(r["cen_d2"]),
                float(r["ext_d0"]), float(r["ext_d1"]), float(r["ext_d2"]))
        if not self.augment:
            from .crops import extract_crop
            return extract_crop(vol, *geom, side=self.fixed_side)
        from .crop_aug import augmented_crop
        rng = np.random.default_rng((self.seed, i))
        return augmented_crop(vol, *geom, rng=rng, side=self.fixed_side)

    def __getitem__(self, i: int):
        return self.crop(i)[None], self.rest[i], self.labels[i], i

    def torch_dataset(self):
        """Wrap as a ``torch.utils.data.Dataset`` (server only)."""
        from torch.utils.data import Dataset

        outer = self

        class _TD(Dataset):
            def __len__(self):
                return len(outer)

            def __getitem__(self, i):
                import torch
                crop, rest, lab, row = outer[i]
                return (torch.from_numpy(np.ascontiguousarray(crop)),
                        torch.from_numpy(np.ascontiguousarray(rest)),
                        torch.tensor(float(lab)), torch.tensor(int(row)))

        return _TD()
