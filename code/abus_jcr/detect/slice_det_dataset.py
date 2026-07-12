"""The per-slice 2.5D detection dataset (Inv. 1, 10, 13).

Wraps :func:`abus_jcr.slice_dataset.get_stack` with the per-slice 2D GT boxes and
the Inv.-13 training augmentation. Membership is an explicit ``volume_ids`` list —
the caller passes ``fold != f`` for a fold detector (Inv. 10) or all Train volumes
for a full-train detector.

**Import discipline:** the correctness-critical helpers (box conversion, sample
enumeration, negative sampling, and the numpy sample loader) are torch-free and
unit-tested on the laptop. Only ``__getitem__`` builds torch tensors, importing
torch lazily, so the module imports without torch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import conventions as C
from .. import cache as K
from ..augment import TRAIN_AUGMENT
from ..slice_dataset import get_stack
from . import augment_ops
from . import copy_paste


def boxes_halfopen_for(slice_boxes_df: pd.DataFrame, vid: int, z: int) -> np.ndarray:
    """Half-open ``(x1, y1, x2, y2)`` boxes for ``(vid, z)`` from the GT table.

    Inclusive iso-voxel ``(r0, c0, r1, c1)`` -> half-open ``(c0, r0, c1+1, r1+1)``
    (``x = d1 = col``, ``y = d0 = row``). Degenerate boxes cannot occur for
    inclusive integer masks and are asserted away.
    """
    sel = slice_boxes_df[(slice_boxes_df["volume_id"] == vid) & (slice_boxes_df["slice_z"] == z)]
    if len(sel) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    r0 = sel["r0"].to_numpy(); c0 = sel["c0"].to_numpy()
    r1 = sel["r1"].to_numpy(); c1 = sel["c1"].to_numpy()
    boxes = np.stack([c0, r0, c1 + 1, r1 + 1], axis=1).astype(np.float32)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    assert keep.all(), "degenerate GT box from an inclusive integer mask (should be impossible)"
    return boxes[keep]


def enumerate_samples(
    cache_root, volume_ids: Sequence[int], slice_boxes_df: pd.DataFrame
) -> List[Tuple[int, int, bool]]:
    """Every ``(vid, z, is_lesion)`` over the given volumes' ``SLICE_AXIS`` length.

    ``is_lesion`` is True iff ``(vid, z)`` carries a GT box.
    """
    lesion = set(zip(slice_boxes_df["volume_id"].tolist(), slice_boxes_df["slice_z"].tolist()))
    out: List[Tuple[int, int, bool]] = []
    for vid in sorted(int(v) for v in volume_ids):
        n = int(K.read_meta(cache_root, vid)["iso_shape"][C.SLICE_AXIS])
        for z in range(n):
            out.append((vid, z, (vid, z) in lesion))
    return out


def sample_epoch(
    samples: Sequence[Tuple[int, int, bool]],
    neg_pos_ratio: int,
    seed: int,
    epoch: int,
) -> List[int]:
    """Indices into ``samples`` for one training epoch (Inv. 10 membership fixed upstream).

    Keeps **all** lesion-bearing slices; samples ``neg_pos_ratio x n_lesion``
    background slices with an RNG seeded by ``(seed, epoch)``; shuffles the union.
    Deterministic per ``(seed, epoch)`` and reshuffled each epoch.
    """
    rng = np.random.default_rng((int(seed), int(epoch)))
    lesion_idx = [i for i, s in enumerate(samples) if s[2]]
    bg_idx = [i for i, s in enumerate(samples) if not s[2]]
    n_bg = min(len(bg_idx), neg_pos_ratio * len(lesion_idx))
    chosen_bg = rng.choice(bg_idx, size=n_bg, replace=False).tolist() if n_bg > 0 else []
    order = lesion_idx + list(chosen_bg)
    rng.shuffle(order)
    return order


class SliceDetectionDataset:
    """torch ``Dataset`` over ``(vid, z)`` samples with GT boxes + Inv.-13 augment.

    Training epochs subsample background slices via :func:`sample_epoch`
    (``set_epoch`` reshuffles); validation/inference iterate **all** slices with no
    augmentation. ``load_numpy_sample`` is the torch-free core; ``__getitem__``
    wraps it in torch tensors (torch imported lazily).
    """

    def __init__(
        self,
        cache_root,
        slice_boxes_df: pd.DataFrame,
        volume_ids: Sequence[int],
        train: bool = False,
        seed: int = 0,
        neg_pos_ratio: int = C.DET_NEG_POS_SLICE_RATIO,
        stack_fn=get_stack,
        policy: dict = TRAIN_AUGMENT,
    ):
        self.cache_root = cache_root
        self.slice_boxes_df = slice_boxes_df
        self.volume_ids = list(volume_ids)
        self.train = train
        self.seed = seed
        self.neg_pos_ratio = neg_pos_ratio
        self.stack_fn = stack_fn
        self.policy = policy
        self.samples = enumerate_samples(cache_root, self.volume_ids, slice_boxes_df)
        # [P2-UPDATE P2] shadow-aware copy-paste bank (built only when enabled; default OFF).
        self._crop_bank = self._build_crop_bank() if (self.train and self.policy.get("lesion_copy_paste", False)) else []
        self.set_epoch(0)

    def _build_crop_bank(self, cap: int = 400):
        """Extract lesion+shadow crops from up to ``cap`` lesion slices (un-augmented)."""
        lesion = [(vid, z) for vid, z, is_les in self.samples if is_les]
        rng = np.random.default_rng((int(self.seed), 777))
        if len(lesion) > cap:
            lesion = [lesion[i] for i in rng.choice(len(lesion), size=cap, replace=False)]
        bank = []
        for vid, z in lesion:
            stack = np.asarray(self.stack_fn(self.cache_root, vid, z), dtype=np.float32)
            boxes = boxes_halfopen_for(self.slice_boxes_df, vid, z)
            for b in boxes:
                bank.append(copy_paste.extract_lesion_crop(stack, b))
        return bank

    def set_epoch(self, epoch: int) -> None:
        """(Re)compute this epoch's sample order. Train subsamples; else keep all."""
        self.epoch = int(epoch)
        if self.train:
            self._order = sample_epoch(self.samples, self.neg_pos_ratio, self.seed, epoch)
        else:
            self._order = list(range(len(self.samples)))

    def __len__(self) -> int:
        return len(self._order)

    def load_numpy_sample(
        self, vid: int, z: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Torch-free: assemble the ``(C, d0, d1)`` stack + half-open boxes, augment if train.

        Augmentation is applied only when ``self.train`` and an ``rng`` is given
        (candidate generation / val run without augmentation, Inv. 13).
        """
        stack = np.asarray(self.stack_fn(self.cache_root, vid, z), dtype=np.float32)
        boxes = boxes_halfopen_for(self.slice_boxes_df, vid, z)
        if self.train and rng is not None:
            stack, boxes = augment_ops.apply_train_augment(stack, boxes, rng, policy=self.policy)
            # [P2-UPDATE P2] shadow-aware copy-paste (default OFF; only when bank built).
            if self._crop_bank and rng.random() < float(self.policy.get("copy_paste_p", 0.0)):
                crop = self._crop_bank[int(rng.integers(0, len(self._crop_bank)))]
                stack, boxes = copy_paste.paste_lesion(stack, boxes, crop, rng)
        return stack, boxes

    def __getitem__(self, i: int):
        import torch

        vid, z, _ = self.samples[self._order[i]]
        rng = np.random.default_rng((int(self.seed), int(self.epoch), int(i))) if self.train else None
        stack, boxes = self.load_numpy_sample(vid, z, rng)
        image = torch.as_tensor(np.ascontiguousarray(stack), dtype=torch.float32)
        target: Dict = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.full((len(boxes),), C.DET_FG_LABEL, dtype=torch.int64),
            "volume_id": int(vid),
            "slice_z": int(z),
        }
        return image, target
