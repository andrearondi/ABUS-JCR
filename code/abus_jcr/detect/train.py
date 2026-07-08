"""Detector training — standard recipe + val-loss early stopping (Inv. 2, 9, 10, 14).

Trained *normally* (Inv. 2): SGD/cosine, early stopping on the official 30-case
Val split for **both** regimes (Inv. 9 — Val is the model-selection split). No FROC
operating point, no competitive threshold/NMS tuning here.

- ``--regime fold  --fold f`` (seed ``DET_FOLD_SEED``): trains on Train volumes with
  ``manifest.fold != f`` only (Inv. 10, out-of-fold) -> ``retinanet_fold{f}.pt``.
- ``--regime full  --seed s`` (s in ``DET_FULL_SEEDS``): trains on all 100 Train
  volumes -> ``retinanet_full_seed{s}.pt`` (Inv. 14, 3 standalone deployment seeds).

Torch is imported lazily; the module imports without torch.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .. import conventions as C
from .retinanet import build_retinanet, save_checkpoint
from .slice_det_dataset import SliceDetectionDataset


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch and set deterministic cudnn where it does not forbid an op."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_volume_ids(manifest: pd.DataFrame, regime: str, fold_or_seed: int) -> List[int]:
    """Volume ids to train on. fold: Train ``fold != f`` (Inv. 10); full: all Train."""
    train = manifest[manifest["split"] == "train"]
    if regime == "fold":
        ids = train[train["fold"] != fold_or_seed]["volume_id"]
    elif regime == "full":
        ids = train["volume_id"]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return sorted(int(v) for v in ids)


def val_volume_ids(manifest: pd.DataFrame) -> List[int]:
    return sorted(int(v) for v in manifest[manifest["split"] == "val"]["volume_id"])


def _collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _to_device_targets(targets, device):
    out = []
    for t in targets:
        out.append({"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)})
    return out


def _lr_lambda(step: int, warmup_iters: int, warmup_factor: float, total_iters: int) -> float:
    """Linear warmup for ``warmup_iters`` then cosine decay to 0 over ``total_iters``."""
    if step < warmup_iters:
        alpha = step / max(1, warmup_iters)
        return warmup_factor * (1 - alpha) + alpha
    progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _val_loss(model, loader, device) -> float:
    """Mean torchvision loss over Val slices, computed in train() mode under no_grad.

    The model only returns losses when given targets; BN in the v2 backbone is
    frozen (GroupNorm elsewhere), so train() mode here is a pure loss readout,
    documented and accepted (Inv. 9 uses this only as the early-stop signal).
    """
    import torch

    model.train()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            tg = _to_device_targets(targets, device)
            losses = model(images, tg)
            total += float(sum(losses.values()).item())
            n += 1
    return total / max(1, n)


def train_detector(
    regime: str,
    fold_or_seed: int,
    cache_root,
    manifest: pd.DataFrame,
    slice_boxes_train: pd.DataFrame,
    slice_boxes_val: pd.DataFrame,
    out_root,
    max_epochs: int = C.DET_MAX_EPOCHS,
    patience: int = C.DET_EARLYSTOP_PATIENCE,
    batch_size: int = C.DET_BATCH_SIZE,
    num_workers: int = 8,
    device: str = "cuda",
) -> Dict:
    """Train one detector; return a summary ``dict`` and write its best checkpoint + jsonl log."""
    import torch
    from torch.utils.data import DataLoader

    seed = C.DET_FOLD_SEED if regime == "fold" else int(fold_or_seed)
    seed_everything(seed)

    run = f"retinanet_fold{fold_or_seed}" if regime == "fold" else f"retinanet_full_seed{fold_or_seed}"
    out_root = Path(out_root)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_path = out_root / "checkpoints" / f"{run}.pt"
    log_path = out_root / "logs" / f"{run}.jsonl"

    tr_ids = train_volume_ids(manifest, regime, fold_or_seed)
    va_ids = val_volume_ids(manifest)

    train_ds = SliceDetectionDataset(cache_root, slice_boxes_train, tr_ids, train=True, seed=seed)
    val_ds = SliceDetectionDataset(cache_root, slice_boxes_val, va_ids, train=False, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_collate)

    model = build_retinanet(c_channels=C.C_CHANNELS, num_classes=C.DET_NUM_CLASSES, pretrained=True)
    model.to(device)

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=C.DET_OPTIMIZER["lr"], momentum=C.DET_OPTIMIZER["momentum"],
        weight_decay=C.DET_OPTIMIZER["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_iters = max_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, C.DET_LR_SCHEDULE["warmup_iters"],
                                  C.DET_LR_SCHEDULE["warmup_factor"], total_iters))

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0

    with log_path.open("w") as logf:
        for epoch in range(max_epochs):
            train_ds.set_epoch(epoch)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, collate_fn=_collate)
            model.train()
            running, nb = 0.0, 0
            for images, targets in train_loader:
                images = [im.to(device) for im in images]
                tg = _to_device_targets(targets, device)
                losses = model(images, tg)
                loss = sum(losses.values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                global_step += 1
                running += float(loss.item()); nb += 1
            train_loss = running / max(1, nb)

            val_loss = _val_loss(model, val_loader, device)
            lr_now = opt.param_groups[0]["lr"]
            rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"[{run}] epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}  lr {lr_now:.2e}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss; best_epoch = epoch; epochs_no_improve = 0
                cfg = {"regime": regime, "fold_or_seed": int(fold_or_seed), "seed": seed,
                       "best_epoch": best_epoch, "best_val_loss": best_val}
                save_checkpoint(ckpt_path, model, cfg)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[{run}] early stop at epoch {epoch} (best epoch {best_epoch}, val {best_val:.4f})")
                    break

    return {"run": run, "checkpoint": str(ckpt_path), "log": str(log_path),
            "best_epoch": best_epoch, "best_val_loss": best_val, "epochs_ran": epoch + 1}
