"""Detector training — standard recipe + val CPM-proxy early stopping (Inv. 2 amended, 9, 10, 14).

[P2-UPDATE B5] Model selection is on a DETECTION metric — **val CPM-proxy**: the mean
per-slice recall at the FROC FP/slice budgets on the official 30-case Val split, a
pre-linking foreshadow of the Inv.-3 CPM (mean recall at fixed FP operating points).
Better-aligned than generic AP for a candidate generator and discriminative where
per-volume recall saturates. val_ap and val_loss stay logged as diagnostics. SGD/cosine
recipe unchanged. No FROC operating point / NMS tuning here (Phase 3 owns the
recall-saturating operating point).

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
from .. import cache as K
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


def _val_gt_df(slice_boxes_val: pd.DataFrame, val_ids: Sequence[int]) -> pd.DataFrame:
    """Val GT as half-open ``(x1,y1,x2,y2)`` per slice for the 2D-AP metric.

    Inclusive iso ``(r0,c0,r1,c1)`` -> half-open ``x1=c0, y1=r0, x2=c1+1, y2=r1+1``.
    """
    df = slice_boxes_val[slice_boxes_val["volume_id"].isin(list(val_ids))]
    return pd.DataFrame({
        "volume_id": df["volume_id"].astype("int64").to_numpy(),
        "slice_z": df["slice_z"].astype("int64").to_numpy(),
        "x1": df["c0"].to_numpy(dtype=float),
        "y1": df["r0"].to_numpy(dtype=float),
        "x2": (df["c1"].to_numpy() + 1).astype(float),
        "y2": (df["r1"].to_numpy() + 1).astype(float),
    })


def _n_val_slices(cache_root, val_ids) -> int:
    """Total val slices scanned (lesion + background) — the FP-per-slice denominator."""
    return sum(int(K.read_meta(cache_root, int(v))["iso_shape"][C.SLICE_AXIS]) for v in val_ids)


def _val_detection_metrics(model, cache_root, val_ids, gt_df, n_slices, device, batch_size):
    """[P2-UPDATE B5] eval-mode val detection metrics. SELECTION = ``val_cpm_proxy``
    (mean per-slice recall at the FROC FP/slice budgets, Inv.-3 foreshadow); ``val_ap``,
    per-volume recall, and the per-budget recalls are logged. Same inference entry Phase 3 reuses."""
    import pandas as _pd

    from . import infer
    from .metrics import val_ap_2d, per_volume_recall_2d, recall_at_fp_budgets_2d

    frames = [
        infer.run_detector_on_volume(
            model, cache_root, int(vid),
            C.DET_DIAG_SCORE_THRESH, C.DET_DIAG_NMS_THRESH, C.DET_DIAG_DETECTIONS_PER_IMG,
            batch_size=batch_size, device=device)
        for vid in val_ids
    ]
    det_df = _pd.concat(frames, ignore_index=True) if frames else frames
    cpm, per_b = recall_at_fp_budgets_2d(det_df, gt_df, n_slices, C.DET_SELECTION_FP_BUDGETS, C.DET_AP_IOU_THRESH)
    ap = val_ap_2d(det_df, gt_df, C.DET_AP_IOU_THRESH)
    vrec = per_volume_recall_2d(det_df, gt_df, C.DET_PER_SLICE_RECALL["score_thresh"], C.DET_AP_IOU_THRESH)
    return {
        "val_cpm_proxy": float(cpm),
        "val_ap": float(ap),
        "val_lesion_recall": float(vrec),
        "val_froc_recall": {str(k): round(float(v), 4) for k, v in per_b.items()},
    }


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

    gt_df_val = _val_gt_df(slice_boxes_val, va_ids)   # [P2-UPDATE B5] selection target
    n_slices_val = _n_val_slices(cache_root, va_ids)  # FP-per-slice denominator

    best_cpm = float("-inf")    # [P2-UPDATE B5] select on MAX val CPM-proxy (was: min val-loss)
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
            # [P2-UPDATE B5] detection-metric selection: eval-mode val CPM-proxy (+ AP, recall logged).
            vm = _val_detection_metrics(
                model, cache_root, va_ids, gt_df_val, n_slices_val, device, batch_size)
            lr_now = opt.param_groups[0]["lr"]
            rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_cpm_proxy": vm["val_cpm_proxy"], "val_ap": vm["val_ap"],
                   "val_lesion_recall": vm["val_lesion_recall"],
                   "val_froc_recall": vm["val_froc_recall"], "lr": lr_now}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"[{run}] epoch {epoch}: train {train_loss:.4f}  val_loss {val_loss:.4f}  "
                  f"cpm_proxy {vm['val_cpm_proxy']:.4f}  val_ap {vm['val_ap']:.4f}  "
                  f"val_rec {vm['val_lesion_recall']:.4f}  lr {lr_now:.2e}")

            if vm["val_cpm_proxy"] > best_cpm + 1e-6:
                best_cpm = vm["val_cpm_proxy"]; best_epoch = epoch; epochs_no_improve = 0
                cfg = {"regime": regime, "fold_or_seed": int(fold_or_seed), "seed": seed,
                       "best_epoch": best_epoch, "best_val_cpm_proxy": best_cpm,
                       "best_val_ap": vm["val_ap"], "best_val_loss": val_loss,
                       "selection_metric": C.DET_SELECTION_METRIC}
                save_checkpoint(ckpt_path, model, cfg)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[{run}] early stop at epoch {epoch} (best epoch {best_epoch}, "
                          f"cpm_proxy {best_cpm:.4f})")
                    break

    return {"run": run, "checkpoint": str(ckpt_path), "log": str(log_path),
            "best_epoch": best_epoch, "best_val_cpm_proxy": best_cpm, "epochs_ran": epoch + 1}
