"""[4.8] Training protocol — fixed annealed schedule, save every epoch, post-hoc selection.

Mirrors the Inv.-2/A1 *shape* that Phase 3 settled on for the detector:

* fixed epoch budget (``RESC_ENC_EPOCHS`` / ``RESC_SET_EPOCHS``), AdamW, cosine to ~0,
  **no early stopping**, every epoch written to disk;
* the deployed epoch chosen **post-hoc on val CPM** (the official ``average_recall``),
  never on val loss; among epochs within ``RESC_SELECT_CPM_TOL`` of the max, the
  **earliest** (``variants.select_epoch_by_val_cpm``);
* the Inv.-2 ceiling tie-break does NOT apply — by Inv. 8 the ceiling is identical for
  every epoch and every rung.

**Seeds.** Rescorer seed ``r`` is PAIRED with detector ``full_seed{r}``: replica ``r``
trains on the (seed-independent) OOF train pool and is evaluated on val pool
``full_seed{r}``. That is the Training-Matrix end-to-end replica and avoids a 3×3 explosion.
``detect.train.seed_everything`` is reused and the seed is logged in every checkpoint.

**Determinism.** ``cudnn.deterministic = True`` (via ``seed_everything``); the crop cache,
the standardisation stats and the cached embeddings are all content-hashed.

**Training data = the PRIMARY ``candidates_train.parquet``.** The ``_hicov`` contingency is
used only if the pre-registered trigger fires (spec Open escalation #3) — and that needs the
user's go-ahead, so it is a CLI flag, never a default.

Torch-only module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from .. import conventions as C
from .losses import rescorer_loss
from .variants import select_epoch_by_val_cpm

__all__ = ["cosine_lr", "make_optimizer", "pretrain_encoder", "train_set_variant",
           "write_epoch_table", "select_epoch_by_val_cpm"]


def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    """Cosine anneal to ~0 over the fixed budget — no early stopping, no restarts."""
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * float(epoch) / float(total_epochs)))


def make_optimizer(model, cfg: Dict, lr: Optional[float] = None):
    """AdamW at the declared ``RESC_*_OPT`` config, with ``lr`` optionally overridden by
    the hyperparameter trial."""
    import torch

    if str(cfg["name"]).lower() != "adamw":
        raise ValueError(f"only AdamW is specified for Phase 4, got {cfg['name']!r}")
    return torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"] if lr is None else lr),
                             weight_decay=float(cfg["weight_decay"]))


def write_epoch_table(rows: Sequence[Dict], out_dir) -> Path:
    """Persist the per-epoch val-CPM table — the selection provenance (exit check 8)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "val_cpm_by_epoch.csv"
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def pretrain_encoder(encoder, head, train_loader, evaluate_epoch: Callable[[int], Dict],
                     out_dir, seed: int, epochs: Optional[int] = None,
                     opt_cfg: Optional[Dict] = None, lr: Optional[float] = None,
                     alpha: float = 0.25, device: str = "cuda") -> Dict:
    """[4.3] Train ``CandidateEncoder + B1Head`` end to end — this run **is** rung B1.

    ``train_loader`` yields ``(crops, rest, labels, rows)``; augmentation lives in the
    dataset (Inv. 13: pretraining only). ``evaluate_epoch(epoch)`` must return
    ``{"val_cpm": float, ...}`` computed through the official oracle on the paired val seed
    pool — the ONLY selection signal.

    Saves every epoch to ``out_dir/epoch{NN}.pt`` and returns the epoch table plus the
    post-hoc selected epoch.
    """
    import torch

    from ..detect.train import seed_everything

    seed_everything(int(seed))
    epochs = int(C.RESC_ENC_EPOCHS if epochs is None else epochs)
    opt_cfg = dict(C.RESC_ENC_OPT if opt_cfg is None else opt_cfg)
    base_lr = float(opt_cfg["lr"] if lr is None else lr)

    model = torch.nn.ModuleDict({"encoder": encoder, "head": head}).to(device)
    opt = make_optimizer(model, opt_cfg, lr=base_lr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(base_lr, ep, epochs)
        model.train()
        tot, nb = 0.0, 0
        for crops, rest, labels, _rows in train_loader:
            crops = crops.to(device).float()
            rest = rest.to(device).float()
            labels = labels.to(device).float()
            a = model["encoder"](crops)
            feats = torch.cat([a, rest], dim=-1)[:, None, :]        # a set of ONE (B1)
            logits = model["head"](feats).squeeze(1)
            # B1 is a CE rung: w_rank = 0 (a singleton set carries no ranking information)
            loss, _ = rescorer_loss(logits[None, :], labels[None, :], None,
                                    w_rank=0.0, lam=1.0, alpha=float(alpha))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1

        ckpt = out_dir / f"epoch{ep:02d}.pt"
        torch.save({"epoch": ep, "seed": int(seed), "encoder": encoder.state_dict(),
                    "head": head.state_dict(), "alpha": float(alpha), "lr": base_lr,
                    "encoder_name": C.RESC_ENCODER}, ckpt)
        metrics = evaluate_epoch(ep)
        rows.append({"epoch": ep, "train_loss": tot / max(1, nb), **metrics})
        print(f"  [enc seed{seed}] epoch {ep:02d} loss {rows[-1]['train_loss']:.4f} "
              f"val_cpm {metrics.get('val_cpm', float('nan')):.4f}", flush=True)

    write_epoch_table(rows, out_dir)
    chosen = select_epoch_by_val_cpm([r["epoch"] for r in rows], [r["val_cpm"] for r in rows])
    return {"epochs": rows, "selected_epoch": int(chosen),
            "selected_ckpt": str(out_dir / f"epoch{chosen:02d}.pt"),
            "selection_rule": f"earliest epoch within {C.RESC_SELECT_CPM_TOL} of max val_cpm"}


def train_set_variant(model, batches: Callable[[int], object], evaluate_epoch: Callable[[int], Dict],
                      out_dir, seed: int, w_rank: float, lam: float, alpha: float,
                      lr: Optional[float] = None, epochs: Optional[int] = None,
                      opt_cfg: Optional[Dict] = None, device: str = "cuda",
                      gamma: Optional[float] = None, soft_targets: bool = False,
                      row_weights=None) -> Dict:
    """[4.6] Train ONE ``(variant, seed, trial)`` over the cached embeddings.

    ``batches(epoch)`` yields dicts from ``datasets.collate_sets`` (already padded + masked);
    the batching unit is a SET, ``RESC_SET_BATCH_SETS`` sets per step. Sets with no positive
    are KEPT — they are the all-negative calibration anchors ``focal_bce`` needs.

    ``gamma`` / ``soft_targets`` / ``row_weights`` are the `[4.2b]` objective-alignment knobs
    and are inert at their defaults. ``row_weights`` is indexed **per record row** and is
    gathered onto each padded batch through ``collate_sets``' ``rows`` map, so it needs no
    change to the batch schema and padding can never pick up a weight.
    """
    import torch

    from .datasets import batch_row_weights
    from ..detect.train import seed_everything

    seed_everything(int(seed))
    epochs = int(C.RESC_SET_EPOCHS if epochs is None else epochs)
    opt_cfg = dict(C.RESC_SET_OPT if opt_cfg is None else opt_cfg)
    base_lr = float(opt_cfg["lr"] if lr is None else lr)

    model = model.to(device)
    opt = make_optimizer(model, opt_cfg, lr=base_lr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(base_lr, ep, epochs)
        model.train()
        tot, rank_tot, bce_tot, nb = 0.0, 0.0, 0.0, 0
        for batch in batches(ep):
            t = {k: torch.as_tensor(v).to(device) for k, v in batch.items()
                 if k in ("feats", "labels", "coord", "length", "mask")}
            logits = model(t["feats"].float(), t["coord"].float(), t["length"].float(), t["mask"])
            bw = batch_row_weights(batch["rows"], row_weights)
            if bw is not None:
                bw = torch.as_tensor(bw).to(logits.device)
            loss, parts = rescorer_loss(logits, t["labels"].float(), t["mask"].float(),
                                        w_rank=float(w_rank), lam=float(lam), alpha=float(alpha),
                                        gamma=gamma, soft=bool(soft_targets), bce_weights=bw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
            rank_tot += float(parts["rank"].detach()); bce_tot += float(parts["bce"].detach())

        torch.save({"epoch": ep, "seed": int(seed), "model": model.state_dict(),
                    "w_rank": float(w_rank), "lam": float(lam), "alpha": float(alpha),
                    "lr": base_lr}, out_dir / f"epoch{ep:02d}.pt")
        metrics = evaluate_epoch(ep)
        rows.append({"epoch": ep, "train_loss": tot / max(1, nb),
                     "train_rank": rank_tot / max(1, nb), "train_bce": bce_tot / max(1, nb),
                     **metrics})
        print(f"  [seed{seed} w{w_rank} l{lam} a{alpha}] epoch {ep:02d} "
              f"loss {rows[-1]['train_loss']:.4f} val_cpm {metrics.get('val_cpm', float('nan')):.4f}",
              flush=True)

    write_epoch_table(rows, out_dir)
    chosen = select_epoch_by_val_cpm([r["epoch"] for r in rows], [r["val_cpm"] for r in rows])
    payload = {"epochs": rows, "selected_epoch": int(chosen),
               "selected_ckpt": str(out_dir / f"epoch{chosen:02d}.pt"),
               "selected_val_cpm": float(rows[chosen]["val_cpm"]),
               "hyperparameters": {"w_rank": float(w_rank), "lam": float(lam),
                                   "alpha": float(alpha), "lr": base_lr, "epochs": epochs},
               "selection_rule": f"earliest epoch within {C.RESC_SELECT_CPM_TOL} of max val_cpm"}
    (out_dir / "selection.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "epochs"}, indent=2, sort_keys=True))
    return payload
