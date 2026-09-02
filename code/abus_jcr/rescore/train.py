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
           "write_epoch_table", "select_epoch_by_val_cpm", "resolve_model",
           "froc_beta", "ref_complement"]


def froc_beta(epoch: int, epochs: int) -> float:
    """The pooled-surrogate β for this epoch: ``RESC_FROC_BETA0 → RESC_FROC_BETA``, linear
    over the first ``RESC_FROC_BETA_ANNEAL_FRAC`` of the budget, constant after.

    Exists because a bounded surrogate has no gradient at initialisation when every lesion
    sits far from every budget (RB_PHASE_4_ISO §12 Step 3). Schedule decided with the user
    2026-09-02, before any ``-P`` rung trained. Pure and torch-free so the schedule is
    pinned by ``tests/test_froc_wiring.py`` on any machine.
    """
    anneal = max(1.0, float(epochs) * float(C.RESC_FROC_BETA_ANNEAL_FRAC))
    t = min(1.0, max(0.0, float(epoch)) / anneal)
    return float(C.RESC_FROC_BETA + (C.RESC_FROC_BETA0 - C.RESC_FROC_BETA) * (1.0 - t))


def ref_complement(rows, n_pool: int):
    """Boolean keep-mask over the pool: True for rows NOT in this batch.

    The reference table is scored once per epoch over the WHOLE pool; a batch's own rows
    must be excluded before it is passed to the loss, or every batch candidate is counted
    twice (once live, once as its stale detached copy). ``rows`` is ``collate_sets``' map,
    ``-1`` on padding — padding excludes nothing.
    """
    import numpy as np

    keep = np.ones(int(n_pool), dtype=bool)
    rr = np.asarray(rows).reshape(-1)
    keep[rr[rr >= 0]] = False
    return keep


def resolve_model(model_or_factory, seed: int, seed_fn=None):
    """Seed **first**, then construct. Never the other way round.

    Callers used to build the module and *then* hand it to a trainer that seeded on entry, so a
    model's initial weights inherited whatever global RNG state the caller happened to leave
    behind. Measured cost: `[4.2b]` and `[4.2c]` scored the same cell on the same seed and the
    same data at **0.5796** and **0.6006** (+0.0210) because the reference block's bootstrap
    draws changed between the runs. Only the FIRST model built in a run is affected — every
    later one is constructed after a ``seed_everything`` plus deterministic training — which is
    exactly the pattern that was observed. The same shape sat in ``run_variant_trial``, so it
    reached `[4.5]`, `[4.6]` and `[4.8]` too.

    Pass a zero-argument **factory** and the ordering cannot be got wrong. A pre-built module is
    still accepted for compatibility, but it *cannot* be made reproducible — its initialisation
    has already happened — so every caller in this repo passes a factory.

    ``seed_fn`` exists so the ordering contract is testable without torch; it defaults to
    ``detect.train.seed_everything``, which is the project-wide one (python/numpy/torch plus
    deterministic cudnn), so the rescorer can never drift from the detector.
    """
    if seed_fn is None:
        from ..detect.train import seed_everything as seed_fn        # noqa: N806 (lazy: torch)
    seed_fn(int(seed))
    return model_or_factory() if callable(model_or_factory) else model_or_factory


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


def pretrain_encoder(model_factory, train_loader, evaluate_epoch: Callable[[int], Dict],
                     out_dir, seed: int, epochs: Optional[int] = None,
                     opt_cfg: Optional[Dict] = None, lr: Optional[float] = None,
                     alpha: float = 0.25, device: str = "cuda",
                     gamma: Optional[float] = None, row_weights=None) -> Dict:
    """[4.3] Train ``CandidateEncoder + B1Head`` end to end — this run **is** rung B1.

    ``model_factory`` is a zero-argument callable returning ``(encoder, head)``. It is invoked
    **after** seeding (:func:`resolve_model`) so the weights do not inherit the caller's RNG
    state — see that function for the measured cost of getting this backwards.

    ``train_loader`` yields ``(crops, rest, labels, rows)``; augmentation lives in the
    dataset (Inv. 13: pretraining only). ``evaluate_epoch(epoch)`` must return
    ``{"val_cpm": float, ...}`` computed through the official oracle on the paired val seed
    pool — the ONLY selection signal.

    ``gamma`` / ``row_weights`` carry the `[4.2c]`-promoted objective; both default to the
    ``conventions`` values, so the deployed behaviour is whatever is recorded there.
    ``row_weights`` is indexed per RECORD ROW and gathered with the loader's own ``rows``.

    Saves every epoch to ``out_dir/epoch{NN}.pt`` and returns the epoch table, the post-hoc
    selected epoch, and the constructed ``encoder``/``head``.
    """
    import numpy as np
    import torch

    encoder, head = resolve_model(model_factory, seed)
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
        # NB `batch_rows`, not `rows` — `rows` is the epoch table being accumulated below
        for crops, rest, labels, batch_rows in train_loader:
            crops = crops.to(device).float()
            rest = rest.to(device).float()
            labels = labels.to(device).float()
            a = model["encoder"](crops)
            feats = torch.cat([a, rest], dim=-1)[:, None, :]        # a set of ONE (B1)
            logits = model["head"](feats).squeeze(1)
            # the loader shuffles CANDIDATES, so the per-lesion weight is gathered per row
            bw = None
            if row_weights is not None:
                idx = np.asarray(batch_rows.cpu() if hasattr(batch_rows, "cpu")
                                 else batch_rows, dtype=int)
                bw = torch.as_tensor(
                    np.asarray(row_weights, dtype=np.float32)[idx][None, :]).to(device)
            # B1 is a CE rung: w_rank = 0 (a singleton set carries no ranking information)
            loss, _ = rescorer_loss(logits[None, :], labels[None, :], None,
                                    w_rank=0.0, lam=1.0, alpha=float(alpha),
                                    gamma=gamma, bce_weights=bw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1

        ckpt = out_dir / f"epoch{ep:02d}.pt"
        torch.save({"epoch": ep, "seed": int(seed), "encoder": encoder.state_dict(),
                    "head": head.state_dict(), "alpha": float(alpha), "lr": base_lr,
                    "encoder_name": C.RESC_ENCODER}, ckpt)
        metrics = evaluate_epoch(ep, encoder, head)
        rows.append({"epoch": ep, "train_loss": tot / max(1, nb), **metrics})
        print(f"  [enc seed{seed}] epoch {ep:02d} loss {rows[-1]['train_loss']:.4f} "
              f"val_cpm {metrics.get('val_cpm', float('nan')):.4f}", flush=True)

    write_epoch_table(rows, out_dir)
    chosen = select_epoch_by_val_cpm([r["epoch"] for r in rows], [r["val_cpm"] for r in rows])
    return {"epochs": rows, "selected_epoch": int(chosen),
            "selected_ckpt": str(out_dir / f"epoch{chosen:02d}.pt"),
            "selection_rule": f"earliest epoch within {C.RESC_SELECT_CPM_TOL} of max val_cpm",
            "encoder": encoder, "head": head}


def train_set_variant(model_factory, batches: Callable[[int], object],
                      evaluate_epoch: Callable[[int], Dict],
                      out_dir, seed: int, w_rank: float, lam: float, alpha: float,
                      lr: Optional[float] = None, epochs: Optional[int] = None,
                      opt_cfg: Optional[Dict] = None, device: str = "cuda",
                      gamma: Optional[float] = None, soft_targets: bool = False,
                      row_weights=None, rank_loss: Optional[str] = None,
                      n_vol: Optional[float] = None, ref_provider=None,
                      init_state=None) -> Dict:
    """[4.6] Train ONE ``(variant, seed, trial)`` over the cached embeddings.

    ``model_factory`` is a zero-argument callable; it is invoked **after** seeding
    (:func:`resolve_model`), which is what makes a rung reproducible across runs.

    ``evaluate_epoch(epoch, model)`` receives the module because the trainer, not the caller,
    now constructs it. ``batches(epoch)`` yields dicts from ``datasets.collate_sets`` (already
    padded + masked); the batching unit is a SET, ``RESC_SET_BATCH_SETS`` sets per step. Sets with no positive
    are KEPT — they are the all-negative calibration anchors ``focal_bce`` needs.

    ``gamma`` / ``soft_targets`` / ``row_weights`` are the `[4.2b]` objective-alignment knobs
    and are inert at their defaults. ``row_weights`` is indexed **per record row** and is
    gathered onto each padded batch through ``collate_sets``' ``rows`` map, so it needs no
    change to the batch schema and padding can never pick up a weight.

    **The pooled rungs (`B1-P`/`A2-P`/`FULL-P`, wired 2026-09-02).** Four opt-in pieces, all
    inert at their defaults so the six pre-registered rungs train bit-identically:

    ``rank_loss="froc"``
        routes the ranking term to :func:`losses.froc_surrogate_loss`; β follows
        :func:`froc_beta`'s anneal.
    ``ref_provider(epoch, model) -> (logits, label_codes)``
        the reference table — the WHOLE train pool scored detached once per epoch; each
        batch receives the :func:`ref_complement` of its own rows, so no candidate is
        counted both live and stale. The provider may flip ``model.eval()``; the loop
        restores ``train()``.
    ``n_vol``
        total train volumes (100), so the cost-per-volume rate is exact, not per-batch.
    ``init_state``
        the CE twin's deployed ``state_dict`` — loaded **after** :func:`resolve_model`
        seeds, so RNG state stays deterministic while the weights warm-start.
    """
    import numpy as np
    import torch

    from .datasets import batch_row_weights

    model = resolve_model(model_factory, seed)
    if init_state is not None:
        model.load_state_dict(init_state)
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
        ref_logits_pool = ref_labels_pool = None
        if ref_provider is not None:
            ref_logits_pool, ref_labels_pool = ref_provider(ep, model)
        beta_ep = froc_beta(ep, epochs) if rank_loss == "froc" else None
        model.train()                       # the ref provider scores under eval(); restore
        tot, rank_tot, bce_tot, nb = 0.0, 0.0, 0.0, 0
        for batch in batches(ep):
            t = {k: torch.as_tensor(v).to(device) for k, v in batch.items()
                 if k in ("feats", "labels", "coord", "length", "mask")}
            logits = model(t["feats"].float(), t["coord"].float(), t["length"].float(), t["mask"])
            bw = batch_row_weights(batch["rows"], row_weights)
            if bw is not None:
                bw = torch.as_tensor(bw).to(logits.device)
            extra: Dict = {}
            if rank_loss is not None:
                extra = {"rank_loss": rank_loss, "n_vol": n_vol, "beta": beta_ep}
                if ref_logits_pool is not None:
                    keep = ref_complement(batch["rows"], len(ref_logits_pool))
                    extra["ref_logits"] = torch.as_tensor(
                        np.ascontiguousarray(ref_logits_pool[keep]),
                        dtype=logits.dtype, device=logits.device)
                    extra["ref_labels"] = torch.as_tensor(
                        np.ascontiguousarray(ref_labels_pool[keep]),
                        dtype=logits.dtype, device=logits.device)
            loss, parts = rescorer_loss(logits, t["labels"].float(), t["mask"].float(),
                                        w_rank=float(w_rank), lam=float(lam), alpha=float(alpha),
                                        gamma=gamma, soft=bool(soft_targets), bce_weights=bw,
                                        **extra)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
            rank_tot += float(parts["rank"].detach()); bce_tot += float(parts["bce"].detach())

        torch.save({"epoch": ep, "seed": int(seed), "model": model.state_dict(),
                    "w_rank": float(w_rank), "lam": float(lam), "alpha": float(alpha),
                    "lr": base_lr}, out_dir / f"epoch{ep:02d}.pt")
        metrics = evaluate_epoch(ep, model)
        rows.append({"epoch": ep, "train_loss": tot / max(1, nb),
                     "train_rank": rank_tot / max(1, nb), "train_bce": bce_tot / max(1, nb),
                     **metrics})
        beta_note = f" beta {beta_ep:.3f}" if beta_ep is not None else ""
        print(f"  [seed{seed} w{w_rank} l{lam} a{alpha}] epoch {ep:02d} "
              f"loss {rows[-1]['train_loss']:.4f} val_cpm {metrics.get('val_cpm', float('nan')):.4f}"
              f"{beta_note}",
              flush=True)

    write_epoch_table(rows, out_dir)
    chosen = select_epoch_by_val_cpm([r["epoch"] for r in rows], [r["val_cpm"] for r in rows])
    payload = {"epochs": rows, "selected_epoch": int(chosen),
               "selected_ckpt": str(out_dir / f"epoch{chosen:02d}.pt"),
               "selected_val_cpm": float(rows[chosen]["val_cpm"]),
               "hyperparameters": {"w_rank": float(w_rank), "lam": float(lam),
                                   "alpha": float(alpha), "lr": base_lr, "epochs": epochs},
               "selection_rule": f"earliest epoch within {C.RESC_SELECT_CPM_TOL} of max val_cpm",
               "model": model}
    (out_dir / "selection.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k not in ("epochs", "model")},
        indent=2, sort_keys=True))
    return payload
