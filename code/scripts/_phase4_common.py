"""Shared path resolution + feature/batching plumbing for the Phase-4 scripts.

Phase 4 consumes the Phase-1 substrate (iso cache, manifest) under ``--phase1-out`` and the
**frozen** Phase-3 candidate pool under ``--phase3-out``; it writes its own artefacts under
``--out-root``. Nothing here regenerates a candidate (Inv. 8) or reads Test (Inv. 9).

Defaults follow ``SERVER_LAYOUT.md``; override for local runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from abus_jcr import conventions as C
from abus_jcr.cache import read_meta
from abus_jcr.candidates.record import read_candidate_record
from abus_jcr.gt_labels import load_gt_documented, to_official_gt
from abus_jcr.rescore.datasets import collate_sets, group_sets
from abus_jcr.rescore.losses import encode_labels
from abus_jcr.rescore.tokens import apply_standardiser, build_feature_matrix, fit_standardiser

DEFAULT_PHASE1_OUT = "/home/maia-user/Andre2/outputs/phase1"
DEFAULT_PHASE3_OUT = "/home/maia-user/Andre2/outputs/phase3"
DEFAULT_PHASE4_OUT = "/home/maia-user/Andre2/outputs/phase4"
DEFAULT_DATA_ROOT = "/home/maia-user/Andre2/data"

_SPLIT_DIR = {"train": "Train", "val": "Validation"}   # Test is Phase 5 only (Inv. 9)


def add_phase4_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase1-out", default=DEFAULT_PHASE1_OUT,
                        help=f"Phase-1 output root (cache/, manifest.csv); default {DEFAULT_PHASE1_OUT}")
    parser.add_argument("--phase3-out", default=DEFAULT_PHASE3_OUT,
                        help=f"Phase-3 output root (candidates/); default {DEFAULT_PHASE3_OUT}")
    parser.add_argument("--out-root", default=DEFAULT_PHASE4_OUT,
                        help=f"Phase-4 output root; default {DEFAULT_PHASE4_OUT}")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"dataset root holding the split dirs; default {DEFAULT_DATA_ROOT}")
    parser.add_argument("--record-suffix", default="",
                        help="'' = the PRIMARY frozen pool (default), which since 2026-08-08 IS "
                             "the promoted corrected-flip pool. The 'hicov' contingency pool is "
                             "RETIRED (the promoted folds reach its 81 TP-bearing sets without its "
                             "costs) — do not promote it. Any substrate change needs the user's "
                             "explicit go-ahead (spec Open escalation #3).")


# ----------------------------------------------------------------------------- paths
def cache_root(args) -> Path:
    return Path(args.phase1_out) / "cache"


def candidates_dir(args) -> Path:
    return Path(args.phase3_out) / "candidates"


def crops_dir(args) -> Path:
    return Path(args.out_root) / "crops"


def encoder_dir(args, seed: int) -> Path:
    return Path(args.out_root) / "encoder" / f"seed{int(seed)}"


def embeddings_dir(args) -> Path:
    return Path(args.out_root) / "embeddings"


def emb_path(args, split: str, seed: int) -> Path:
    return embeddings_dir(args) / f"emb_{split}_seed{int(seed)}.npy"


def variant_dir(args, variant: str, seed: int, trial: int) -> Path:
    return Path(args.out_root) / "variants" / f"{variant}_seed{int(seed)}_trial{int(trial)}"


def grid_dir(args) -> Path:
    return Path(args.out_root) / "grid"


def split_root(args, split: str) -> Path:
    return Path(args.data_root) / _SPLIT_DIR[split]


# ----------------------------------------------------------------------------- data
def load_record(args, split: str) -> pd.DataFrame:
    """The frozen Phase-3 candidate record for a split (Parquet, CSV fallback)."""
    if split not in _SPLIT_DIR:
        raise SystemExit(f"Phase 4 reads train/val only (Inv. 9); got {split!r}")
    suffix = f"_{args.record_suffix}" if getattr(args, "record_suffix", "") else ""
    base = candidates_dir(args) / f"candidates_{split}{suffix}"
    rec = read_candidate_record(base).reset_index(drop=True)
    print(f"# record {base.name}: {len(rec)} rows, {rec['public_id'].nunique()} volumes, "
          f"{rec.groupby(['detector_of_origin','public_id']).ngroups} sets, "
          f"pos={int((rec.label=='pos').sum())} neg={int((rec.label=='neg').sum())} "
          f"ignore={int((rec.label=='ignore').sum())}", flush=True)
    return rec


def load_gt(args, split: str) -> pd.DataFrame:
    """Official GT boxes from the split's shipped ``bbx_labels.csv`` (Inv. 3)."""
    return to_official_gt(load_gt_documented(split_root(args, split) / "bbx_labels.csv"))


def iso_shape_map(args, record_df: pd.DataFrame) -> Dict[int, Sequence[int]]:
    """``public_id -> META['iso_shape']`` for the ``abs_geom`` centroid normalisation."""
    return {int(p): read_meta(cache_root(args), int(p))["iso_shape"]
            for p in sorted(record_df["public_id"].unique())}


def seed_detector(seed: int) -> str:
    """Rescorer seed ``r`` is PAIRED with detector ``full_seed{r}`` (spec §4.8)."""
    return f"full_seed{int(seed)}"


def val_pool_for_seed(record_val: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The one seed pool this replica is evaluated on — never the 3 concatenated (Inv. 14)."""
    det = seed_detector(seed)
    sub = record_val[record_val["detector_of_origin"] == det]
    if not len(sub):
        raise SystemExit(f"no val candidates for {det}; available: "
                         f"{sorted(record_val['detector_of_origin'].unique())}")
    return sub.reset_index(drop=True)


def gt_for_pool(gt_df: pd.DataFrame, record_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the GT table to the volumes this pool actually covers.

    On the promoted pool the TRAIN split covers **100/100** volumes (the archived arm covered
    97). The gap that remains is per-seed on VAL: ``full_seed0`` produces zero candidates for
    **vol 129**, so it contributes 29 sets, not 30 ([F.7]). Scoring an uncovered volume as a
    silent miss would understate every rung by the same amount, but it would also make the
    Phase-4 numbers incomparable to B0, which is computed on the pool's own volumes — so this
    mirrors B0, and the covered-volume count is printed for the record.

    Note the official oracle still divides recall by the **30** GT lesions in the split, so
    seed0's missing set is not silently inflating its recall ([F.8] audit).
    """
    pids = set(int(p) for p in record_df["public_id"].unique())
    return gt_df[gt_df["public_id"].astype(int).isin(pids)].reset_index(drop=True)


# ----------------------------------------------------------------------------- features
def rest_blocks(use_blocks: Sequence[str]) -> Tuple[str, ...]:
    """The non-appearance blocks (what the encoder-pretraining loader supplies as ``rest``)."""
    return tuple(b for b in use_blocks if b != "appearance")


def build_features(record_df: pd.DataFrame, emb: Optional[np.ndarray], use_blocks: Sequence[str],
                   iso_shapes: Dict[int, Sequence[int]],
                   stats: Optional[Dict] = None) -> Tuple[np.ndarray, List[str], Dict]:
    """Assemble + standardise the token matrix. ``stats=None`` FITS (train only, Inv. 9/10)."""
    X, names = build_feature_matrix(record_df, emb=emb, use_blocks=use_blocks,
                                    iso_shape_by_pid=iso_shapes)
    if stats is None:
        stats = fit_standardiser(X)
    return apply_standardiser(X, stats), names, stats


def boxes_of(record_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """The OFFICIAL box columns the Axis-A descriptor consumes (never the iso ones)."""
    return (record_df[["coordX", "coordY", "coordZ"]].to_numpy(float),
            record_df[["x_length", "y_length", "z_length"]].to_numpy(float))


def label_codes(record_df: pd.DataFrame) -> np.ndarray:
    return encode_labels(record_df["label"].to_numpy())


def set_batches(record_df: pd.DataFrame, feats: np.ndarray, seed: int,
                batch_sets: Optional[int] = None, max_set_size: Optional[int] = None,
                shuffle: bool = True):
    """A ``batches(epoch)`` callable for ``rescore.train.train_set_variant``.

    The batching unit is a SET (Inv. 7). No negative subsampling
    (``RESC_NEG_POS_RATIO is None``) and no set is dropped: on the promoted train pool the
    **19 of 100** all-negative sets (81/100 are TP-bearing, [F.9] §4) are the cross-volume
    calibration anchors ``focal_bce`` needs, and dropping them would remove the only term
    that supplies a level.

    ``max_set_size=None`` pads each batch to the **batch** max, not to the global
    ``RESC_MAX_SET_SIZE`` — see ``datasets.collate_sets``.
    """
    batch_sets = int(C.RESC_SET_BATCH_SETS if batch_sets is None else batch_sets)
    groups = list(group_sets(record_df).values())
    labels = label_codes(record_df)
    coord, length = boxes_of(record_df)
    feats32 = np.ascontiguousarray(feats, dtype=np.float32)

    def batches(epoch: int):
        order = np.arange(len(groups))
        if shuffle:
            np.random.default_rng((int(seed), int(epoch))).shuffle(order)
        for s in range(0, len(order), batch_sets):
            chunk = [groups[i] for i in order[s:s + batch_sets]]
            yield collate_sets(chunk, feats32, labels, coord, length, max_set_size=max_set_size)

    return batches


def set_index_lists(record_df: pd.DataFrame) -> List[np.ndarray]:
    return list(group_sets(record_df).values())


# ----------------------------------------------------------------------------- the runner
def load_variant_inputs(args, seed: int, use_blocks: Sequence[str]):
    """Everything a set-module run needs: records, standardised tokens, GT, boxes.

    The standardiser is fitted on the TRAIN pool and applied unchanged to val (Inv. 9/10).
    """
    rec_tr = load_record(args, "train")
    rec_va_all = load_record(args, "val")
    rec_va = val_pool_for_seed(rec_va_all, seed)

    emb_tr = np.load(emb_path(args, "train", seed))
    emb_va_all = np.load(emb_path(args, "val", seed))
    va_rows = rec_va_all.index[rec_va_all["detector_of_origin"] == seed_detector(seed)].to_numpy()
    emb_va = emb_va_all[va_rows]

    use_emb = "appearance" in use_blocks
    Ztr, names, stats = build_features(rec_tr, emb_tr if use_emb else None, use_blocks,
                                       iso_shape_map(args, rec_tr), stats=None)
    Zva, _, _ = build_features(rec_va, emb_va if use_emb else None, use_blocks,
                               iso_shape_map(args, rec_va), stats=stats)
    gt_va = gt_for_pool(load_gt(args, "val"), rec_va)
    gt_tr = gt_for_pool(load_gt(args, "train"), rec_tr)
    return {"rec_tr": rec_tr, "rec_va": rec_va, "Ztr": Ztr, "Zva": Zva, "names": names,
            "stats": stats, "gt_va": gt_va, "gt_tr": gt_tr, "d_in": int(Ztr.shape[1])}


def set_module_params(d_in: int, capacity, use_geometry: bool = False) -> int:
    """Parameter count of the set module at ``capacity`` — the fairness reference."""
    from abus_jcr.rescore.setmodel import RGFrocRescorer

    _tag, n_layers, d_model, n_heads = capacity
    m = RGFrocRescorer(d_in=d_in, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                       use_geometry=use_geometry)
    return int(sum(p.numel() for p in m.parameters()))


def run_variant_trial(args, variant: str, seed: int, trial: Dict, capacity, out_dir,
                      use_blocks: Sequence[str] = C.RESC_TOKEN_BLOCKS,
                      geom_mechanism: Optional[str] = None, device: str = "cuda",
                      epochs: Optional[int] = None, n_boot: int = 200,
                      inputs: Optional[Dict] = None) -> Dict:
    """Train ONE ``(variant, seed, hyperparameter trial)`` and return its selection payload.

    Shared by [4.5] capacity selection, [4.6] the grid, and [4.8] the sub-ablations, so the
    schedule, the selection rule and the fairness accounting cannot drift between them.
    """
    import torch

    from abus_jcr.rescore.evaluate import evaluate_variant, score_pool
    from abus_jcr.rescore.setmodel import build_rescorer
    from abus_jcr.rescore.train import train_set_variant
    from abus_jcr.rescore.variants import VARIANTS, match_b1_capacity

    inputs = inputs or load_variant_inputs(args, seed, use_blocks)
    d_in = inputs["d_in"]
    _tag, n_layers, d_model, n_heads = capacity

    hidden = None
    if VARIANTS[variant]["module"] == "mlp":
        target = set_module_params(d_in, capacity, use_geometry=False)
        hidden = match_b1_capacity(d_in, d_model, target)
        print(f"# fairness: B1 hidden {hidden} matched to the set module's {target} params")
    model = build_rescorer(variant, d_in, capacity, hidden=hidden, geom_mechanism=geom_mechanism)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(f"# {variant} seed{seed} {trial}: d_in={d_in}, capacity={_tag}, params={n_params}")

    batches = set_batches(inputs["rec_tr"], inputs["Ztr"], seed=seed)
    va_sets = set_index_lists(inputs["rec_va"])
    va_coord, va_length = boxes_of(inputs["rec_va"])
    Zva32 = np.ascontiguousarray(inputs["Zva"], dtype=np.float32)

    def evaluate_epoch(epoch: int) -> Dict:
        prob = score_pool(model, Zva32, va_coord, va_length, va_sets,
                          n_rows=len(inputs["rec_va"]), device=device)
        res = evaluate_variant(inputs["rec_va"], prob, inputs["gt_va"],
                               seed_tag=f"{variant}_seed{seed}", n_boot=n_boot)
        model.train()
        return {"val_cpm": res["cpm"], "val_ceiling": res["ceiling"],
                "val_ci_lo": res["ci"]["lo"], "val_ci_hi": res["ci"]["hi"]}

    payload = train_set_variant(model, batches, evaluate_epoch, out_dir, seed=seed,
                                w_rank=trial["w_rank"], lam=trial["lam"], alpha=trial["alpha"],
                                lr=trial["lr"], epochs=epochs, device=device)
    payload.update({"variant": variant, "seed": int(seed), "capacity": list(capacity),
                    "params": n_params, "b1_hidden": hidden, "d_in": d_in,
                    "use_blocks": list(use_blocks),
                    "geom_mechanism": geom_mechanism or C.RESC_GEOM_MECHANISM})
    return payload


def load_deployed_report(args, variant: str, seed: int) -> Dict:
    """The ``[4.6]`` report for one ``(variant, seed)`` — the deployed trial + checkpoint."""
    p = Path(args.out_root) / "variants" / f"{variant}_seed{int(seed)}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} — run [4.6] for {variant} seed {seed} first")
    return json.loads(p.read_text())


def b1_hidden_for(args, d_in: int, capacity) -> int:
    """B1's fairness-matched hidden width: the [4.5] value if frozen, else recomputed."""
    from abus_jcr.rescore.variants import match_b1_capacity

    p = Path(args.out_root) / "capacity" / "capacity_choice.json"
    if p.exists():
        return int(json.loads(p.read_text())["b1_hidden"])
    return match_b1_capacity(d_in, capacity[2], set_module_params(d_in, capacity))


def load_deployed_model(args, variant: str, seed: int, d_in: int, device: str = "cuda"):
    """Rebuild a rung's module and load its deployed checkpoint."""
    import torch

    from abus_jcr.rescore.setmodel import build_rescorer

    rep = load_deployed_report(args, variant, seed)
    capacity = tuple(rep["capacity"])
    hidden = b1_hidden_for(args, d_in, capacity) if variant == "B1" else None
    model = build_rescorer(variant, d_in, capacity, hidden=hidden,
                           geom_mechanism=rep.get("geom_mechanism"))
    state = torch.load(rep["deployed"]["selected_ckpt"], map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval(), rep


# ----------------------------------------------------------------------------- misc
def assert_device(device: str) -> None:
    """Fail loudly if CUDA is requested but unavailable (reused Phase-2/3 guard)."""
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"--device {device} requested but torch is not installed ({e}). "
                         "Activate the abus-jcr env on the GPU host.")
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {device} requested but torch reports no CUDA GPU "
                         f"(device_count={torch.cuda.device_count()}). Move to the A6000 host.")


def dump_json(payload, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))
    print(f"# wrote {path}", flush=True)
    return path


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)
