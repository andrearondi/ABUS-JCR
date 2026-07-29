"""[4.1] The 3D crop the shared encoder sees — the Inv.-5/6 coordinate contract.

The crop is cut from the **0.4 mm isotropic cache** in **storage order ``(d0, d1, d2)``**
using the record's frozen ``cen_d*``/``ext_d*`` — the very numbers
``link.reconstruct.iso_centre_of_tube`` / ``iso_extents_of_tube`` wrote. The official
``coordX..z_length`` columns are NEVER used to place a crop; they are the scoring box and
live in the permuted ITK order (``PERM_STORAGE_TO_ITK = (2,1,0)``). Applying that
permutation before indexing the iso volume is the classic silent bug — it returns an
essentially empty crop (spec "Check 1": empty in 5/6 Val cases) and is pinned as a
negative control in ``tests/test_rescore_crops.py``.

**Adaptive CUBIC ROI, not a fixed 48³ voxel box.** The Train size audit rules the literal
48³ out: union-box in-plane diag p50 = 54.1 iso px, p99 = 231.1, and GT in-plane extent
p99 = 228 — a 48-voxel (19.2 mm) box is smaller than the MEDIAN lesion's in-plane
diagonal. So the ROI side is sized from the candidate's own iso extents,
``clip(RESC_CROP_CONTEXT * max(ext), RESC_CROP_MIN_SIDE, RESC_CROP_MAX_SIDE)``, and
trilinearly resampled onto the fixed ``RESC_CROP_OUT³`` grid. The ROI is **cubic** (not
per-axis normalised) so the crop stays isotropic in PHYSICAL space and a thin
depth-column shadow still looks like a thin depth-column shadow; absolute size is not
lost, it is carried explicitly by the ``abs_geom`` + ``score_stats`` token blocks.

Outside the volume is **zero** (anechoic black), never edge-replicated: replicating the
skin line downward is acoustically impossible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import conventions as C
from ..preprocess import preprocess_hash

__all__ = ["roi_side_iso", "sample_grid", "extract_crop", "pad_fraction", "crop_hash",
           "build_crop_cache", "open_crop_cache", "crop_cache_dir"]

_SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------- geometry
def roi_side_iso(ext_d0: float, ext_d1: float, ext_d2: float,
                 context: Optional[float] = None,
                 min_side: Optional[float] = None,
                 max_side: Optional[float] = None) -> float:
    """Cubic ROI side in iso voxels = ``clip(context * max(ext), min_side, max_side)``.

    Defaults are read from :mod:`conventions` at call time (so a monkeypatched constant is
    honoured, and the value can never drift from ``crop_hash``).
    """
    context = C.RESC_CROP_CONTEXT if context is None else context
    min_side = C.RESC_CROP_MIN_SIDE if min_side is None else min_side
    max_side = C.RESC_CROP_MAX_SIDE if max_side is None else max_side
    raw = float(context) * float(max(ext_d0, ext_d1, ext_d2))
    return float(min(max(raw, float(min_side)), float(max_side)))


def sample_grid(cen: float, side: float, out: int) -> np.ndarray:
    """Cell-centred sample coordinates along one axis: ``lo + (side/out)*(k+0.5)``.

    ``lo = cen - side/2``. The grid is symmetric about ``cen``, so the ROI centre lands on
    the crop's geometric centre ``(out-1)/2`` regardless of ``side``.
    """
    lo = float(cen) - float(side) / 2.0
    step = float(side) / int(out)
    return lo + step * (np.arange(int(out), dtype=float) + 0.5)


def extract_crop(vol_iso, cen_d0: float, cen_d1: float, cen_d2: float,
                 ext_d0: float, ext_d1: float, ext_d2: float,
                 out: Optional[int] = None, side: Optional[float] = None) -> np.ndarray:
    """``(out, out, out)`` float32 crop in storage order, centred on ``cen``.

    ``side`` overrides the adaptive ROI (used by the [4.2] fixed-crop control run).
    Interpolation is ``map_coordinates(order=RESC_CROP_INTERP, mode='constant',
    cval=RESC_CROP_PAD_VALUE)``; values are clipped back to the cache's ``[0, 1]``
    contract (linear interpolation cannot overshoot, but rounding can).
    """
    out = C.RESC_CROP_OUT if out is None else int(out)
    side = roi_side_iso(ext_d0, ext_d1, ext_d2) if side is None else float(side)
    vol = np.asarray(vol_iso)
    g = [sample_grid(c, side, out) for c in (cen_d0, cen_d1, cen_d2)]
    coords = np.stack(np.meshgrid(g[0], g[1], g[2], indexing="ij"), axis=0)
    crop = ndimage.map_coordinates(
        vol, coords, order=int(C.RESC_CROP_INTERP), mode="constant",
        cval=float(C.RESC_CROP_PAD_VALUE),
    ).astype(np.float32)
    np.clip(crop, 0.0, 1.0, out=crop)
    return crop


def pad_fraction(shape: Tuple[int, int, int], cen, side: float, out: Optional[int] = None) -> float:
    """Fraction of the crop's sample points that fall OUTSIDE the volume (→ zero padding).

    The grid is separable, so the inside-fraction is the product of the per-axis
    inside-fractions. Reported by [4.1] because the ``RESC_CROP_MAX_SIDE = 160`` clamp
    exceeds the 158-voxel depth axis for the largest lesions.
    """
    out = C.RESC_CROP_OUT if out is None else int(out)
    inside = 1.0
    for a in range(3):
        g = sample_grid(cen[a], side, out)
        inside *= float(((g >= 0.0) & (g <= shape[a] - 1)).sum()) / out
    return float(1.0 - inside)


# ----------------------------------------------------------------------------- hashing
def _canonical_crop_cfg() -> Dict:
    """The dict the crop-cache hash is taken over. Every entry is crop-invalidating."""
    return {
        "RESC_CROP_OUT": C.RESC_CROP_OUT,
        "RESC_CROP_CONTEXT": C.RESC_CROP_CONTEXT,
        "RESC_CROP_MIN_SIDE": C.RESC_CROP_MIN_SIDE,
        "RESC_CROP_MAX_SIDE": C.RESC_CROP_MAX_SIDE,
        "RESC_CROP_INTERP": C.RESC_CROP_INTERP,
        "RESC_CROP_PAD_VALUE": C.RESC_CROP_PAD_VALUE,
        "preprocess_hash": preprocess_hash(),
        "schema_version": _SCHEMA_VERSION,
    }


def crop_hash() -> str:
    """SHA-256 over the crop config + ``preprocess_hash`` — names the crop-cache directory.

    Mirrors :func:`preprocess.preprocess_hash`: a stale crop cache can never be silently
    reused, because changing any crop constant (or the iso cache itself) yields a new dir.
    """
    blob = json.dumps(_canonical_crop_cfg(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def crop_cache_dir(out_dir, split: str) -> Path:
    return Path(out_dir) / crop_hash() / str(split)


# ----------------------------------------------------------------------------- the cache
def _default_vol_loader(cache_root, pid: int):
    from ..cache import open_vol
    return open_vol(cache_root, int(pid))


def build_crop_cache(record_df: pd.DataFrame, cache_root, out_dir, split: str,
                     vol_loader: Optional[Callable] = None, progress: bool = True) -> Dict:
    """Materialise one float16 memmap ``crops.npy`` of shape ``(n_rows, out, out, out)``.

    Row ``k`` of the memmap ↔ row ``k`` of the record — the same construction-preserved
    join the official pred CSV uses (``candidates.record.to_official_pred_csv``). Also
    writes ``index.csv`` (candidate_id, row, public_id, detector_of_origin) and
    ``CROP_META.json``.

    Asserts every record row's ``preprocess_hash`` equals the current
    :func:`preprocess.preprocess_hash` — a crop cut from a different iso cache would be a
    silent Inv.-6 violation.

    Returns the [4.1] report: row/volume counts, the ROI-side histogram, the pad-fraction
    distribution, and the **max set size** over the record (a SET is one
    ``(detector_of_origin, public_id)`` pair, Inv. 7) — asserted against
    ``RESC_MAX_SET_SIZE`` by the caller.
    """
    vol_loader = vol_loader or _default_vol_loader
    n = len(record_df)
    current = preprocess_hash()
    bad = record_df[record_df["preprocess_hash"].astype(str) != current]
    if len(bad):
        raise ValueError(
            f"{len(bad)}/{n} record rows carry preprocess_hash != {current}; the record was "
            f"generated against a different iso cache. Regenerate, do not crop.")

    cdir = crop_cache_dir(out_dir, split)
    cdir.mkdir(parents=True, exist_ok=True)
    out = int(C.RESC_CROP_OUT)
    crops = np.lib.format.open_memmap(cdir / "crops.npy", mode="w+", dtype=np.float16,
                                      shape=(n, out, out, out))

    sides = np.zeros(n, dtype=float)
    pads = np.zeros(n, dtype=float)
    pid_col = record_df["public_id"].to_numpy()
    # group by volume so each iso volume is read from disk exactly once; `rows` are POSITIONAL
    # record indices, so memmap row k stays candidate k regardless of grouping order
    for pid in pd.unique(pid_col):
        rows = np.flatnonzero(pid_col == pid)
        vol = np.asarray(vol_loader(cache_root, int(pid)))
        if progress:
            print(f"  vol {int(pid)}: {len(rows)} candidates, iso shape {vol.shape}", flush=True)
        for k in rows:
            r = record_df.iloc[int(k)]
            cen = (float(r["cen_d0"]), float(r["cen_d1"]), float(r["cen_d2"]))
            ext = (float(r["ext_d0"]), float(r["ext_d1"]), float(r["ext_d2"]))
            side = roi_side_iso(*ext)
            crops[int(k)] = extract_crop(vol, *cen, *ext, side=side).astype(np.float16)
            sides[k] = side
            pads[k] = pad_fraction(vol.shape, cen, side)
    crops.flush()

    idx = pd.DataFrame({
        "candidate_id": record_df["candidate_id"].to_numpy(),
        "row": np.arange(n),
        "public_id": record_df["public_id"].to_numpy(),
        "detector_of_origin": record_df["detector_of_origin"].to_numpy(),
        "roi_side": sides,
        "pad_frac": pads,
    })
    idx.to_csv(cdir / "index.csv", index=False)

    set_sizes = record_df.groupby(["detector_of_origin", "public_id"], sort=False).size()
    stats = {
        "split": split,
        "crop_hash": crop_hash(),
        "preprocess_hash": current,
        "n_rows": int(n),
        "n_volumes": int(record_df["public_id"].nunique()),
        "n_sets": int(len(set_sizes)),
        "max_set_size": int(set_sizes.max()) if len(set_sizes) else 0,
        "median_set_size": float(set_sizes.median()) if len(set_sizes) else float("nan"),
        "roi_side_min": float(sides.min()) if n else float("nan"),
        "roi_side_median": float(np.median(sides)) if n else float("nan"),
        "roi_side_max": float(sides.max()) if n else float("nan"),
        "roi_side_hist": {str(int(b)): int(c) for b, c in
                          zip(*np.unique(np.round(sides).astype(int), return_counts=True))},
        "pad_frac_median": float(np.median(pads)) if n else float("nan"),
        "pad_frac_p90": float(np.percentile(pads, 90)) if n else float("nan"),
        "pad_frac_max": float(pads.max()) if n else float("nan"),
        "bytes": int(crops.nbytes),
    }
    (cdir / "CROP_META.json").write_text(json.dumps(
        {**stats, "config": _canonical_crop_cfg()}, sort_keys=True, indent=2))
    return stats


def open_crop_cache(out_dir, split: str):
    """Memmap an existing crop cache (read-only) + its ``CROP_META.json``.

    Refuses a cache whose recorded hashes disagree with the current config — the same
    guard :func:`cache.assert_hash` applies to the iso cache.
    """
    cdir = crop_cache_dir(out_dir, split)
    meta_path = cdir / "CROP_META.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no CROP_META.json under {cdir}; run scripts/phase4_build_crops.py")
    meta = json.loads(meta_path.read_text())
    if meta.get("crop_hash") != crop_hash() or meta.get("preprocess_hash") != preprocess_hash():
        raise ValueError(f"stale crop cache at {cdir}: recorded {meta.get('crop_hash')} / "
                         f"{meta.get('preprocess_hash')} vs current {crop_hash()} / {preprocess_hash()}")
    return np.load(cdir / "crops.npy", mmap_mode="r"), meta
