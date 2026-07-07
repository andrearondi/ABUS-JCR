"""On-disk isotropic cache: layout, hash guard, memmap slice access (Inv. 6).

Layout::

    <cache_root>/<preprocess_hash>/
    ├── CACHE_META.json            # hash inputs + schema version
    ├── vol/VOL_<id>.npy           # float32 [0,1], iso shape (d0,d1,d2)
    ├── mask/MASK_<id>.npy         # uint8 {0,1}, same shape
    └── meta/META_<id>.json        # per-volume META (inverse affine etc.)

The directory is named by :func:`abus_jcr.preprocess.preprocess_hash`; changing
any cache-invalidating input yields a new directory. :func:`assert_hash` refuses
to read a cache whose recorded hash disagrees with the current config, so a stale
cache is never silently reused. Volumes are memmapped so the slice dataset reads
a single ``[:, :, z]`` frame without loading the whole array.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from . import conventions as C
from .preprocess import preprocess_hash, _canonical_cfg

_SCHEMA_VERSION = 1


class CacheHashMismatch(RuntimeError):
    """Raised when a cache's recorded preprocess hash != the current config."""


def cache_dir(cache_root) -> Path:
    """The hash-named cache directory under ``cache_root``."""
    return Path(cache_root) / preprocess_hash()


def _write_cache_meta(cdir: Path) -> None:
    meta_path = cdir / "CACHE_META.json"
    if meta_path.exists():
        return
    payload = {
        "preprocess_hash": preprocess_hash(),
        "schema_version": _SCHEMA_VERSION,
        "config": _canonical_cfg(C.ISO_SPACING_MM),
    }
    meta_path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def write_case(cache_root, volume_id: int, vol_iso: np.ndarray, mask_iso: np.ndarray, meta: Dict) -> None:
    """Write one case (vol, mask, meta) into the hash-named cache directory."""
    cdir = cache_dir(cache_root)
    (cdir / "vol").mkdir(parents=True, exist_ok=True)
    (cdir / "mask").mkdir(parents=True, exist_ok=True)
    (cdir / "meta").mkdir(parents=True, exist_ok=True)
    _write_cache_meta(cdir)

    np.save(cdir / "vol" / f"VOL_{volume_id}.npy", np.ascontiguousarray(vol_iso, dtype=np.float32))
    np.save(cdir / "mask" / f"MASK_{volume_id}.npy", np.ascontiguousarray(mask_iso, dtype=np.uint8))
    (cdir / "meta" / f"META_{volume_id}.json").write_text(json.dumps(meta, sort_keys=True, indent=2))


def open_vol(cache_root, volume_id: int) -> np.memmap:
    """Memmap the isotropic volume (read-only)."""
    return np.load(cache_dir(cache_root) / "vol" / f"VOL_{volume_id}.npy", mmap_mode="r")


def open_mask(cache_root, volume_id: int) -> np.memmap:
    """Memmap the isotropic mask (read-only)."""
    return np.load(cache_dir(cache_root) / "mask" / f"MASK_{volume_id}.npy", mmap_mode="r")


def read_meta(cache_root, volume_id: int) -> Dict:
    """Load one case's per-volume META."""
    return json.loads((cache_dir(cache_root) / "meta" / f"META_{volume_id}.json").read_text())


def assert_hash(cache_root) -> None:
    """Refuse to proceed if the cache's recorded hash != the current config."""
    cm_path = cache_dir(cache_root) / "CACHE_META.json"
    if not cm_path.exists():
        raise CacheHashMismatch(f"no CACHE_META.json under {cache_dir(cache_root)}")
    recorded = json.loads(cm_path.read_text()).get("preprocess_hash")
    current = preprocess_hash()
    if recorded != current:
        raise CacheHashMismatch(
            f"cache hash {recorded} != current preprocess_hash {current}; "
            f"regenerate the cache (use --force) instead of reusing a stale one"
        )
