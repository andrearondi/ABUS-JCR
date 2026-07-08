"""Shared path resolution + loaders for the Phase-2 scripts.

Phase 2 consumes the Phase-1 substrate (iso cache, manifest, slice-box tables)
under ``--phase1-out`` and writes its own artifacts under ``--out-root``. Paths
default to SERVER_LAYOUT.md; override for local runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PHASE1_OUT = "/home/maia-user/Andre2/outputs/phase1"
DEFAULT_PHASE2_OUT = "/home/maia-user/Andre2/outputs/phase2"


def add_phase2_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase1-out", default=DEFAULT_PHASE1_OUT,
                        help=f"Phase-1 output root (cache/, labels/, manifest.csv); default {DEFAULT_PHASE1_OUT}")
    parser.add_argument("--out-root", default=DEFAULT_PHASE2_OUT,
                        help=f"Phase-2 output root; default {DEFAULT_PHASE2_OUT}")


def cache_root(args) -> Path:
    return Path(args.phase1_out) / "cache"


def load_manifest(args) -> pd.DataFrame:
    return pd.read_csv(Path(args.phase1_out) / "manifest.csv")


def load_slice_boxes(args, split_label: str) -> pd.DataFrame:
    """Load ``slice_boxes_<split>`` (Parquet preferred, CSV mirror fallback)."""
    base = Path(args.phase1_out) / "labels" / f"slice_boxes_{split_label}"
    pq = base.with_suffix(".parquet")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(base.with_suffix(".csv"))
