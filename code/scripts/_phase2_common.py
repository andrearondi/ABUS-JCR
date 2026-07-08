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


def assert_device(device: str) -> None:
    """Fail loudly if a CUDA device is requested but unavailable (Inv.: A6000 runs).

    Prevents a CPU-only node (e.g. a JupyterHub pod with no NVIDIA driver) from
    silently loading the whole dataset/model before crashing — the training/cost/
    dump steps MUST run on the GPU host, not a login/notebook pod.
    """
    if not str(device).startswith("cuda"):
        return
    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"--device {device} requested but torch is not installed ({e}). "
                         "Activate the abus-jcr env on the GPU host.")

    if not torch.cuda.is_available():
        raise SystemExit(
            f"--device {device} requested but torch reports no CUDA GPU "
            f"(is_available=False, device_count={torch.cuda.device_count()}).\n"
            "You are almost certainly on a CPU-only node (check `nvidia-smi`). "
            "Move to the A6000 GPU host, `conda activate "
            "/home/maia-user/Andre2/envs/abus-jcr`, verify "
            "`python -c \"import torch; print(torch.cuda.is_available())\"` prints True, "
            "then re-run. (The [2.0b] Train-stats probe is CPU-only and can run anywhere.)"
        )


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
