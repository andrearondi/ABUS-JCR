"""Shared CLI plumbing for the Phase-0a scripts.

Split-root resolution order:
1. ``--split-root PATH`` explicit override (used for local Validation runs);
2. else ``<--data-root>/<--split>`` (the server layout, per SERVER_LAYOUT.md;
   ``--data-root`` defaults to ``$WORK/data``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _default_root(rel: str) -> str:
    """``$WORK/<rel>``, or a placeholder that names its own fix when ``$WORK`` is unset.

    **Never hardcode an absolute default here.** Until 2026-08-31 these constants named
    ``/home/maia-user/...``, the home directory on the *pre-migration* server. Every one of
    them therefore pointed at a machine that no longer exists, and any runbook line that
    omitted a flag inherited that. It surfaced when ``[4.2]`` spent several minutes of GPU
    time before its wrapped ``phase4_pretrain_encoder`` failed on
    ``--data-root $WORK/data``.

    ``$WORK`` is exported by ``~/.bashrc`` on the login node and re-exported by the sbatch
    harness inside a job, so it is set on both paths a script can run down. When it is not
    set — a laptop, where none of these roots exist anyway — the returned string is not a
    path but an instruction, so the failure names the two ways out (export ``$WORK``, or pass
    the flag) instead of pointing at somebody's old home directory.
    """
    work = os.environ.get("WORK")
    return f"{work}/{rel}" if work else f"<unset: export WORK, or pass the flag>/{rel}"


DEFAULT_DATA_ROOT = _default_root("data")


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=["Train", "Validation", "Test"], default="Validation",
                        help="split name under --data-root")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"dataset root holding the split dirs (default {DEFAULT_DATA_ROOT})")
    parser.add_argument("--split-root", default=None,
                        help="explicit path to the split dir; overrides --data-root/--split")


def resolve_split_root(args: argparse.Namespace) -> Path:
    if args.split_root:
        return Path(args.split_root)
    return Path(args.data_root) / args.split


def split_label(args: argparse.Namespace) -> str:
    if args.split_root:
        return Path(args.split_root).name
    return args.split
