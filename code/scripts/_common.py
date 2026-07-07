"""Shared CLI plumbing for the Phase-0a scripts.

Split-root resolution order:
1. ``--split-root PATH`` explicit override (used for local Validation runs);
2. else ``<--data-root>/<--split>`` (the server layout, per SERVER_LAYOUT.md
   ``--data-root`` defaults to ``/home/maia-user/Andre2/data``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DATA_ROOT = "/home/maia-user/Andre2/data"


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
