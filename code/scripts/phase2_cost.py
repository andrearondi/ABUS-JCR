"""Persist params / FLOPs / latency for the RetinaNet (do-not-drift #16).

Builds the model (optionally loading a trained checkpoint for identical arch),
measures cost at the fixed input ``(1, C, DET_MIN_SIZE, DET_MAX_SIZE)`` on the
A6000, writes ``phase2_cost.json`` and echoes a Markdown block for RESULTS.

Usage (server):
    python scripts/phase2_cost.py --out-root /home/maia-user/Andre2/outputs/phase2 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from abus_jcr import conventions as C
from abus_jcr.detect import cost as CST
from abus_jcr.detect.retinanet import build_retinanet, load_checkpoint
from _phase2_common import add_phase2_paths, assert_device, cache_root, load_manifest


def _mean_slices_per_volume(args) -> float:
    """Mean iso SLICE_AXIS length over Train+Val volumes (for per-volume latency)."""
    from abus_jcr import cache as K

    try:
        manifest = load_manifest(args)
        vids = manifest[manifest["split"].isin(["train", "val"])]["volume_id"].tolist()
        ns = [int(K.read_meta(cache_root(args), int(v))["iso_shape"][C.SLICE_AXIS]) for v in vids]
        return sum(ns) / len(ns) if ns else 1.0
    except Exception:
        return 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 cost instrumentation")
    add_phase2_paths(parser)
    parser.add_argument("--checkpoint", default=None, help="optional trained checkpoint to load")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    assert_device(args.device)
    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint)
    else:
        model = build_retinanet(pretrained=False)
    model.to(args.device)

    spv = _mean_slices_per_volume(args)
    rec = CST.measure_cost(model, c_channels=C.C_CHANNELS,
                           min_size=C.DET_MIN_SIZE, max_size=C.DET_MAX_SIZE,
                           device=args.device, k=args.k, warmup=args.warmup,
                           slices_per_volume=spv)
    path = CST.write_cost(rec, Path(args.out_root) / "cost")

    lat = rec["latency"]
    print(f"# Phase 2 cost -> {path}\n")
    print("```markdown")
    print(f"| metric | value |")
    print(f"|---|---|")
    print(f"| params (total) | {rec['params_total']:,} |")
    print(f"| params (trainable) | {rec['params_trainable']:,} |")
    print(f"| GFLOPs @ {rec['flop_input']} | {rec['gflops']:.2f} |")
    print(f"| per-slice latency (ms) | {lat['per_slice_ms_mean']:.2f} ± {lat['per_slice_ms_std']:.2f} ({lat['device']}) |")
    print(f"| per-volume latency (ms) | {lat['per_volume_ms_mean']:.1f} (× {lat['slices_per_volume']:.1f} slices) |")
    print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
