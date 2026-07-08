"""Cost instrumentation — params / FLOPs / latency (do-not-drift #16, Inv. Phase 2).

Persisted NOW because a model is built in Phase 2 (the checklist requires cost in
every model-building phase). FLOPs and latency are input-size dependent, so the
fixed measurement input ``(1, C, min_size, max_size)`` is recorded alongside.
Torch is imported lazily.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict

from .. import conventions as C


def count_params(model) -> Dict[str, int]:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def measure_gflops(model, c_channels: int, min_size: int, max_size: int, device=None) -> Dict:
    """GFLOPs of one ``eval()`` forward at ``(1, C, min_size, max_size)``.

    Uses ``torch.utils.flop_counter.FlopCounterMode``. Returns NaN GFLOPs (with a
    note) if the counter is unavailable in the installed torch build.
    """
    import torch

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x = torch.rand(c_channels, int(min_size), int(max_size), device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model([x])
        flops = counter.get_total_flops()
        return {"gflops": float(flops) / 1e9, "flop_input": [1, c_channels, int(min_size), int(max_size)]}
    except Exception as e:  # counter absent / op unsupported
        return {"gflops": float("nan"), "flop_input": [1, c_channels, int(min_size), int(max_size)],
                "flop_note": f"{type(e).__name__}: {e}"}


def measure_latency(model, c_channels: int, min_size: int, max_size: int,
                    device=None, k: int = 50, warmup: int = 10,
                    slices_per_volume: float = 1.0) -> Dict:
    """Per-slice latency (mean±std ms) over ``k`` timed ``eval()`` forwards.

    Warms up ``warmup`` iters, synchronises CUDA around each timed forward, and
    reports per-volume = per-slice × ``slices_per_volume``. CPU timing is recorded
    if no GPU is present (flagged).
    """
    import time

    import torch

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    is_cuda = torch.device(device).type == "cuda"
    x = torch.rand(c_channels, int(min_size), int(max_size), device=device)

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model([x])
        _sync()
        times = []
        for _ in range(k):
            t0 = time.perf_counter()
            model([x])
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times)
    std_ms = statistics.pstdev(times)
    return {
        "device": str(device),
        "is_cuda": is_cuda,
        "per_slice_ms_mean": float(mean_ms),
        "per_slice_ms_std": float(std_ms),
        "per_volume_ms_mean": float(mean_ms * slices_per_volume),
        "slices_per_volume": float(slices_per_volume),
        "k": int(k), "warmup": int(warmup),
    }


def measure_cost(model, c_channels: int = C.C_CHANNELS,
                 min_size: int = C.DET_MIN_SIZE, max_size: int = C.DET_MAX_SIZE,
                 device=None, k: int = 50, warmup: int = 10,
                 slices_per_volume: float = 1.0) -> Dict:
    """Full cost record: params + GFLOPs + latency at the fixed input size."""
    rec = {"input_size": [1, int(c_channels), int(min_size), int(max_size)]}
    rec.update(count_params(model))
    rec.update(measure_gflops(model, c_channels, min_size, max_size, device))
    rec["latency"] = measure_latency(model, c_channels, min_size, max_size, device,
                                     k=k, warmup=warmup, slices_per_volume=slices_per_volume)
    return rec


def write_cost(rec: Dict, out_dir) -> Path:
    """Persist the cost record to ``<out_dir>/phase2_cost.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase2_cost.json"
    path.write_text(json.dumps(rec, sort_keys=True, indent=2))
    return path
