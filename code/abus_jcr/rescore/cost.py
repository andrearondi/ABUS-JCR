"""[4.10] Cost instrumentation — params / GFLOPs / latency (do-not-drift #16).

Persisted because Phase 4 builds a model. The Phase-4 rows sit next to the recorded Phase-3
detector cost ([P3U3.5], seed 0, 30 val vols, op 0.03): **detector 6.688 s/vol, linking
0.364 s/vol, aggregation 0.019 s/vol; detector 36.34 M params / 22.64 GFLOPs**. The point of
the comparison is that the rescorer's cost must be read against a 6.7 s/vol detector — a
per-set forward measured in milliseconds is a rounding error on the deployed pipeline, and
that has to be *shown*, not asserted.

Measured here:
* encoder params/GFLOPs at ``(1, 1, 48, 48, 48)``;
* set-module params/GFLOPs at three PROMOTED-pool set sizes — the val **median (85)**, the
  val **max (292)** and the overall **max (509**, train fold0 vol14), see ``COSTED_SET_SIZES``;
* per-set rescoring latency (mean ± std over 50 timed forwards);
* per-volume crop-extraction latency (the CPU cost of re-extracting a set's crops).

Reuses ``detect.cost.count_params`` so Phase-2/3/4 parameter counts are produced by one
function. Torch is imported lazily.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .. import conventions as C
from ..detect.cost import count_params

__all__ = ["MEDIAN_SET_SIZE", "MAX_SET_SIZE", "MAX_TRAIN_SET_SIZE", "COSTED_SET_SIZES",
           "encoder_cost", "setmodule_cost", "crop_extraction_cost", "write_cost"]

#: PROMOTED-pool set sizes (updated 2026-08-09; were 92 / 253 from the ARCHIVED pool).
#: Val median 85.0 and val max 292 come from ``[F.9]`` §4 / ``[F.7]``'s per-volume log;
#: the overall max is **train fold0 vol14 at 509**, which the set module sees during
#: training, so it is costed too — the archived pool never had a set that large.
MEDIAN_SET_SIZE = 85          # val, the size the deployed rescorer meets most often
MAX_SET_SIZE = 292            # val, the worst single evaluation set
MAX_TRAIN_SET_SIZE = 509      # train fold0 vol14 — the worst set anywhere in either pool

#: What ``setmodule_cost`` times by default: typical, worst-at-inference, worst-anywhere.
COSTED_SET_SIZES = (MEDIAN_SET_SIZE, MAX_SET_SIZE, MAX_TRAIN_SET_SIZE)


def _gflops(model, make_inputs) -> Dict:
    import torch

    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(*make_inputs())
        return {"gflops": float(counter.get_total_flops()) / 1e9}
    except Exception as e:  # counter absent / op unsupported
        return {"gflops": float("nan"), "flop_note": f"{type(e).__name__}: {e}"}


def _latency(model, make_inputs, k: int = 50, warmup: int = 10, device=None) -> Dict:
    import torch

    is_cuda = device is not None and str(device).startswith("cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(*make_inputs())
        _sync()
        times = []
        for _ in range(k):
            t0 = time.perf_counter()
            model(*make_inputs())
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {"ms_mean": float(statistics.mean(times)), "ms_std": float(statistics.pstdev(times)),
            "k": int(k), "warmup": int(warmup), "device": str(device), "is_cuda": bool(is_cuda)}


def encoder_cost(encoder, device: str = "cuda", k: int = 50, batch: int = 1) -> Dict:
    """Params/GFLOPs/latency of the shared 3D encoder at ``(batch, 1, 48, 48, 48)``."""
    import torch

    encoder = encoder.to(device).eval()
    out = int(C.RESC_CROP_OUT)

    def _mk():
        return (torch.rand(batch, 1, out, out, out, device=device),)

    rec = {"input_size": [batch, 1, out, out, out], "encoder": C.RESC_ENCODER}
    rec.update(count_params(encoder))
    rec.update(_gflops(encoder, _mk))
    rec["latency"] = _latency(encoder, _mk, k=k, device=device)
    return rec


def setmodule_cost(model, d_in: int, set_sizes: Sequence[int] = COSTED_SET_SIZES,
                   device: str = "cuda", k: int = 50, use_geometry: bool = False) -> Dict:
    """Params + per-set GFLOPs/latency at the promoted pool's median / max / worst-anywhere
    set sizes (:data:`COSTED_SET_SIZES`)."""
    import torch

    model = model.to(device).eval()
    rec = {"d_in": int(d_in), "use_geometry": bool(use_geometry), "per_set": {}}
    rec.update(count_params(model))
    for n in set_sizes:
        def _mk(n=n):
            feats = torch.rand(1, n, d_in, device=device)
            coord = torch.rand(1, n, 3, device=device) * 300.0
            length = torch.rand(1, n, 3, device=device) * 40.0 + 1.0
            mask = torch.ones(1, n, dtype=torch.bool, device=device)
            return (feats, coord, length, mask)

        entry = {"set_size": int(n)}
        entry.update(_gflops(model, _mk))
        entry["latency"] = _latency(model, _mk, k=k, device=device)
        rec["per_set"][str(int(n))] = entry
    return rec


def crop_extraction_cost(vol_iso, record_rows, k: int = 20) -> Dict:
    """CPU cost of re-extracting one volume's crops from the iso cache (Inv. 5's price)."""
    from .crops import extract_crop, roi_side_iso

    times = []
    for _ in range(k):
        t0 = time.perf_counter()
        for _, r in record_rows.iterrows():
            side = roi_side_iso(r["ext_d0"], r["ext_d1"], r["ext_d2"])
            extract_crop(vol_iso, float(r["cen_d0"]), float(r["cen_d1"]), float(r["cen_d2"]),
                         float(r["ext_d0"]), float(r["ext_d1"]), float(r["ext_d2"]), side=side)
        times.append(time.perf_counter() - t0)
    n = int(len(record_rows))
    return {"n_candidates": n, "k": int(k),
            "seconds_per_volume_mean": float(np.mean(times)),
            "seconds_per_volume_std": float(np.std(times)),
            "ms_per_candidate": float(np.mean(times) * 1000.0 / max(1, n))}


def write_cost(rec: Dict, out_dir) -> Path:
    """Persist to ``<out_dir>/phase4_cost.json`` (mirrors ``detect.cost.write_cost``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase4_cost.json"
    rec = dict(rec)
    rec["phase3_reference"] = {"detector_s_per_vol": 6.688, "linking_s_per_vol": 0.364,
                               "aggregation_s_per_vol": 0.019, "detector_params_M": 36.34,
                               "detector_gflops": 22.64, "source": "[P3U3.5]"}
    path.write_text(json.dumps(rec, sort_keys=True, indent=2, default=float))
    return path
