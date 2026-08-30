"""[4.1] Build the Phase-4 crop cache for the train + val pools (Inv. 5, 6).

Materialises one float16 memmap per split, in **record row order**, named by ``crop_hash``
so a stale cache can never be silently reused. Prints the [4.1] report: size, ROI-side
histogram, pad-fraction distribution, and the **max set size over both pools**, which is
ASSERTED against ``RESC_MAX_SET_SIZE`` (576). On the promoted pool the worst single set is
**train fold0 vol14 at 509** and val's worst is **292** ([F.7]); the archived pool's max was
253, which is why the constant was 320 and would have failed this assertion on the first run.
This is the place the true max gets pinned — a FAIL here means raise the constant and rebuild,
never truncate a frozen pool.

Usage:
    python scripts/phase4_build_crops.py --splits train val \\
        --phase1-out $WORK/outputs/phase1 \\
        --phase3-out $WORK/outputs/phase3 \\
        --out-root  $WORK/outputs/phase4
"""

from __future__ import annotations

import argparse
import sys

from abus_jcr import conventions as C
from abus_jcr.cache import assert_hash
from abus_jcr.preprocess import preprocess_hash
from abus_jcr.rescore.crops import build_crop_cache, crop_hash

from _phase4_common import add_phase4_paths, cache_root, crops_dir, dump_json, load_record


def main() -> int:
    ap = argparse.ArgumentParser(description="[4.1] build the Phase-4 3D crop cache")
    add_phase4_paths(ap)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = ap.parse_args()

    assert_hash(cache_root(args))            # refuse a stale iso cache before writing 4 GB
    print(f"# preprocess_hash = {preprocess_hash()}")
    print(f"# crop_hash       = {crop_hash()}")
    print(f"# crop config     = out {C.RESC_CROP_OUT}^3, side = clip({C.RESC_CROP_CONTEXT} * "
          f"max(ext), {C.RESC_CROP_MIN_SIDE}, {C.RESC_CROP_MAX_SIDE}) iso vox, "
          f"interp order {C.RESC_CROP_INTERP}, pad {C.RESC_CROP_PAD_VALUE}")

    report, global_max = {}, 0
    for split in args.splits:
        rec = load_record(args, split)
        print(f"\n# [4.1] building crops for {split} ({len(rec)} rows)")
        stats = build_crop_cache(rec, cache_root(args), crops_dir(args), split)
        report[split] = stats
        global_max = max(global_max, int(stats["max_set_size"]))
        print(f"\n# {split}: {stats['n_rows']} crops, {stats['bytes'] / 1e9:.2f} GB, "
              f"{stats['n_sets']} sets (median {stats['median_set_size']:.1f}, "
              f"max {stats['max_set_size']})")
        print(f"  ROI side  min/median/max = {stats['roi_side_min']:.0f} / "
              f"{stats['roi_side_median']:.0f} / {stats['roi_side_max']:.0f} iso vox")
        print(f"  ROI side histogram       = {stats['roi_side_hist']}")
        print(f"  pad fraction median/p90/max = {stats['pad_frac_median']:.3f} / "
              f"{stats['pad_frac_p90']:.3f} / {stats['pad_frac_max']:.3f}")

    print(f"\n# [4.1] MAX SET SIZE over {args.splits} = {global_max} "
          f"(pad width RESC_MAX_SET_SIZE = {C.RESC_MAX_SET_SIZE})")
    ok = global_max <= C.RESC_MAX_SET_SIZE
    print("# EXIT CHECK 2:", "PASS" if ok else "FAIL")
    report["max_set_size_over_splits"] = global_max
    report["max_set_size_ok"] = bool(ok)
    dump_json(report, crops_dir(args) / "crop_build_report.json")

    if not ok:
        print(f"FAIL: a set of {global_max} candidates exceeds the pad width "
              f"{C.RESC_MAX_SET_SIZE}. Do NOT truncate a frozen pool — raise "
              f"RESC_MAX_SET_SIZE and rebuild.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
