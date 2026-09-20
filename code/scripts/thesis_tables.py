#!/usr/bin/env python3
"""Table bodies of thesis_polimi ch. 7 / App. A, generated from the mirrored JSON artefacts.

Every number in the grid and comparison tables is read from an artefact, never typed. A table
body lives in its chapter between two marker comments,

    % BEGIN GENERATED <name>
    ...rows...
    % END GENERATED <name>

``--write`` replaces what is between the markers; ``--check`` (the default) fails if a chapter
differs from what the artefacts produce. Captions, labels and column specs stay hand-written.

    python scripts/thesis_tables.py --write        # after the artefacts change
    python scripts/thesis_tables.py                # check only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abus_jcr.rescore.factor_grid import (APPENDIX_PAIRS, ARCH, FACTOR_PAIRS, GRID, OBJ,  # noqa: E402
                                          OVERALL, pair_key)

MACRO = {"B1": r"\cIC", "B2": r"\cJC", "A1": r"\cGC", "B1-P": r"\cIP", "A2-P": r"\cJP",
         "FULL-P": r"\cGP", "A2": r"\cJV", "FULL": r"\cGV", "B0": "detector score",
         "B0-rank": "rank rule", "B0-spread": "detector score, spread over the grid"}
KEY_FP = [0.125, 0.25, 0.5, 1, 2, 4, 8]
HELD = {"CE": "under CE", "pooled": "under pooled", "Independent": "Independent",
        "Joint": "Joint", "Joint+Geo": "Joint+Geo", "-": ""}


def s3(x: float) -> str:
    return "$0.000$" if abs(x) < 0.0005 else f"${x:+.3f}$"


def pm(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


def interval(lo: float, hi: float) -> str:
    return f"[{s3(lo)}, {s3(hi)}]"


# ------------------------------------------------------------------------------- the grid
def grid_rows(g: dict) -> list[str]:
    pr = g["per_rung"]
    seeds = lambda r: " / ".join(f"{v:.3f}" for v in pr[r]["per_seed_cpm"])  # noqa: E731
    rows = []
    for arch in ARCH:
        ids = [GRID[(arch, o)] for o in OBJ]
        rows.append(f"{arch} & " + " & ".join(pm(pr[i]["cpm_mean"], pr[i]["cpm_std"]) for i in ids) + r" \\")
        rows.append(" & " + " & ".join(r"{\scriptsize " + seeds(i) + "}" for i in ids) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    rows[-1] = r"\midrule"
    for ref in ("B0", "B0-rank"):
        rows.append(f"{MACRO[ref]} & \\multicolumn{{2}}{{c}}{{{pm(pr[ref]['cpm_mean'], pr[ref]['cpm_std'])}"
                    f" \\quad {{\\scriptsize {seeds(ref)}}}}} \\\\")
    rows.append(f"recall ceiling & \\multicolumn{{2}}{{c}}{{{pm(pr['B0']['ceiling_mean'], pr['B0']['ceiling_std'])}"
                + r" \quad {\scriptsize " + " / ".join(f"{g['per_seed'][s]['B0']['ceiling']:.3f}"
                                                        for s in sorted(g["per_seed"])) + r"}} \\")
    return rows


# ------------------------------------------------------------------- one-switch pair tables
def pair_rows(rep: dict, factor: str) -> list[str]:
    rows = []
    for p in [q for q in rep["pairs"] if q["factor"] == factor]:
        ps = sorted(p["per_seed"], key=lambda r: r["seed"])
        label = f"{MACRO[p['a']]} $-$ {MACRO[p['b']]}"
        rows.append(f"{label} & ${p['delta_mean']:+.3f} \\pm {p['delta_std']:.3f}$ & "
                    + " & ".join(f"{s3(r['delta'])} ({100 * r['frac_positive']:.0f}\\,\\%)" for r in ps) + r" \\")
        rows.append(" & & " + " & ".join(r"{\scriptsize " + interval(r["lo"], r["hi"]) + "}" for r in ps) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    return rows[:-1]


def rate_rows(g: dict) -> list[str]:
    """Per-rate sensitivity differences, mean over seeds: exact, read from the grid itself."""
    rows, seeds = [], sorted(g["per_seed"])
    for p in [q for q in FACTOR_PAIRS if q["factor"] == "objective"] + [OVERALL]:
        kr = lambda s, r, k: g["per_seed"][s][r]["key_recall"][str(k)]  # noqa: E731
        means = [float(np.mean([kr(s, p["a"], k) - kr(s, p["b"], k) for s in seeds])) for k in KEY_FP]
        rows.append(f"{MACRO[p['a']]} $-$ {MACRO[p['b']]} & " + " & ".join(s3(m) for m in means) + r" \\")
    return rows


# --------------------------------------------------------------------- appendix: stored pairs
def stored_rows(g: dict) -> list[str]:
    rows = []
    for a, b in APPENDIX_PAIRS:
        c = g["comparisons"].get(pair_key(a, b))
        if c is None:
            continue
        rows.append(f"{MACRO[a]} $-$ {MACRO[b]} & ${c['delta_mean']:+.3f} \\pm {c['delta_std']:.3f}$ & "
                    + " & ".join(f"{s3(r['delta'])} {{\\scriptsize {interval(r['lo'], r['hi'])}}}"
                                 for r in c["per_seed"]) + r" \\")
    return rows


def val_diff_rows(gv: dict, gt: dict) -> list[str]:
    """Exact per-seed CPM differences on validation, with the test mean beside them."""
    rows = []
    for p in tuple(FACTOR_PAIRS) + (OVERALL,):
        dv = np.array(gv["per_rung"][p["a"]]["per_seed_cpm"]) - np.array(gv["per_rung"][p["b"]]["per_seed_cpm"])
        dt = np.array(gt["per_rung"][p["a"]]["per_seed_cpm"]) - np.array(gt["per_rung"][p["b"]]["per_seed_cpm"])
        rows.append(f"{MACRO[p['a']]} $-$ {MACRO[p['b']]} & " + " & ".join(s3(v) for v in dv)
                    + f" & {s3(dv.mean())} & {s3(dt.mean())}" + r" \\")
    return rows


# ------------------------------------------------------------------------------------ main
def build(evo: Path) -> dict[str, list[str]]:
    gt = json.loads((evo / "maia_47_stage/phase5/grid/grid_TEST.json").read_text())
    gv = json.loads((evo / "maia_47_stage/phase4/grid/grid.json").read_text())
    out = {"test-grid": grid_rows(gt), "val-grid": grid_rows(gv), "stored-pairs": stored_rows(gt),
           "val-diffs": val_diff_rows(gv, gt), "test-rates": rate_rows(gt)}
    d = evo / "maia_47_stage/phase5/grid"
    fp = next((d / n for n in ("factor_comparisons_TEST.json", "factor_comparisons_TEST_laptop.json")
               if (d / n).exists()), None)
    if fp is not None:
        rep = json.loads(fp.read_text())
        for f in ("jointness", "geometry", "objective", "overall"):
            out[f"pairs-{f}"] = pair_rows(rep, f)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo", type=Path, default=Path("/Volumes/Evo"))
    ap.add_argument("--chapters", type=Path,
                    default=Path(__file__).resolve().parents[2] / "thesis_polimi/chapters")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--show", nargs="*", default=None, help="print these bodies and exit")
    a = ap.parse_args(argv)
    bodies = build(a.evo)
    if a.show is not None:
        for k in (a.show or bodies):
            print(f"% --- {k}\n" + "\n".join(bodies[k]) + "\n")
        return 0
    bad, seen = [], set()
    for tex in sorted(a.chapters.glob("*.tex")):
        t = tex.read_text()
        new = t
        for m in re.finditer(r"% BEGIN GENERATED (\S+)\n(.*?)% END GENERATED \1\n", t, flags=re.S):
            name, body = m.group(1), m.group(2)
            seen.add(name)
            if name not in bodies:
                bad.append(f"{tex.name}: no artefact yet for '{name}'")
                continue
            want = "\n".join(bodies[name]) + "\n"
            if body != want:
                if a.write:
                    new = new.replace(m.group(0), f"% BEGIN GENERATED {name}\n{want}% END GENERATED {name}\n")
                else:
                    bad.append(f"{tex.name}: table '{name}' differs from the artefacts")
        if a.write and new != t:
            tex.write_text(new)
            print(f"updated {tex.name}")
    for msg in bad:
        print("MISMATCH " + msg)
    print(f"# {len(seen)} generated tables found in {a.chapters}; {'OK' if not bad else 'FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
