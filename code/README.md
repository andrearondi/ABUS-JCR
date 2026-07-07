# abus_jcr — Phase 0a

Pre-flight gating checks for the ABUS-JCR thesis: GT-box double-check, lesion
audit, and the frozen official-metric (FROC/CPM) oracle. This package defines
the coordinate conventions, I/O contract, and scoring wrapper that Phases 1, 3,
and 5 consume. Nothing here trains a model.

## Install

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -e .          # Phase-0 runtime deps
pip install -e ".[test]"  # + pytest
```

## Run the Phase-0a scripts

Each script takes either `--split {Train,Validation,Test} [--data-root PATH]`
(server layout, `--data-root` defaults to `/home/maia-user/Andre2/data`) or an
explicit `--split-root PATH` (used for the local Validation copy).

```bash
python scripts/phase0a_gt_doublecheck.py --split Train   # residual table + PASS/FAIL (tol 0)
python scripts/phase0a_lesion_audit.py   --split Train   # 26-conn lesions/volume + dominance verdict
python scripts/phase0a_spacing_table.py  --split Train   # shape+spacing table + d0>=d1>=d2 assertion
```

The exact server sequence (env creation, both splits, pytest) is in
[`runbooks/RB_PHASE_0.md`](../runbooks/RB_PHASE_0.md); paste raw output into
`results/RESULTS_PHASE_0.md`.

## Tests

```bash
pytest tests            # set ABUS_SPLIT_ROOT to run the data-driven double-check on the server
```

`test_gt_box_doublecheck` is data-driven: it reads `ABUS_SPLIT_ROOT` if set
(the server points this at `Train`), else the local Validation split, else
skips.

## Key modules

| Module | Responsibility |
|---|---|
| `conventions.py` | Single source of truth: permutation, injected spacing, CSV schemas, thresholds |
| `io_nrrd.py` | pynrrd read (storage order), recursive case discovery, spacing injection |
| `geometry.py` | `BoxStorage` ↔ `OfficialBox`, `mask → box` transform, IoU delegate |
| `gt_labels.py` | GT column adapter + per-case mask↔box double-check (tol 0) |
| `lesions.py` | 26-connectivity component audit |
| `eval/froc.py` | `evaluate()` wrapper, CPM/ceiling accessors, prediction writer, bootstrap CI |
| `eval/_official_det_score.py` | **byte-identical** vendored `Final_Evaluation/det_score.py` (never edited) |

**Coordinate space (Inv. 6):** official scoring space = native voxel indices,
ITK `(x, y, z)` order, centre + full extent. Storage order is `(d0, d1, d2)`;
the permutation to ITK is the self-inverse `(2, 1, 0)`. Physical spacing is the
injected constant `(0.073, 0.200, 0.475674)` mm (storage order) — the NRRD
header's identity placeholder is always ignored.
