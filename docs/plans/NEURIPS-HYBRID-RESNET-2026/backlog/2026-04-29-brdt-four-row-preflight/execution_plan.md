# BRDT Four-Row Preflight Execution Plan

## Scope

Run the bounded BRDT decision-support preflight after the operator, dataset, and
task adapters are ready. This plan does not authorize Rytov, limited-angle,
FFNO, physics-only, external FDTD mismatch, or multi-seed rows.

## Required Outputs

- Classical Born, U-Net, FNO vanilla, and SRU/Hybrid-family rows under one
  dataset/operator/input/split/metric contract.
- Table JSON/CSV, metric schema, fixed-sample visuals, source arrays, runtime
  metadata, and row statuses.
- Claim boundary explicitly labeled as decision-support preflight only.

## Checks

- `pytest -q tests/studies/test_born_rytov_dt_preflight.py`
- `python -m compileall -q scripts/studies/born_rytov_dt`
