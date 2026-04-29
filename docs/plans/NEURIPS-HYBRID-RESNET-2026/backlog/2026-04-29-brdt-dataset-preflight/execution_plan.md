# BRDT Dataset Preflight Execution Plan

## Scope

Create the minimal BRDT smoke dataset path after the Born operator validation
passes. This plan is limited to physical-target locking, geometry validation,
train-only normalization, and a reproducible preflight dataset manifest.

## Required Outputs

- Dataset generator or dry-run manifest path for weak-scattering phantoms.
- Manifest recording physical `q`, normalization statistics, split policy,
  operator version, generation command, git state, and environment.
- Tests proving physical `q` is unnormalized before the forward loss.

## Checks

- `pytest -q tests/studies/test_born_rytov_dt_dataset.py`
- `python -m compileall -q scripts/studies/born_rytov_dt`
