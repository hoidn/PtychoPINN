# BRDT Operator Validation Execution Plan

## Scope

Implement and validate the differentiable 2D Born forward operator for the BRDT
candidate lane. This plan is limited to operator correctness and validation
artifacts; it does not authorize dataset generation or neural training.

## Required Outputs

- PyTorch Born operator implementation or prototype.
- Independent validation report with analytic/tiny-grid/gradient checks.
- Optional ODTbrain inverse-side consistency check when dependencies are
  available.
- Tests covering operator shape, dtype/device behavior, gradients, and at least
  one independent numerical oracle.

## Checks

- `pytest -q tests/studies/test_born_rytov_dt_operator.py tests/studies/test_born_rytov_dt_validation.py`
- `python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt`
