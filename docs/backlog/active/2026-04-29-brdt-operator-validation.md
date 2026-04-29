---
priority: 120
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_born_rytov_dt_operator.py tests/studies/test_born_rytov_dt_validation.py
  - python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt
prerequisites: []
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - BRDT cannot be trusted if synthetic data and physics loss use the same unvalidated operator.
  - The Born forward model needs independent checks before any neural row is trained.
  - This item is the first BRDT execution step and is on equal candidate footing with WaveBench.
---

# Backlog Item: BRDT Operator Validation

## Objective

- Implement and validate the differentiable 2D Born forward operator required by
  the BRDT candidate lane.

## Scope

- Consume
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`.
- Add or prototype `BornRytovForward2D` under the repo's task-appropriate
  physics/study surface.
- Lock the coordinate convention, FFT normalization, angle convention,
  detector-frequency convention, and real/imag output layout.
- Validate the operator with independent checks:
  - analytic point or Gaussian phantom checks;
  - tiny-grid direct Born integral comparison;
  - finite-difference or autograd `gradcheck`;
  - dtype/device reproducibility checks;
  - ODTbrain inverse-side consistency when the optional dependency is available.
- Emit an operator validation report with tolerances, sample counts, command
  provenance, and any known scale or phase convention offset.

## Notes for Reviewer

- Do not accept a self-consistency-only test where generated data and validation
  both use the same PyTorch operator.
- Do not start dataset generation or neural training until this item emits a
  clear pass/fail operator validation report.
- Keep Rytov mode out of scope unless Born mode has already passed.
