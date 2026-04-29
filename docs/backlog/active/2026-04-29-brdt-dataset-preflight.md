---
priority: 121
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_born_rytov_dt_dataset.py
  - python -m compileall -q scripts/studies/born_rytov_dt
prerequisites:
  - 2026-04-29-brdt-operator-validation
related_roadmap_phases:
  - candidate-brdt-preflight
signals_for_selection:
  - BRDT needs a locked physical target, split policy, and normalization contract before adapter or training work.
  - The dataset must preserve physical `q` for the forward loss and train-only normalization for model outputs.
  - This item turns the design into a small reproducible smoke dataset, not a paper-grade benchmark.
---

# Backlog Item: BRDT Dataset Preflight

## Objective

- Create the minimal BRDT dataset-generation and manifest path needed for
  operator/data/model preflight work.

## Scope

- Consume the passed BRDT operator validation report and the candidate design.
- Lock the physical target:

  ```math
  q(x,z)=k_m^2\left(\left(\frac{n(x,z)}{n_m}\right)^2-1\right).
  ```

- Generate only a small preflight dataset first, using weak-scattering phantoms
  and the validated Born operator.
- Store physical `q_true`, normalized `q_true`, complex sinograms, angle masks,
  split metadata, train-only normalization statistics, generation command, git
  state, and environment information.
- Define the physics-loss normalization rule explicitly:

  ```math
  L_{\mathrm{phys}} = \|A(\mathrm{unnormalize}(\hat q))-y\|.
  ```

- Emit a dataset manifest and a dry-run/geometry validation summary.

## Notes for Reviewer

- Do not let normalized `q` enter the physical forward operator without an
  explicit unnormalize step.
- Do not generate the larger `128 x 128` decision-support split until the smoke
  dataset and manifest pass.
- Do not reuse CDI line-pattern objects as the only BRDT phantom family.
