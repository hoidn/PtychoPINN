---
priority: 3
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/execution_plan.md
check_commands:
  - pytest --collect-only -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
  - pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode or model"
  - pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or input_mode"
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites: []
related_roadmap_phases:
  - candidate-brdt-sinogram-input
signals_for_selection:
  - The manuscript BRDT setup is being changed so learned models consume the measured complex sinogram, not a Born-derived image.
  - This item must land before the 40-epoch sinogram-input evidence run.
  - It is higher priority than WaveBench items but behind the active priority-1 and priority-2 SRU-Net mechanism items unless the operator promotes it.
---

# Backlog Item: BRDT Sinogram-Input Adapter Contract

## Objective

Implement and test the BRDT learned-model input contract in which SRU-Net and
FFNO consume the measured complex sinogram directly.

## Scope

- Add `input_mode="sinogram"` as a supported BRDT input contract while
  preserving `input_mode="born_init_image"` for historical runners.
- Keep `direct_sinogram` rejected as a legacy alias.
- Add task-local sinogram-input adapters that map `(B, 2, 64, 128)` measured
  real/imaginary sinograms to `(B, 1, 128, 128)` target-grid predictions.
- Do not compute a fixed Born inverse in the learned-model input path.
- Preserve the Born inverse only as a non-learned baseline and visualization
  reference.

## Required Verification

- Shape tests prove SRU-Net and FFNO accept complex sinograms and output target
  grid predictions.
- Input-mode tests prove `sinogram` is accepted and old Born-image rows are not
  selected by the sinogram-input runner.
- Smoke execution proves the measured sinogram is used as model input and as
  the Born-consistency target.
