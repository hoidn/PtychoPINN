# Execution Report

## Completed In This Pass

- Added `ptycho_torch/generators/neuralop_uno.py` as the external NeuralOperator U-NO adapter for the locked Lines128 CDI contract, with lazy dependency loading, frozen preflight constructor settings, `real_imag` output adaptation to `(B, H, W, 1, 2)`, and fail-closed checks for `N=128`, `gridsize=1`, and `C=1`.
- Wired `neuralop_uno` through the Torch generator registry, both config architecture literal surfaces, checkpoint rebuild, and the direct grid-lines Torch runner label/CLI surface so both PINN and supervised routes use the same U-NO body.
- Added focused regression coverage for adapter shape/dependency failures, registry resolution, supervised and PINN Lightning construction, checkpoint reload, and runner acceptance/labeling.
- Updated durable architecture/workflow/config docs to describe the supported `neuralop_uno` surface and its current limits. `docs/index.md` was not changed because no new durable document was added and discovery paths did not change.
- No benchmark row, append-only bundle, compare-wrapper model-ID work, or paper-bundle mutation was launched in this pass.

## Completed Plan Tasks

- Task 1: Added red-first tests for registry resolution, adapter behavior, supervised and PINN construction, checkpoint rebuild, and runner labeling.
- Task 2: Implemented the external U-NO adapter with the locked Lines128 CDI contract and actionable dependency/shape errors.
- Task 3: Wired registry, config, checkpoint rebuild, and direct runner support for `neuralop_uno`.
- Task 4: Updated durable docs for the supported architecture surface and its intentional limits.
- Task 5: Ran the focused selector plus all required deterministic gates and archived the logs under the item verification root.

## Remaining Required Plan Tasks

- None for the approved scope of this backlog item.

## Verification

- Focused selector: `pytest -q tests/torch/test_neuralop_uno_generator.py tests/torch/test_generator_registry.py tests/torch/test_lightning_checkpoint.py tests/torch/test_grid_lines_torch_runner.py`
  - Result: `146 passed`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/pytest_focused.log`
- Required deterministic pytest gate: `pytest -q tests/torch/test_generator_registry.py tests/torch/test_loss_modes.py`
  - Result: `12 passed`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/pytest_required.log`
- Required compile gate: `python -m compileall -q ptycho_torch scripts/studies`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/compileall.log`
- Required output check:
  - Command verifies `ptycho_torch/generators/neuralop_uno.py` and `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md` exist.
  - Result: `U-NO integration outputs present`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/required_outputs_check.log`

## Residual Risks

- `neuralop_uno` intentionally supports only the locked Lines128 CDI lane (`N=128`, `gridsize=1`, `C=1`, `generator_output_mode=real_imag`). Any broader U-NO contract still requires a separate approved plan.
- The architecture now exists for direct runner and checkpoint use, but compare-wrapper model-ID integration and append-only U-NO benchmark-row execution remain explicit follow-on work.
- The runtime depends on external `neuraloperator==2.0.0`; future environment drift will fail closed with actionable errors rather than silently substituting another model surface.
