# Execution Report

## Completed In This Pass

- Added the missing direct-runner regression coverage for `architecture="neuralop_uno"` with `training_procedure="supervised"` so the runner now proves acceptance for both approved procedures.
- Added red-first runner contract tests proving `setup_torch_configs()` rejects unsupported `neuralop_uno` combinations at setup time for:
  - `N != 128`
  - `gridsize != 1`
  - `generator_output_mode != "real_imag"`
- Narrowly updated `scripts/studies/grid_lines_torch_runner.py` to enforce that locked Lines128 U-NO contract during config setup instead of failing later during model construction.
- Corrected the durable supported-architecture documentation in:
  - `docs/CONFIGURATION.md`
  - `docs/workflows/pytorch.md`
  - `docs/architecture_torch.md`
  - `ptycho_torch/generators/README.md`
  so the documented live Torch architecture surface now matches the actual registry/config literals, including `ffno` and `spectral_resnet_bottleneck_net`.
- Kept scope unchanged: no benchmark rows, compare-wrapper model-ID work, YAML/prompt edits, or paper-bundle mutation were introduced.

## Completed Current-Scope Work

- Task 1 review gap closed: direct-runner coverage now includes `TorchRunnerConfig(architecture="neuralop_uno", training_procedure="supervised")`.
- Task 4 review gap closed: the durable docs named above now agree with the live supported architecture surface rather than omitting registered architectures.
- The non-blocking maintainability defect identified in review was fixed in scope: `neuralop_uno` invalid runner inputs now fail closed in `setup_torch_configs()` with actionable errors instead of surviving until deeper model construction.
- All approved current-scope implementation work for this backlog item is complete in the current checkout.

## Follow-Up Work

- Compare-wrapper model-ID routing and append-only U-NO paper-row execution remain separate follow-on items by design.
- Any expansion of `neuralop_uno` beyond the locked Lines128 CDI lane (`N=128`, `gridsize=1`, `C=1`, `real_imag`) still requires a new approved plan.
- A lightweight doc-surface regression check tying architecture tables to the registry/config literals would reduce future drift, but it was not required to approve this backlog item.

## Residual Risks

- `neuralop_uno` still depends on external `neuraloperator==2.0.0`; future environment drift will fail closed rather than silently substituting a different implementation.
- The adapter and runner intentionally reject broader CDI contracts; users attempting `gridsize>1`, non-`real_imag`, or non-`128x128` U-NO runs will need separate implementation work.
- The broader paper-table extension path is still unimplemented here, so direct runner support should not be read as completed benchmark integration.

## Verification

- Red-first runner selector:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k 'neuralop_uno and (supervised_training or non_lines128 or non_unit_gridsize or non_real_imag_output)'`
  - Result: `3 failed, 1 passed` before the runner guard was added, then `4 passed` after the fix.
- Focused selector:
  - `pytest -q tests/torch/test_neuralop_uno_generator.py tests/torch/test_generator_registry.py tests/torch/test_lightning_checkpoint.py tests/torch/test_grid_lines_torch_runner.py`
  - Result: `150 passed`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/pytest_focused.log`
- Required deterministic pytest gate:
  - `pytest -q tests/torch/test_generator_registry.py tests/torch/test_loss_modes.py`
  - Result: `12 passed`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/pytest_required.log`
- Required compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/compileall.log`
- Required output check:
  - Result: `U-NO integration outputs present`
  - Log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/required_outputs_check.log`
