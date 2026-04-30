# Execution Report

## Completed In This Pass

- Added a checkpoint-path regression in `tests/torch/test_lightning_checkpoint.py` that tampers saved `neuralop_uno` hyperparameters and proves reload now fails closed for both unsupported saved modes:
  - `generator_output_mode="amp_phase"`
  - `generator_output_mode="amp_phase_logits"`
- Narrowly updated `ptycho_torch/model.py` so checkpoint/module rebuild preserves the saved `neuralop_uno` output mode verbatim for validation instead of coercing every non-`amp_phase` mode to `real_imag`.
- Re-ran the focused U-NO selector plus the backlog item’s deterministic gates and refreshed the archived verification logs under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-generator-integration/verification/`.
- Kept scope unchanged: no benchmark rows, compare-wrapper model-ID work, YAML/prompt edits, or paper-bundle mutation were introduced.

## Completed Current-Scope Work

- The blocking implementation-review defect is fixed in scope: `neuralop_uno` checkpoint rebuild no longer accepts an invalid saved output mode by silently coercing `amp_phase_logits` to `real_imag`.
- Checkpoint reload now matches the approved plan/design contract for the locked Lines128 CDI lane: `neuralop_uno` supports only `generator_output_mode="real_imag"` and raises a clear error for other saved modes.
- The previously completed direct-runner guardrails and documentation updates remain intact, and the current checkout now satisfies the remaining approved current-scope work for this backlog item.

## Follow-Up Work

- Compare-wrapper model-ID routing and append-only U-NO paper-row execution remain separate follow-on items by design.
- Any expansion of `neuralop_uno` beyond the locked Lines128 CDI lane (`N=128`, `gridsize=1`, `C=1`, `real_imag`) still requires a new approved plan.
- A lightweight doc-surface regression check tying architecture tables to the registry/config literals would reduce future drift, but it was not required to approve this backlog item.

## Residual Risks

- `neuralop_uno` still depends on external `neuraloperator==2.0.0`; future environment drift will fail closed rather than silently substituting a different implementation.
- The adapter and runner intentionally reject broader CDI contracts; users attempting `gridsize>1`, non-`real_imag`, or non-`128x128` U-NO runs will need separate implementation work.
- The broader paper-table extension path is still unimplemented here, so direct runner support should not be read as completed benchmark integration.

## Verification

- Red-first checkpoint selector:
  - `pytest -q tests/torch/test_lightning_checkpoint.py -k 'neuralop_uno_checkpoint_rejects_invalid_saved_output_mode'`
  - Result before patch: `1 failed, 1 passed`
  - Result after patch: `2 passed`
- Red-first runner selector:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k 'neuralop_uno and (supervised_training or non_lines128 or non_unit_gridsize or non_real_imag_output)'`
  - Result: `3 failed, 1 passed` before the runner guard was added, then `4 passed` after the fix.
- Focused selector:
  - `pytest -q tests/torch/test_neuralop_uno_generator.py tests/torch/test_generator_registry.py tests/torch/test_lightning_checkpoint.py tests/torch/test_grid_lines_torch_runner.py`
  - Result: `152 passed`
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
