---
priority: 2
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from ptycho_torch.generators.ffno import FfnoGeneratorModule
    model = FfnoGeneratorModule(cnn_blocks=0)
    assert len(model.refiners) == 0
    print("CDI FFNO no-refiner generator instantiates")
    PY
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - Completed Lines128 CDI FFNO rows used `fno_cnn_blocks=2`, which adds local CNN refinement after the factorized Fourier stack.
  - Pure FFNO manuscript claims require a corrected row with `fno_cnn_blocks=0`.
  - This item is higher priority than WaveBench candidate work because it repairs an already manuscript-facing CDI comparator.
---

# Backlog Item: Rerun CDI Lines128 FFNO With No Local Refiner

## Objective

- Rerun the Lines128 CDI `pinn_ffno` row under the locked paper contract with
  `fno_cnn_blocks=0`.
- Preserve the completed `pinn_ffno` artifact roots as historical
  FFNO-local-refiner proxy evidence; do not overwrite them.
- Produce the corrected pure-FFNO comparator needed before the manuscript or
  tables can make canonical CDI FFNO claims.

## Scope

- Use the completed Lines128 CDI table as the fixed contract source for every
  field except the corrected local-refiner count.
- Fixed fields include:
  - `N=128`, `gridsize=1`, synthetic grid-lines, Run1084 fixed probe;
  - `seed=3`, `nimgs_train=2`, `nimgs_test=2`, fixed sample ids;
  - `40` epochs, batch `16`, Adam `2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`;
  - `torch_loss_mode=mae`, `torch_output_mode=real_imag`;
  - `fno_modes=12`, `fno_width=32`, `fno_blocks=4`.
- Required correction:
  - `fno_cnn_blocks=0`.
- Run only the corrected FFNO row. Reuse non-FFNO rows by lineage in later
  table-refresh work.

## Outputs

- Item-local artifacts under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/`.
- Row artifacts for corrected `pinn_ffno`:
  invocation, config, history, metrics, checkpoint, recon arrays, visual panels,
  and provenance.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`.

## Completion Gate

- The row config and invocation must record `fno_cnn_blocks=0`.
- The model must contain no `_LocalResidualRefiner` modules.
- The summary must compare corrected no-refiner metrics against the historical
  `fno_cnn_blocks=2` proxy row without relabeling the historical row.
- Any table promotion must be left to
  `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`.
