---
priority: 2
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from ptycho_torch.generators.ffno import FfnoGeneratorModule
    model = FfnoGeneratorModule(cnn_blocks=0)
    assert len(model.refiners) == 0
    print("CDI supervised FFNO no-refiner generator instantiates")
    PY
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The completed supervised FFNO control also used the old local-refiner proxy profile.
  - Training-procedure comparisons must compare no-refiner FFNO + PINN against no-refiner FFNO + supervised.
  - This item is higher priority than WaveBench candidate work because it repairs an already manuscript-facing objective-control row.
---

# Backlog Item: Rerun Supervised CDI FFNO With No Local Refiner

## Objective

- Rerun the Lines128 CDI `supervised_ffno` control with `fno_cnn_blocks=0`.
- Compare it only against the corrected no-refiner `pinn_ffno` row, not the
  historical local-refiner proxy row.
- Preserve the completed supervised extension as provenance-bearing historical
  proxy evidence.

## Scope

- Use the locked Lines128 supervised-extension contract and change only the
  FFNO local-refiner count:
  - `fno_cnn_blocks=0`.
- Keep dataset, split, probe, visual samples, metrics, scheduler, and epoch
  budget identical to the completed supervised extension.
- Do not rerun CNN, FNO, SRU-Net, U-NO, or historical proxy FFNO rows.

## Outputs

- Item-local artifacts under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/`.
- Corrected `supervised_ffno` row artifacts and a two-row objective-control
  comparison against corrected no-refiner `pinn_ffno`.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`.

## Completion Gate

- The supervised row config and invocation must record `fno_cnn_blocks=0`.
- The objective-control table must use corrected no-refiner FFNO rows for both
  training procedures.
- Historical `FFNO-local proxy` rows may appear only as caveated context.
