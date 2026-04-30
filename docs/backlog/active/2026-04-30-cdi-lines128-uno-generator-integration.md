---
priority: 36
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md
check_commands:
  - pytest -q tests/torch/test_generator_registry.py tests/torch/test_loss_modes.py
  - python -m compileall -q ptycho_torch scripts/studies
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("ptycho_torch/generators/neuralop_uno.py"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing U-NO integration outputs: {missing}")
    print("U-NO integration outputs present")
    PY
prerequisites:
  - 2026-04-30-cdi-lines128-uno-design-preflight
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - U-NO table execution is not valid until `neuralop_uno` is a real generator architecture in both PINN and supervised paths.
  - The integration must prove architecture identity, not just row labels.
---

# Backlog Item: Integrate NeuralOperator U-NO Generator

## Objective

- Add `neuralop_uno` as a real Torch generator architecture that can be used by
  both Lines128 PINN and supervised training procedures.

## Scope

- Add a `ptycho_torch/generators/neuralop_uno.py` adapter around
  `neuralop.models.UNO`.
- Register `neuralop_uno` in the generator registry and model architecture
  configuration.
- Ensure `architecture=neuralop_uno, mode=Unsupervised` uses the U-NO generator
  body.
- Ensure `architecture=neuralop_uno, mode=Supervised` uses the same U-NO
  generator body, not the legacy supervised autoencoder path.
- Preserve the existing `real_imag` generator output contract and fail closed
  if U-NO cannot emit the required shape.
- Add focused tests for:
  - registry resolution
  - U-NO forward shape
  - supervised and PINN model construction
  - missing/incompatible NeuralOperator dependency behavior
- Do not run full Lines128 benchmark rows in this item.

## Notes for Reviewer

- The row is not integrated merely because `neuralop.models.UNO` imports.
  Tests must prove the actual generator body is used in both training modes.
- Do not change existing Hybrid/FNO/FFNO/CNN row behavior.
- Do not tune U-NO hyperparameters in this item except to implement the
  preflight-frozen constructor defaults.
