---
priority: 35
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing U-NO preflight inputs: {missing}")
    print("U-NO preflight inputs present")
    PY
  - |
    python - <<'PY'
    import neuralop
    from neuralop.models import UNO
    print(f"neuralop {neuralop.__version__}: {UNO}")
    PY
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-04-29-cdi-lines128-supervised-equivalent-rows
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - U-NO is an optional append-only Lines128 CDI comparator, not a replacement for the completed six-row table.
  - The external NeuralOperator package is already present in ptycho311 locally, but its constructor defaults and shape contract must be frozen before any benchmark rows run.
  - This item should resolve environment/API readiness without editing model code or running paper rows.
---

# Backlog Item: Preflight Lines128 NeuralOperator U-NO Extension

## Objective

- Verify the local U-NO environment/API contract and freeze the exact
  `neuralop_uno` row settings needed for a later append-only Lines128 table
  extension.

## Scope

- Consume
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`.
- Verify `ptycho311` has `neuraloperator==2.0.0`, import module `neuralop`,
  and `neuralop.models.UNO`.
- Record the local `UNO` constructor signature, package provenance, Python,
  Torch, CUDA, and GPU environment details.
- Freeze initial U-NO settings before metrics are available:
  `hidden_channels`, `lifting_channels`, `projection_channels`, `n_layers`,
  `uno_n_modes`, input/output channel policy, and `generator_output_mode`.
- Run only tiny API/shape probes. Do not add registry support or benchmark
  rows in this item.
- Emit a durable preflight summary with one of:
  - `ready_for_uno_generator_integration`
  - `blocked_neuraloperator_missing_or_incompatible`
  - `blocked_uno_shape_contract_mismatch`

## Notes for Reviewer

- Do not substitute the repo's internal `HybridUNOGenerator` for external
  NeuralOperator U-NO.
- Do not tune U-NO defaults after seeing benchmark metrics.
- A blocker is acceptable if the external package API cannot satisfy the
  locked Lines128 CDI contract.
