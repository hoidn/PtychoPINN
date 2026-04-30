---
priority: 32
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-forward-model-physics-validation/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing WaveBench preflight summary: {missing}")
    print("wavebench preflight summary present")
    PY
prerequisites:
  - 2026-04-29-wavebench-inverse-source-preflight
related_roadmap_phases:
  - wavebench-additional-inverse-wave-extension
signals_for_selection:
  - Select only if preflight classifies physics readiness as `exact_physics_loop_ready` or identifies a narrow solver-alignment task likely to reach that status.
  - This item is required before any WaveBench row can be called physics-informed.
  - Steering on 2026-04-30 moved this WaveBench follow-up ahead of remaining optional U-NO table-extension work, subject to the preflight outcome.
---

# Backlog Item: Validate WaveBench Forward Model For Physics-Informed Rows

## Objective

- Validate a differentiable WaveBench inverse-source forward model well enough
  to support later physics-informed or hybrid supervised-plus-physics training.

## Scope

- Consume the selected WaveBench variant, tensor contract, and solver metadata
  from preflight.
- Implement, adapt, or wrap the minimal differentiable forward map:

  `F_c(q_0) = M u(q_0; c)`

  for the selected inverse-source variant.
- Reproduce WaveBench boundary measurements from held-out ground-truth `q_0`
  examples.
- Emit a reproduction report containing:
  - dataset variant and file identifiers
  - grid/time/boundary/receiver/initial-condition contract
  - normalization policy
  - sample count
  - waveform MAE, RMSE, relative L2, and normalized residual
  - pass/fail status against the thresholds in the WaveBench design
- If exact reproduction fails, classify the result as approximate-model
  regularization only and do not authorize physics-informed benchmark claims.

## Notes for Reviewer

- This item is a validity gate, not a model-training item.
- Do not loosen thresholds after seeing failures without a checked-in rationale
  based on dataset scale or metadata.
- Do not label any downstream row `physics-informed` unless this item passes.
- If solver code is unavailable or not differentiable, record
  `physics_loop_deferred` rather than creating a weak surrogate without
  disclosure.
