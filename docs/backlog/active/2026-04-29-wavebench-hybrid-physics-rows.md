---
priority: 34
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-hybrid-physics-rows/execution_plan.md
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
  - 2026-04-29-wavebench-shared-encoder-supervised-benchmark
  - 2026-04-29-wavebench-forward-model-physics-validation
related_roadmap_phases:
  - wavebench-additional-inverse-wave-extension
signals_for_selection:
  - Select only if the supervised WaveBench rows exist and the forward-model validation gate passed.
  - This item tests whether wave-equation consistency improves selected learned inverse models.
  - Steering on 2026-04-30 moved this WaveBench follow-up ahead of remaining optional U-NO table-extension work, subject to the supervised and forward-model gates.
---

# Backlog Item: Run WaveBench Hybrid Physics-Informed Rows

## Objective

- Add physics-informed and hybrid supervised-plus-physics rows for selected
  WaveBench inverse-source models after the forward-model validity gate passes.

## Scope

- Start with the strongest one or two supervised shared-encoder rows, plus one
  reference row such as U-Net or FNO.
- Train hybrid supervised-plus-physics rows:

  `lambda_q L_sup + lambda_y L_phys`

- Add physics-only rows only if the hybrid rows are stable and the forward
  solver cost is acceptable.
- Report source-reconstruction metrics and waveform-consistency metrics
  separately.
- Emit boundary-data visual comparisons:
  observed `y`, predicted `F_c(hat q_0)`, and residual.

## Notes for Reviewer

- Do not start this item without a passed forward-model reproduction report.
- Do not call approximate-model rows physics-informed benchmark evidence.
- Do not bury cases where waveform residual improves but source reconstruction
  degrades; both metric families must be visible.
- Keep this as an additional WaveBench evidence lane alongside CDI and CNS.
