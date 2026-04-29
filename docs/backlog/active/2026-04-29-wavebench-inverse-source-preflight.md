---
priority: 41
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/execution_plan.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing WaveBench preflight inputs: {missing}")
    print("wavebench preflight inputs present")
    PY
prerequisites: []
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The manuscript may benefit from an additional 2D known-forward-model inverse-wave benchmark candidate alongside CDI and CNS.
  - WaveBench inverse source reconstruction has a 2D target, known wave forward model, and published FNO/U-Net baseline infrastructure, but local dataset/checkpoint/solver compatibility is not yet verified.
  - This preflight decides whether WaveBench should become an additional manuscript evidence target before any roadmap-level addition is attempted.
---

# Backlog Item: Preflight WaveBench Inverse Source Benchmark

## Objective

- Determine whether WaveBench inverse source reconstruction is a practical
  secondary inverse-wave benchmark for the SRU-Net manuscript, covering both
  supervised and physics-informed variants.

## Scope

- Consume
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md`.
- Inspect the WaveBench repository, dataset availability, inverse-source
  variants, loader shapes, native FNO/U-Net checkpoint availability, baseline
  scripts/notebooks, and forward-model implementation surfaces.
- Decide the first runnable inverse-source variant and record exact input,
  target, and measurement tensor shapes.
- Validate whether a differentiable local forward model can reproduce
  WaveBench boundary measurements from ground-truth `q_0` closely enough to
  support physics-informed training.
- Produce a preflight summary with one of:
  - `ready_for_supervised_plan`
  - `ready_for_supervised_and_physics_plan`
  - `needs_dataset_or_checkpoint_decision`
  - `not_suitable_for_current_manuscript`
- Do not run the full benchmark and do not alter the CDI or CNS paper lanes in this
  item.

## Notes for Reviewer

- Keep WaveBench as a candidate evidence lane until data, baseline, and forward
  solver compatibility are verified.
- Do not describe WaveBench inverse source as geology, full waveform inversion,
  or material-property inversion.
- Do not accept a physics-informed row unless `F_c(q_0) ~= y` is demonstrated
  on ground-truth examples under the selected dataset variant.
- Keep native WaveBench FNO/U-Net reference rows separate from shared-encoder
  internal comparison rows.
- If exact physics-loop reproduction is not practical, the supervised benchmark
  may still be useful, but physics-informed claims must be deferred or framed
  as approximate-model regularization.
