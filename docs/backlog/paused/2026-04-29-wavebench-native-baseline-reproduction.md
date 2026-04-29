---
priority: 42
plan_path: TBD_AFTER_PREFLIGHT
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight.md"),
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
  - Select only if the WaveBench preflight status is `ready_for_supervised_plan` or `ready_for_supervised_and_physics_plan`.
  - Native WaveBench FNO/U-Net rows are needed before repo-local shared-encoder rows can be interpreted against the published benchmark context.
---

# Backlog Item: Reproduce Native WaveBench FNO/U-Net Baselines

## Objective

- Reproduce or evaluate the native WaveBench FNO and U-Net baselines on the
  selected inverse-source variant from the preflight.

## Scope

- Consume the checked-in WaveBench preflight summary and metadata.
- Use the selected inverse-source variant, dataset files, split, normalization,
  and loader path from preflight.
- Load provided native WaveBench FNO/U-Net checkpoints if compatible; otherwise
  run the shortest native-baseline reproduction needed for a fair reference.
- Emit table-ready metrics and a concise summary under the NeurIPS planning
  tree.
- Keep native WaveBench rows separate from shared-encoder rows.

## Notes for Reviewer

- Do not treat this as an SRU-Net comparison item.
- Do not change the selected WaveBench variant after seeing baseline metrics.
- Do not add WaveBench rows to manuscript tables unless a roadmap/evidence
  package amendment has added WaveBench as an additional evidence lane.
- If native checkpoints are unavailable or incompatible, record
  `native_checkpoint_unavailable` and route to a retraining decision rather
  than inventing an incomparable baseline.
