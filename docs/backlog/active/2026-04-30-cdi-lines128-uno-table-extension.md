---
priority: 70
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md
check_commands:
  - pytest -q tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
  - python -m compileall -q ptycho_torch scripts/studies
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
        Path(".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/metrics.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing U-NO table-extension inputs: {missing}")
    print("U-NO table-extension inputs present")
    PY
prerequisites:
  - 2026-04-30-cdi-lines128-uno-generator-integration
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - This item appends U-NO rows to the completed Lines128 CDI table without rerunning already completed rows.
  - The output should be a new extended bundle, not a rewrite of the authoritative six-row root.
  - Steering on 2026-04-30 moved WaveBench candidate work ahead of this optional U-NO table extension; do not select this until eligible WaveBench items have been attempted or blocked by their own gates.
---

# Backlog Item: Append U-NO Rows To Lines128 CDI Table

## Objective

- Run U-NO in both PINN and supervised modes under the locked Lines128 CDI
  contract and append those rows to the existing complete paper table without
  rerunning completed rows.

## Scope

- Consume the completed six-row base root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`.
- Launch only the new rows:
  - `pinn_neuralop_uno`
  - `supervised_neuralop_uno`
- Use the exact locked Lines128 contract from the completed table:
  dataset, split, probe, seed, epochs, scheduler, loss, output mode, fixed
  visual sample ids, and shared-scale policy.
- Promote/audit the existing six completed rows into a new extended root by
  lineage reference; do not rerun or rewrite them.
- Emit a new derived bundle such as
  `complete_table_plus_uno_<timestamp>` with:
  - merged JSON/CSV/TeX metrics tables
  - model and paper benchmark manifests
  - fixed-sample visual bundle
  - row-local U-NO invocation/config/history/metrics/reconstruction artifacts
  - explicit base-row lineage and fresh U-NO row provenance
- Update durable summaries and indexes so the U-NO extension is discoverable.

## Notes for Reviewer

- Reject any implementation that reruns all completed rows just to add U-NO.
- Reject any implementation that overwrites
  `complete_table_20260430T150757Z_repair_tmux`.
- Do not accept the extended bundle as complete if U-NO row-local launcher
  proof, metrics, visuals, or environment provenance are missing.
- Keep the claim boundary distinct:
  `complete_lines128_cdi_benchmark_plus_uno_extension`.
