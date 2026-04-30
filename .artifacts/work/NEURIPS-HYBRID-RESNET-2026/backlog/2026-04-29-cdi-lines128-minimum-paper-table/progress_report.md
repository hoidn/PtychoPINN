# Active Work

- none; the backlog item is now complete

# Current Status

- final state: `paper_complete`
- authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z`
- execution path used:
  `same_root_recovery`
- same-root recovery command:
  - `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T235811Z --reuse-existing-recons`
  - exit status: `0`
- the fresh root now contains:
  - merged bundle artifacts:
    `metrics.json`, `metric_schema.json`, `model_manifest.json`,
    `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`,
    `paper_benchmark_manifest.json`
  - TensorFlow row-local provenance for `baseline` and `pinn`:
    `invocation.json`, `invocation.sh`, `config.json`, `history.json`,
    `metrics.json`, `stdout.log`, `stderr.log`
  - recovered Torch row-local provenance now also includes:
    `runs/pinn_hybrid_resnet/config.json`,
    `runs/pinn_fno_vanilla/config.json`
  - final required visuals:
    per-row `amp_phase_*.png`, per-row `amp_phase_error_*.png`,
    `compare_amp_phase.png`, `frc_curves.png`
- archived verification logs for this pass:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_review_fix_current.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates_review_fix_current.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_review_fix_current.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_same_root_recovery_20260429T235811Z.log`

# Next Resume Condition

- none for this backlog item
- later complete-table follow-up remains separate:
  `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno`

# Blocker

- none

# Blocker Class

- n/a
