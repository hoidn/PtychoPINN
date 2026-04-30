# Active Work

- none; the backlog item is now complete

# Current Status

- final state: `paper_complete`
- authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z`
- execution path used:
  `fresh_rerun_after_review_fix`
- review-fix note:
  the earlier same-root recovery claim was rejected because reused-row
  provenance was synthetic. The accepted bundle now comes from a fresh rerun
  that emitted real row-local logs, completed invocation metadata, and
  exit-code proofs for every required row.
- accepted rerun command:
  - `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z`
  - exit status: `0`
- the authoritative root now contains:
  - merged bundle artifacts:
    `metrics.json`, `metric_schema.json`, `model_manifest.json`,
    `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`,
    `paper_benchmark_manifest.json`
  - root-level provenance artifacts:
    `invocation.json`, `invocation.sh`, `dataset_identity_manifest.json`,
    `split_manifest.json`, `live_stdout.log`, `live_stderr.log`
  - per-row provenance for `baseline`, `pinn`, `pinn_hybrid_resnet`,
    `pinn_fno_vanilla`:
    `invocation.json`, `invocation.sh`, `config.json`, `history.json`,
    `metrics.json`, `stdout.log`, `stderr.log`, `exit_code_proof.json`
  - final required visuals:
    per-row `amp_phase_*.png`, per-row `amp_phase_error_*.png`,
    `compare_amp_phase.png`, `frc_curves.png`
- archived verification logs for the accepted pass:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/focused_pytest_review_fix_20260430b.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/backlog_required_pytest_review_fix_20260430b.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_review_fix_20260430b.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_fresh_rerun_20260430T035104Z.log`

# Next Resume Condition

- none for this backlog item
- later complete-table follow-up remains separate:
  `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno`

# Blocker

- none

# Blocker Class

- n/a
