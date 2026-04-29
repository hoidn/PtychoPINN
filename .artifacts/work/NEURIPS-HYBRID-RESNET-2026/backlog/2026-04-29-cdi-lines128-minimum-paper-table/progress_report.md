# Active Work

- Minimum-subset execution authority is checked in at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`.
- The derived execution manifest is written at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`.
- The prior root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T204642Z`
  is classified `failed_recoverable`: TensorFlow training/inference and stitched metrics completed, but row collation crashed before the torch rows launched.
- The current four-row minimum-subset rerun is active in tmux session `lines128-min-rerun` on socket
  `/tmp/claude-tmux-sockets/claude.sock`, output root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z`.

# Current Status

- Recoverable benchmark crash root cause is identified and fixed:
  - `ptycho/workflows/grid_lines_workflow.py` emitted TF paper-row payloads with `"N": null`.
  - `scripts/studies/grid_lines_compare_wrapper.py` now treats explicit `None` as missing and backfills the locked `n_value`.
  - Regression coverage:
    `tests/test_grid_lines_compare_wrapper.py::test_wrapper_backfills_tf_row_n_when_payload_emits_none`
- Required deterministic gates are green and archived:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall.log`
- Focused minimum-subset selector is green and archived:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_postfix.log`
- Workflow integration marker is green and archived:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_integration_postfix.log`
- The current launch is live:
  - tmux shell PID: `1158849`
  - Python worker PID: `1158854`
  - stdout/stderr log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z/live_stdout.log`
  - observed phase at this handoff: TensorFlow `baseline` and `pinn` completed, Torch `pinn_hybrid_resnet` completed with row-local artifacts, and Torch `pinn_fno_vanilla` is now the active row.
  - observed state: root-level `metrics.json` still reflects only the completed TensorFlow rows, `recons/pinn_hybrid_resnet/recon.npz` plus `runs/pinn_hybrid_resnet/{metrics.json,history.json,model.pt,randomness_contract.json}` are present, and `runs/pinn_fno_vanilla/invocation.{json,sh}` has been created for the final row.
  - latest live-log observation (`2026-04-29T22:15:12Z`): `pinn_fno_vanilla` Lightning training is in progress around `Epoch 25/40`, so the chosen root remains an in-flight partial bundle rather than a completed four-row result.

# Next Resume Condition

- Resume when tmux session `lines128-min-rerun` prints `__EXIT__:0` and the run root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T213028Z`
  contains fresh row-local artifacts for `pinn_fno_vanilla` plus the merged bundle surfaces (`metrics.json`, `metric_schema.json`, `model_manifest.json`, CSV/TeX tables, visuals, FRC/source artifacts as emitted by the harness).
- If the tmux run exits nonzero, capture the pane, diagnose the concrete failure, patch narrowly, and relaunch into a fresh `minimum_subset_<timestamp>` root without changing the frozen contract.
