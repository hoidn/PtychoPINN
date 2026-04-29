# Active Work

- Minimum-subset execution authority is checked in at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`.
- The derived execution manifest is written at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`.
- The four-row minimum subset is running in tmux session `lines128-min-run` on socket
  `/tmp/claude-tmux-sockets/claude.sock`, output root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T204642Z`.

# Current Status

- Focused selectors are green:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_workflow.py`
- Required deterministic gates are green and archived:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall.log`
- The first launch attempt failed immediately with a recoverable import-path bug in `scripts/studies/lines128_paper_benchmark.py`; that was patched and a direct script-path CLI test now passes.
- The second launch showed a real contract bug: explicit-model execution left the TF `cnn` rows at the wrapper default `60` epochs instead of the locked `40`. That wrapper bug is now fixed and covered by `tests/test_grid_lines_compare_wrapper.py::test_wrapper_uses_locked_epoch_budget_for_tf_rows_in_explicit_model_mode`.
- A short relaunch wrapper mistake around a shell-local run-root variable failed before the benchmark started; that was corrected without changing the study contract.
- The current launch is live:
  - tmux wrapper PID: `1125671`
  - Python worker PID: `1125694`
  - stdout/stderr log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T204642Z/live_stdout.log`
  - observed state: fresh startup completed under the corrected wrapper; the valid run is in progress and no semantic blocker is currently known.

# Next Resume Condition

- Resume when the live tmux run prints `__EXIT__:0` and the run root contains fresh row-local artifacts plus the merged bundle surfaces (`metrics.json`, `metric_schema.json`, `model_manifest.json`, CSV/TeX tables, visuals, FRC/source artifacts as emitted by the harness).
- If the tmux run exits nonzero, capture the pane, diagnose the concrete failure, patch narrowly, and relaunch into a fresh `minimum_subset_<timestamp>` root without changing the frozen contract.
