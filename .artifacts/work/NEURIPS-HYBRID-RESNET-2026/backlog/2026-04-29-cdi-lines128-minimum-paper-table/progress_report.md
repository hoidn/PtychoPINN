# Active Work

- Minimum-subset execution authority is checked in at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md`.
- The derived execution manifest is written at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json`.
- The four-row minimum subset is running in tmux session `lines128-min` on socket
  `/tmp/claude-tmux-sockets/claude.sock`, output root
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T204000Z`.

# Current Status

- Focused selectors are green:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_workflow.py`
- Required deterministic gates are green and archived:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall.log`
- The first launch attempt failed immediately with a recoverable import-path bug in `scripts/studies/lines128_paper_benchmark.py`; that was patched and a direct script-path CLI test now passes.
- The current launch is live:
  - tmux wrapper PID: `1116511`
  - Python worker PID: `1116529`
  - observed state: dataset/probe setup complete and execution has moved into the benchmark workflow; no semantic blocker has appeared yet.

# Next Resume Condition

- Resume when the live tmux run prints `__EXIT__:0` and the run root contains fresh row-local artifacts plus the merged bundle surfaces (`metrics.json`, `metric_schema.json`, `model_manifest.json`, CSV/TeX tables, visuals, FRC/source artifacts as emitted by the harness).
- If the tmux run exits nonzero, capture the pane, diagnose the concrete failure, patch narrowly, and relaunch into a fresh `minimum_subset_<timestamp>` root without changing the frozen contract.
