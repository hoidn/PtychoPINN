# Active Work

- Fixed the minimum-subset bundle writer so nested NumPy-backed scalars can be serialized when the final paper bundle is written:
  `scripts/studies/metrics_tables.py`.
- Added a regression test covering nested NumPy scalar payloads at the paper-bundle boundary:
  `tests/studies/test_metrics_tables.py::test_write_paper_bundle_serializes_numpy_scalars_in_nested_payloads`.
- Re-audited recovered roots under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/`
  and recorded the classification note at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/recovered_root_audit.md`.

# Current Status

- The new red/green serialization fix is complete and verified:
  - focused red test passed after the patch:
    `tests/studies/test_metrics_tables.py -k numpy_scalars_in_nested_payloads`
  - focused plan selectors are green and archived:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_focused_postjsonfix.log`
  - backlog-required deterministic gates are green and archived:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/pytest_deterministic_gates_postjsonfix.log`
  - compile gate is green and archived:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_postjsonfix.log`
- Recovered-root status after this pass:
  - `minimum_subset_20260429T204000Z`: `stale_do_not_reuse`
  - `minimum_subset_20260429T204642Z`: `failed_recoverable`
  - `minimum_subset_20260429T213028Z`: `failed_recoverable` after all four rows completed and final bundle collation failed on `numpy.float32` JSON serialization
- A compliant fresh rerun was attempted in tmux session `lines128-min-rerun2` on socket `/tmp/claude-tmux-sockets/claude.sock` with tracked shell job PID `1188880`.
  - output root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260429T224103Z`
  - failure point:
    dataset NPZ writing during `build_grid_lines_datasets(...)`
  - observed error:
    `OSError: [Errno 28] No space left on device`
  - filesystem state at failure time:
    `/dev/nvme0n1p2 457G used 434G, avail 0, use% 100%`

# Next Resume Condition

- Resume after enough disk space is freed on `/` to permit a fresh `minimum_subset_<timestamp>` root. Current failed roots for this item already occupy approximately:
  - `1.6G` for `minimum_subset_20260429T204000Z`
  - `1.7G` for `minimum_subset_20260429T204642Z`
  - `2.2G` for `minimum_subset_20260429T213028Z`
  - `151M` for `minimum_subset_20260429T224103Z`
- Once space is available, relaunch the minimum-subset benchmark into a new root under the same frozen contract and confirm `__EXIT__:0` plus fresh merged bundle surfaces (`metrics.json`, `metric_schema.json`, `model_manifest.json`, CSV/TeX tables, visuals, and source artifacts).

# Blocker

- Missing local disk space on `/` prevents the required fresh same-root rerun from writing dataset artifacts. The post-fix rerun is not currently active.

# Blocker Class

- `missing_resource`
