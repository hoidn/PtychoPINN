# Completed In This Pass

- Patched the shared compare/reporting surfaces so grid-lines compare runs now
  emit `metrics_table.csv`, wrapper invocations record runtime/commit/completion
  metadata, and library-driven Torch runner calls persist per-row
  `invocation.json` / `invocation.sh` artifacts.
- Backfilled the completed stable root at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
  without relaunching training by writing the missing CSV table, row-level
  invocation artifacts for `pinn_hybrid_resnet` and `pinn_ffno`, and repaired
  wrapper completion metadata.
- Updated the durable preflight, summary, and studies index so they describe the
  repaired stable-root contract instead of the earlier incomplete artifact set.

# Completed Current-Scope Work

- The implementation-review blockers are closed:
  - wrapper root now includes `metrics.json`, `metrics_table.csv`,
    `metrics_table.tex`, `metrics_table_best.tex`, and the required compare
    visuals
  - both `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/` now include
    `invocation.json`, `invocation.sh`, `metrics.json`, `history.json`,
    `model.pt`, and `randomness_contract.json`
  - wrapper `invocation.json` now records runtime provenance, git commit, and
    completion metadata
- Fresh verification for this repair pass:
  - `pytest -q tests/test_grid_lines_invocation_logging.py tests/test_grid_lines_compare_wrapper.py`
    -> `50 passed, 23 warnings in 7.76s`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py`
    -> `111 passed, 20 warnings in 19.49s`
  - required backlog gate:
    `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `162 passed, 43 warnings in 283.25s`
  - required backlog gate:
    `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
  - workflow-policy integration marker:
    `pytest -q -m integration`
    -> `5 passed, 4 skipped, 1645 deselected in 299.91s`
- Archived logs for this pass:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T104852Z_pytest.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T105343Z_compileall.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T105350Z_integration.log`

# Follow-Up Work

- Add a small stable-root validator so future backlog items cannot be marked
  complete while required files such as `metrics_table.csv` or per-row
  invocation artifacts are still missing.

# Residual Risks

- The per-row invocation artifacts in the repaired stable root were backfilled
  from the fixed wrapper contract after the original in-process compare had
  already finished, so their timestamps reflect the repair pass rather than the
  original training start time.
- The integration-marker run still carries four pre-existing skips documented by
  the test suite itself; this pass did not change those dependencies.
