# Execution Report

## Completed In This Pass

- Root-caused implementation review High #1: the shared
  `paper_provenance.write_launcher_completion_evidence` helper silently
  returned `None` for fresh single-row training launches because
  `parsed_args.reuse_existing_recons` was false and `parsed_args.mode` was not
  `complete_table` or `extend_with_uno`. The wrapper finalization path
  (`grid_lines_compare_wrapper._finalize_root_launcher_completion_artifacts`)
  was therefore unable to emit the required row-local
  `launcher_completion.json` for this rerun, even though the wrapper
  invocation was completed cleanly with `exit_code=0` and the launcher
  stderr already contained both required completion markers.
- Applied the narrow root-cause fix in
  `scripts/studies/paper_provenance.py` by removing the
  `parsed_args.reuse_existing_recons` / `parsed_args.mode` early return.
  The function's other safety checks (wrapper completion + `exit_code == 0`,
  row metrics/history/recon present, launcher log freshness vs. invocation
  start, `Saved artifacts to .../runs/<row>` plus
  `Torch runner complete. Artifacts in .../runs/<row>` markers in stderr or
  `DEBUG eval_reconstruction [<row>]:` markers in stdout) are sufficient for
  correctness and remain in force.
- Added focused regression coverage:
  - `tests/studies/test_paper_provenance.py::test_write_launcher_completion_evidence_emits_for_fresh_single_row_training`
  - `tests/test_grid_lines_compare_wrapper.py::test_main_finalizes_launcher_completion_for_fresh_single_row_training`
- Reverified end-to-end against the real preserved wrapper artifact root for
  this item: removed the prior hand-written
  `runs/supervised_ffno/launcher_completion.json`, called the patched
  `_finalize_root_launcher_completion_artifacts` against the unchanged
  `runs/supervised_ffno_no_refiner_20260506T232535Z` root, and confirmed the
  helper auto-wrote the row artifact with
  `evidence_source = "wrapper_launcher_stderr_row_completion_markers"`. Full
  reverification record at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/verification/launcher_completion_auto_emit_reverification.md`.
- Re-ran the deterministic preflight gate from the plan:
  - `python` FFNO no-refiner instantiation proof
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"` (3 passed)
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"` (4 passed)
  - `python -m compileall -q ptycho_torch scripts/studies` (clean)
  - new launcher_completion regressions:
    `pytest -q tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py -k "launcher_completion or fresh_single_row or stdout_eval or row_completion_markers or stale_current_root"` (6 passed)
- Disclosed the previously omitted abandoned earlier launch
  `runs/supervised_ffno_no_refiner_20260506T232355Z` (cancelled before the
  torch runner started; only `datasets/N128/gs1/train.npz` was written; no
  `metrics.json`, `history.json`, `recon.npz`, model checkpoint, or
  lightning logs were produced) in both the durable summary and the
  reverification record.
- Updated the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`
  to describe the real fix, drop the resolved residual risk, and point at
  the reverification artifact.

## Completed Current-Scope Work

- Plan Task 1: Freeze authorities and audit the existing launch path.
- Plan Task 2: Apply only the minimal runner/wrapper/test fixes needed.
  - This pass replaced the prior "no production code edit was required"
    determination after implementation review showed the shared completion
    helper was the actual gap. The narrow fix is contained to one early-
    return guard in `paper_provenance.py` plus two focused regression tests.
- Plan Task 3: Launch the fresh supervised FFNO no-refiner row.
  - Reused the existing successful 232535Z launch (tracked-PID exit `0`,
    required row-local artifacts present and freshly written). The earlier
    abandoned 232355Z attempt is now disclosed.
- Plan Task 4: Audit no-refiner purity and same-contract fairness.
- Plan Task 5: Refresh objective-control outputs without rewriting the base
  table.
- Plan Task 6: Write the durable summary and update discoverability.

## Follow-Up Work

- The corrected no-refiner objective-control pair is active manuscript-facing
  evidence, but the immutable six-row CDI authority and any broader
  manuscript wording that still cites the historical FFNO-local-refiner
  objective-control rows require a later explicit table-refresh sweep. That
  work is intentionally separate from this fix loop, per the implementation
  review's "Follow-Up Work" note and the plan's explicit non-goal that this
  item must not silently rewrite the canonical six-row table.

## Residual Risks

- Manuscript surfaces outside this item (the immutable six-row CDI table and
  prose that still references the local-refiner FFNO objective-control rows)
  remain unrefreshed; they will continue to point to historical proxy
  evidence until the deferred no-refiner table-refresh sweep runs.
