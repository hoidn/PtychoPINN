## Completed In This Pass

- fixed the implementation-review blocker in code by moving TensorFlow
  provenance to real row-scoped workflow captures, so each TF row now writes
  its own `stdout.log` and `stderr.log` instead of inheriting a duplicated
  shared compare-wrapper capture
- stopped the compare wrapper from overwriting TensorFlow row logs with one
  shared workflow transcript
- tightened paper-grade validation so required rows are downgraded when they
  reuse identical `stdout.log` payloads, which makes the earlier accepted root
  fail honestly under the stricter contract
- added regression coverage for both failure modes across
  `tests/test_grid_lines_compare_wrapper.py`,
  `tests/studies/test_metrics_tables.py`,
  `tests/studies/test_lines128_paper_benchmark.py`, and the existing workflow
  fixture coverage touched by the new TF row-log path
- reran the focused and required backlog verification suites and archived fresh
  logs for this pass
- started a fresh post-fix benchmark rerun in
  `runs/minimum_subset_20260430T051928Z`, then stopped it after confirming the
  long-running TensorFlow training phase would extend beyond the scope of this
  review-fix commit; that root is incomplete and not paper-grade evidence

## Completed Current-Scope Work

- the implementation-review findings are resolved at the code and validator
  level
- fresh verification for this pass:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/test_grid_lines_workflow.py tests/torch/test_grid_lines_torch_runner.py`
    -> `254 passed, 53 warnings in 44.74s`
    (`verification/focused_pytest_tf_provenance_fix_20260429.log`)
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `172 passed, 47 warnings in 303.97s`
    (`verification/backlog_required_pytest_tf_provenance_fix_20260429.log`)
  - `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
    (`verification/compileall_tf_provenance_fix_20260429.log`)
- the historical root
  `runs/minimum_subset_20260430T035104Z` is no longer treated as paper-grade
  under the current validator because its TensorFlow `baseline` and `pinn`
  `stdout.log` artifacts are duplicated shared-workflow captures rather than
  row-local execution proof
- no new authoritative paper-grade bundle root was produced in this pass

## Follow-Up Work

- rerun the minimum-subset benchmark with the fixed workflow to create a fresh
  authoritative root whose TensorFlow rows carry distinct row-scoped logs
- revalidate that fresh root and promote it only if
  `benchmark_status=paper_complete`, every required row remains
  `paper_grade`, and `missing_fields_by_row` is empty
- treat the interrupted root `runs/minimum_subset_20260430T051928Z` as an
  incomplete follow-up artifact unless a later pass resumes or replaces it

## Residual Risks

- this backlog item is not currently closed as paper-grade evidence because the
  post-fix benchmark rerun has not finished
- older summaries and manifests that cite
  `runs/minimum_subset_20260430T035104Z` as `paper_complete` are historical and
  must not be used as the current evidence contract
- the required rerun remains a long-running GPU job whose completion still
  depends on the local `ptycho311` runtime stack and available device time
