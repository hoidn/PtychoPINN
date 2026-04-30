## Completed In This Pass

- fixed the implementation-review blocker by rejecting synthetic reused-row
  execution proofs and requiring completed invocation metadata plus real
  row-local `stdout.log` and `stderr.log` artifacts before a row can remain
  `paper_grade`
- updated fresh-run collation so TensorFlow and Torch rows now emit honest
  row-local logs and exit-code proof payloads, and TensorFlow row provenance now
  records completed invocation state instead of placeholder logs
- added regression coverage for the stricter provenance contract across
  `tests/studies/test_metrics_tables.py`,
  `tests/test_grid_lines_compare_wrapper.py`,
  `tests/studies/test_lines128_paper_benchmark.py`, and the existing workflow
  tests that caught and validated the follow-up `datetime` import fix
- discarded the earlier recovered-root claim for
  `runs/minimum_subset_20260429T235811Z` because its `--reuse-existing-recons`
  path fabricated row logs and exit-code proof data
- reran the minimum-subset benchmark from a fresh root
  `runs/minimum_subset_20260430T035104Z`, producing an honest completed bundle
  with row-local invocation, config, history, metrics, logs, and proof artifacts

## Completed Current-Scope Work

- authoritative root:
  `runs/minimum_subset_20260430T035104Z`
- execution path used:
  `fresh_rerun_after_review_fix`
- final bundle state:
  `benchmark_status=paper_complete`,
  `claim_boundary=minimum_draftable_cdi_subset`,
  `missing_bundle_artifacts=[]`
- required rows remain complete with empty `missing_fields_by_row`:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`
- required bundle artifacts now exist in the authoritative root:
  `metrics.json`, `model_manifest.json`, `paper_benchmark_manifest.json`,
  `dataset_identity_manifest.json`, `split_manifest.json`, `live_stdout.log`,
  `live_stderr.log`, `metrics_table.csv`, `metrics_table.tex`,
  `metrics_table_best.tex`, and `visuals/frc_curves.png`
- each required row now carries honest paper-grade provenance under `runs/*/`:
  `invocation.json`, `invocation.sh`, `config.json`, `history.json`,
  `metrics.json`, `stdout.log`, `stderr.log`, and `exit_code_proof.json`
- verification:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/test_grid_lines_workflow.py tests/torch/test_grid_lines_torch_runner.py`
    -> `252 passed, 53 warnings in 46.35s`
    (`verification/focused_pytest_review_fix_20260430b.log`)
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `171 passed, 47 warnings in 301.98s`
    (`verification/backlog_required_pytest_review_fix_20260430b.log`)
  - `python -m compileall -q ptycho_torch scripts/studies ptycho/workflows`
    -> exit `0`
    (`verification/compileall_review_fix_20260430b.log`)
  - fresh rerun command:
    `python scripts/studies/lines128_paper_benchmark.py --mode minimum_subset --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --execution-authority-note docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_execution_authority.md --execution-manifest .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/execution/benchmark_execution_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T035104Z`
    -> exit `0`
    (`verification/lines128_fresh_rerun_20260430T035104Z.log`)

## Follow-Up Work

- later complete-table CDI rows `pinn_spectral_resnet_bottleneck_net` and
  `pinn_ffno` remain intentionally out of scope for this minimum-subset item
- the stale root `runs/minimum_subset_20260429T235811Z` remains useful only as
  a historical failed-recovery artifact and should not be cited as paper-grade
  evidence in later summaries or manuscript tables

## Residual Risks

- this item still closes only the minimum draftable CDI subset, not the later
  complete `lines128` paper table
- reproduction still depends on the recorded local runtime stack
  (`ptycho311`, local GPU, local TF/Torch installs) even though the bundle now
  enforces honest runtime and provenance gates
- deleting row-local logs, invocation metadata, or required visuals from
  `runs/minimum_subset_20260430T035104Z` would correctly downgrade the bundle on
  the next validation pass
