## Completed In This Pass

- reconfirmed that the checked-in launch authority, readiness-only harness
  note, and execution JSON still agree on the frozen four-row contract,
  comparator, seed policy, sample IDs, and shared visual-scale policy
- retained the narrow validator change in `scripts/studies/metrics_tables.py`
  and its regression coverage in `tests/studies/test_metrics_tables.py`, which
  keeps shared TensorFlow diagnostic logs admissible when structured row
  provenance is complete
- reran the plan-required deterministic verification gates and archived fresh
  logs for this completion pass
- revalidated the accepted root `runs/minimum_subset_20260430T035104Z` as the
  authoritative minimum-table bundle because its bundle manifests still report
  `paper_complete`, `claim_boundary=minimum_draftable_cdi_subset`, and empty
  `missing_fields_by_row`

## Completed Plan Tasks

- Task 1: authority alignment and recovery-path audit remain satisfied; the
  accepted root is `runs/minimum_subset_20260430T035104Z`
- Task 2: no additional code change was required in this pass beyond the
  existing narrow validator/test update already present in the checkout
- Task 3: reran the focused selector, required backlog selector, and compile
  gate; all passed and were archived under this item's verification root
- Task 4/5: no new write-side recovery or fresh rerun was required because the
  accepted root already contains the complete four-row paper bundle
- Task 6: the durable summary and audit remain aligned to the accepted root and
  current evidence contract

## Remaining Required Plan Tasks

- none for backlog item `2026-04-29-cdi-lines128-minimum-paper-table`
- later complete-table rows remain intentionally out of scope here:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`

## Verification

- `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  -> `200 passed, 47 warnings in 41.27s`
  (`verification/focused_pytest_final_20260430.log`)
- `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  -> `172 passed, 47 warnings in 301.72s (0:05:01)`
  (`verification/backlog_required_pytest_final_20260430.log`)
- `python -m compileall -q ptycho_torch scripts/studies`
  -> exit `0`
  (`verification/compileall_final_20260430.log`)
- accepted evidence root check:
  `runs/minimum_subset_20260430T035104Z` contains `metrics.json`,
  `metric_schema.json`, `model_manifest.json`, `metrics_table.csv`,
  `metrics_table.tex`, `metrics_table_best.tex`, fixed-sample visuals,
  `frc_curves.png`, and manifests reporting `paper_complete` with empty
  `missing_fields_by_row`

## Residual Risks

- diagnostic logs remain useful for debugging, but their uniqueness is not part
  of the current paper-grade evidence contract
- the interrupted follow-up root `runs/minimum_subset_20260430T051928Z` must
  not be cited as authoritative minimum-table evidence
