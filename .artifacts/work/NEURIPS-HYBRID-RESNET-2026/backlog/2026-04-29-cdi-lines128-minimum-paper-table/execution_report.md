## Completed In This Pass

- closed the blocking paper-grade provenance hole in
  `scripts/studies/metrics_tables.py` by requiring `splits.manifest_json` to
  exist, load, and match the row contract for `nimgs_train`, `nimgs_test`, and
  `seed`
- tightened output-proof validation in
  `scripts/studies/metrics_tables.py` so `exit_code_proof.json` must bind to the
  validated row via matching `model_id`, `invocation_json`, `stdout_log`, and
  `stderr_log`
- added regression coverage in
  `tests/studies/test_metrics_tables.py` for the missing split-manifest path
  and for mismatched exit-code proof identity
- updated the synthetic minimum-subset fixture in
  `tests/studies/test_lines128_paper_benchmark.py` to emit the now-required
  split-manifest fields and row-bound exit-code proofs
- reran the focused selector and the required backlog checks, and archived fresh
  logs under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`

## Completed Current-Scope Work

- the implementation-review blocking defect is fixed: paper-grade promotion no
  longer accepts placeholder split provenance or an exit-code proof copied from
  another row
- the review’s follow-on validation gap is also narrowed for current scope:
  output-proof identity must now match the row being certified, not just any
  completed invocation with existing logs
- current verification evidence for this pass:
  - `pytest_focused_20260430T100751Z.log`
  - `pytest_required_20260430T100845Z.log`
  - `compileall_required_20260430T100845Z.log`

## Follow-Up Work

- later complete-table rows remain out of scope for this item:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- if future paper-grade validators add more provenance fields, the synthetic
  minimum-subset fixtures will need to keep tracking that contract explicitly

## Residual Risks

- the required test gates still emit the known non-fatal warning set from
  `tight_layout`, `skimage` SSIM, and FRC calculations
- this pass tightened row-level provenance validation only; it did not broaden
  the minimum CDI subset scope or relaunch benchmark roots
