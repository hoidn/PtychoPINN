## Completed In This Pass

- hardened `scripts/studies/paper_provenance.py` so `write_exit_code_proof()`
  only emits proof when the referenced invocation records `exit_code == 0`
- tightened `scripts/studies/metrics_tables.py` so the paper-grade split
  validator now requires `gridsize` and `set_phi` to match the recorded split
  manifest alongside `nimgs_train`, `nimgs_test`, and `seed`
- added regression coverage in
  `tests/studies/test_paper_provenance.py` and
  `tests/studies/test_metrics_tables.py` for the reviewed exit-code and
  split-contract failure modes
- updated the affected synthetic fixtures in
  `tests/studies/test_lines128_paper_benchmark.py` and
  `tests/test_grid_lines_compare_wrapper.py` so they emit the now-required
  split fields and completed-invocation metadata
- recorded independent persisted Torch-row completion evidence for
  `runs/minimum_subset_20260430T084339Z` in
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/torch_row_exit_evidence_20260430T104510Z.md`
- updated the recovered-root audit and durable summary to cite the independent
  launcher evidence note rather than relying on the repaired row-local Torch
  invocation/proof pair alone
- reran the focused selector, required backlog pytest gate, and `compileall`,
  archiving fresh logs under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`

## Completed Current-Scope Work

- the implementation-review code blockers are fixed: future paper-grade
  promotion no longer accepts row-exit proof without a recorded zero exit code,
  and split provenance now enforces the full fixed contract including
  `gridsize` and `set_phi`
- the artifact-level approval blocker is addressed for the current authoritative
  root by attaching independent persisted launcher evidence for the two Torch
  rows, satisfying the review’s alternative to a fresh rerun
- current verification evidence for this pass:
  - `pytest_focused_20260430T104510Z.log`
  - `pytest_required_20260430T103703Z.log`
  - `compileall_required_20260430T103703Z.log`
  - `torch_row_exit_evidence_20260430T104510Z.md`

## Follow-Up Work

- later complete-table rows remain out of scope for this item:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- if future paper-grade validators add more provenance fields, the synthetic
  minimum-subset fixtures will need to keep tracking that contract explicitly
- the existing `084339Z` root still contains historically repaired Torch
  row-local invocation files; the new launcher-evidence note is what makes the
  retained root acceptable under the current review-fix decision

## Residual Risks

- the required test gates still emit the known non-fatal warning set from
  `tight_layout`, `skimage` SSIM, and FRC calculations
- the retained paper-grade claim for `084339Z` now depends on a documented
  cross-artifact evidence argument, not on a regenerated fresh four-row root
- this pass tightened provenance validation and resolved the review blocker; it
  did not broaden the minimum CDI subset scope or launch later complete-table
  rows
