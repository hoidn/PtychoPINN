## Completed In This Pass

- hardened `scripts/studies/paper_provenance.py` so `write_exit_code_proof()`
  only emits proof when the referenced invocation records `exit_code == 0`
- added `write_launcher_completion_evidence()` and wired recovered Torch rows
  in `scripts/studies/grid_lines_compare_wrapper.py` to persist
  `runs/*/launcher_completion.json` when the authoritative evidence comes from
  a `--reuse-existing-recons` wrapper recovery pass
- tightened `scripts/studies/metrics_tables.py` so the paper-grade split
  validator now requires `gridsize` and `set_phi` to match the recorded split
  manifest alongside `nimgs_train`, `nimgs_test`, and `seed`
- tightened `scripts/studies/metrics_tables.py` again so recovered PyTorch rows
  under `reuse_existing_recons` must carry machine-readable launcher
  completion evidence, and fixed relative artifact resolution to prefer the
  current output root over unrelated cwd-relative files
- added regression coverage in
  `tests/studies/test_paper_provenance.py` and
  `tests/studies/test_metrics_tables.py` for the reviewed exit-code and
  split-contract failure modes
- updated the affected synthetic fixtures in
  `tests/studies/test_lines128_paper_benchmark.py` and
  `tests/test_grid_lines_compare_wrapper.py` so they emit the now-required
  split fields and completed-invocation metadata
- recorded companion human-readable Torch-row completion evidence for
  `runs/minimum_subset_20260430T084339Z` in
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/torch_row_exit_evidence_20260430T104510Z.md`
- regenerated the authoritative `084339Z` bundle in place so
  `metrics.json`, `model_manifest.json`, and `paper_benchmark_manifest.json`
  now consume the structured
  `runs/pinn_hybrid_resnet/launcher_completion.json` and
  `runs/pinn_fno_vanilla/launcher_completion.json` artifacts directly and
  report `paper_complete`
- updated the recovered-root audit, progress report, and durable summary to
  describe the shipped machine-readable launcher-completion contract rather
  than relying on the earlier prose-only explanation
- completed the remaining Task 6 handoff contract in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md`
  by recording the fixed `seed=3` policy and the required note that CDI `cnn`
  and PDEBench CNS `unet_strong` serve analogous local-baseline roles while
  remaining task-local, non-identical implementations
- reran the focused selector, required backlog pytest gate, and `compileall`,
  archiving fresh logs under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`

## Completed Current-Scope Work

- the implementation-review code blockers are fixed: future paper-grade
  promotion no longer accepts row-exit proof without a recorded zero exit code,
  split provenance now enforces the full fixed contract including
  `gridsize` and `set_phi`, and recovered Torch rows can no longer promote by
  prose alone because the bundle validator requires structured launcher
  completion evidence
- the artifact-level approval blocker is addressed for the current authoritative
  root because the root itself now carries machine-readable
  `launcher_completion.json` evidence for the two recovered Torch rows and the
  bundle manifests validate against it successfully
- the durable summary is now self-contained for downstream paper assembly on
  the reviewed points: it names the fixed comparator, fixed `seed=3` policy,
  and the required CDI `cnn` versus PDEBench CNS `unet_strong` labeling note
- current verification evidence for this pass:
  - `pytest_focused_20260430T1130Z.log`
  - `pytest_required_20260430T1130Z.log`
  - `compileall_required_20260430T1130Z.log`
  - `lines128_same_root_bundle_regen_20260430T084339Z_review_fix_final.log`
  - `torch_row_exit_evidence_20260430T104510Z.md`

## Follow-Up Work

- later complete-table rows remain out of scope for this item:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- if future paper-grade validators add more provenance fields, the synthetic
  minimum-subset fixtures will need to keep tracking that contract explicitly
- if downstream manifests need to distinguish fresh-rerun recovery from
  historical-artifact recovery more mechanically, extend the new recovered-row
  launcher-completion lineage into a first-class freshness/repair summary
- the existing `084339Z` root still contains historically repaired Torch
  row-local invocation files; the new `launcher_completion.json` artifacts are
  what make the retained root acceptable under the current review-fix decision

## Residual Risks

- the required test gates still emit the known non-fatal warning set from
  `tight_layout`, `skimage` SSIM, and FRC calculations
- the retained paper-grade claim for `084339Z` still depends on a documented
  cross-artifact evidence argument, but that dependency is now encoded
  machine-readably in the accepted root instead of living only in prose
- this pass tightened provenance validation and resolved the review blocker; it
  did not broaden the minimum CDI subset scope or launch later complete-table
  rows
