## Completed In This Pass

- fixed recovered TensorFlow runtime provenance in
  `scripts/studies/grid_lines_compare_wrapper.py` so same-root recovery now
  reads the row-local invocation instead of the top-level bundle-regeneration
  invocation and explicitly marks runtime as unavailable when the recovered TF
  row only records wrapper-library completion
- added a regression in `tests/test_grid_lines_compare_wrapper.py` that fails
  if recovered TF rows ever inherit `command_wall_time_sec` from the wrapper
  repair pass again
- reran the focused selector, required backlog pytest gate, and `compileall`,
  archiving fresh logs under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`:
  `pytest_focused_20260430T115123Z.log`,
  `pytest_required_20260430T115123Z.log`,
  `compileall_required_20260430T115123Z.log`
- regenerated the authoritative
  `runs/minimum_subset_20260430T084339Z` same-root bundle with
  `--reuse-existing-recons`, then refreshed the retained root’s launcher
  contract and structured Torch output references so
  `metrics.json`, `model_manifest.json`, and
  `paper_benchmark_manifest.json` return to `paper_complete` with the corrected
  TF runtime summaries
- archived the regeneration logs under
  `lines128_same_root_bundle_regen_20260430T115123Z_runtime_fix.log` and
  `lines128_same_root_bundle_regen_20260430T115123Z_runtime_fix_tmux.log`

## Completed Current-Scope Work

- the blocking implementation-review defect is fixed: recovered TF rows no
  longer certify wrapper bundle-repair duration as row runtime provenance
- the authoritative `084339Z` minimum-subset root is again `paper_complete`
  after same-root regeneration, and both required recovered TF rows now report
  `runtime_source=unavailable_under_recovery` with no misleading numeric
  `command_wall_time_sec`
- the current verification evidence for this pass is:
  `pytest_focused_20260430T115123Z.log`,
  `pytest_required_20260430T115123Z.log`,
  `compileall_required_20260430T115123Z.log`,
  `lines128_same_root_bundle_regen_20260430T115123Z_runtime_fix.log`,
  `lines128_same_root_bundle_regen_20260430T115123Z_runtime_fix_tmux.log`

## Follow-Up Work

- later complete-table rows remain out of scope for this item:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- if future recovery passes need numeric runtime provenance for TF rows, add a
  first-class row-runtime source instead of inferring it from wrapper-level or
  library-mode invocation timestamps
- if the launcher-completion contract should survive one-off local reruns more
  directly, move the post-run attachment of `launcher_completion.json` outputs
  into a dedicated finalization step that runs after the wrapper invocation is
  fully recorded

## Residual Risks

- the required test gates still emit the known non-fatal warning set from
  `tight_layout`, `skimage` SSIM, and FRC calculations
- recovered TF rows remain paper-grade on the basis of complete invocation,
  config, dataset, split, outputs, and visual provenance while runtime stays
  explicitly unavailable under recovery; this is honest for the current
  retained root but not a substitute for future fresh standalone TF reruns if a
  paper workflow later requires numeric TF runtime claims
- this pass fixed the runtime-provenance blocker only; it did not broaden the
  minimum CDI subset scope or launch later complete-table rows
