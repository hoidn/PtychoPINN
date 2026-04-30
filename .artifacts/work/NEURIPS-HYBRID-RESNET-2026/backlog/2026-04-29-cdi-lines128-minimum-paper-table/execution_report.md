## Completed In This Pass

- fixed the review-blocking wrapper-finalization gap in
  `scripts/studies/lines128_paper_benchmark.py`: recovered Torch rows now gain
  `outputs.launcher_completion_json` only after the wrapper invocation is
  marked `completed`, and the finalizer rewrites `metrics.json`,
  `model_manifest.json`, and `paper_benchmark_manifest.json` from that
  refreshed evidence surface
- added a regression in
  `tests/studies/test_lines128_paper_benchmark.py` that reproduces the stale
  retained-root failure mode and fails unless post-completion finalization
  upgrades recovered Torch rows to a paper-grade-complete bundle
- reran the focused suite, required backlog pytest gate, and `compileall`,
  archiving the final evidence set under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/`:
  `pytest_focused_20260430T122925Z.log`,
  `pytest_required_20260430T122925Z.log`,
  `compileall_required_20260430T122925Z.log`
- regenerated the authoritative
  `runs/minimum_subset_20260430T084339Z` same-root bundle under a dedicated
  tmux-backed launcher contract so `metrics.json`, `model_manifest.json`, and
  `paper_benchmark_manifest.json` now all report `paper_complete` with no
  missing bundle artifacts
- archived the regeneration log under
  `lines128_same_root_bundle_regen_20260430T122925Z.log`

## Completed Current-Scope Work

- the blocking implementation-review defect is fixed: the authoritative
  retained root no longer contradicts itself about paper-grade status, and the
  recovered Torch rows now carry the launcher-completion evidence required by
  the current bundle validator
- the authoritative `084339Z` minimum-subset root is now honestly
  `paper_complete`; `metrics.json`, `model_manifest.json`, and
  `paper_benchmark_manifest.json` all agree on empty `missing_fields_by_row`
  and `missing_bundle_artifacts=[]`
- the current verification evidence for this pass is:
  `pytest_focused_20260430T122925Z.log`,
  `pytest_required_20260430T122925Z.log`,
  `compileall_required_20260430T122925Z.log`,
  `lines128_same_root_bundle_regen_20260430T122925Z.log`

## Follow-Up Work

- later complete-table rows remain out of scope for this item:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- if future recovery passes add new row-level evidence that depends on wrapper
  completion, keep it in the same post-completion finalization stage rather
  than trying to emit it during the initial bundle build
- if future recovery passes need numeric runtime provenance for TF rows, add a
  first-class row-runtime source instead of inferring it from wrapper-level or
  library-mode invocation timestamps

## Residual Risks

- the required test gates still emit the known non-fatal warning set from
  `tight_layout`, `skimage` SSIM, and FRC calculations
- recovered TF rows remain paper-grade on the basis of complete invocation,
  config, dataset, split, outputs, and visual provenance while runtime stays
  explicitly unavailable under recovery; this is honest for the current
  retained root but not a substitute for future fresh standalone TF reruns if a
  paper workflow later requires numeric TF runtime claims
- this pass fixed the retained-root consistency blocker only; it did not
  broaden the minimum CDI subset scope or launch later complete-table rows
