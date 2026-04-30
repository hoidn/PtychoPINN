## Completed In This Pass

- resolved the remaining review-fix correctness gap in
  `scripts/studies/grid_lines_torch_runner.py` by persisting `exit_code` in
  torch row invocation artifacts on both success and failure
- resolved the wrapper-finalization race in
  `scripts/studies/lines128_paper_benchmark.py` by recomputing bundle status
  after the top-level invocation is marked `completed`
- added regression coverage for the torch runner exit-code contract, the
  compare-wrapper top-level exit-code contract, and the minimum-subset
  wrapper-finalization path after invocation completion
- repaired the completed fresh rerun root
  `runs/minimum_subset_20260430T084339Z` in place by adding `exit_code: 0`
  only to the two completed torch row invocation JSONs whose matching
  `exit_code_proof.json` already recorded successful completion
- reran the same-root minimum-subset bundle regeneration under tmux with the
  tracked launched PID `1367035`; the process exited `0` and rewrote the root
  manifests to `paper_complete`
- reran the focused selector, mandatory backlog selector, and `compileall`
  gates and archived fresh logs for this pass

## Completed Current-Scope Work

- blocking implementation-review findings are closed for the approved minimum
  CDI subset:
  - wrapper launcher provenance is now required and republished after wrapper
    completion
  - TensorFlow and Torch rows now share a replayable seed/exit-code provenance
    contract that the validator enforces
  - regression coverage now exercises the repaired wrapper and row invocation
    contracts
- the authoritative minimum-table evidence root is now
  `runs/minimum_subset_20260430T084339Z`
- that root currently reports:
  - `benchmark_status=paper_complete`
  - `missing_bundle_artifacts=[]`
  - empty `missing_fields_by_row` for `baseline`, `pinn`,
    `pinn_hybrid_resnet`, and `pinn_fno_vanilla`
  - presence of `metrics.json`, `metric_schema.json`, `model_manifest.json`,
    `metrics_table.csv`, `metrics_table.tex`, `metrics_table_best.tex`,
    `visuals/amp_phase_gt.png`, `visuals/compare_amp_phase.png`, and
    `visuals/frc_curves.png`
- the durable summary, audit, progress note, and docs index now point at the
  repaired `084339Z` root instead of the earlier superseded rerun

## Follow-Up Work

- later complete-table rows remain intentionally out of scope for this item:
  `pinn_spectral_resnet_bottleneck_net`, `pinn_ffno`
- any later paper-evidence packaging work should treat
  `runs/minimum_subset_20260430T084339Z` as the authoritative minimum CDI
  subset source root
- the stale `035104Z` rerun and stopped `051928Z` follow-up root should remain
  historical only and must not be cited as the current minimum-table evidence

## Residual Risks

- verification still emits the known non-fatal warning set from
  `tight_layout`, `skimage` SSIM, and FRC calculations; the required tests pass
  despite those warnings
- the same-root repair path depended on honest pre-existing row completion
  evidence (`exit_code_proof.json`) for the two Torch rows; future launcher
  regressions would need the same contract rechecked before root promotion
- this item establishes only the minimum draftable CDI subset, not the later
  complete `lines128` table or `/home/ollie/Documents/neurips/` evidence bundle
