# Active Work

- fresh post-fix minimum-subset rerun still required before this backlog item
  can be promoted again as paper-grade evidence

# Current Status

- final state: `implementation_fixed_fresh_rerun_required`
- authoritative paper-grade root: none in this pass
- execution path used:
  `tf_row_log_fix_with_followup_rerun_required`
- review-fix note:
  the earlier historical `paper_complete` claim was invalid under the stricter
  validator because TensorFlow `baseline` and `pinn` reused duplicated shared
  workflow logs. The implementation now emits honest row-scoped TF logs and the
  validator rejects duplicated required-row stdout payloads, but a fresh bundle
  rerun is still required before the item can be promoted again.
- interrupted follow-up rerun:
  - root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T051928Z`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/lines128_fresh_rerun_tf_provenance_fix_20260430T051928Z.log`
  - state: `stopped_in_training_not_authoritative`
- archived verification logs for this pass:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/focused_pytest_tf_provenance_fix_20260429.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/backlog_required_pytest_tf_provenance_fix_20260429.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/verification/compileall_tf_provenance_fix_20260429.log`

# Next Resume Condition

- complete the fresh post-fix minimum-subset rerun and revalidate the resulting
  root before restoring any `paper_complete` claim
- later complete-table follow-up remains separate:
  `pinn_spectral_resnet_bottleneck_net` and `pinn_ffno`

# Blocker

- rerun still required for paper-grade re-promotion

# Blocker Class

- long-running benchmark follow-up
