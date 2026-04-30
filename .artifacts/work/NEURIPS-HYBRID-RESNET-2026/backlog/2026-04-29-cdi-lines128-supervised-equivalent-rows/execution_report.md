## Completed In This Pass

- fixed the implementation-review correctness bug in
  `ptycho_torch/config_factory.py` by mapping training-payload
  `model_type` overrides onto `PTModelConfig.mode`
- fixed the real supervised-path crash in `ptycho_torch/model.py` by allowing
  `Ptycho_Supervised.forward(...)` to accept `experiment_ids`
- added targeted regressions for the supervised mode/loss handoff and the
  supervised FFNO loss path, then reran the required same-contract
  `supervised_ffno` row under the frozen `lines128` contract
- promoted the preserved `pinn_ffno` comparator into the corrected
  `170808Z` root, replayed the compare-wrapper recovery path in tmux, and
  rebuilt `metrics.json`, `model_manifest.json`,
  `paper_benchmark_manifest.json`, and
  `execution/supervised_ffno_parity_audit.json` from corrected artifacts

## Completed Current-Scope Work

- the row is now genuinely supervised:
  `lightning_logs/version_0/hparams.yaml` under
  `runs/supervised_ffno_extension_20260430T170808Z/` records
  `mode: Supervised` and `loss_function: MAE`
- the adjacent extension root is now truthfully `paper_complete` at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T170808Z`
- the stale parity claim is retired: the rebuilt comparison audit records
  `comparison_outcome: non_identical_same_contract_comparison`
- durable summary and studies-index entries now point at the corrected
  `170808Z` root instead of the superseded `160218Z` root
- archived verification for the completed scope is current:
  - targeted review regressions:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_targeted_20260430T173424Z.log`
  - focused selector:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focused_20260430T173446Z.log`
  - required deterministic gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_required_20260430T173548Z.log`
  - compile gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_required_20260430T174105Z.log`

## Follow-Up Work

- none is required for current-scope backlog approval
- if paper-facing interpretation needs a stronger supervised FFNO control row,
  that is new experimental scope rather than remaining implementation debt on
  this item

## Residual Risks

- the extension still reuses the accepted `pinn_ffno` comparator by promotion
  instead of rerunning it in this pass
- any manuscript, table, or downstream summary that still references the
  superseded `160218Z` root or the earlier exact-parity claim is now wrong and
  must be updated to the corrected `170808Z` root plus the rebuilt comparison
  audit
- verification logs still contain the known non-fatal `tight_layout`,
  `skimage` SSIM, and FRC warnings already present on related Lines128 study
  surfaces
