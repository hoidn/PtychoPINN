## Completed In This Pass

- fixed the implementation-review correctness bug in `ptycho_torch/model.py`
  by allowing supervised Lightning runs to use the configured generator
  module instead of always instantiating the legacy `Autoencoder`
- passed the selected generator and output mode through
  `PtychoPINN_Lightning(... mode='Supervised' ...)`, so `supervised_ffno`
  now executes the real FFNO supervised path
- fixed `scripts/studies/grid_lines_compare_wrapper.py` so the wrapper script
  bootstraps repository imports from the repo root and current-root Torch-row
  recovery can materialize missing `stdout.log`, `stderr.log`, and
  `exit_code_proof.json` artifacts for direct-runner rows
- added targeted regressions for the supervised FFNO generator wiring, fresh
  current-root Torch-row recovery semantics, recovered row-log enrichment, and
  direct script import bootstrap; reran the required same-contract
  `supervised_ffno` row under the frozen `lines128` contract
- replayed the reuse-only compare-wrapper pass in tmux for the corrected
  `180217Z` root and rebuilt `metrics.json`, `model_manifest.json`,
  `paper_benchmark_manifest.json`, and
  `execution/supervised_ffno_parity_audit.json` from the corrected artifacts

## Completed Current-Scope Work

- the row is now genuinely supervised:
  `lightning_logs/version_0/hparams.yaml` under
  `runs/supervised_ffno_extension_20260430T180217Z/` records
  `mode: Supervised`, `architecture: ffno`, and `generator_output: real_imag`
- the authoritative adjacent extension root is now truthfully
  `paper_complete` at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`
- the rebuilt supervised row is `paper_grade` with no missing bundle fields;
  `model_manifest.json` and `paper_benchmark_manifest.json` at the corrected
  root now both validate cleanly
- the stale parity claim is retired: the rebuilt comparison audit records
  `comparison_outcome: non_identical_same_contract_comparison`
- durable summary, studies-index, execution-manifest, and protocol-audit
  entries now point at the corrected `180217Z` root
- archived verification for the completed scope is current:
  - supervised rerun launch:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_launch_20260430T180217Z.log`
  - reuse-only wrapper regeneration:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430T180217Z.log`
  - targeted regressions:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_targeted_20260430_supervised_equivalent_rows.log`
  - required deterministic gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_20260430_supervised_equivalent_rows.log`
  - compile gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_20260430_supervised_equivalent_rows.log`

## Follow-Up Work

- none is required for current-scope backlog approval
- if paper-facing interpretation needs a stronger supervised FFNO control row,
  that is new experimental scope rather than remaining implementation debt on
  this item

## Residual Risks

- the extension still reuses the accepted `pinn_ffno` comparator by promotion
  instead of rerunning it in this pass
- any manuscript, table, or downstream summary that still references the
  superseded `170808Z` root or the earlier exact-parity claim is now wrong
  and must be updated to the corrected `180217Z` root plus the rebuilt
  comparison audit
- verification logs still contain the known non-fatal `tight_layout`,
  `skimage` SSIM, and FRC warnings already present on related Lines128 study
  surfaces
