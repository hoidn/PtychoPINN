## Completed In This Pass

- fixed the remaining implementation-review provenance defect in
  `scripts/studies/paper_provenance.py` by rejecting stale current-root
  launcher logs that predate the wrapper invocation claiming completion
- updated `scripts/studies/grid_lines_compare_wrapper.py` so the wrapper root
  always writes fresh `launcher_stdout.log` / `launcher_stderr.log` artifacts
  and performs a post-completion launcher-evidence refresh after
  `invocation.json` is marked `completed`
- added regressions for:
  - stale current-root launcher-log rejection
  - root launcher-log emission from `grid_lines_compare_wrapper.main()`
  - post-completion `launcher_completion.json` materialization for reused Torch rows
- reran the focused harness selectors plus the mandatory backlog gate and
  compileall after the wrapper/provenance patch
- replayed the reuse-only compare-wrapper pass in tmux for the authoritative
  `supervised_ffno_extension_20260430T180217Z` root, which refreshed the root
  launcher logs and rewrote `runs/supervised_ffno/launcher_completion.json`
  with the current supervised FFNO eval markers

## Completed Current-Scope Work

- the authoritative adjacent extension root remains
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`
  and is now internally consistent again
- `runs/supervised_ffno/launcher_completion.json` in that root was rewritten at
  `2026-04-30T19:18:37Z` and now cites the current root
  `launcher_stdout.log` markers with
  `DEBUG eval_reconstruction [supervised_ffno]: amp_pred stats: mean=26.012983`
- current verification evidence for approval is archived at:
  - focused harness suite:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_focus_20260430_wrapper_provenance.log`
    Note: command result was `226 passed, 47 warnings in 44.93s`
  - required deterministic gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/pytest_20260430_supervised_equivalent_rows_final.log`
    Note: tmux-tracked PID `1524068` exited `0`; command result was
    `182 passed, 47 warnings in 303.60s`
  - compile gate:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/compileall_20260430_supervised_equivalent_rows_final.log`
  - authoritative root refresh:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/verification/supervised_ffno_bundle_20260430_root_refresh_final.log`

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
