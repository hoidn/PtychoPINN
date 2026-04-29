# Execution Report

## Completed In This Pass

- Fixed the review gaps in
  `scripts/studies/pdebench_image128/cns_paper_bundle.py`:
  - missing `1024` headline rows now emit `pilot`-mode rerun commands, so
    successful reruns can satisfy the existing
    `capped_decision_support_only` authority contract instead of being trapped
    in `smoke_feasibility_only`
  - expensive rerun launch is now gated by the plan-required
    `pytest`/`compileall` preflight commands, with durable evidence written to
    `verification/rerun_preflight_checks.json`
  - `bundle_validation.json` now records the accepted table roster, the visual
    bundle roster, and explicit agreement checks between the table payload,
    `fixed_sample_manifest.json`, `figure_manifest.json`, and the rendered
    figure entries
- Added focused regression coverage in
  `tests/studies/test_pdebench_image128_runner.py` for:
  - successful post-rerun `1024` audit promotion through a real tiny
    `pilot`-mode rerun
  - rejection of non-authoritative matching `1024` roots
  - enforced pre-rerun verification gating
  - durable bundle-validation roster agreement
- Rebuilt the real CNS bundle artifact root with the repaired code so the
  checked-out artifacts now include the refreshed
  `1024_same_cap_audit.{json,md}`, `1024_rerun_manifest.json`,
  `bundle_validation.json`, and fresh verification logs.

## Completed Current-Scope Work

- High review item resolved: the approved rerun-capable `1024` upgrade path now
  exists end-to-end in code under the approved capped contract. The audit can
  remain `upgrade_ready`, advance to `upgrade_ready_after_reruns`, or fall back
  cleanly to `512`, and the rerun path now points at a mode that can become
  authoritative after completion.
- Medium review item resolved: the only path that can spend GPU time now
  enforces the required local verification gate before launch instead of
  relying on operator discipline.
- Medium review item resolved: the durable validation payload now proves row
  roster, claim-boundary, and figure/table agreement inside the bundle root
  rather than relying on an external verification note.
- The authoritative checked-in bundle still stays on the locked
  `512 / 64 / 64` lane because this pass fixed the implementation and refreshed
  the fallback audit, but did not launch the expensive real `1024` reruns.

## Follow-Up Work

- Launch the missing same-contract `1024 / 128 / 128` reruns for
  `fno_base`, `unet_strong`, and `author_ffno_cns_base` when compute budget is
  available, then refresh the contract decision and row-lock authority only if
  those reruns finish under one contract.
- If a later pass uses `--execute-missing-1024-reruns` against real data,
  preserve the emitted tmux/PID/exit-code evidence in the same bundle root
  before attempting any `1024` authority widening.

## Verification

- `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/verification/pytest_required.log`
  Result: `51 passed in 65.23s`
- `pytest -q tests/studies/test_pdebench_image128_visualization.py`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/verification/pytest_visualization.log`
  Result: `6 passed in 0.96s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/verification/compileall.log`
  Result: exit `0`
- `python scripts/studies/pdebench_image128/cns_paper_bundle.py --locked-rows-path .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle`
  Log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/verification/bundle_run.log`
  Result: exit `0`
- Bundle artifact verification log:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/verification/bundle_verify.log`
  Result: `sample_ids=[0]`, `figure_entries=44`,
  `validation_benchmark_status=paper_complete`,
  `validation_table_visual_agreement=True`,
  `validation_sample_figure_agreement=True`,
  `validation_figure_entry_agreement=True`,
  `audit_outcome=fallback_to_512_required`

## Residual Risks

- The bundle is still bounded `capped_decision_support_only` evidence, not a
  same-protocol full-training or `paper_grade` CNS benchmark.
- The preferred `1024 / 128 / 128` headline lane still lacks same-contract
  `fno_base`, `unet_strong`, and `author_ffno_cns_base` rows; the refreshed
  audit now emits a durable rerun manifest for those gaps, but the real reruns
  themselves were not executed in this pass.
- The tmux/PID-tracked rerun executor is covered by unit tests and artifact
  validation logic, but it has not yet been exercised against a real long GPU
  run in this pass.
- Reused run roots still lack standalone repo git SHA, dirty-state, run-log,
  exit-code, and precise accelerator artifacts, so the table records explicit
  missing-hardware notes instead of guessing provenance.
