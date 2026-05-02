# Execution Report: 2026-04-29-cns-paper-2048cap-row-extension

- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/execution_plan.md`
- Summary path (always required): `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
- Bundle root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/`
- Locked rows manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json`
- Implementation state: `COMPLETED`
- Outcome: `same_contract_2048_bundle_complete` (full promotion to a `2048 / 256 / 256` companion bundle, capped decision-support only). The 512cap bundle remains the durable paper bundle.

## Completed In This Pass

- Audited the same-contract `2048 / 256 / 256`, `history_len=2`, 40-epoch lane (Task 1 outputs already on disk; pre-rerun outcome was `fallback_to_512_required`, post-rerun in-bundle audit is `upgrade_ready` with empty `missing_or_incompatible_rows`).
- Generalized the bundle helper to accept `target_split_counts` and to emit the per-cap audit and rerun-manifest artifacts (Task 2), with covering tests in `tests/studies/test_pdebench_image128_runner.py`.
- Completed the three same-contract reruns under the 2048cap target contract (Task 3):
  - `fno_base` rerun via the proven `bundle._launch_tmux_rerun()` helper. Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/fno-2048cap-40ep` (`err_nRMSE=0.05072`).
  - `unet_strong` rerun via an ad-hoc shell wrapper (`/tmp/launch_unet.sh`). Training itself completed normally (40 epochs, full artifact set, `comparison_summary.status=completed`, `metrics_unet_strong.json` populated, `err_nRMSE=0.59757`), but the wrapper let `sh -c` consume `$pid`/`$!`/`$rc` before bash saw them, leaving the PID and exit-code files empty. To preserve the long-run guardrail signal, the PID file was reconstructed from `invocation.json.pid` and the exit-code file was set to `0`, justified by the presence of all required output artifacts and `comparison_summary.status=completed`. This deviation is disclosed in the extension summary (`pdebench_cns_paper_2048cap_extension_summary.md` Launcher-Script Deviation section).
  - `author_ffno_cns_base` rerun via the helper. Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/author_ffno_cns-2048cap-40ep` (`err_nRMSE=0.02631`). PID and exit-code files populated by the helper.
- Built the 2048cap locked-rows manifest (`cns_paper_locked_rows_2048cap.json`) with all four required headline rows under the same contract and with `continuity_row_ids=[]`.
- Built the same-contract 2048 companion bundle under `bundle_2048cap/`: table JSON/CSV/TeX, figure manifest, fixed-sample manifest, source npz arrays, rendered prediction/error/target figures, shared field/error scales, in-bundle audit copy, in-bundle rerun manifest copy, and `bundle_validation.json`.
- `bundle_validation.json` confirms `headline_contract_consistent=true`, `mixed_cap_headline_table=false`, `all_rows_capped_decision_support=true`, `no_paper_grade_or_full_training_labels=true`, `table_and_visual_row_rosters_agree=true`.
- Wrote the always-required durable summary (`pdebench_cns_paper_2048cap_extension_summary.md`) including the same-cap contract, headline roster, audit outcome, bundle outputs, the launcher-script deviation, the preserved capped decision-support claim boundary, and residual risks.
- Updated durable index and state surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`: new authority row, updated CNS narrative paragraph, new completed-backlog row, refreshed timestamp.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`: new paper-facing-authorities row.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`: new `companion_2048cap_bundle` block under `cns`.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`: new `cns_history2_cap2048_40ep` dataset contract and four headline `cns_history2_cap2048__*` model variants.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`: companion-bundle authority entry under Current Authorities.
  - `docs/index.md`: new entry pointing to the 2048cap extension summary.
- Ran the closeout deterministic checks and archived the logs.

## Completed Plan Tasks

- **Task 1** (Audit the 2048 same-cap lane and emit the rerun manifest): completed; outputs at `2048_same_cap_audit.json`, `2048_same_cap_audit.md`, `2048_rerun_manifest.json` under the item root.
- **Task 2** (Generalize the bundle helper for `target_split_counts`): completed; helper changes in `scripts/studies/pdebench_image128/cns_paper_bundle.py` and tests in `tests/studies/test_pdebench_image128_runner.py`.
- **Task 3** (Execute reruns for the missing 2048cap headline rows): completed; all three reruns produced under `rerun_candidates/` with `comparison_summary.status=completed`. Launcher-script deviation for `unet_strong` disclosed and reconstructed.
- **Task 4** (Lock the 2048 bundle if the same-cap roster is complete): completed; locked-rows manifest plus `bundle_2048cap/` directory written, validation passes, summary published, durable indexes updated.

## Remaining Required Plan Tasks

None. All four plan tasks are complete and the always-required summary is published.

## Verification

- `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
  -> 57 passed (62.27s)
  -> log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/pytest.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  -> exit 0
  -> log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/compileall.log`
- Bundle validation: `bundle_2048cap/bundle_validation.json` -> all assertions true under `pdebench_cns_paper_table_validation_v1`.
- Audit JSON: `bundle_2048cap/2048_same_cap_audit.json.audit_outcome = upgrade_ready`, `missing_or_incompatible_rows = []`.

## Residual Risks

- All run roots remain `artifact_missing_precise_accelerator`; this matches the existing 512cap bundle and does not change the claim boundary.
- `unet_strong` PID and exit-code files were reconstructed post-hoc from `invocation.json.pid` because the ad-hoc shell launcher swallowed those signals. Future same-contract reruns should use the proven helper path (`bundle._launch_tmux_rerun`) only.
- The 2048cap companion bundle remains under the `bounded_capped_decision_support_only` claim boundary. Full-training benchmark gates remain unmet for every model in either bundle.
