# Execution Report: 2026-04-29-cns-paper-2048cap-row-extension

- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/execution_plan.md`
- Summary path (always required): `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
- Bundle root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/`
- Locked rows manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json`
- Implementation state: `COMPLETED`
- Outcome: `same_contract_2048_bundle_complete` (full promotion to the `2048 / 256 / 256` capped CNS authority bundle). The 512cap bundle is preserved as historical provenance only.

## Completed In This Pass

- Synced the remaining Task 4 authority surfaces so the promoted `2048 / 256 / 256` bundle is now the current discoverability and audit authority across `paper_evidence_manifest.json`, `paper_evidence_package_audit_summary.md`, `paper_evidence_index.md`, `evidence_matrix.md`, `model_variant_index.json`, `docs/index.md`, and this execution report.
- Replaced the invalid `unet_strong` provenance with a fresh helper-launched rerun at `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/unet_strong-2048cap-40ep`, verified under PID `1845354` and exit code `0`, with proof at `verification/unet_rerun_replacement_verification.json`.
- Fixed the bundle-helper freshness verifier so helper-tracked reruns are not rejected when the launch marker content is a few microseconds newer than the marker/pid file mtimes. Regression coverage was added in `tests/studies/test_pdebench_image128_runner.py`.
- Regenerated the required item-root audit artifacts so `2048_same_cap_audit.json.audit_outcome = upgrade_ready` and `2048_rerun_manifest.json.rerun_candidates = []`.
- Refreshed `cns_paper_locked_rows_2048cap.json`, `bundle_2048cap/`, and `pdebench_cns_paper_2048cap_extension_summary.md` so the promoted 2048cap bundle reflects the verified replacement `unet_strong` metrics (`err_nRMSE=0.64315`) and no longer relies on reconstructed PID/exit-code evidence.

## Completed Current-Scope Work

- Closed the review’s first blocking issue by bringing the required item-root audit/manifests into sync with the delivered 2048cap state.
- Closed the review’s second blocking issue by replacing the reconstructed `unet_strong` launch evidence with a valid helper-tracked rerun and by fixing the verifier bug exposed during that rerun.
- Closed the remaining Task 4 authority-sync gap by repointing the durable discoverability surfaces to the complete 2048cap bundle while preserving the 512cap bundle as historical provenance.
- Preserved the approved scope boundary: the promoted 2048cap bundle remains capped decision-support only, and the older 512cap bundle remains available as historical fallback evidence.

## Follow-Up Work

- Rename the still-1024-specific execution surface (`execute_missing_1024_reruns` / `--execute-missing-1024-reruns`) so the helper API no longer advertises a misleading 1024-only contract.

## Verification

- `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
  -> 58 passed in 61.93s
  -> log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/pytest.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  -> exit 0
  -> log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/compileall.log`
- Helper verification: `verification/unet_rerun_replacement_verification.json`
  -> PID `1845354`, exit code `0`
- Bundle validation: `bundle_2048cap/bundle_validation.json` -> all assertions true under `pdebench_cns_paper_table_validation_v1`.
- Audit JSON: item-root and in-bundle `2048_same_cap_audit.json.audit_outcome = upgrade_ready`, `missing_or_incompatible_rows = []`.

## Residual Risks

- All run roots remain `artifact_missing_precise_accelerator`; this matches the existing 512cap bundle and does not change the claim boundary.
- The promoted 2048cap authority bundle remains under the `bounded_capped_decision_support_only` claim boundary. Full-training benchmark gates remain unmet for every model in either bundle.
